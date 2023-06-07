import torch
import argparse
import os
import yaml
import pathlib
import pickle
import logging
import sys
import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torchvision
import glob
from torch.serialization import default_restore_location
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import numpy as np

import utils
import models 

from utils.data_helpers.load_datasets_helpers import *
from utils.meters import *
from utils.progress_bar import *
from utils.noise_model import get_noise
from utils.metrics import ssim,psnr
from utils.util_calculate_psnr_ssim import calculate_psnr,calculate_ssim
from utils.test_metrics import *



def load_model(args):
    USE_CUDA = True
    device = torch.device('cuda') if (torch.cuda.is_available() and USE_CUDA) else torch.device('cpu')
    
    checkpoint_path = glob.glob(args.output_dir +'/unet*')
    if len(checkpoint_path) != 1:
        raise ValueError("There is either no or more than one model to load")
    checkpoint_path = pathlib.Path(checkpoint_path[0] + f"/checkpoints/checkpoint_{args.restore_mode}.pt")
    state_dict = torch.load(checkpoint_path, map_location=lambda s, l: default_restore_location(s, "cpu"))
    args = argparse.Namespace(**{ **vars(state_dict["args"]), "no_log": True})

    model = models.unet_fastMRI(
        in_chans=args.in_chans,
        chans = args.chans,
        num_pool_layers = args.num_pool_layers,
        drop_prob = 0.0,
        residual_connection = args.residual,
    ).to(device)
    model.load_state_dict(state_dict["model"][0])
    model.eval()
    return model

def cli_main_test(args):
    USE_CUDA = True
    device = torch.device('cuda') if (torch.cuda.is_available() and USE_CUDA) else torch.device('cpu')
    
    model = load_model(args)
    
    # evaluate test performance over following noise range
    noise_std_range = np.linspace(args.test_noise_std_min, args.test_noise_std_max, 
                                  ((args.test_noise_std_max-args.test_noise_std_min)//args.test_noise_stepsize)+1,dtype=int)/255.
    
    metrics_path = os.path.join(args.output_dir, args.test_mode + '_' + str(args.test_noise_std_min)+'-'+str(args.test_noise_std_max)+f'_metrics_{args.restore_mode}.p')
    
    metrics_dict = metrics_avg_on_noise_range(model, args, noise_std_range, device = device)
    pickle.dump( metrics_dict, open(metrics_path, "wb" ) )

def cli_main(args):
    available_models = glob.glob(f'{args.output_dir}/*')
        
    if not args.resume_training and available_models:
        raise ValueError('There exists already a trained model and resume_training is set False')
    if args.resume_training: 
        f_restore_file(args)
        
    # reset the attributes of the function save_checkpoint
    mode = "max"
    default_score = float("inf") if mode == "min" else float("-inf")
    utils.save_checkpoint.best_score =  default_score
    utils.save_checkpoint.best_step = -1
    utils.save_checkpoint.best_epoch = -1
    utils.save_checkpoint.last_step = -1
    utils.save_checkpoint.current_lr = args.lr
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Set the name of the directory for saving results 
    utils.setup_experiment(args) 
    
    utils.init_logging(args)
    
    # Build data loaders, a model and an optimizer
    model = models.unet_fastMRI(
        in_chans=args.in_chans,
        chans = args.chans,
        num_pool_layers = args.num_pool_layers,
        drop_prob = 0.0,
        residual_connection = args.residual,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=args.lr_gamma, patience=args.lr_patience, 
        threshold=args.lr_threshold, threshold_mode='abs', cooldown=0, 
        min_lr=args.lr_min, eps=1e-08, verbose=True
    )
    
    
    logging.info(f"Built a model consisting of {sum(p.numel() for p in model.parameters()):,} parameters")

    trainset = ImagenetSubdataset(args.train_size,args.path_to_ImageNet_train,mode='train',patch_size=args.patch_size,val_crop=args.val_crop)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True,generator=torch.Generator().manual_seed(args.seed))
    
    valset = ImagenetSubdataset(args.val_size,args.path_to_ImageNet_train,mode='val',patch_size=args.patch_size,val_crop=args.val_crop)
    val_loader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True,generator=torch.Generator().manual_seed(args.seed))
    
    print(optimizer.param_groups[0]["lr"])
    if args.resume_training:
        state_dict = utils.load_checkpoint(args, model, optimizer, scheduler)
        global_step = state_dict['last_step']
        start_epoch = int(state_dict['last_step']/(len(train_loader)))+1
        start_decay = True
    elif args.no_annealing:
        global_step = -1
        start_epoch = 0
        start_decay = True
    else:
        global_step = -1
        start_epoch = 0
        start_decay = False
    
    print(optimizer.param_groups[0]["lr"])
    args.log_interval = min(len(trainset), 100) # len(train_loader)=log once per epoch
    args.no_visual = False # True for not logging to tensorboard
    
    # Track moving average of loss values

    train_meters = {"train_loss":RunningAverageMeter(0.98)}
    valid_meters = {name: AverageMeter() for name in (["valid_psnr", "valid_ssim", "valid_psnr_self_supervised", "valid_ssim_self_supervised"])}
    # Create tensorflow event file
    writer = SummaryWriter(log_dir=args.experiment_dir) if not args.no_visual else None
    
    break_counter = 0
    # store the best val performance from lr-interval before the last lr decay
    best_val_last = 0
    # track the best val performance for the current lr-inerval
    best_val_current = 0
    # count for how many lr intervals there was no improvement and break only if there was no improvement for 2
    lr_interval_counter = 0
    # if best_val_current at the end of the current lr interval is smaller than best_val_last we perform early stopping
    
    for epoch in range(start_epoch, args.num_epochs):
        start = time.process_time()
        train_bar = ProgressBar(train_loader, epoch)
        # At beginning of each epoch reset the train meters
        for meter in train_meters.values():
            meter.reset()
        
        for inputs, noise_seed in train_bar:
            model.train() #Sets the module in training mode.

            global_step += 1
            inputs = inputs.to(device)
            
            noise = get_noise(inputs,noise_seed, fix_noise = args.fix_noise, noise_std = args.noise_std/255.)
            noisier_noise = get_noise(inputs,torch.mul(noise_seed,10),fix_noise = args.fix_noisier, noise_std = args.noise_std_noisier/255.)
                        

            noisy_inputs = noise + inputs
            noisier_image = noisy_inputs + noisier_noise
            outputs = model(noisier_image)
            loss = F.mse_loss(outputs, noisy_inputs, reduction="sum") / torch.prod(torch.tensor(inputs.size())) #(inputs.size(0) * 2)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            train_meters["train_loss"].update(loss.item())
            train_bar.log(dict(**train_meters, lr=optimizer.param_groups[0]["lr"]), verbose=True)

        # Add to tensorflow event file:
        if writer is not None:
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("loss/train", train_meters["train_loss"].avg, global_step)
            sys.stdout.flush()
                
            

        if epoch % args.valid_interval == 0:
            model.eval()
            gen_val = torch.Generator()
            gen_val = gen_val.manual_seed(10)
            for meter in valid_meters.values():
                meter.reset()

            valid_bar = ProgressBar(val_loader)
            for sample, noise_seed in valid_bar:

                with torch.no_grad():
                    sample = sample.to(device)
                    
                    # Self-supervised validation with fixed noise 
                    noise_self_supervised = get_noise(sample,noise_seed, fix_noise = args.fix_noise, noise_std = args.noise_std/255.)
                    noisier_noise = get_noise(sample,torch.mul(noise_seed,10),fix_noise = args.fix_noisier, noise_std = args.noise_std_noisier/255.)
                    
                    noisy_input_fixed = sample + noise_self_supervised
                    noisier_input = noisy_input_fixed + noisier_noise

                    model_output = model(noisier_input)
                    prediction = 2*model_output - noisier_input

                    valid_psnr_self_supervised = psnr(model_output, noisy_input_fixed)
                    valid_ssim_self_supervised = ssim(model_output, noisy_input_fixed)
                    valid_meters["valid_psnr_self_supervised"].update(valid_psnr_self_supervised.item())
                    valid_meters["valid_ssim_self_supervised"].update(valid_ssim_self_supervised.item())

                    # Ground truth validation wit fixed noise

                    # It uses the same input and output as in the self-supervised case since the noise seed is fixed
                    valid_psnr = psnr(prediction, sample) 
                    valid_ssim = ssim(prediction, sample)
                    valid_meters["valid_psnr"].update(valid_psnr.item())
                    valid_meters["valid_ssim"].update(valid_ssim.item())

            if writer is not None:
                # Average is correct valid_meters['valid_psnr'].avg since .val would be just the psnr of last sample in val set.
                writer.add_scalar("psnr/valid", valid_meters['valid_psnr'].avg, global_step)
                writer.add_scalar("ssim/valid", valid_meters['valid_ssim'].avg, global_step)
                
                writer.add_scalar("psnr_selfsupervised/valid", valid_meters['valid_psnr_self_supervised'].avg, global_step)
                writer.add_scalar("ssim_selfsupervised/valid", valid_meters["valid_ssim_self_supervised"].avg, global_step)
                
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)
                sys.stdout.flush()
            
            if args.val_flag == 0: # if we do self-supervised validation
                val_loss = valid_meters["valid_psnr_self_supervised"].avg 
            else: # if we do supervised validation
                val_loss = valid_meters["valid_psnr"].avg

            
            if utils.save_checkpoint.best_score < val_loss and not start_decay:                
                utils.save_checkpoint(args, global_step, epoch, model, optimizer, score=val_loss, mode="max")
                current_lr = utils.save_checkpoint.current_lr
                optimizer.param_groups[0]["lr"] = current_lr*args.lr_beta
                utils.save_checkpoint.current_lr = current_lr*args.lr_beta
                annealing_counter = 0
            elif not start_decay:
                annealing_counter += 1
                current_lr = utils.save_checkpoint.current_lr
                if annealing_counter == args.lr_patience_annealing:
                    
                    available_models = glob.glob(f'{args.output_dir}/*')
                    if not available_models:
                        raise ValueError('No file to restore')
                    elif len(available_models)>1:
                        raise ValueError('Too many files to restore from')
                        
                    model_path = os.path.join(available_models[0], "checkpoints/checkpoint_best.pt")    
                    state_dict = torch.load(model_path, map_location=lambda s, l: default_restore_location(s, "cpu"))
                    model = [model] if model is not None and not isinstance(model, list) else model
                    for m, state in zip(model, state_dict["model"]):
                        m.load_state_dict(state)
                    model = model[0]
                    
                    optimizer.param_groups[0]["lr"] = current_lr/(args.lr_beta*args.inital_decay_factor)
                    start_decay = True
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='max', factor=args.lr_gamma, patience=args.lr_patience, 
                        threshold=args.lr_threshold, threshold_mode='abs', cooldown=0, 
                        min_lr=args.lr_min, eps=1e-08, verbose=True
                    )
            else:
                utils.save_checkpoint(args, global_step, epoch, model, optimizer, score=val_loss, mode="max")
                current_lr = optimizer.param_groups[0]["lr"]
                if val_loss > best_val_current:
                    best_val_current = val_loss
                
            
        
        if writer is not None:            
            writer.add_scalar("epoch", epoch, global_step)
            sys.stdout.flush()
        
        if start_decay:
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]["lr"]
            
            #At every lr decay check if the model did not improve during the current or the previous lr interval and break if it didn't.
            if new_lr < current_lr: 
                if best_val_current < best_val_last and lr_interval_counter==1:
                    logging.info('Break training due to convergence of val loss!')
                    break
                elif best_val_current < best_val_last and lr_interval_counter==0:
                    lr_interval_counter += 1
                    logging.info('Do not yet break due to convergence of val loss!')
                else:
                    best_val_last = best_val_current
                    best_val_current = 0
                    lr_interval_counter = 0
            
        end = time.process_time() - start
        logging.info(train_bar.print(dict(**train_meters, **valid_meters, lr=current_lr, time=np.round(end/60,3))))
        
        if optimizer.param_groups[0]["lr"] == args.lr_min and start_decay:
            break_counter += 1
        if break_counter == args.break_counter:
            print('Break training due to minimal learning rate constraint!')
            break

    logging.info(f"Done training! Best PSNR {utils.save_checkpoint.best_score:.3f} obtained after step {utils.save_checkpoint.best_step} (epoch {utils.save_checkpoint.best_epoch}).")




def get_args(hp,ee,rr):
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Add data arguments
    parser.add_argument("--train-size", default=None, help="number of examples in training set")
    parser.add_argument("--val-size", default=40, help="number of examples in validation set")
    parser.add_argument("--test-size", default=100, help="number of examples in test set")
    parser.add_argument("--val-crop", default=True, type=bool, help="Crop validation images to train size.")
    
    parser.add_argument("--patch-size", default=128, help="size of the center cropped HR image")
    
    parser.add_argument("--batch-size", default=128, type=int, help="train batch size")

    # Add model arguments
    parser.add_argument("--model", default="unet", help="model architecture")

    # Add noise arguments
    parser.add_argument('--noise_std', default = 15, type = float, 
                help = 'noise level')
    parser.add_argument('--test_noise_std_min', default = 15, type = float, 
                help = 'minimal noise level for testing')
    parser.add_argument('--test_noise_std_max', default = 15, type = float, 
                help = 'maximal noise level for testing')
    parser.add_argument('--test_noise_stepsize', default = 5, type = float, 
                help = 'Stepsize between test_noise_std_min and test_noise_std_max')

    # Add optimization arguments
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--lr-gamma", default=0.5, type=float, help="factor by which to reduce learning rate")
    parser.add_argument("--lr-beta", default=2, type=float, help="factor by which to increase learning rate")
    parser.add_argument("--lr-patience", default=5, type=int, help="epochs without improvement before lr decay")
    parser.add_argument("--no_annealing", default=True, type=bool, help="Use lr annealing or not.")
    parser.add_argument("--lr-patience-annealing", default=3, type=int, help="epochs without improvement before lr annealing stops")
    parser.add_argument("--lr-min", default=1e-5, type=float, help="Once we reach this learning rate continue for break_counter many epochs then stop.")
    parser.add_argument("--lr-threshold", default=0.003, type=float, help="Improvements by less than this threshold are not counted for decay patience.")
    parser.add_argument("--break-counter", default=9, type=int, help="Once smallest learning rate is reached, continue for so many epochs before stopping.")
    parser.add_argument("--inital-decay-factor", default=2, type=int, help="After annealing found a lr for which val loss does not improve, go back initial_decay_factor many lrs")
    
    parser.add_argument("--num-epochs", default=100, type=int, help="force stop training at specified epoch")
    parser.add_argument("--valid-interval", default=1, type=int, help="evaluate every N epochs")
    parser.add_argument("--save-interval", default=1, type=int, help="save a checkpoint every N steps")
    

    # Add model arguments
    parser = models.unet_fastMRI.add_args(parser)
    
    parser = utils.add_logging_arguments(parser)

    #args = parser.parse_args()
    args, _ = parser.parse_known_args()
    
    # Set arguments specific for this experiment
    dargs = vars(args)
    for key in hp.keys():
        dargs[key] = hp[key][ee]
    args.seed = int(42 + 10*rr)
    
    return args


def f_restore_file(args):
    #available_models = glob.glob(f'{args.output_dir}/{args.experiment}-*')
    available_models = glob.glob(f'{args.output_dir}/*')
    if not available_models:
        raise ValueError('No file to restore')
    if not args.restore_mode:
        raise ValueError("Pick restore mode either 'best' 'last' or '\path\to\checkpoint\dir'")
    if args.restore_mode=='best':
        mode = "max"
        best_score = float("inf") if mode == "min" else float("-inf")
        best_model = None
        for modelp in available_models:
            model_path = os.path.join(modelp, "checkpoints/checkpoint_best.pt")
            if os.path.isfile(model_path):
                state_dict = torch.load(model_path, map_location=lambda s, l: default_restore_location(s, "cpu"))
                score = state_dict["best_score"]
                if (score < best_score and mode == "min") or (score > best_score and mode == "max"):
                    best_score = score
                    best_model = model_path
                    best_modelp = modelp
                    best_step = state_dict["best_step"]
                    best_epoch = state_dict["best_epoch"]
        args.restore_file = best_model
        args.experiment_dir = best_modelp
        #logging.info(f"Prepare to restore best model {best_model} with PSNR {best_score} at step {best_step}, epoch {best_epoch}")
    elif args.restore_mode=='last':
        last_step = -1
        last_model = None
        for modelp in available_models:
            model_path = os.path.join(modelp, "checkpoints/checkpoint_last.pt")
            if os.path.isfile(model_path):
                state_dict = torch.load(model_path, map_location=lambda s, l: default_restore_location(s, "cpu"))
                step = state_dict["last_step"]
                if step > last_step:
                    last_step = step
                    last_model = model_path
                    last_modelp = modelp
                    score = state_dict["score"]
                    last_epoch = state_dict["epoch"]
        args.restore_file = last_model
        args.experiment_dir = last_modelp
        #logging.info(f"Prepare to restore last model {last_model} with PSNR {score} at step {last_step}, epoch {last_epoch}")
    else:
        args.restore_file = args.restore_mode
        args.experiment_dir = args.restore_mode[:args.restore_mode.find('/checkpoints')]
            



def infer_images(args):
    USE_CUDA = True
    device = torch.device('cuda') if (torch.cuda.is_available() and USE_CUDA) else torch.device('cpu')
    net = load_model(args) # the denoiser

    seed_dict = {
        "val":10,
        "test":20,
        "cbsd68":30,
        "urban100":40,
        "mcmaster18":50,
        "kodak24":60,
        "CBSD68":70,
    }
    gen = torch.Generator()
    gen = gen.manual_seed(seed_dict[args.test_mode])



    # Load the test images 
    load_path = '../training_set_lists/'
    if args.test_mode == 'test':
        files_source = torch.load(load_path+f'ImageNetTest{args.test_size}_filepaths.pt') 
        #files_source.sort()       
    elif args.test_mode == 'val':
        files_source = torch.load(load_path+f'ImageNetVal{args.val_size}_filepaths.pt') 
        #files_source.sort()      
    else:
        files_source = torch.load(load_path+f'{args.test_mode}_filepaths.pt')

    
    if not os.path.isdir(args.output_dir+'/test_images'):
        os.mkdir(args.output_dir+'/test_images')

    counter = 0
    transformT = transforms.ToTensor()
    transformIm = transforms.ToPILImage()
    for f in files_source:
        counter = counter + 1
        if counter > 3:
            break
        # Create noise
        ISource = torch.unsqueeze(transformT(Image.open(f).convert("RGB")),0).to(device)
        noise =  torch.randn(ISource.shape,generator = gen) * args.noise_std/255.
        INoisy = noise.to(device) + ISource
        
        out = torch.clamp(net(INoisy), 0., 1.).cpu()
        out = torch.squeeze(out,0) # Get rid of the 1 in dim 0.
        im = transformIm(out)

        INoisy = torch.clamp(torch.squeeze(INoisy,0), 0., 1.).cpu()
        #INoisy = torch.squeeze(INoisy,0).cpu()
        INoisy = transformIm(INoisy)
        clean_image = Image.open(f).convert("RGB")
        im.save(args.output_dir+f'/test_images/im{counter}_denoised_notclamped.png')
        clean_image.save(args.output_dir+f'/test_images/im{counter}_ground_truth_notclamped.png')
        INoisy.save(args.output_dir+f'/test_images/im{counter}_noisy_notclamped.png')

        im.save(args.output_dir+f'/test_images/im{counter}_denoised_notclamped.pdf')
        clean_image.save(args.output_dir+f'/test_images/im{counter}_ground_truth_notclamped.pdf')
        INoisy.save(args.output_dir+f'/test_images/im{counter}_noisy_notclamped.pdf')


   
    