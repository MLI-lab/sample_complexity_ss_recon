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
from tensorboard.backend.event_processing import event_accumulator

import utils
import models 

from utils.data_helpers.load_datasets_helpers import *
from utils.meters import *
from utils.progress_bar import *
from utils.noise_model import get_noise
from utils.metrics import ssim,psnr
from utils.util_calculate_psnr_ssim import calculate_psnr,calculate_ssim,calculate_psnr_neighbor, calculate_upsnr
from utils.test_metrics import *

import torchvision.transforms.functional as FF


operation_seed_counter = 0

def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)


def generate_subimages(img, mask):
    n, c, h, w = img.shape
    subimage = torch.zeros(n,
                           c,
                           h // 2,
                           w // 2,
                           dtype=img.dtype,
                           layout=img.layout,
                           device=img.device)
    # per channel
    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
            n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
    return subimage


def get_generator(seed = None):
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    if seed == None:
        g_cuda_generator.manual_seed(operation_seed_counter)
    else:
        g_cuda_generator.manual_seed(seed)
    return g_cuda_generator


def generate_mask_pair(img,seed=None):
    # prepare masks (N x C x H/2 x W/2)
    n, c, h, w = img.shape
    mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    # prepare random mask pairs
    idx_pair = torch.tensor(
        [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
        dtype=torch.int64,
        device=img.device)
    rd_idx = torch.zeros(size=(n * h // 2 * w // 2, ),
                         dtype=torch.int64,
                         device=img.device)
    if seed == None:
        torch.randint(low=0,
                    high=8,
                    size=(n * h // 2 * w // 2, ),
                    generator=get_generator(),
                    out=rd_idx)
    else:
        torch.randint(low=0,
                    high=8,
                    size=(n * h // 2 * w // 2, ),
                    generator=get_generator(seed),
                    out=rd_idx)

    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(start=0,
                                end=n * h // 2 * w // 2 * 4,
                                step=4,
                                dtype=torch.int64,
                                device=img.device).reshape(-1, 1)
    # get masks
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    return mask1, mask2

def generate_val_mask(img):
    n, c, h, w = img.shape
    mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    mask3 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    mask4 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    mask1[torch.arange(start=0,end=n * h // 2 * w // 2 * 4, 
                        step=4, dtype=torch.int64,device=img.device)]=1
    mask2[torch.arange(start=1,end=n * h // 2 * w // 2 * 4, 
                        step=4, dtype=torch.int64,device=img.device)]=1
    mask3[torch.arange(start=2,end=n * h // 2 * w // 2 * 4, 
                        step=4, dtype=torch.int64,device=img.device)]=1
    mask4[torch.arange(start=3,end=n * h // 2 * w // 2 * 4, 
                        step=4, dtype=torch.int64,device=img.device)]=1

    return mask1, mask2, mask3, mask4

def load_model(args):
    USE_CUDA = True
    device = torch.device('cuda') if (torch.cuda.is_available() and USE_CUDA) else torch.device('cpu')
    
    checkpoint_path = glob.glob(args.output_dir +'/unet*')
    if len(checkpoint_path) != 1:
        raise ValueError("There is either no or more than one model to load")
    checkpoint_path = pathlib.Path(checkpoint_path[0] + f"/checkpoints/checkpoint_{args.restore_mode}.pt")
    state_dict = torch.load(checkpoint_path, map_location=lambda s, l: default_restore_location(s, "cpu"))
    args = argparse.Namespace(**{ **vars(state_dict["args"]), "no_log": True})

    #model = models.build_model(args).to(device)
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
    
    
    num_epoch = args.n_epoch
    ratio = num_epoch / 100
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                     milestones=[
                                         int(20 * ratio) - 1,
                                         int(40 * ratio) - 1,
                                         int(60 * ratio) - 1,
                                         int(80 * ratio) - 1
                                     ],
                                     gamma=args.lr_gamma)
    
    
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
    #train_meters = {name: RunningAverageMeter(0.98) for name in (["train_loss_running_avg"])}
    #train_meters = {name: AverageMeter() for name in (["train_loss"])}
    train_meters = {"train_loss": AverageMeter(), "train_loss_running_avg":RunningAverageMeter(0.98),"reconstruction_loss": AverageMeter(), "regularization_loss":AverageMeter()}
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

            noisy_inputs = noise + inputs
            mask1, mask2 = generate_mask_pair(noisy_inputs)
            noisy_sub1 = generate_subimages(noisy_inputs, mask1)
            noisy_sub2 = generate_subimages(noisy_inputs, mask2)
            with torch.no_grad():
                noisy_denoised = model(noisy_inputs)
            noisy_sub1_denoised = generate_subimages(noisy_denoised, mask1)
            noisy_sub2_denoised = generate_subimages(noisy_denoised, mask2)

            noisy_output = model(noisy_sub1)
            noisy_target = noisy_sub2
            Lambda = (epoch / args.n_epoch)* args.increase_ratio
            diff = noisy_output - noisy_target
            exp_diff = noisy_sub1_denoised - noisy_sub2_denoised

            loss1 = torch.mean(diff**2)
            reg = torch.mean((diff - exp_diff)**2)
            loss2 = Lambda * reg
            loss = args.Lambda1 * loss1 + args.Lambda2 * loss2
            

            model.zero_grad()
            loss.backward()
            optimizer.step()

            train_meters["train_loss"].update(loss.item())
            train_meters["reconstruction_loss"].update(loss1.item())
            train_meters["regularization_loss"].update(reg.item())
            train_meters["train_loss_running_avg"].update(loss.item())
            train_bar.log(dict(**train_meters, lr=optimizer.param_groups[0]["lr"]), verbose=True)

        # Add to tensorflow event file:
        if writer is not None:
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("loss/train", train_meters["train_loss"].avg, global_step)
            writer.add_scalar("loss/train_running_avg", train_meters["train_loss_running_avg"].avg, global_step)
            writer.add_scalar("loss/rec",train_meters["reconstruction_loss"].avg,global_step)
            writer.add_scalar("loss/reg",train_meters["regularization_loss"].avg,global_step)
            #gradients = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None], dim=0)
            #writer.add_histogram("gradients", gradients, global_step)
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
                    sample_np = sample.cpu().detach().numpy()
                    shape = np.shape(sample_np)
                    if shape[3] % 2 == 1:
                        sample_np = sample_np[:,:,:,:-1]
                    sample = torch.from_numpy(sample_np)
                    sample = sample.to(device)
                    #print(sample.size())
                    
                    # Self-supervised validation with fixed noise 
                    noise_self_supervised = get_noise(sample,noise_seed, fix_noise = args.fix_noise, noise_std = args.noise_std/255.)
                    noisy_input_fixed = sample + noise_self_supervised
                    output_self_supervised = model(noisy_input_fixed)

                    #noisy_target = sample + noise_target_self_supervised
                    #if args.val_crop:
                    mask1, mask2, mask3, mask4 = generate_val_mask(noisy_input_fixed)
                    noisy_sub1 = generate_subimages(noisy_input_fixed, mask1)
                    noisy_sub2 = generate_subimages(noisy_input_fixed, mask2)
                    noisy_sub3 = generate_subimages(noisy_input_fixed, mask3)
                    noisy_sub4 = generate_subimages(noisy_input_fixed, mask4)

                    noisy_sub1_denoised = generate_subimages(output_self_supervised, mask1)
                    noisy_sub2_denoised = generate_subimages(output_self_supervised, mask2)

                    #noisy_sub1_denoised = generate_subimages(output_self_supervised, mask1)
                    noisy_output = model(noisy_sub1)
                    valid_psnr_self_supervised = calculate_psnr_neighbor(noisy_output,noisy_sub2,noisy_sub1_denoised,noisy_sub2_denoised,args.increase_ratio) 
                    valid_ssim_self_supervised = ssim(noisy_output, noisy_sub2)
                    valid_meters["valid_psnr_self_supervised"].update(valid_psnr_self_supervised.item())
                    valid_meters["valid_ssim_self_supervised"].update(valid_ssim_self_supervised.item())

                    # Ground truth validation wit fixed noise

                    # It uses the same input and output as in the self-supervised case since the noise seed is fixed
                    valid_psnr = psnr(output_self_supervised, sample) 
                    valid_ssim = ssim(output_self_supervised, sample)
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

                    num_epoch = args.n_epoch
                    ratio = num_epoch / 100
                    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                     milestones=[
                                         int(20 * ratio) - 1,
                                         int(40 * ratio) - 1,
                                         int(60 * ratio) - 1,
                                         int(80 * ratio) - 1
                                     ],
                                     gamma=args.lr_gamma)
   
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
            scheduler.step()
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
        INoisy = transformIm(INoisy)
        clean_image = Image.open(f).convert("RGB")
        im.save(args.output_dir+f'/test_images/im{counter}_denoised_notclamped.png')
        clean_image.save(args.output_dir+f'/test_images/im{counter}_ground_truth_notclamped.png')
        INoisy.save(args.output_dir+f'/test_images/im{counter}_noisy_notclamped.png')

        im.save(args.output_dir+f'/test_images/im{counter}_denoised_notclamped.pdf')
        clean_image.save(args.output_dir+f'/test_images/im{counter}_ground_truth_notclamped.pdf')
        INoisy.save(args.output_dir+f'/test_images/im{counter}_noisy_notclamped.pdf')


   
    