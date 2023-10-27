import torch
import argparse
import os
import pathlib
import pickle
import logging
import sys
import time
import io
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torchvision
import glob
from torch.serialization import default_restore_location
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import numpy as np

import copy
import utils
import models 

from utils.data_helpers.load_datasets_helpers import *
from utils.meters import *
from utils.progress_bar import *
from utils.metrics import psnr_gpu


def get_figure(image,figsize,title):
    """Return a matplotlib figure of a given image."""
    if len(image.shape) != 3:
        raise ValueError("Image dimensions not suitable for logging to tensorboard.")
    if image.shape[0] == 1 or image.shape[0] == 3:
        image = np.rollaxis(image,0,3)
    # Create a figure to contain the plot.
    if figsize:
        figure = plt.figure(figsize=figsize)
    else:
        figure = plt.figure()
    # Start next subplot.
    plt.subplot(1, 1, 1, title=title)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap='gray')
    figure.tight_layout()

    return figure

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)

    frameTensor = torch.tensor(np.frombuffer(buf.getvalue(), dtype=np.uint8), device='cpu')
    image = torchvision.io.decode_png(frameTensor)
    return image

def stack2raw(var):
    _, h, w = var.shape
    if var.is_cuda:
        res = torch.cuda.FloatTensor(h * 2, w * 2)
    else:
        res = torch.FloatTensor(h * 2, w * 2)
    res[0::2, 0::2] = var[0]
    res[0::2, 1::2] = var[1]
    res[1::2, 0::2] = var[2]
    res[1::2, 1::2] = var[3]
    return res

def add_img_to_tensorboard(writer, epoch, name, psnr, input_img,output,target,actual_output):

    error = torch.abs(target - output)
    actual_output = actual_output - actual_output.min()
    actual_output = actual_output / actual_output.max()
    input_img = input_img - input_img.min() 
    input_img = input_img / input_img.max()
    output = output - output.min() 
    output = output / output.max()
    target = target - target.min()
    target = target / target.max()
    error = error - error.min() 
    error = error / error.max()
    image = torch.cat([input_img, target, output, actual_output], dim=1)
    image = torchvision.utils.make_grid(image, nrow=1, normalize=False)

    figure = get_figure(image.cpu().numpy(),figsize=(3,12),title=f"psnr={psnr:.4f}")

    writer.add_image(name+"_abs", plot_to_image(figure), epoch)

    plt.close()


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


def save_test_image_with_dc(experiment_path, ground_truth_image, input_img, output, fname, track_meter, do_stack2raw=False):

    if do_stack2raw:
        input_img = stack2raw(input_img).unsqueeze(0).unsqueeze(0).detach()
        output = stack2raw(output).unsqueeze(0).unsqueeze(0).detach()
        ground_truth_image = stack2raw(ground_truth_image).unsqueeze(0).unsqueeze(0).detach()

    save_path = experiment_path + '/test_figures/'

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    error = torch.abs(ground_truth_image - output)
    output = output - output.min() 
    output = output / output.max()
    ground_truth_image = ground_truth_image - ground_truth_image.min()
    ground_truth_image = ground_truth_image / ground_truth_image.max()
    input_img = input_img - input_img.min()
    input_img = input_img / input_img.max()
    error = error - error.min() 
    error = error / error.max()

    image = torch.cat([ground_truth_image, input_img, output, error], dim=0)
    image = torchvision.utils.make_grid(image, nrow=2, normalize=False, value_range=(0,1), pad_value=1)
    psnr_score = track_meter["PSNR"].val[-1]
    figure = get_figure(image.cpu().numpy(),figsize=(8,12),title=f"psnr={psnr_score:.3f}") 

    plt.savefig(experiment_path + '/test_figures/' + f"{fname}.png", dpi='figure')
    plt.close()


def cli_main_test(args):
    USE_CUDA = True
    device = torch.device('cuda') if (torch.cuda.is_available() and USE_CUDA) else torch.device('cpu')
    model = load_model(args)

    args.resume_training = True
    f_restore_file(args)

    args.log_dir = os.path.join(args.experiment_dir, "logs")
    os.makedirs(args.log_dir, exist_ok=True)
    args.log_file = os.path.join(args.log_dir, "train.log")

    utils.init_logging(args)
    

    valset = SIDDSubdataset_fromArray(args.val_size,args.seed,mode='val',patch_size=args.patch_size,supervised=True)
    val_loader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False,generator=torch.Generator().manual_seed(args.seed))
    testset = SIDDSubdataset_fromArray(args.test_size,args.seed,mode='test',patch_size=args.patch_size,supervised=True)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False,generator=torch.Generator().manual_seed(args.seed))

    val_tracks = {'PSNR' : TrackMeter_testing()}
    test_tracks = {'PSNR' : TrackMeter_testing()}

    model.eval()

    valid_bar = ProgressBar(val_loader)
    for inputs_val, targets_val in valid_bar:

        outputs_val = model(inputs_val)

        valid_psnr = psnr_gpu(outputs_val[0], targets_val[0]) 
        val_tracks["PSNR"].update(valid_psnr)

    test_bar = ProgressBar(test_loader)
    idx=0
    for inputs_val, targets_val in test_bar:

        outputs_val = model(inputs_val)

        valid_psnr = psnr_gpu(outputs_val[0], targets_val[0]) 
        test_tracks["PSNR"].update(valid_psnr)

        if idx in [0,100,200,300,400]:
            fname = f"test_{idx}_stack2raw"
            save_test_image_with_dc(args.output_dir, targets_val[0], inputs_val[0], outputs_val[0], fname, test_tracks, do_stack2raw=True)
            for ch in range(4):
                fname = f"test_{idx}_ch{ch}"
                save_test_image_with_dc(args.output_dir, targets_val[:,ch:ch+1], inputs_val[:,ch:ch+1], outputs_val[:,ch:ch+1], fname, test_tracks)
        idx+=1

    logging.info(f"\nEvaluate validationset of length {len(val_loader)}:")
    for metric in val_tracks.keys():
        logging.info(f"{metric}: avg {val_tracks[metric].avg:.6f}, std {val_tracks[metric].std:.6f}")

    logging.info(f"\nEvaluate testset of length {len(test_loader)}:")
    for metric in test_tracks.keys():
        logging.info(f"{metric}: avg {test_tracks[metric].avg:.6f}, std {test_tracks[metric].std:.6f}")

    pickle.dump( val_tracks, open(args.output_dir + '/test_validationSet_metrics.pkl', "wb" ) , pickle.HIGHEST_PROTOCOL )
    pickle.dump( test_tracks, open(args.output_dir + '/test_testSet_metrics.pkl', "wb" ) , pickle.HIGHEST_PROTOCOL )


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
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.5)
    
    logging.info(f"Built a model consisting of {sum(p.numel() for p in model.parameters()):,} parameters")

    trainset = SIDDSubdataset_fromArray(args.train_size,args.seed,mode='train',patch_size=args.patch_size,supervised=args.supervised)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False,generator=torch.Generator().manual_seed(args.seed))
    
    valset = SIDDSubdataset_fromArray(args.val_size,args.seed,mode='val',patch_size=args.patch_size,supervised=True)
    val_loader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False,generator=torch.Generator().manual_seed(args.seed))
    
    print(optimizer.param_groups[0]["lr"])
    if args.resume_training:
        state_dict = utils.load_checkpoint(args, model, optimizer, scheduler)
        global_step = state_dict['last_step']
        start_epoch = int(state_dict['last_step']/(len(train_loader)))+1
    else:
        global_step = -1
        start_epoch = 0
    
    print(optimizer.param_groups[0]["lr"])
    args.log_interval = min(len(trainset), 100) # len(train_loader)=log once per epoch
    args.no_visual = False # True for not logging to tensorboard
    
    # Track moving average of loss values
    train_meters = { "train_loss":RunningAverageMeter(0.98)}
    valid_meters = {name: AverageMeter() for name in (["valid_psnr"])}
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
    
    log_image_interval_tb = 20
    for epoch in range(start_epoch, args.num_epochs):
        start = time.process_time()
        train_bar = ProgressBar(train_loader, epoch)
        # At beginning of each epoch reset the train meters
        for meter in train_meters.values():
            meter.reset()
        
        idx=0
        for inputs, targets in train_bar:
            model.train() #Sets the module in training mode.

            global_step += 1

            outputs = model(inputs)
            loss = F.mse_loss(outputs, targets, reduction="sum") / torch.prod(torch.tensor(inputs.size())) #(inputs.size(0) * 2)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            train_meters["train_loss"].update(loss.item())
            train_bar.log(dict(**train_meters, lr=optimizer.param_groups[0]["lr"]), verbose=True)

        # Add to tensorflow event file:
        if writer is not None:
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("loss/train", train_meters["train_loss"].avg, epoch)

            if idx ==0 and epoch % log_image_interval_tb == 0: # log a random train image to tensorboard
                idx+=1
                name = "train_0_img"
                psnr_train = psnr_gpu(outputs[0], targets[0])
                ch=0
                for ch in range(4):
                    name = f"train_0_img_{ch}"
                    add_img_to_tensorboard(writer, epoch, name, psnr_train, inputs[0,ch:ch+1].detach(),outputs[0,ch:ch+1].detach(),targets[0,ch:ch+1].detach()) if epoch % log_image_interval_tb == 0 else None

            sys.stdout.flush()
                
            

        if epoch % args.valid_interval == 0:
            model.eval()
            for meter in valid_meters.values():
                meter.reset()

            valid_bar = ProgressBar(val_loader)
            idx=0
            for inputs_val, targets_val in valid_bar:
                with torch.no_grad():
                    outputs_val = model(inputs_val)
                    valid_psnr = psnr_gpu(outputs_val[0], targets_val[0]) 
                    valid_meters["valid_psnr"].update(valid_psnr)

            if writer is not None:
                writer.add_scalar("psnr/valid", valid_meters['valid_psnr'].avg, epoch)

                
                if idx ==0 and epoch % log_image_interval_tb == 0: # log a val image to tensorboard
                    idx+=1
                    name = "val_0_img"
                    for ch in range(4):
                        name = f"val_0_img_{ch}"
                        add_img_to_tensorboard(writer, epoch, name, valid_psnr, inputs_val[0,ch:ch+1].detach(),outputs_val[0,ch:ch+1].detach(),targets_val[0,ch:ch+1].detach()) if epoch % log_image_interval_tb == 0 else None
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
                sys.stdout.flush()

            val_loss = valid_meters["valid_psnr"].avg

        
            utils.save_checkpoint(args, global_step, epoch, model, optimizer, score=val_loss, mode="max")
            current_lr = optimizer.param_groups[0]["lr"]
            if val_loss > best_val_current:
                best_val_current = val_loss
        
        if writer is not None:            
            writer.add_scalar("epoch", epoch, global_step)
            sys.stdout.flush()
        

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
        #if args.val_on_testset:
        #    logging.info(train_bar.print(dict(**train_meters, **valid_meters, **test_meters, lr=current_lr, time=np.round(end/60,3))))
        #else:
        logging.info(train_bar.print(dict(**train_meters, **valid_meters, lr=current_lr, time=np.round(end/60,3))))
        
        if optimizer.param_groups[0]["lr"] < args.lr_min:
            break_counter += 1
        if break_counter == args.break_counter:
            print('Break training due to minimal learning rate constraint!')
            break

    logging.info(f"Done training! Best PSNR {utils.save_checkpoint.best_score:.3f} obtained after step {utils.save_checkpoint.best_step} (epoch {utils.save_checkpoint.best_epoch}).")


def cli_main_gradDiff(args):
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

    writer = SummaryWriter(log_dir=args.experiment_dir) if not args.no_visual else None
    
    # Build data loaders, a model and an optimizer
    model = models.unet_fastMRI(
        in_chans=args.in_chans,
        chans = args.chans,
        num_pool_layers = args.num_pool_layers,
        drop_prob = 0.0,
        residual_connection = args.residual,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.5)
    
    logging.info(f"Built a model consisting of {sum(p.numel() for p in model.parameters()):,} parameters")

    trainset = SIDDSubdataset_fromArray_gradDiff(args.train_size,args.seed,mode='train',patch_size=args.patch_size)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False,generator=torch.Generator().manual_seed(args.seed))
    

    print(optimizer.param_groups[0]["lr"])
    if args.resume_training:
        state_dict = utils.load_checkpoint(args, model, optimizer, scheduler)
        global_step = state_dict['last_step']
        start_epoch = int(state_dict['last_step']/(len(train_loader)))+1
    else:
        global_step = -1
        start_epoch = 0
    
    print(optimizer.param_groups[0]["lr"])
    args.log_interval = min(len(trainset), 100) # len(train_loader)=log once per epoch
    args.no_visual = False # True for not logging to tensorboard
    
    # Track moving average of loss values
    train_meters = { "train_loss":RunningAverageMeter(0.98)}
    sup_diff_tracks = {'divide_by_norm_of_risk_grad': TrackMeter_testing(), 'take_mse': TrackMeter_testing()}
    ss_diff_tracks = {'divide_by_norm_of_risk_grad': TrackMeter_testing(), 'take_mse': TrackMeter_testing()}
    
    for epoch in range(start_epoch, args.num_epochs):
        
        # compute gradient histogrms
        model_copy = copy.deepcopy(model)
        train_bar_hist = ProgressBar(train_loader, epoch)
        model_copy.train()

        for meter in sup_diff_tracks.values():
            meter.reset()
        for meter in ss_diff_tracks.values():
            meter.reset()
        
        idx=0
        # estimate ground truth gradient based on whole dataset
        for inputs, targets_sup, targets_self in train_bar_hist:
            model_copy.train() #Sets the module in training mode.

            outputs = model_copy(inputs)
            loss_sup = F.mse_loss(outputs, targets_sup, reduction="sum") / torch.prod(torch.tensor(inputs.size())) #(inputs.size(0) * 2)

            param = list(model_copy.parameters())

            model_copy.zero_grad()
            loss_sup.backward(retain_graph=True)

            if idx == 0:
                for p in param:
                    p.grad_true_risk = p.grad
                    p.grad = None
            else:
                for p in param:
                    p.grad_true_risk += p.grad
                    p.grad = None

            if idx in [0,1,2,3,4]:
                for ch in range(4):
                    name = f"train_{idx}_img_{ch}"
                    add_img_to_tensorboard(writer, epoch, name, 0, inputs[0,ch:ch+1].detach(),
                                           targets_sup[0,ch:ch+1].detach(),targets_self[0,ch:ch+1].detach(),
                                           outputs[0,ch:ch+1].detach()) 
            idx+=1

        for p in param:
            p.grad_true_risk = p.grad_true_risk/len(train_loader)

        # compute stochastic supervised and self-supervised gradients based on the same dataset
        train_bar_hist = ProgressBar(train_loader, epoch)
        for inputs, targets_sup, targets_self in train_bar_hist:

            outputs = model_copy(inputs)
            loss_sup = F.mse_loss(outputs, targets_sup, reduction="sum") / torch.prod(torch.tensor(inputs.size())) #(inputs.size(0) * 2)
            loss_self = F.mse_loss(outputs, targets_self, reduction="sum") / torch.prod(torch.tensor(inputs.size())) #(inputs.size(0) * 2)

            param = list(model_copy.parameters())

            model_copy.zero_grad()

            loss_sup.backward(retain_graph=True)
            for p in param:
                p.grad_sup = p.grad
                p.grad = None

            loss_self.backward(retain_graph=True) 
            for p in param:
                p.grad_self = p.grad
                p.grad = None

            diff_sup = torch.zeros(1).to(device)
            diff_self = torch.zeros(1).to(device)
            norm_grad_of_risk = torch.zeros(1).to(device)

            for p in param:
                diff_sup += torch.sum(torch.square(torch.sub(p.grad_sup,p.grad_true_risk)))
                diff_self += torch.sum(torch.square(torch.sub(p.grad_self,p.grad_true_risk)))
                norm_grad_of_risk += torch.sum(torch.square(p.grad_true_risk))

            sup_diff_tracks['divide_by_norm_of_risk_grad'].update(torch.div(diff_sup,norm_grad_of_risk).item())
            sup_diff_tracks['take_mse'].update(torch.mean(diff_sup).item())

            ss_diff_tracks['divide_by_norm_of_risk_grad'].update(torch.div(diff_self,norm_grad_of_risk).item())
            ss_diff_tracks['take_mse'].update(torch.mean(diff_self).item())

        pickle.dump( sup_diff_tracks, open(args.output_dir + f"/sup_diff_tracks_ep{epoch}.pkl", "wb" ) , pickle.HIGHEST_PROTOCOL )
        pickle.dump( ss_diff_tracks, open(args.output_dir + f"/ss_diff_tracks_ep{epoch}.pkl", "wb" ) , pickle.HIGHEST_PROTOCOL )

        
        train_bar = ProgressBar(train_loader, epoch)
        # At beginning of each epoch reset the train meters
        for meter in train_meters.values():
            meter.reset()
        
        for inputs, targets_sup, targets_self in train_bar:
            model.train() #Sets the module in training mode.

            outputs = model(inputs)
            loss = F.mse_loss(outputs, targets_sup, reduction="sum") / torch.prod(torch.tensor(inputs.size())) #(inputs.size(0) * 2)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            train_meters["train_loss"].update(loss.item())
            train_bar.log(dict(**train_meters, lr=optimizer.param_groups[0]["lr"]), verbose=True)




def get_args(hp,ee,rr):
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Add data arguments
    parser.add_argument("--train-size", default=None, help="number of examples in training set")
    parser.add_argument("--val-size", default=40, help="number of examples in validation set")
    parser.add_argument("--test-size", default=100, help="number of examples in test set")
    parser.add_argument("--patch-size", default=128, help="size of the center cropped HR image")
    parser.add_argument("--batch-size", default=1, type=int, help="train batch size")

    # Add model arguments
    parser.add_argument("--model", default="unet", help="model architecture")

    # Add optimization arguments
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--lr-gamma", default=0.5, type=float, help="factor by which to reduce learning rate")
    parser.add_argument("--lr-min", default=1e-5, type=float, help="Once we reach this learning rate continue for break_counter many epochs then stop.")
    parser.add_argument("--lr-threshold", default=0.003, type=float, help="Improvements by less than this threshold are not counted for decay patience.")
    parser.add_argument("--break-counter", default=9, type=int, help="Once smallest learning rate is reached, continue for so many epochs before stopping.")
    parser.add_argument("--num-epochs", default=100, type=int, help="force stop training at specified epoch")
    parser.add_argument("--valid-interval", default=1, type=int, help="evaluate every N epochs")
    parser.add_argument("--save-interval", default=1, type=int, help="save a checkpoint every N steps")
    
    # Add model arguments
    parser = models.unet_fastMRI.add_args(parser)
    
    parser = utils.add_logging_arguments(parser)

    args, _ = parser.parse_known_args()
    
    # Set arguments specific for this experiment
    dargs = vars(args)
    for key in hp.keys():
        dargs[key] = hp[key][ee]
    args.seed = int(42 + 10*rr)
    
    return args


def f_restore_file(args):
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
            
