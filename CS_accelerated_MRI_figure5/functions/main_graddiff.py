
#################
# Import python packages
import torch
import logging
import time
from torch.utils.tensorboard import SummaryWriter
import sys
import os
from torch.serialization import default_restore_location
from collections import defaultdict
import numpy as np
import torchvision
import pickle
import matplotlib.pyplot as plt
from packaging import version
import copy

from torch.nn import L1Loss, MSELoss

from functions.math import complex_abs, complex_mul, complex_conj
from functions.data.transforms import center_crop_to_smallest, normalize_to_given_mean_std
if version.parse(torch.__version__) >= version.parse("1.7.0"):
    from functions.fftc import fft2c_new as fft2c
    from functions.fftc import ifft2c_new as ifft2c
else:
    from functions.fftc import fft2c_old as fft2c
    from functions.fftc import ifft2c_old as ifft2c

# Implementation of SSIMLoss
from functions.training.losses import SSIMLoss

from functions.training.debug_helper import print_tensor_stats, save_figure

# Set seeds, create directories, set path to checkpoints if available
from functions.train_utils import setup_experiment

from functions.train_utils import load_checkpoint,save_checkpoint, init_logging

# Function that returns a MaskFunc object
from functions.data.subsample import create_mask_for_mask_type

from functions.data.transforms import UnetDataTransform_hist

from functions.data.mri_dataset import SliceDataset

from functions.models.unet import Unet

# Create scheduler and optimizer objects
from functions.training.training_functions import configure_optimizers, Compute_batch_train_loss

# Class that allows to  track the average of some quantity over an epoch
from functions.training.meters import AverageMeter, TrackMeter_testing

# Gives a customized tqdm object that can be used as iterable instead of train_loader
from functions.training.progress_bar import ProgressBar

# Functions to log images with a header to tensorboard
from functions.log_save_image_utils import plot_to_image, get_figure

def add_img_to_tensorboard(writer, epoch, name, input_img, target, output, val_ssim_fct, max_value, crop):

    output, _ = center_crop_to_smallest(output, target)
    input_img, _ = center_crop_to_smallest(input_img, target)

    # Normalize output to mean and std of target
    #target, output = normalize_to_given_mean_std(target, output)               
    ssim_loss = 1-val_ssim_fct(output, target, data_range=max_value)

    error = torch.abs(target - output)
    input_img = input_img - input_img.min() 
    input_img = input_img / input_img.max()
    output = output - output.min() 
    output = output / output.max()
    target = target - target.min()
    target = target / target.max()
    error = error - error.min() 
    error = error / error.max()
    image = torch.cat([input_img, target, output, error], dim=0)
    image = torchvision.utils.make_grid(image, nrow=1, normalize=False)
    if crop:
        figure = get_figure(image.cpu().numpy(),figsize=(3,12),title=f"ssim={ssim_loss.item():.6f}")
    else:
        figure = get_figure(image.cpu().numpy(),figsize=(3,20),title=f"ssim={ssim_loss.item():.6f}")
    writer.add_image(name, plot_to_image(figure), epoch)


def main_train(hp_exp):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # ------------
    # setup:
    # Set seeds, create directories, set path to checkpoints if available
    # ------------
    hp_exp = setup_experiment(hp_exp)
    init_logging(hp_exp)

    writer = SummaryWriter(log_dir=hp_exp['log_path']) if hp_exp['tb_logging'] else None

    train_log_filenames_list = []
    for k in hp_exp['log_train_images'].keys():
        train_log_filenames_list.append(k)

    mode_lookup = {
        'SSIM' : 'max',
        'PSNR' : 'max',
        'L1' : 'min',
        'L2' : 'min',
        'MSE' : 'min',
        'L2_kspace' : 'min',
        'L1_kspace' : 'min',
    }

    # ------------
    # data
    # ------------
    mask_func = create_mask_for_mask_type(
        hp_exp['mask_type'], hp_exp['selfsup'], hp_exp['center_fraction'], hp_exp['acceleration'], hp_exp['acceleration_total']
    )

    data_transform_train = UnetDataTransform_hist(hp_exp['challenge'],mask_func=mask_func, use_seed=hp_exp['use_mask_seed_for_training'], hp_exp=hp_exp,mode="train")

    def _init_fn(worker_id):
        np.random.seed(12 + worker_id)

    trainset =  SliceDataset(
            dataset=hp_exp['train_set'],
            path_to_dataset=hp_exp['data_path'],
            path_to_sensmaps=hp_exp['smaps_path'],
            provide_senmaps=hp_exp['provide_senmaps'],
            challenge=hp_exp['challenge'],
            transform=data_transform_train,
            use_dataset_cache=True,
        )

    train_loader = torch.utils.data.DataLoader(
            dataset=trainset,
            batch_size=hp_exp['batch_size'],
            num_workers=hp_exp['num_workers'],
            shuffle=True,
            generator=torch.Generator().manual_seed(hp_exp['seed']),
            pin_memory =True,
        )

    # ------------
    # model
    # ------------
    if hp_exp['two_channel_imag_real']:
        in_chans = 2
    else:
        in_chans = 1

    model = Unet(
            in_chans=in_chans,
            out_chans=in_chans,
            chans=hp_exp['chans'],
            num_pool_layers=hp_exp['num_pool_layers'],
            drop_prob=0.0,
        ).to(device)
    logging.info(f"Built a model consisting of {sum(p.numel() for p in model.parameters()):,} parameters")#

    # ------------
    # trainer
    # ------------
    optimizer, scheduler = configure_optimizers(hp_exp, model.parameters())

    compute_batch_train_loss = Compute_batch_train_loss()

    train_meters = {'train_' + name: AverageMeter() for name in (hp_exp['loss_functions'])}
    if len(hp_exp['loss_functions']) > 1:
        train_meters['cumulated_loss'] = AverageMeter()
    train_meters['train_L2_gt_abs'] = AverageMeter()
    
    if hp_exp['two_channel_imag_real']:
        train_meters['train_L2_gt_comp'] = AverageMeter()


    #########
    # Training
    save_train_figures = True
    save_val_figures = False

    compute_gradients_interval = 1

    loss_fct = MSELoss(reduction='sum')

    sup_diff_tracks = {'divide_by_norm_of_risk_grad': TrackMeter_testing(), 'take_mse': TrackMeter_testing()}
    ss_diff_tracks = {'divide_by_norm_of_risk_grad': TrackMeter_testing(), 'take_mse': TrackMeter_testing()}

    for epoch in range(save_checkpoint.start_epoch, hp_exp['num_epochs']):

        # compute gradient histogrms
        if epoch % compute_gradients_interval == 0:
            model_copy = copy.deepcopy(model)
            train_bar_hist = ProgressBar(train_loader, epoch)
            model_copy.train()

            for meter in sup_diff_tracks.values():
                meter.reset()
            for meter in ss_diff_tracks.values():
                meter.reset()

            # estimate ground truth gradient based on whole dataset
            for id,sample in enumerate(train_bar_hist):

                gt_kspace, binary_background_mask, input_image, input_kspace, input_mask, target_image, target_kspace, target_mask, target_mask_weighted, ground_truth_image, sens_maps, mean, std, fname, slice_num = sample
                
                gt_kspace=gt_kspace.to(device)
                input_image=input_image.to(device)
                target_image=target_image.to(device)
                target_kspace=target_kspace.to(device)
                input_kspace=input_kspace.to(device)
                input_mask=input_mask.to(device)
                target_mask=target_mask.to(device)
                target_mask_weighted=target_mask_weighted.to(device)
                ground_truth_image=ground_truth_image.to(device)
                sens_maps=sens_maps.to(device)
                mean=mean.to(device)
                std=std.to(device)
                binary_background_mask=binary_background_mask.to(device)

                # prediction
                x_output = model_copy(input_image)
                x_output = x_output * std + mean

                # move to kspace
                output_per_coil_imgs = torch.moveaxis(x_output , 1, -1 )
                output_per_coil_imgs = complex_mul(output_per_coil_imgs, sens_maps)
                y_output_sup = fft2c(output_per_coil_imgs)

                y_output_ss = y_output_sup * target_mask_weighted + 0.0
                y_target_ss = target_kspace * target_mask_weighted + 0.0

                y_target_sup = gt_kspace

                save_figure(torch.log(torch.abs(input_kspace[0,0,:,:,0].detach().cpu())+ 1e-9),"y_input_real",hp_exp,save=save_train_figures) if id==0 else None
                save_figure(torch.log(torch.abs(y_output_sup[0,0,:,:,0].detach().cpu())+ 1e-9),"y_output_sup_real",hp_exp,save=save_train_figures) if id==0 else None
                save_figure(torch.log(torch.abs(y_output_ss[0,0,:,:,0].detach().cpu())+ 1e-9),"y_output_ss_real",hp_exp,save=save_train_figures) if id==0 else None
                save_figure(torch.log(torch.abs(y_target_sup[0,0,:,:,0].detach().cpu())+ 1e-9),"y_target_sup_real",hp_exp,save=save_train_figures) if id==0 else None
                save_figure(torch.log(torch.abs(y_target_ss[0,0,:,:,0].detach().cpu())+ 1e-9),"y_target_ss_real",hp_exp,save=save_train_figures) if id==0 else None

                # compute loss
                train_loss_sup = loss_fct(y_output_sup,y_target_sup) / torch.sum(torch.abs(y_target_sup)**2)

                #train_loss_ss = loss_fct(y_output_ss,y_target_ss) / torch.sum(torch.abs(y_target_ss)**2)

                param = list(model_copy.parameters())

                model_copy.zero_grad()
                train_loss_sup.backward(retain_graph=True)

                if id == 0:
                    for p in param:
                        p.grad_true_risk = p.grad
                        p.grad = None
                else:
                    for p in param:
                        p.grad_true_risk += p.grad
                        p.grad = None
                
            for p in param:
                p.grad_true_risk = p.grad_true_risk/len(train_loader)

            # compute stochastic supervised and self-supervised gradients based on the same dataset
            train_bar_hist = ProgressBar(train_loader, epoch)
            for id,sample in enumerate(train_bar_hist):

                gt_kspace, binary_background_mask, input_image, input_kspace, input_mask, target_image, target_kspace, target_mask, target_mask_weighted, ground_truth_image, sens_maps, mean, std, fname, slice_num = sample
                
                gt_kspace=gt_kspace.to(device)
                input_image=input_image.to(device)
                target_image=target_image.to(device)
                target_kspace=target_kspace.to(device)
                input_kspace=input_kspace.to(device)
                input_mask=input_mask.to(device)
                target_mask=target_mask.to(device)
                target_mask_weighted=target_mask_weighted.to(device)
                ground_truth_image=ground_truth_image.to(device)
                sens_maps=sens_maps.to(device)
                mean=mean.to(device)
                std=std.to(device)
                binary_background_mask=binary_background_mask.to(device)

                # prediction
                x_output = model_copy(input_image)
                x_output = x_output * std + mean

                # move to kspace
                output_per_coil_imgs = torch.moveaxis(x_output , 1, -1 )
                output_per_coil_imgs = complex_mul(output_per_coil_imgs, sens_maps)
                y_output_sup = fft2c(output_per_coil_imgs)

                y_output_ss = y_output_sup * target_mask_weighted + 0.0
                y_target_ss = target_kspace * target_mask_weighted + 0.0

                y_target_sup = gt_kspace

                # compute loss
                train_loss_sup = loss_fct(y_output_sup,y_target_sup) / torch.sum(torch.abs(y_target_sup)**2)

                train_loss_ss = loss_fct(y_output_ss,y_target_ss) / torch.sum(torch.abs(y_target_ss)**2)

                param = list(model_copy.parameters())

                model_copy.zero_grad()
                train_loss_sup.backward(retain_graph=True)

                for p in param:
                    p.grad_sup = p.grad
                    p.grad = None

                train_loss_ss.backward(retain_graph=True) 
                for p in param:
                    p.grad_ss = p.grad
                    p.grad = None

                diff_sup = torch.zeros(1).to(device)
                diff_ss = torch.zeros(1).to(device)
                norm_grad_of_risk = torch.zeros(1).to(device)

                for p in param:
                    diff_sup += torch.sum(torch.square(torch.sub(p.grad_sup,p.grad_true_risk)))
                    diff_ss += torch.sum(torch.square(torch.sub(p.grad_ss,p.grad_true_risk)))
                    norm_grad_of_risk += torch.sum(torch.square(p.grad_true_risk))

                sup_diff_tracks['divide_by_norm_of_risk_grad'].update(torch.div(diff_sup,norm_grad_of_risk).item())
                sup_diff_tracks['take_mse'].update(torch.mean(diff_sup).item())

                ss_diff_tracks['divide_by_norm_of_risk_grad'].update(torch.div(diff_ss,norm_grad_of_risk).item())
                ss_diff_tracks['take_mse'].update(torch.mean(diff_ss).item())

            pickle.dump( sup_diff_tracks, open(hp_exp['log_path'] + f"sup_diff_tracks_ep{epoch}.pkl", "wb" ) , pickle.HIGHEST_PROTOCOL )
            pickle.dump( ss_diff_tracks, open(hp_exp['log_path'] + f"ss_diff_tracks_ep{epoch}.pkl", "wb" ) , pickle.HIGHEST_PROTOCOL )

        
        # perform one training epoch
        train_bar = ProgressBar(train_loader, epoch)
        for meter in train_meters.values():
            meter.reset()

        for id,sample in enumerate(train_bar):
            model.train()

            gt_kspace, binary_background_mask, input_image, input_kspace, input_mask, target_image, target_kspace, target_mask, target_mask_weighted, ground_truth_image, sens_maps, mean, std, fname, slice_num = sample
                
            gt_kspace=gt_kspace.to(device)
            input_image=input_image.to(device)
            target_image=target_image.to(device)
            target_kspace=target_kspace.to(device)
            input_kspace=input_kspace.to(device)
            input_mask=input_mask.to(device)
            target_mask=target_mask.to(device)
            target_mask_weighted=target_mask_weighted.to(device)
            ground_truth_image=ground_truth_image.to(device)
            sens_maps=sens_maps.to(device)
            mean=mean.to(device)
            std=std.to(device)
            binary_background_mask=binary_background_mask.to(device)


            # prediction
            x_output = model(input_image)
            x_output = x_output * std + mean

            # move to kspace
            output_per_coil_imgs = torch.moveaxis(x_output , 1, -1 )
            output_per_coil_imgs = complex_mul(output_per_coil_imgs, sens_maps)
            y_output_sup = fft2c(output_per_coil_imgs)

            y_target_sup = gt_kspace

            # compute loss
            train_loss_sup = loss_fct(y_output_sup,y_target_sup) / torch.sum(torch.abs(y_target_sup)**2)

            model.zero_grad()
            train_loss_sup.backward()
            optimizer.step()

            train_meters["train_L2_kspace"].update(train_loss_sup.item())
            train_bar.log(dict(**train_meters), verbose=True)
        logging.info(train_bar.print(dict(**train_meters,)))


    val_metric_dict = {}
    return train_meters, val_metric_dict


