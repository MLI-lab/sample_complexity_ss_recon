
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

from functions.data.transforms import UnetDataTransform

from functions.data.mri_dataset import SliceDataset

from functions.models.unet import Unet

# Create scheduler and optimizer objects
from functions.training.training_functions import configure_optimizers, Compute_batch_train_loss

# Class that allows to  track the average of some quantity over an epoch
from functions.training.meters import AverageMeter

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

    # Get list of filenames logged to tensorboard during validation (from the validation set)
    val_log_filenames_list = []
    for k in hp_exp['log_val_images'].keys():
        val_log_filenames_list.append(k)

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

    data_transform_train = UnetDataTransform(hp_exp['challenge'],mask_func=mask_func, use_seed=hp_exp['use_mask_seed_for_training'], hp_exp=hp_exp,mode="train")
    data_transform_val = UnetDataTransform(hp_exp['challenge'],mask_func=mask_func, use_seed=True, hp_exp=hp_exp,mode="val")

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

    valset =  SliceDataset(
            dataset=hp_exp['val_set'],
            path_to_dataset=hp_exp['data_path'],
            path_to_sensmaps=hp_exp['smaps_path'],
            provide_senmaps=hp_exp['provide_senmaps'],
            challenge=hp_exp['challenge'],
            transform=data_transform_val,
            use_dataset_cache=True,
        )

    val_loader = torch.utils.data.DataLoader(
            dataset=valset,
            batch_size=1,
            num_workers=hp_exp['num_workers'],
            shuffle=False,
            generator=torch.Generator().manual_seed(hp_exp['seed']),
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

    valid_meters = {'val_SSIM' : AverageMeter(), 'val_PSNR' : AverageMeter(), 'val_L1' : AverageMeter(), 'val_L2' : AverageMeter(), 
                    'val_L2_kspace': AverageMeter(), 'val_L2_gt_abs': AverageMeter()}
    if hp_exp['two_channel_imag_real']:
        train_meters['train_L2_gt_comp'] = AverageMeter()
        valid_meters['val_L2_gt_comp'] = AverageMeter() 

    val_ssim_fct = SSIMLoss()
    val_l1_fct = L1Loss(reduction='sum')
    val_mse_fct = MSELoss()
    val_mse_reduceSum_fct = MSELoss(reduction='sum')

    # ------------
    # load a stored model if available
    # ------------
    if hp_exp['restore_file']:
        load_checkpoint(hp_exp, model, optimizer, scheduler)


    #########
    # Training
    save_train_figures = True
    save_val_figures = False
    mask_dict = {}
    for epoch in range(save_checkpoint.start_epoch, hp_exp['num_epochs']):
        start = time.process_time()
        train_bar = ProgressBar(train_loader, epoch)

        


        for meter in train_meters.values():
            meter.reset()

        for batch_id, batch in enumerate(train_bar):

            hp_exp['mode'] = 'train'
            model.train()

            save_checkpoint.global_step +=1 

            binary_background_mask, input_image, input_kspace, input_mask, target_image, target_kspace, target_mask, target_mask_weighted, ground_truth_image, sens_maps, mean, std, fname, slice_num = batch

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

            output = model(input_image)
            output = output * std + mean
            output_tensorboard = output.detach().clone()

            ################################
            # Compute the training loss
            ################################
   
            if hp_exp['selfsup'] or hp_exp['compute_sup_loss_in_kspace']:
                # move complex dim to end
                output_per_coil_imgs = torch.moveaxis(output , 1, -1 )

                output_per_coil_imgs = complex_mul(output_per_coil_imgs, sens_maps)

                # Transform coil images to kspace
                output_kspace = fft2c(output_per_coil_imgs)

                output_kspace = output_kspace * target_mask_weighted + 0.0
                target_kspace = target_kspace * target_mask_weighted + 0.0
                output_train_loss = output_kspace
                target_train_loss = target_kspace
            else:
                output_train_loss = output
                target_train_loss = target_image

            # Use max value per ground truth slice instead of per volume to compute ssim and psnr in image domain
            max_value = ground_truth_image.max().unsqueeze(0)

            train_loss = compute_batch_train_loss.get_batch_train_loss(hp_exp, output_train_loss, target_train_loss, max_value, train_meters)
            

            model.zero_grad()
            train_loss.backward()
            optimizer.step()

            ################################
            # Compute train metrics that can be compared over all different setupts
            ################################
            
            # Apply center cropping to outputs if necessary
            output_train_metrics, _ = center_crop_to_smallest(output_tensorboard, ground_truth_image)
            target_image_train_metrics, _ = center_crop_to_smallest(target_image, ground_truth_image)


            # Apply binary masking to outputs if binary masks are given
            binary_background_mask, _ = center_crop_to_smallest(binary_background_mask, ground_truth_image)
            output_train_metrics = output_train_metrics * binary_background_mask

            if hp_exp['two_channel_imag_real']:
                # target_image is already masked (if possible)
                loss = val_mse_reduceSum_fct(output_train_metrics, target_image_train_metrics) / torch.sum(torch.abs(target_image_train_metrics)**2)
                train_meters["train_L2_gt_comp"].update(loss.item())

                # prepare for train_L2_gt_abs
                output_train_metrics = complex_abs(torch.moveaxis(output_train_metrics , 1, -1 )).unsqueeze(1)

            loss = val_mse_reduceSum_fct(output_train_metrics, ground_truth_image) / torch.sum(torch.abs(ground_truth_image)**2)
            train_meters["train_L2_gt_abs"].update(loss.item())

            train_bar.log(dict(**train_meters), verbose=True)

            ################################
            # Log some training images to tensorboard
            ################################
            
            if hp_exp['tb_logging'] and fname[0] in train_log_filenames_list and epoch % hp_exp['log_image_interval'] == 0:
                if slice_num.item() == hp_exp['log_train_images'][fname[0]]:
                    with torch.no_grad():
                        if hp_exp['two_channel_imag_real']:
                            crop = False

                            name = f"train_{fname[0]}_s{slice_num.item()}_ch1/"+hp_exp['exp_name']
                            inp = input_image[:,0,:,:].unsqueeze(1)
                            tar = target_image[:,0,:,:].unsqueeze(1)
                            out = output_tensorboard[:,0,:,:].unsqueeze(1)
                            # If we want to look at center crop then crop target
                            if crop:
                                tar, _ = center_crop_to_smallest(tar, target_image)
                            add_img_to_tensorboard(writer, epoch, name, inp, tar, out, val_ssim_fct, max_value, crop)

                            name = f"train_{fname[0]}_s{slice_num.item()}_ch2/"+hp_exp['exp_name']
                            inp = input_image[:,1,:,:].unsqueeze(1)
                            tar = target_image[:,1,:,:].unsqueeze(1)
                            out = output_tensorboard[:,1,:,:].unsqueeze(1)
                            # If we want to look at center crop then crop target
                            if crop:
                                tar, _ = center_crop_to_smallest(tar, target_image)
                            add_img_to_tensorboard(writer, epoch, name, inp, tar, out, val_ssim_fct, max_value, crop)

                            name = f"train_{fname[0]}_s{slice_num.item()}_abs/"+hp_exp['exp_name']
                            inp = complex_abs(torch.moveaxis(input_image , 1, -1 )).unsqueeze(1)
                            tar = complex_abs(torch.moveaxis(target_image , 1, -1 )).unsqueeze(1)
                            out = complex_abs(torch.moveaxis(output_tensorboard , 1, -1 )).unsqueeze(1)
                            # If we want to look at center crop then crop target
                            if crop:
                                tar, _ = center_crop_to_smallest(tar, target_image)
                            add_img_to_tensorboard(writer, epoch, name, inp, tar, out, val_ssim_fct, max_value, crop)

                        else:
                            name = f"train_{fname[0]}_s{slice_num.item()}_abs/"+hp_exp['exp_name']
                            add_img_to_tensorboard(writer, epoch, name, input_image, target_image, output_tensorboard, val_ssim_fct, max_value, crop=True)
        


        if epoch % hp_exp['val_interval'] == 0:
            hp_exp['mode'] = 'val'
            model.eval()
            for meter in valid_meters.values():
                meter.reset()

            valid_bar = ProgressBar(val_loader, epoch)

            for sample_id, sample in enumerate(valid_bar):
                with torch.no_grad():
                    binary_background_mask, input_image, input_kspace, input_mask, target_image, target_kspace, target_mask, target_mask_weighted, ground_truth_image, sens_maps, mean, std, fname, slice_num = sample

                    input_image=input_image.to(device)
                    input_kspace=input_kspace.to(device)
                    input_mask=input_mask.to(device)
                    target_image=target_image.to(device)
                    target_kspace=target_kspace.to(device)
                    target_mask=target_mask.to(device)
                    target_mask_weighted=target_mask_weighted.to(device)
                    ground_truth_image=ground_truth_image.to(device)
                    sens_maps=sens_maps.to(device)
                    mean=mean.to(device)
                    std=std.to(device)
                    binary_background_mask=binary_background_mask.to(device)
                
                    output = model(input_image)

                    output = output * std + mean


                    ###############################################################
                    # Validation L1 and L2 are computed as during training
                    if hp_exp['selfsup'] or hp_exp['compute_sup_loss_in_kspace']:
                        # move complex dim to end
                        output_per_coil_imgs = torch.moveaxis(output , 1, -1 )

                        output_per_coil_imgs = complex_mul(output_per_coil_imgs, sens_maps)

                        # Transform coil images to kspace
                        output_kspace = fft2c(output_per_coil_imgs)
                        
                        output_kspace_fully_sampled = output_kspace.clone()

                        output_kspace = output_kspace * target_mask_weighted + 0.0
                        target_kspace = target_kspace * target_mask_weighted + 0.0
                        L2kspace = val_mse_reduceSum_fct(output_kspace, target_kspace) / torch.sum(torch.abs(target_kspace)**2)
                        valid_meters["val_L2_kspace"].update(L2kspace.item())

                        # L1 and L2 validation are computed on the full images without cropping and complex absolute value and without masking or dc
                        # L1 validation loss 
                        loss = val_l1_fct(output, target_image) / torch.sum(torch.abs(target_image))
                        valid_meters["val_L1"].update(loss.item())

                        # L2 validation loss 
                        loss = val_mse_reduceSum_fct(output, target_image) / torch.sum(torch.abs(target_image)**2)
                        valid_meters["val_L2"].update(loss.item())
                    else:
                        if hp_exp['two_channel_imag_real']:
                            # To enable data consistency later on
                            # move complex dim to end
                            output_per_coil_imgs = torch.moveaxis(output , 1, -1 )
                            output_per_coil_imgs = complex_mul(output_per_coil_imgs, sens_maps)
                            # Transform coil images to kspace
                            output_kspace = fft2c(output_per_coil_imgs)
                            #save_figure(torch.log(complex_abs(output_kspace[0,0,:,:]) + 1e-9).detach().cpu(),'output_kspace_val',hp_exp,save=save_val_figures)
                            output_kspace_fully_sampled = output_kspace.clone()
                            L2kspace = val_mse_reduceSum_fct(output_kspace, target_kspace) / torch.sum(torch.abs(target_kspace)**2)
                            valid_meters["val_L2_kspace"].update(L2kspace.item())

                        # L1 and L2 validation are computed on the full images without cropping and complex absolute value and without masking or dc
                        # L1 validation loss 
                        loss = val_l1_fct(output, target_image) / torch.sum(torch.abs(target_image))
                        valid_meters["val_L1"].update(loss.item())

                        # L2 validation loss 
                        loss = val_mse_reduceSum_fct(output, target_image) / torch.sum(torch.abs(target_image)**2)
                        valid_meters["val_L2"].update(loss.item())
                    ###############################################################

                    
                    ###############################################################
                    # Validation PSNR and SSIM are computed on masked, cropped and real images
                    # Apply masking before computing scores in the image domain in order to eliminate artifacts in the background
                        
                    output = output * binary_background_mask

                    output_tensorboard = output.clone()

                    # PSNR and SSIM are computed on the center cropped magnitude reconstruction
                    output, _ = center_crop_to_smallest(output, ground_truth_image)
                    if hp_exp['two_channel_imag_real']:
                        target_image_train_metrics, _ = center_crop_to_smallest(target_image, ground_truth_image)

                        loss = val_mse_reduceSum_fct(output, target_image_train_metrics) / torch.sum(torch.abs(target_image_train_metrics)**2)
                        valid_meters["val_L2_gt_comp"].update(loss.item())

                        # Move complex dim to end, apply complex abs, insert channel dimension
                        output = complex_abs(torch.moveaxis(output , 1, -1 ))
                        output = output.unsqueeze(1)

                    # Use max value per ground truth slice instead of per volume
                    max_value = ground_truth_image.max().unsqueeze(0)

                    # SSIM                    
                    ssim_loss = 1-val_ssim_fct(output, ground_truth_image, data_range=max_value)
                    valid_meters["val_SSIM"].update(ssim_loss.item())

                    loss = val_mse_reduceSum_fct(output, ground_truth_image) / torch.sum(torch.abs(ground_truth_image)**2)
                    valid_meters["val_L2_gt_abs"].update(loss.item())

                    # MSE for PSNR
                    loss = val_mse_fct(output, ground_truth_image) # reduce with mean

                    # PSNR
                    psnr = 20 * torch.log10(torch.tensor(max_value.item()))- 10 * torch.log10(loss)
                    valid_meters["val_PSNR"].update(psnr.item())

                    valid_bar.log(dict(**valid_meters), verbose=True)

                    if hp_exp['tb_logging'] and fname[0] in val_log_filenames_list and epoch % hp_exp['log_image_interval'] == 0:
                        if slice_num.item() == hp_exp['log_val_images'][fname[0]]:
                            if hp_exp['two_channel_imag_real']:
                                crop = False

                                name = f"val_{fname[0]}_s{slice_num.item()}_ch1/"+hp_exp['exp_name']
                                inp = input_image[:,0,:,:].unsqueeze(1)
                                tar = target_image[:,0,:,:].unsqueeze(1)
                                out = output_tensorboard[:,0,:,:].unsqueeze(1)
                                # If we want to look at center crop then crop target
                                if crop:
                                    tar, _ = center_crop_to_smallest(tar, ground_truth_image)
                                add_img_to_tensorboard(writer, epoch, name, inp, tar, out, val_ssim_fct, max_value, crop)

                                name = f"val_{fname[0]}_s{slice_num.item()}_ch2/"+hp_exp['exp_name']
                                inp = input_image[:,1,:,:].unsqueeze(1)
                                tar = target_image[:,1,:,:].unsqueeze(1)
                                out = output_tensorboard[:,1,:,:].unsqueeze(1)
                                # If we want to look at center crop then crop target
                                if crop:
                                    tar, _ = center_crop_to_smallest(tar, ground_truth_image)
                                add_img_to_tensorboard(writer, epoch, name, inp, tar, out, val_ssim_fct, max_value, crop)

                                name = f"val_{fname[0]}_s{slice_num.item()}_abs/"+hp_exp['exp_name']
                                inp = complex_abs(torch.moveaxis(input_image , 1, -1 )).unsqueeze(1)
                                tar = complex_abs(torch.moveaxis(target_image , 1, -1 )).unsqueeze(1)
                                out = complex_abs(torch.moveaxis(output_tensorboard , 1, -1 )).unsqueeze(1)
                                # If we want to look at center crop then crop target
                                if crop:
                                    tar, _ = center_crop_to_smallest(tar, ground_truth_image)
                                add_img_to_tensorboard(writer, epoch, name, inp, tar, out, val_ssim_fct, max_value, crop)

                            else:
                                name = f"val_{fname[0]}_s{slice_num.item()}_abs/"+hp_exp['exp_name']
                                add_img_to_tensorboard(writer, epoch, name, input_image, target_image, output_tensorboard, val_ssim_fct, max_value, crop=True)

            if hp_exp['two_channel_imag_real']:
                val_metric_dict = { #keys should have the same name as the keys used to pick a training loss in hp_exp['loss_functions']
                    'SSIM' : valid_meters['val_SSIM'].avg,
                    'L1' : valid_meters['val_L1'].avg,
                    'L2' : valid_meters['val_L2'].avg,
                    'PSNR' : valid_meters['val_PSNR'].avg,
                    'L2_kspace' : valid_meters['val_L2_kspace'].avg,
                    'L2_gt_abs' : valid_meters["val_L2_gt_abs"].avg,
                    'L2_gt_comp' : valid_meters["val_L2_gt_comp"].avg,
                }
            else:
                val_metric_dict = { #keys should have the same name as the keys used to pick a training loss in hp_exp['loss_functions']
                    'SSIM' : valid_meters['val_SSIM'].avg,
                    'L1' : valid_meters['val_L1'].avg,
                    'L2' : valid_meters['val_L2'].avg,
                    'PSNR' : valid_meters['val_PSNR'].avg,
                    'L2_kspace' : valid_meters['val_L2_kspace'].avg,
                    'L2_gt_abs' : valid_meters["val_L2_gt_abs"].avg,
                }

            
            current_lr = save_checkpoint.current_lr
            current_best_score = save_checkpoint.best_score

            # Logging to tensorboard
            if hp_exp['tb_logging']:
                writer.add_scalar("lr", current_lr, epoch)
                writer.add_scalar("epoch", epoch, epoch)
                for tr_loss_name in train_meters.keys():
                    writer.add_scalar(tr_loss_name, train_meters[tr_loss_name].avg, epoch)
                for val_loss_name in val_metric_dict.keys():
                    writer.add_scalar('val_'+val_loss_name, val_metric_dict[val_loss_name], epoch)
                sys.stdout.flush()

            #### Learning rate decay
            score = val_metric_dict[hp_exp['decay_metric']]
            if hp_exp['lr_scheduler'] == 'MultiStepLR':
                scheduler.step()
            elif hp_exp['lr_scheduler'] == 'ReduceLROnPlateau':
                scheduler.step(score)
            else:
                raise ValueError('Scheduler is not defined')

            save_checkpoint(hp_exp, epoch, model, optimizer, scheduler, score=score) # This potentially updates save_checkpoint.best_score

            end = time.process_time() - start
            # Logging to train.log
            if (val_metric_dict[hp_exp['decay_metric']] < current_best_score and save_checkpoint.mode == "min") or (val_metric_dict[hp_exp['decay_metric']] > current_best_score and save_checkpoint.mode == "max"):
                logging.info(train_bar.print(dict(**train_meters, **valid_meters, lr=current_lr, time=np.round(end/60,3), New='Highscore')))
            else:
                logging.info(train_bar.print(dict(**train_meters, **valid_meters, lr=current_lr, time=np.round(end/60,3))))

            new_lr = optimizer.param_groups[0]["lr"]
            save_checkpoint.current_lr = new_lr # current lr during next epoch

            if hp_exp['early_stop_lr_deacy']:

                if (score < current_best_score and save_checkpoint.mode == "min") or (score > current_best_score and save_checkpoint.mode == "max"):
                    save_checkpoint.best_val_current_lr_interval = score
                #At every lr decay check if the model did not improve during the lr_convergence_break_counter many lastt lr intervals and break if it didn't.
                if new_lr < current_lr: 
                    if save_checkpoint.best_val_current_lr_interval != save_checkpoint.best_score:
                        save_checkpoint.lr_interval_counter += 1
                        if save_checkpoint.lr_interval_counter == hp_exp['lr_convergence_break_counter']:
                            logging.info(f'lr decayed to {new_lr}. Break training due to convergence of val loss!')
                            break
                        else:
                            logging.info(f'lr decayed to {new_lr}. lr_interval_counter increased but do not yet break due to convergence of val loss!')
                    else:
                        save_checkpoint.best_val_current_lr_interval = float("inf") if  mode_lookup[hp_exp['decay_metric']] == "min" else float("-inf")
                        save_checkpoint.lr_interval_counter = 0
                        logging.info(f'lr decayed to {new_lr}. No convergence detected. Reset lr_interval_counter.')
                
                if np.round(current_lr,10) <= hp_exp['lr_min']:
                    if hp_exp['lr_min_break_counter'] == save_checkpoint.break_counter:
                        logging.info('Break training due to minimal learning rate constraint!')
                        break
                    else:
                        save_checkpoint.break_counter += 1

        else:
            current_lr = save_checkpoint.current_lr
            if hp_exp['lr_scheduler'] == 'MultiStepLR':
                scheduler.step()
            else:
                raise ValueError('Scheduler is not defined')
            if hp_exp['tb_logging']:
                writer.add_scalar("lr", current_lr, epoch)
                writer.add_scalar("epoch", epoch, epoch)
                for tr_loss_name in train_meters.keys():
                    writer.add_scalar(tr_loss_name, train_meters[tr_loss_name].avg, epoch)

            new_lr = optimizer.param_groups[0]["lr"]
            save_checkpoint.current_lr = new_lr # current lr during next epoch
            end = time.process_time() - start
            logging.info(train_bar.print(dict(**train_meters, lr=current_lr, time=np.round(end/60,3))))

    logging.info(f"Done training! Best PSNR {save_checkpoint.best_score:.5f} obtained after epoch {save_checkpoint.best_epoch}.")

    # return names of matrics logged to tensorboard
    return train_meters, val_metric_dict


def main_test(hp_exp):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # ------------
    # setup:
    # Set seeds, create directories, set path to checkpoints if available
    # ------------
    hp_exp = setup_experiment(hp_exp)
    init_logging(hp_exp)

    # Get list of filenames saved during testing (from the test set)
    test_log_filenames_list = []
    for k in hp_exp['save_test_images'].keys():
        test_log_filenames_list.append(k)

    # ------------
    # data
    # ------------
    # For testing we want the target mask to be all ones. 
    # This can be achieved either by setting self_sup=False or acceleration_total=1.0
    mask_func = create_mask_for_mask_type(
        hp_exp['mask_type'], self_sup=False, center_fraction=hp_exp['center_fraction'], acceleration=hp_exp['acceleration'], acceleration_total=1.0
    )

    data_transform = UnetDataTransform(hp_exp['challenge'],mask_func=mask_func, use_seed=True, hp_exp=hp_exp, mode="test")

    testset =  SliceDataset(
            dataset=hp_exp['test_set'],
            path_to_dataset=hp_exp['data_path'],
            path_to_sensmaps=hp_exp['smaps_path'],
            provide_senmaps=hp_exp['provide_senmaps'],
            challenge=hp_exp['challenge'],
            transform=data_transform,
            use_dataset_cache=True,
        )

    test_loader = torch.utils.data.DataLoader(
            dataset=testset,
            batch_size=1,
            num_workers=hp_exp['num_workers'],
            shuffle=False,
            generator=torch.Generator().manual_seed(hp_exp['seed']),
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

    # ------------
    # load a stored model if available
    # ------------
    if hp_exp['restore_file']:
        load_checkpoint(hp_exp, model, None, None)


    test_ssim_fct = SSIMLoss()
    test_l1_fct_sum = L1Loss(reduction='sum')
    test_mse_fct_sum = MSELoss(reduction='sum')
    test_mse_fct_mean = MSELoss(reduction='mean')

    model.eval()

    test_bar = ProgressBar(test_loader, epoch=0)
    # Collect scores
    ssim_vals = []
    L1_vals = []
    psnr_vals = []
    L2_vals = []

    # Always compute both, scores after binary masking and after data consistency.
    ssim_vals_dc = []
    L1_vals_dc = []
    psnr_vals_dc = []
    L2_vals_dc = []

    for sample_id, sample in enumerate(test_bar):
        with torch.no_grad():
            binary_background_mask, input_image, input_kspace, input_mask, target_image, target_kspace, target_mask, target_mask_weighted, ground_truth_image, sens_maps, mean, std, fname, slice_num = sample

            input_image=input_image.to(device)
            input_kspace=input_kspace.to(device)
            input_mask=input_mask.to(device)
            target_kspace=target_kspace.to(device)
            target_mask=target_mask.to(device)
            ground_truth_image=ground_truth_image.to(device)
            sens_maps=sens_maps.to(device)
            mean=mean.to(device)
            std=std.to(device)
            binary_background_mask=binary_background_mask.to(device)
        
            output = model(input_image)

            output = output * std + mean

            #####################        
            sens_maps_conj = complex_conj(sens_maps)
            output_per_coil_imgs = torch.moveaxis(output.clone() , 1, -1 )
            output_per_coil_imgs = complex_mul(output_per_coil_imgs, sens_maps)

            # Transform coil images to kspace
            output_kspace = fft2c(output_per_coil_imgs)

            ################
            
            ################
            # Get scores in image domain after data consistency
            output_image_data_consistency = ifft2c(output_kspace* (1-input_mask) + input_kspace)
            output_image_data_consistency = complex_mul(output_image_data_consistency, sens_maps_conj)
            output_image_data_consistency = output_image_data_consistency.sum(dim=1, keepdim=False)

            output_image_data_consistency = torch.moveaxis(output_image_data_consistency , -1, 1)
            output_image_data_consistency, _ = center_crop_to_smallest(output_image_data_consistency, ground_truth_image)
            output_image_data_consistency = complex_abs(torch.moveaxis(output_image_data_consistency , 1, -1 ))
            output_image_dc = output_image_data_consistency.unsqueeze(1)

            # L1 
            loss = test_l1_fct_sum(output_image_dc, ground_truth_image) / torch.sum(torch.abs(ground_truth_image))
            L1_vals_dc.append(loss.item())

            # L2 
            loss = test_mse_fct_sum(output_image_dc, ground_truth_image) / torch.sum(torch.abs(ground_truth_image)**2)
            L2_vals_dc.append(loss.item())

            max_value = ground_truth_image.max().unsqueeze(0)

            # MSE for PSNR
            mse = test_mse_fct_mean(output_image_dc, ground_truth_image)
            
            # PSNR
            psnr = 20 * torch.log10(torch.tensor(max_value.item()))- 10 * torch.log10(mse)
            psnr_vals_dc.append(psnr.item())

            # SSIM
            ssim_loss = 1-test_ssim_fct(output_image_dc, ground_truth_image, data_range=max_value)
            ssim_vals_dc.append(ssim_loss.item())

            output = output * binary_background_mask
            ######################

            ######################
            # Get scores after binary masking without data consistency
            # at test time L1, L2, PSNR and SSIM are all computed on center cropped magnitude values
            output, _ = center_crop_to_smallest(output, ground_truth_image)
            if hp_exp['two_channel_imag_real']:
                # Move complex dim to end, apply complex abs, insert channel dimension
                output = complex_abs(torch.moveaxis(output , 1, -1 ))
                output = output.unsqueeze(1)

            # L1 
            loss = test_l1_fct_sum(output, ground_truth_image) / torch.sum(torch.abs(ground_truth_image))
            L1_vals.append(loss.item())

            # L2 
            loss = test_mse_fct_sum(output, ground_truth_image) / torch.sum(torch.abs(ground_truth_image)**2)
            L2_vals.append(loss.item())

            # Normalize output to mean and std of target
            #target, output = normalize_to_given_mean_std(target, output)

            # Use max value per ground truth slice instead of per volume
            max_value = ground_truth_image.max().unsqueeze(0)

            # MSE for PSNR
            mse = test_mse_fct_mean(output, ground_truth_image)
            
            # PSNR
            psnr = 20 * torch.log10(torch.tensor(max_value.item()))- 10 * torch.log10(mse)
            psnr_vals.append(psnr.item())

            # SSIM
            ssim_loss = 1-test_ssim_fct(output, ground_truth_image, data_range=max_value)
            ssim_vals.append(ssim_loss.item())
            ######################

            # Save some test images
            if fname[0] in test_log_filenames_list:
                if slice_num.item() == hp_exp['save_test_images'][fname[0]]:
                    error = torch.abs(ground_truth_image - output)
                    error_dc = torch.abs(ground_truth_image - output_image_dc)
                    output = output - output.min() 
                    output = output / output.max()
                    output_image_dc = output_image_dc - output_image_dc.min() 
                    output_image_dc = output_image_dc / output_image_dc.max()
                    ground_truth_image = ground_truth_image - ground_truth_image.min()
                    ground_truth_image = ground_truth_image / ground_truth_image.max()
                    error = error - error.min() 
                    error_dc = error_dc - error_dc.min() 
                    max_norm = torch.stack([error,error_dc]).max()
                    error = error / max_norm
                    error_dc = error_dc / max_norm

                    image = torch.cat([ground_truth_image, ground_truth_image, output, output_image_dc, error, error_dc], dim=0)
                    image = torchvision.utils.make_grid(image, nrow=2, normalize=False, value_range=(0,1), pad_value=1)
                    figure = get_figure(image.cpu().numpy(),figsize=(8,12),title=f"ssim={ssim_loss.item():.6f}, ssim_dc={ssim_vals_dc[-1]:.6f}") 
                    if not os.path.isdir(hp_exp['log_path'] + 'test_imgs/'):
                        os.mkdir(hp_exp['log_path'] + 'test_imgs/')
                    plt.savefig(hp_exp['log_path'] + f"test_imgs/{fname[0]}_s{slice_num.item()}.png", dpi='figure')
                    plt.close()


    test_metric_dict = {
        'ssim_m' : np.mean(np.array(ssim_vals)),
        'ssim_s' : np.std(np.array(ssim_vals)),
        'L1_m' : np.mean(np.array(L1_vals)),
        'L1_s' : np.std(np.array(L1_vals)),
        'psnr_m' : np.mean(np.array(psnr_vals)),
        'psnr_s' : np.std(np.array(psnr_vals)),
        'L2_m' : np.mean(np.array(L2_vals)),
        'L2_s' : np.std(np.array(L2_vals)),
    }
    print(test_metric_dict)

    test_metric_dict_dc = {
        'ssim_m' : np.mean(np.array(ssim_vals_dc)),
        'ssim_s' : np.std(np.array(ssim_vals_dc)),
        'L1_m' : np.mean(np.array(L1_vals_dc)),
        'L1_s' : np.std(np.array(L1_vals_dc)),
        'psnr_m' : np.mean(np.array(psnr_vals_dc)),
        'psnr_s' : np.std(np.array(psnr_vals_dc)),
        'L2_m' : np.mean(np.array(L2_vals_dc)),
        'L2_s' : np.std(np.array(L2_vals_dc)),
    }
    print(test_metric_dict_dc)
    
    testset_name = hp_exp['log_path'] + hp_exp['test_set'][hp_exp['test_set'].find('datasets/')+9 : hp_exp['test_set'].find('.yaml')]
    pickle.dump( test_metric_dict, open(testset_name + '_metrics_' + hp_exp['resume_from_which_checkpoint'] + '.p', "wb" ) )

    pickle.dump( test_metric_dict_dc, open(testset_name + '_metrics_DC_' + hp_exp['resume_from_which_checkpoint'] + '.p', "wb" ) )

    logging.info("Evaluate testset : {}".format(testset_name))
    for test_metric in test_metric_dict.keys():
        logging.info("{}: {}".format(test_metric, test_metric_dict[test_metric]))
    logging.info("Evaluate testset : {} with data consistency".format(testset_name))
    for test_metric in test_metric_dict_dc.keys():
        logging.info("{}: {}".format(test_metric, test_metric_dict_dc[test_metric]))


    

