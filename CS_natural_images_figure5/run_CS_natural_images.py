# %%
import torch
import h5py
import numpy as np
import os
import yaml
import logging
import glob
import json
import random
import pickle
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt
from torch.nn import MSELoss

from argparse import ArgumentParser

from torch.utils.tensorboard import SummaryWriter
import sys

from CS_natural_images_functions.unet import Unet
from CS_natural_images_functions.fftc import fft2c, ifft2c
from CS_natural_images_functions.losses import SSIMLoss
from CS_natural_images_functions.progress_bar import ProgressBar, init_logging, AverageMeter, TrackMeter, TrackMeter_testing
from CS_natural_images_functions.log_progress_helpers import save_figure, add_img_to_tensorboard, save_test_image_with_dc
from CS_natural_images_functions.load_save_model_helpers import setup_experiment_or_load_checkpoint, save_checkpoint

from CS_natural_images_functions.data_transforms import UnetDataTransform, CropDataset
from CS_natural_images_functions.data_transforms import compute_number_of_lines_in_input_target_kspace


# %%


def read_args():
    parser = ArgumentParser()
    
    # Required arguments
    parser.add_argument(
        '--config_file',
        type=str,
        help='Name of a config file in the experiment_configs folder.',
        required=True
    )

    parser.add_argument(
        '--path_to_ImageNet_train',
        type=str,
        help='Path to ImageNet train directory.',
        required=True
    )

    parser.add_argument(
        '--experiment_number',
        type=str,
        help="Set a unique identifier for the folder containing the experimental results. Start number with '001'. ",
        required=True
    )

    parser.add_argument(
        '--run_which_seeds',
        type=str,
        choices=('run_best_seed','run_all_seeds'),
        help='Choose to run either only the best seed or all seeds shown in our results.',
        required=True
    )

    # Optional arguments
    parser.add_argument(
        '--training',
        default=True,
        action='store_false',
        help='Add this flag to disable training.'
    )

    parser.add_argument(
        '--testing',
        default=True,
        action='store_false',
        help='Add this flag to disable testing.'
    )

    parser.add_argument(
        '--gpu',
        choices=(0, 1, 2, 3),
        default=3,
        type=int,
        help='Pick one out of four gpus.'
    )

    parser.add_argument(
        '--trainset_size',
        choices=(50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000),
        default=50,
        type=int,
        help='Set training set size.'
    )

    parser.add_argument(
        '--img_size',
        default=100,
        type=int,
        help='Set img_size for downsampling.'
    )

    parser.add_argument(
        '--num_epochs',
        default=1000,
        type=int,
        help='Set number of training epochs.'
    )

    parser.add_argument(
        '--val_epoch_interval',
        default=2,
        type=int,
        help='Set how often the validation loss is computed.'
    )

    parser.add_argument(
        '--patience',
        default=10,
        type=int,
        help='Patience parameter for the learning rate scheduler.'
    )

    parser.add_argument(
        '--acceleration',
        default=4.0,
        type=float,
        help='Undersampling of training and test inputs.'
    )

    parser.add_argument(
        '--acceleration_total',
        default=1.0,
        type=float,
        help='Undersampling of data available for input target split. Set to 1 for supervised training.'
    )

    parser.add_argument(
        '--center_fraction',
        default=0.08,
        type=float,
        help='Fraction of lines that are always sample from the center (input and target). Set to 0.0 for sampling all lines randomly.'
    )

    parser.add_argument(
        '--fix_split',
        default=True,
        action='store_true',
        help='Add this flag to set use_seed=True for fixed input target split for self-supervised and fixed input for supervised training.'
    )

    args = parser.parse_args()

    with open("experiment_configs/"+args.config_file) as handle:
        config_file = json.load(handle)

    args.acceleration_total = config_file['acceleration_total']
    args.acceleration = config_file['acceleration']
    args.trainset_size = config_file['trainset_size']
    args.val_epoch_interval = config_file['val_epoch_interval']
    args.patience = config_file['patience']

    if args.run_which_seeds == 'run_best_seed':
        seeds = [config_file['best_seed']]
    elif args.run_which_seeds == 'run_all_seeds':
        seeds = config_file['all_seeds']

    for seed in seeds:

        experiment_name = f"N{args.experiment_number}_t{args.trainset_size}_"
        if args.acceleration_total==1.0:
            experiment_name+="sup_" 
        else: 
            experiment_name+="selfsup_"
        if args.fix_split:
            experiment_name+="fixInput_" 
        else: 
            experiment_name+="RandInput_"
        experiment_name += f"run{seed}"
        
        experiment_path = experiment_name+"/"

        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

        if args.training:
            run_training(experiment_path=experiment_path,
                        acceleration=args.acceleration,
                        center_fraction=args.center_fraction,
                        acceleration_total=args.acceleration_total,
                        seed=seed,
                        img_size=args.img_size,
                        fix_split=args.fix_split,
                        val_epoch_interval=args.val_epoch_interval,
                        num_epochs=args.num_epochs,
                        patience=args.patience,
                        trainset_size=args.trainset_size,
                        path_to_ImageNet_train=args.path_to_ImageNet_train)
        if args.testing:
            run_testing(experiment_path=experiment_path,
                    acceleration=args.acceleration,
                    center_fraction=args.center_fraction,
                    acceleration_total=args.acceleration_total,
                    img_size=args.img_size,
                    path_to_ImageNet_train=args.path_to_ImageNet_train)

################################################################################################

def run_training(experiment_path,
                 acceleration,
                 center_fraction,
                 acceleration_total,
                 seed,
                 img_size,
                 fix_split,
                 val_epoch_interval,
                 num_epochs,
                 patience,
                 trainset_size,
                 path_to_ImageNet_train):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Create directory that holds train files
    if not os.path.isdir(experiment_path):
        os.mkdir(experiment_path)
    else:
        print(experiment_path)
        #raise ValueError("Experiment already exists!!") 
        print("Warning: Experiment already exists!!") 

    # Init train.log file
    init_logging(experiment_path)

    logging.info("Training...")

    # Log sanity checks on the number of lines in the input/target kspaces
    input_size, target_size, overlap_size_high, size_low, p, q, mu, nu, weight_on_random_lines = compute_number_of_lines_in_input_target_kspace(p=1/acceleration,mu=1/acceleration_total,nu=center_fraction, n=img_size)
    logging.info(f"mu: {mu}, p: {p}, q: {q}, nu: {nu}, weight_on_random_lines: {weight_on_random_lines}")
    logging.info(f"\n Lines in kspace: {img_size} \n Lines in input: {input_size} \n Lines in target: {target_size} \n Number of high freq overlapping lines: {overlap_size_high} \n Number of low freq lines: {size_low}")

    # train loss function
    loss_fct = MSELoss(reduction='sum')
    val_ssim_fct = SSIMLoss()

    # Init model
    model = Unet(
                in_chans=2,
                out_chans=2,
                chans=24,
                num_pool_layers=3,
                drop_prob=0.0,).to(device)
    logging.info(f"Built a model consisting of {sum(p.numel() for p in model.parameters()):,} parameters")#

    # Init optimizer and scheduler
    optimizer = torch.optim.Adam( params=model.parameters(),  lr=0.001,  betas=(0.9, 0.999),  eps=1e-08,  weight_decay=0.0,  amsgrad=False)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, 
            mode='max', 
            factor=0.1, 
            patience=patience, 
            threshold=0.0001, 
            threshold_mode='abs', 
            cooldown=0, 
            min_lr=1e-5, 
            eps=1e-08, 
            verbose=True
            )

    # If a checkpoint exists, it is automtically loaded
    setup_experiment_or_load_checkpoint(experiment_path, resume_from='best', model=model, optimizer=optimizer, scheduler=scheduler)

    # Load train set
    train_pool = torch.load('CS_natural_images_functions/training_set_lists/trsize1000000_filepaths.pt')
    zero_norm_files = ['train/n03729826/n03729826_6483.JPEG',
        'train/n04515003/n04515003_24673.JPEG',
        'train/n02111277/n02111277_12490.JPEG',
        'train/n03888605/n03888605_9775.JPEG',
        'train/n02992529/n02992529_3197.JPEG',
        'train/n01930112/n01930112_18908.JPEG',
        'train/n06874185/n06874185_3219.JPEG',
        'train/n06785654/n06785654_17232.JPEG',
        'train/n04033901/n04033901_29617.JPEG',
        'train/n07920052/n07920052_14729.JPEG',
        'train/n03729826/n03729826_40479.JPEG',
        'train/n03729826/n03729826_10716.JPEG',
        'train/n04286575/n04286575_74296.JPEG',
        'train/n03937543/n03937543_10198.JPEG',
        'train/n03063599/n03063599_3942.JPEG',
        'train/n04152593/n04152593_13802.JPEG',
        'train/n04522168/n04522168_24105.JPEG',
        'train/n03532672/n03532672_78983.JPEG',
        'train/n04404412/n04404412_12316.JPEG',
        'train/n04330267/n04330267_18003.JPEG',
        'train/n04118776/n04118776_37671.JPEG',
        'train/n04591713/n04591713_3568.JPEG',
        'train/n02437616/n02437616_12697.JPEG',
        'train/n02799071/n02799071_54867.JPEG',
        'train/n02883205/n02883205_26196.JPEG',
        'train/n02667093/n02667093_2919.JPEG',
        'train/n03196217/n03196217_1135.JPEG',
        'train/n03196217/n03196217_3568.JPEG',
        'train/n15075141/n15075141_19601.JPEG',
        'train/n01943899/n01943899_24166.JPEG']
    for zero_norm_file in zero_norm_files:
        train_pool.remove(zero_norm_file)

    rng_dataset = np.random.default_rng(seed)
    if trainset_size == 1000000:
        train_set = train_pool
    else:
        train_set = rng_dataset.choice(train_pool, size=trainset_size, replace=False, p=None)

    torch.save(train_set,experiment_path+'train_set.pt')

    validation_set = torch.load('CS_natural_images_functions/training_set_lists/ImageNetVal80_filepaths.pt')

    # Train loader
    data_transform_train = UnetDataTransform(acceleration=acceleration,acceleration_total=acceleration_total, fix_split=fix_split, experiment_path=experiment_path,center_fraction=center_fraction)
    trainset = CropDataset(dataset=train_set, path_to_ImageNet_train=path_to_ImageNet_train, transform=data_transform_train, experiment_path=experiment_path, img_size=img_size)
    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=1, num_workers=0, shuffle=True, generator=torch.Generator().manual_seed(0))

    # Val loader
    data_transform_val = UnetDataTransform(acceleration=acceleration,acceleration_total=acceleration_total, fix_split=True, experiment_path=experiment_path,center_fraction=center_fraction)
    valset = CropDataset(dataset=validation_set, path_to_ImageNet_train=path_to_ImageNet_train, transform=data_transform_val, experiment_path=experiment_path, img_size=img_size)
    val_loader = torch.utils.data.DataLoader( dataset=valset, batch_size=1, num_workers=0, shuffle=False, generator=torch.Generator().manual_seed(0))

    # store training loss metrics
    train_meters = {'train_L2': AverageMeter()}
    train_tracks = {'train_L2': TrackMeter('decaying')}
    # store validation metrics
    valid_meters = {'val_SSIM' : AverageMeter(), 'val_PSNR' : AverageMeter(), 'val_L2' : AverageMeter(), 'val_L2_kspace': AverageMeter()}
    valid_tracks = {'val_SSIM' : TrackMeter('increasing'), 'val_PSNR' : TrackMeter('increasing'), 'val_L2' : TrackMeter('decaying'), 'val_L2_kspace': TrackMeter('decaying')}

    # Init tensorboard
    writer = SummaryWriter(log_dir=experiment_path)
    log_image_interval_tb = 30

    break_counter=0
    # Start training
    for epoch in range(save_checkpoint.start_epoch, num_epochs):
        train_bar = ProgressBar(train_loader, epoch)

        for meter in train_meters.values():
            meter.reset()

        for id,sample in enumerate(train_bar):
            model.train()

            y_input, x_input, y_target, x_target, x, input_mask, target_mask, mean, std, fname = sample


            # sanity check on number of lines in input and target mask
            if epoch==0 and id==0:
                tm = target_mask.detach()
                target_mask_no_zeros = torch.where(tm != 0., tm , torch.tensor(1, dtype=tm.dtype).to(device))
                target_mask_norm_to_one = tm / target_mask_no_zeros
                logging.info(f"\n Mask sanity check! Lines in kspace: {input_mask.shape[-2]} \n Lines in input: {torch.sum(input_mask)} \n Lines in target: {torch.sum(target_mask_norm_to_one)} \n Number of all overlapping lines: {torch.sum(input_mask*target_mask_norm_to_one)}")

            # prediction
            x_output = torch.moveaxis(model(torch.moveaxis( x_input , -1, 1 )), 1, -1)

            # unnormalize
            x_output = x_output * std + mean

            # move to kspace
            y_output = fft2c(x_output)

            # apply target mask (all ones for supervised training)
            y_output = y_output * target_mask + 0.0

            # compute loss
            train_loss = loss_fct(y_output,y_target) / torch.sum(torch.abs(y_target)**2)

            model.zero_grad()
            train_loss.backward()
            optimizer.step()

            # log train metrics
            train_meters['train_L2'].update(train_loss.item())
            train_bar.log(dict(**train_meters), verbose=True)

            if id ==0: # log a random train image to tensorboard
                name = f"train_0_img"
                add_img_to_tensorboard(writer, epoch, name, x_input.detach(),x_output.detach(),x_target.detach(),ksp=False) if epoch % log_image_interval_tb == 0 else None
                name = f"train_0_ksp"
                add_img_to_tensorboard(writer, epoch, name, y_input.detach(),y_output.detach(),y_target.detach(),ksp=True) if epoch % log_image_interval_tb == 0 else None
            if id ==1: # log a specific train image to tensorboard
                name = f"train_1_img"
                add_img_to_tensorboard(writer, epoch, name, x_input.detach(),x_output.detach(),x_target.detach(),ksp=False) if epoch % log_image_interval_tb == 0 else None
                name = f"train_1_ksp"
                add_img_to_tensorboard(writer, epoch, name, y_input.detach(),y_output.detach(),y_target.detach(),ksp=True) if epoch % log_image_interval_tb == 0 else None

        train_tracks['train_L2'].update(train_meters['train_L2'].avg,epoch)
        current_lr = optimizer.param_groups[0]["lr"]
        #scheduler.step()
        
        ############################################################################################################################
        if epoch % val_epoch_interval == 0: # set this value such that it works with save_at_epochs and log_image_interval_tb
            model.eval()
            for meter in valid_meters.values():
                meter.reset()

            valid_bar = ProgressBar(val_loader, epoch)

            rand_id = random.randint(0, len(val_loader)) # draw id to log a random slice to tensorboard
            for id, sample in enumerate(valid_bar):
                with torch.no_grad():
                    y_input, x_input, y_target, x_target, x, input_mask, target_mask, mean, std, fname = sample

                    # prediction
                    x_output = torch.moveaxis(model(torch.moveaxis( x_input , -1, 1 )), 1, -1)

                    # unnormalize
                    x_output = x_output * std + mean

                    # move to kspace
                    y_output = fft2c(x_output)

                    if id ==0: # log one fixed and one random validation image to tensorboard
                        name = f"val_0_img"
                        add_img_to_tensorboard(writer, epoch, name, x_input.detach(),x_output.detach(),x.detach(),ksp=False) if epoch % log_image_interval_tb == 0 else None
                        name = f"val_0_ksp"
                        add_img_to_tensorboard(writer, epoch, name, y_input.detach(),y_output.detach(),y_target.detach(),ksp=True) if epoch % log_image_interval_tb == 0 else None
                    elif  id==rand_id: # log one fixed and one random validation image to tensorboard
                        name = f"val_1_img"
                        add_img_to_tensorboard(writer, epoch, name, x_input.detach(),x_output.detach(),x.detach(),ksp=False) if epoch % log_image_interval_tb == 0 else None
                        name = f"val_1_ksp"
                        add_img_to_tensorboard(writer, epoch, name, y_input.detach(),y_output.detach(),y_target.detach(),ksp=True) if epoch % log_image_interval_tb == 0 else None

                    # apply target mask (all ones for supervised training)
                    y_output = y_output * target_mask + 0.0

                    # val loss in kspace (L2)
                    val_loss = loss_fct(y_output,y_target) / torch.sum(torch.abs(y_target)**2)
                    valid_meters['val_L2_kspace'].update(val_loss)
                    
                    # L2 in image domain between complex output and target image
                    val_loss = loss_fct(x_output,x) / torch.sum(torch.abs(x)**2)
                    valid_meters['val_L2'].update(val_loss)
                    
                    output_magnitude = (x_output ** 2).sum(dim=-1).sqrt()
                    x_magnitude = (x ** 2).sum(dim=-1).sqrt() # since x is real, this operation is identity
                    x_magnitude = x_magnitude.unsqueeze(1)
                    output_magnitude = output_magnitude.unsqueeze(1)
                    
                    # psnr
                    max_value = x.max().unsqueeze(0)
                    mse = torch.mean(torch.abs(output_magnitude-x_magnitude)**2)
                    psnr = 20 * torch.log10(torch.tensor(max_value.item()))- 10 * torch.log10(mse)
                    valid_meters["val_PSNR"].update(psnr.item())
                    
                    # ssim
                    ssim_loss = 1-val_ssim_fct(output_magnitude, x_magnitude, data_range=max_value)
                    valid_meters["val_SSIM"].update(ssim_loss.item())

            # log progrss
            valid_tracks['val_L2_kspace'].update(valid_meters['val_L2_kspace'].avg,epoch)
            valid_tracks['val_L2'].update(valid_meters['val_L2'].avg,epoch)
            valid_tracks['val_PSNR'].update(valid_meters['val_PSNR'].avg,epoch)
            valid_tracks['val_SSIM'].update(valid_meters['val_SSIM'].avg,epoch)
            valid_bar.log(dict(**valid_meters), verbose=True)

            scheduler.step(valid_meters['val_PSNR'].avg)
            if current_lr > optimizer.param_groups[0]["lr"]:
                scheduler.patience= scheduler.patience//2
            if current_lr == scheduler.min_lrs[0]:
                break_counter+=1
                if break_counter == 3:
                    break
            

            if save_checkpoint.best_score < valid_meters['val_PSNR'].avg:
                if scheduler.num_bad_epochs == 0:
                    logging.info(train_bar.print(dict(**train_meters, **valid_meters, lr=current_lr, New='Highscore', Scheduler_patience='reset')))
                else:
                    logging.info(train_bar.print(dict(**train_meters, **valid_meters, lr=current_lr, New='Highscore')))
            else:
                if scheduler.num_bad_epochs == 0:
                    logging.info(train_bar.print(dict(**train_meters, **valid_meters, lr=current_lr, Scheduler_patience='reset')))
                else:
                    logging.info(train_bar.print(dict(**train_meters, **valid_meters, lr=current_lr)))

            writer.add_scalar("lr", current_lr, epoch)
            writer.add_scalar("epoch", epoch, epoch)
            writer.add_scalar("train_L2", train_meters["train_L2"].avg, epoch)
            for val_loss_name in valid_meters.keys():
                writer.add_scalar(val_loss_name, valid_meters[val_loss_name].avg, epoch)
            sys.stdout.flush()

            # Save checkpoint
            save_checkpoint(experiment_path, epoch, model, optimizer=optimizer, scheduler=scheduler, score=valid_meters['val_PSNR'].avg, save_at_epochs=[])
        else:
            logging.info(train_bar.print(dict(**train_meters, lr=current_lr)))
            writer.add_scalar("lr", current_lr, epoch)
            writer.add_scalar("epoch", epoch, epoch)
            writer.add_scalar("train_L2", train_meters["train_L2"].avg, epoch)
            sys.stdout.flush()

    logging.info(f"Done training! Best Val score {valid_tracks['val_PSNR'].best_val:.5f} obtained after epoch {valid_tracks['val_PSNR'].best_count}.")

    pickle.dump( valid_tracks, open(experiment_path + 'valid_tracks_metrics.pkl', "wb" ) , pickle.HIGHEST_PROTOCOL )
    pickle.dump( train_tracks, open(experiment_path + 'train_tracks_metrics.pkl', "wb" ) , pickle.HIGHEST_PROTOCOL )



################################################################################################

def run_testing(experiment_path,
                 acceleration,
                 center_fraction,
                 acceleration_total,
                 img_size,
                 path_to_ImageNet_train):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Init train.log file
    init_logging(experiment_path)

    # Log sanity checks on the number of lines in the input/target kspaces
    logging.info("Testing...")
    input_size, target_size, overlap_size_high, size_low, p, q, mu, nu, weight_on_random_lines = compute_number_of_lines_in_input_target_kspace(p=1/acceleration,mu=1/acceleration_total,nu=center_fraction, n=img_size)
    logging.info(f"mu: {mu}, p: {p}, q: {q}, nu: {nu}, weight_on_random_lines: {weight_on_random_lines}")
    logging.info(f"\n Lines in kspace: {img_size} \n Lines in input: {input_size} \n Lines in target: {target_size} \n Number of high freq overlapping lines: {overlap_size_high} \n Number of low freq lines: {size_low}")


    # train loss function
    loss_fct = MSELoss(reduction='sum')
    val_ssim_fct = SSIMLoss()

    # Init model
    model = Unet(
                in_chans=2,
                out_chans=2,
                chans=24,
                num_pool_layers=3,
                drop_prob=0.0,).to(device)
    
    validation_set = torch.load('CS_natural_images_functions/training_set_lists/ImageNetVal80_filepaths.pt')
    test_set = torch.load('CS_natural_images_functions/training_set_lists/ImageNetTest300_filepaths.pt')

    # test loader
    data_transform_test = UnetDataTransform(acceleration=acceleration,acceleration_total=acceleration_total, fix_split=True, experiment_path=experiment_path,center_fraction=center_fraction)
    testset = CropDataset(dataset=test_set, path_to_ImageNet_train=path_to_ImageNet_train,  transform=data_transform_test, experiment_path=experiment_path, img_size=img_size)
    test_loader = torch.utils.data.DataLoader( dataset=testset, batch_size=1, num_workers=0, shuffle=False, generator=torch.Generator().manual_seed(0), )

    # Val loader
    data_transform_val = UnetDataTransform(acceleration=acceleration,acceleration_total=acceleration_total, fix_split=True, experiment_path=experiment_path,center_fraction=center_fraction)
    valset = CropDataset(dataset=validation_set, path_to_ImageNet_train=path_to_ImageNet_train,  transform=data_transform_val, experiment_path=experiment_path, img_size=img_size)
    val_loader = torch.utils.data.DataLoader( dataset=valset, batch_size=1, num_workers=0, shuffle=False, generator=torch.Generator().manual_seed(0), )


    setup_experiment_or_load_checkpoint(experiment_path, resume_from='best', model=model, optimizer=None, scheduler=None)

    test_validationSet_tracks = {'SSIM' : TrackMeter_testing(), 'PSNR' : TrackMeter_testing(), 'L2' : TrackMeter_testing(),'SSIM_dc' : TrackMeter_testing(), 'PSNR_dc' : TrackMeter_testing(), 'L2_dc' : TrackMeter_testing()}
    test_testSet_tracks = {'SSIM' : TrackMeter_testing(), 'PSNR' : TrackMeter_testing(), 'L2' : TrackMeter_testing(),'SSIM_dc' : TrackMeter_testing(), 'PSNR_dc' : TrackMeter_testing(), 'L2_dc' : TrackMeter_testing()}

    model.eval()
    tmp=0
    for data_loader, track_meter in zip([val_loader, test_loader],[test_validationSet_tracks, test_testSet_tracks]):
        tmp+=1
        test_bar = ProgressBar(data_loader, epoch=0)

        for id, sample in enumerate(test_bar):

            y_input, x_input, y_target, x_target, x, input_mask, target_mask, mean, std, fname = sample

            # prediction
            x_output = torch.moveaxis(model(torch.moveaxis( x_input , -1, 1 )), 1, -1)

            # unnormalize
            x_output = x_output * std + mean

            # Apply data consistency
            y_output = fft2c(x_output)
            y_output_dc = y_output * (1-input_mask) + y_input
            x_output_dc = ifft2c(y_output_dc)

            # L2 in image domain between complex output and target image
            val_loss = loss_fct(x_output,x) / torch.sum(torch.abs(x)**2)
            track_meter['L2'].update(val_loss)
            val_loss_dc = loss_fct(x_output_dc,x) / torch.sum(torch.abs(x)**2)
            track_meter['L2_dc'].update(val_loss_dc)
            
            output_magnitude = (x_output ** 2).sum(dim=-1).sqrt()
            output_dc_magnitude = (x_output_dc ** 2).sum(dim=-1).sqrt()
            x_magnitude = (x ** 2).sum(dim=-1).sqrt() # since x is real, this operation is identity
            x_magnitude = x_magnitude.unsqueeze(1)
            output_magnitude = output_magnitude.unsqueeze(1)
            output_dc_magnitude = output_dc_magnitude.unsqueeze(1)
            
            # psnr
            max_value = x.max().unsqueeze(0)
            mse = torch.mean(torch.abs(output_magnitude-x_magnitude)**2)
            psnr = 20 * torch.log10(torch.tensor(max_value.item()))- 10 * torch.log10(mse)
            track_meter["PSNR"].update(psnr.item())
            mse_dc = torch.mean(torch.abs(output_dc_magnitude-x_magnitude)**2)
            psnr_dc = 20 * torch.log10(torch.tensor(max_value.item()))- 10 * torch.log10(mse_dc)
            track_meter["PSNR_dc"].update(psnr_dc.item())
            
            # ssim
            ssim_loss = 1-val_ssim_fct(output_magnitude, x_magnitude, data_range=max_value)
            track_meter["SSIM"].update(ssim_loss.item())
            ssim_loss_dc = 1-val_ssim_fct(output_dc_magnitude, x_magnitude, data_range=max_value)
            track_meter["SSIM_dc"].update(ssim_loss_dc.item())

            # Save the first image in test set
            if (tmp==1 and id==1) or (tmp==1 and id==22):
                x_input_abs = (x_input ** 2).sum(dim=-1).sqrt()
                x_input_abs = x_input_abs.unsqueeze(1)
                save_test_image_with_dc(experiment_path, ground_truth_image=x_magnitude, input_img=x_input_abs, output=output_magnitude, output_image_dc=output_dc_magnitude, fname=fname, track_meter=track_meter)

    pickle.dump( test_validationSet_tracks, open(experiment_path + 'test_validationSet_metrics.pkl', "wb" ) , pickle.HIGHEST_PROTOCOL )
    pickle.dump( test_testSet_tracks, open(experiment_path + 'test_testSet_metrics.pkl', "wb" ) , pickle.HIGHEST_PROTOCOL )

    logging.info(f"\nEvaluate validationset of length {len(val_loader)}:")
    for metric in test_validationSet_tracks.keys():
            logging.info(f"{metric}: avg {test_validationSet_tracks[metric].avg:.6f}, std {test_validationSet_tracks[metric].std:.6f}")

    logging.info(f"\nEvaluate testset of length {len(test_loader)}:")
    for metric in test_testSet_tracks.keys():
            logging.info(f"{metric}: avg {test_testSet_tracks[metric].avg:.6f}, std {test_testSet_tracks[metric].std:.6f}")

if __name__ == '__main__':
    read_args()
# %%



