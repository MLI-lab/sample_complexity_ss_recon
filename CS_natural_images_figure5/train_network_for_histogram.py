# %%
import torch
import h5py
import numpy as np
import os
import yaml
import logging
import glob
import random
import pickle
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt
from torch.nn import MSELoss
import copy

from argparse import ArgumentParser

from torch.utils.tensorboard import SummaryWriter
import sys

from tqdm import tqdm

import torchvision.transforms as transforms
import PIL.Image as Image

from skimage.transform import resize

from CS_natural_images_functions.unet import Unet
from CS_natural_images_functions.fftc import fft2c, ifft2c
from CS_natural_images_functions.losses import SSIMLoss
from CS_natural_images_functions.progress_bar import ProgressBar, init_logging, AverageMeter, TrackMeter, TrackMeter_testing
from CS_natural_images_functions.log_progress_helpers import save_figure, add_img_to_tensorboard, save_test_image_with_dc
from CS_natural_images_functions.load_save_model_helpers import setup_experiment_or_load_checkpoint, save_checkpoint

from CS_natural_images_functions.data_transforms import UnetDataTransform
from CS_natural_images_functions.data_transforms import compute_number_of_lines_in_input_target_kspace

# %%
class CropDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    In this case we are only interested in downsampled magnitude images.
    """

    def __init__(
        self,
        dataset: List,
        path_to_ImageNet_train: str,
        transforms_list: List, 
        experiment_path: str,
        img_size: int,
    ):
        """
        Args:
            dataset: A list containing one entry for every slice in the dataset. 
                Each entry is a dictionary with keys 'path','slice','filename'ArithmeticError
            path_to_mridata: Path to fastMRI data on the server.
            transform: Function that transforms the ground truth image x into training input and target.
        """
        self.transform_30 = transforms_list[0]
        self.transform_35 = transforms_list[1]
        self.experiment_path = experiment_path
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Store downsampled ground truth training images here
        self.examples = []

        load_transform = transforms.Compose([      
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),                  
                ]) 

        # Load mri magnitude images, downsample and store
        for datapath in dataset:
            image = Image.open(path_to_ImageNet_train+datapath).convert("L")

            filename = datapath[16:-5]

            self.examples.append((load_transform(image)[0].to(device),filename))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int): 
        # Determine input, target and ground truth
        x,filename = self.examples[i]

        y_input, x_input, y_target, x_target, x_gt, input_mask, target_mask, mean, std, fname = self.transform_30(x,filename,i)
        _, _, _, _, _, _, target_mask_35, _, _, _ = self.transform_35(x,filename,i)

        return y_input, x_input, y_target, x_target, x_gt, input_mask, target_mask, mean, std, fname,target_mask_35
# %%


def read_args():
    parser = ArgumentParser()

    parser.add_argument(
        '--path_to_ImageNet_train',
        type=str,
        help='Path to ImageNet train directory.',
        required=True
    )

    parser.add_argument(
        '--training',
        default=True,
        action='store_false',
        help='Add this flag to disable training.'
    )

    parser.add_argument(
        '--testing',
        default=False,
        action='store_false',
        help='Add this flag to disable testing.'
    )

    parser.add_argument(
        '--experiment_number',
        default='300',
        type=str,
        help='Set consecutive numbering for the experiments.'
    )

    parser.add_argument(
        '--gpu',
        choices=(0, 1, 2, 3),
        default=1,
        type=int,
        help='Pick one out of four gpus.'
    )

    parser.add_argument(
        '--seed',
        default=0,
        type=int,
        help='Set seed for network initialization.'
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
        '--acceleration',
        default=4.0,
        type=float,
        help='Undersampling of training and test inputs.'
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
        help='Add this flag to set fix_split=True for fixed input target split for self-supervised and fixed input for supervised training.'
    )

    args = parser.parse_args()

    exp_nums =  ['992']

    # validation every second epoch
    # lr decay on plateau

    hyperparameters = {
        #'acceleration_total' : [
        #3.0,
        #],
        'trainset_size' : [
        10000,
        ],
        'center_fraction' : [
        0.08,
        ],
        'seed' : [
        1,
        ],
        'fix_split' : [
        True,
        ],
        'num_epochs' : [
        2,
        ],
        'patience' : [ # as we validate every second epoch a patience of 10 actually means 20 epochs
        15,
        ]
    }

    # Sanity checks
    for key in hyperparameters.keys():

        if len(hyperparameters[key]) != len(exp_nums):
            print(key)
            raise ValueError("Specify hyperparameters for every experiment!") 

    for i in range(len(exp_nums)):
        args.trainset_size = hyperparameters['trainset_size'][i]  
        args.center_fraction = hyperparameters['center_fraction'][i]
        args.seed = hyperparameters['seed'][i]
        args.fix_split = hyperparameters['fix_split'][i]
        args.num_epochs = hyperparameters['num_epochs'][i]
        args.patience = hyperparameters['patience'][i]
        args.experiment_number = exp_nums[i]
        

        experiment_name = f"N{args.experiment_number}_t{args.trainset_size}_"
        experiment_name+="sup_VS_ss3035_" 

        if args.center_fraction==0.0:
            experiment_name+="RandCenter_" 
        else: 
            experiment_name+="FixCenter_"

        experiment_name+="grad_diff_"

        experiment_name += f"run{args.seed}"
        
        experiment_path = experiment_name+"/"

        #dataset_path = f"../datasets/train_{args.trainset_size}_selfsup_slice.yaml"

        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

        if args.training:
            run_training(experiment_path=experiment_path,
                        acceleration=args.acceleration,
                        center_fraction=args.center_fraction,
                        seed=args.seed,
                        img_size=args.img_size,
                        fix_split=args.fix_split,
                        num_epochs=args.num_epochs,
                        patience=args.patience,
                        trainset_size=args.trainset_size,
                        path_to_ImageNet_train=args.path_to_ImageNet_train)

################################################################################################

def run_training(experiment_path,
                 acceleration,
                 center_fraction,
                 seed,
                 img_size,
                 fix_split,
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

    # train loss function
    loss_fct = MSELoss(reduction='sum')
    #val_ssim_fct = SSIMLoss()

    # Init model
    model = Unet(
                in_chans=2,
                out_chans=2,
                chans=24,
                num_pool_layers=3,
                drop_prob=0.0,).to(device)

    # Init optimizer and scheduler
    optimizer = torch.optim.Adam( params=model.parameters(),  lr=0.001,  betas=(0.9, 0.999),  eps=1e-08,  weight_decay=0.0,  amsgrad=False)

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
    train_set = rng_dataset.choice(train_pool, size=trainset_size, replace=False, p=None)
    torch.save(train_set,experiment_path+'train_set.pt')
    

    # Train loader
    data_transform_train_35 = UnetDataTransform(acceleration=acceleration,acceleration_total=3.5, fix_split=fix_split, experiment_path=experiment_path,center_fraction=center_fraction)
    data_transform_train_30 = UnetDataTransform(acceleration=acceleration,acceleration_total=3.0, fix_split=fix_split, experiment_path=experiment_path,center_fraction=center_fraction)
    trainset = CropDataset(dataset=train_set, path_to_ImageNet_train=path_to_ImageNet_train, transforms_list=[data_transform_train_30,data_transform_train_35], experiment_path=experiment_path, img_size=img_size)
    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=1, num_workers=0, shuffle=True, generator=torch.Generator().manual_seed(0))

    # store training loss metrics
    train_meters = {'train_L2': AverageMeter()}
    sup_diff_tracks = {'divide_by_norm_of_risk_grad': TrackMeter_testing(), 'take_mse': TrackMeter_testing()}
    ss_diff_30_tracks = {'divide_by_norm_of_risk_grad': TrackMeter_testing(), 'take_mse': TrackMeter_testing()}
    ss_diff_35_tracks = {'divide_by_norm_of_risk_grad': TrackMeter_testing(), 'take_mse': TrackMeter_testing()}

    # Init tensorboard
    #writer = SummaryWriter(log_dir=experiment_path)
    #log_image_interval_tb = 10

    # when to compute gradient histograms
    compute_gradients_interval = 1

    # Start training
    break_counter=0
    for epoch in range(num_epochs):
        
        # compute gradient histogrms
        if epoch % compute_gradients_interval == 0:
            model_copy = copy.deepcopy(model)
            train_bar_hist = ProgressBar(train_loader, epoch)
            model_copy.train()

            for meter in sup_diff_tracks.values():
                meter.reset()
            for meter in ss_diff_30_tracks.values():
                meter.reset()
            for meter in ss_diff_35_tracks.values():
                meter.reset()

            # estimate ground truth gradient based on whole dataset
            for id,sample in enumerate(train_bar_hist):
                y_input, x_input, y_target, x_target, x, input_mask, target_mask_30, mean, std, fname, target_mask_35 = sample

                # prediction
                x_output = torch.moveaxis(model_copy(torch.moveaxis( x_input , -1, 1 )), 1, -1)

                # unnormalize
                x_output = x_output * std + mean

                # move to kspace
                y_output_sup = fft2c(x_output)

                # apply target mask (all ones for supervised training)
                y_output_ss_30 = y_output_sup * target_mask_30 + 0.0
                y_output_ss_35 = y_output_sup * target_mask_35 + 0.0

                y_target_sup = fft2c(x)

                y_target_ss_30 = y_target_sup * target_mask_30 + 0.0
                y_target_ss_35 = y_target_sup * target_mask_35 + 0.0

                save_figure(torch.log(torch.abs(y_input[0,:,:,0].detach().cpu())+ 1e-9),"y_input_real",experiment_path) if id==0 else None
                save_figure(torch.log(torch.abs(y_output_sup[0,:,:,0].detach().cpu())+ 1e-9),"y_output_sup_real",experiment_path) if id==0 else None
                save_figure(torch.log(torch.abs(y_output_ss_30[0,:,:,0].detach().cpu())+ 1e-9),"y_output_ss_30_real",experiment_path) if id==0 else None
                save_figure(torch.log(torch.abs(y_output_ss_35[0,:,:,0].detach().cpu())+ 1e-9),"y_output_ss_35_real",experiment_path) if id==0 else None
                save_figure(torch.log(torch.abs(y_target_sup[0,:,:,0].detach().cpu())+ 1e-9),"y_target_sup_real",experiment_path) if id==0 else None
                save_figure(torch.log(torch.abs(y_target_ss_30[0,:,:,0].detach().cpu())+ 1e-9),"y_target_ss_30_real",experiment_path) if id==0 else None
                save_figure(torch.log(torch.abs(y_target_ss_35[0,:,:,0].detach().cpu())+ 1e-9),"y_target_ss_35_real",experiment_path) if id==0 else None

                # compute loss
                train_loss_sup = loss_fct(y_output_sup,y_target_sup) / torch.sum(torch.abs(y_target_sup)**2)

                #train_loss_ss_30 = loss_fct(y_output_ss_30,y_target_ss_30) / torch.sum(torch.abs(y_target_ss_30)**2)
                #train_loss_ss_35 = loss_fct(y_output_ss_35,y_target_ss_35) / torch.sum(torch.abs(y_target_ss_35)**2)

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
                y_input, x_input, y_target, x_target, x, input_mask, target_mask_30, mean, std, fname, target_mask_35 = sample

                # prediction
                x_output = torch.moveaxis(model_copy(torch.moveaxis( x_input , -1, 1 )), 1, -1)

                # unnormalize
                x_output = x_output * std + mean

                # move to kspace
                y_output_sup = fft2c(x_output)

                # apply target mask (all ones for supervised training)
                y_output_ss_30 = y_output_sup * target_mask_30 + 0.0
                y_output_ss_35 = y_output_sup * target_mask_35 + 0.0

                y_target_sup = fft2c(x)

                y_target_ss_30 = y_target_sup * target_mask_30 + 0.0
                y_target_ss_35 = y_target_sup * target_mask_35 + 0.0

                # compute loss
                train_loss_sup = loss_fct(y_output_sup,y_target_sup) / torch.sum(torch.abs(y_target_sup)**2)

                train_loss_ss_30 = loss_fct(y_output_ss_30,y_target_ss_30) / torch.sum(torch.abs(y_target_ss_30)**2)
                train_loss_ss_35 = loss_fct(y_output_ss_35,y_target_ss_35) / torch.sum(torch.abs(y_target_ss_35)**2)

                param = list(model_copy.parameters())

                model_copy.zero_grad()
                train_loss_sup.backward(retain_graph=True)

                for p in param:
                    p.grad_sup = p.grad
                    p.grad = None

                train_loss_ss_30.backward(retain_graph=True) 
                for p in param:
                    p.grad_ss_30 = p.grad
                    p.grad = None

                train_loss_ss_35.backward(retain_graph=True) 
                for p in param:
                    p.grad_ss_35 = p.grad
                    p.grad = None
    
                diff_sup = torch.zeros(1).to(device)
                diff_ss_30 = torch.zeros(1).to(device)
                diff_ss_35 = torch.zeros(1).to(device)
                norm_grad_of_risk = torch.zeros(1).to(device)

                for p in param:
                    diff_sup += torch.sum(torch.square(torch.sub(p.grad_sup,p.grad_true_risk)))
                    diff_ss_30 += torch.sum(torch.square(torch.sub(p.grad_ss_30,p.grad_true_risk)))
                    diff_ss_35 += torch.sum(torch.square(torch.sub(p.grad_ss_35,p.grad_true_risk)))
                    norm_grad_of_risk += torch.sum(torch.square(p.grad_true_risk))

                sup_diff_tracks['divide_by_norm_of_risk_grad'].update(torch.div(diff_sup,norm_grad_of_risk).item())
                sup_diff_tracks['take_mse'].update(torch.mean(diff_sup).item())

                ss_diff_30_tracks['divide_by_norm_of_risk_grad'].update(torch.div(diff_ss_30,norm_grad_of_risk).item())
                ss_diff_30_tracks['take_mse'].update(torch.mean(diff_ss_30).item())

                ss_diff_35_tracks['divide_by_norm_of_risk_grad'].update(torch.div(diff_ss_35,norm_grad_of_risk).item())
                ss_diff_35_tracks['take_mse'].update(torch.mean(diff_ss_35).item())

            pickle.dump( sup_diff_tracks, open(experiment_path + f"sup_diff_tracks_ep{epoch}.pkl", "wb" ) , pickle.HIGHEST_PROTOCOL )
            pickle.dump( ss_diff_30_tracks, open(experiment_path + f"ss_diff_30_tracks_ep{epoch}.pkl", "wb" ) , pickle.HIGHEST_PROTOCOL )
            pickle.dump( ss_diff_35_tracks, open(experiment_path + f"ss_diff_35_tracks_ep{epoch}.pkl", "wb" ) , pickle.HIGHEST_PROTOCOL )

        # perform one training epoch
        train_bar = ProgressBar(train_loader, epoch)
        for meter in train_meters.values():
            meter.reset()
        for id,sample in enumerate(train_bar):
            model.train()

            y_input, x_input, y_target, x_target, x, input_mask, target_mask_30, mean, std, fname, target_mask_35 = sample

            # prediction
            x_output = torch.moveaxis(model(torch.moveaxis( x_input , -1, 1 )), 1, -1)

            # unnormalize
            x_output = x_output * std + mean

            # move to kspace
            y_output = fft2c(x_output)

            # apply target mask (all ones for supervised training)
            # DO SUPERVISED TRAINING HERE
            #y_output = y_output #* target_mask + 0.0
            y_target = fft2c(x)

            # compute loss
            train_loss = loss_fct(y_output,y_target) / torch.sum(torch.abs(y_target)**2)

            model.zero_grad()
            train_loss.backward()
            optimizer.step()

            # log train metrics
            train_meters['train_L2'].update(train_loss.item())
            train_bar.log(dict(**train_meters), verbose=True)

    



################################################################################################

if __name__ == '__main__':
    read_args()
# %%



