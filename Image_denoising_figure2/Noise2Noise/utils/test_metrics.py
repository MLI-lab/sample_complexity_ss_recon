import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
#import cv2
from utils.noise_model import get_noise
from utils.metrics import ssim,psnr
from utils.util_calculate_psnr_ssim import calculate_psnr,calculate_ssim
from skimage import color
import PIL.Image as Image
import torchvision.transforms as transforms
from utils.utils_image import *

metrics_key = ['psnr_m', 'psnr_s', 'psnr_delta_m', 'psnr_delta_s', 'ssim_m', 'ssim_s', 'ssim_delta_m', 'ssim_delta_s'];

def tensor_to_image(torch_image, low=0.0, high = 1.0, clamp = True):
    if clamp:
        torch_image = torch.clamp(torch_image, low, high);
    return torch_image[0,0].cpu().data.numpy()


def normalize(data):
    return data/255.

def convert_dict_to_string(metrics):
    return_string = '';
    for x in metrics.keys():
        return_string += x+': '+str(round(metrics[x], 3))+' ';
    return return_string



def get_all_comparison_metrics(denoised, source, noisy = None, scale=None, return_title_string = False, clamp = True):

    metrics = {};
    metrics['psnr'] = np.zeros(len(denoised))
    metrics['ssim'] = np.zeros(len(denoised))
    if noisy is not None:
        metrics['psnr_delta'] = np.zeros(len(denoised))
        metrics['ssim_delta'] = np.zeros(len(denoised))

    if clamp:
        denoised = torch.clamp(denoised, 0.0, 1.0)


    metrics['psnr'] = psnr(source, denoised);
    metrics['ssim'] = ssim(source, denoised);

    if noisy is not None:
        metrics['psnr_delta'] = metrics['psnr'] - psnr(source, noisy);
        metrics['ssim_delta'] = metrics['ssim'] - ssim(source, noisy);

    if return_title_string:
        return convert_dict_to_string(metrics)
    else:
        return metrics


def average_on_folder(args, net, noise_std,
            verbose=True, device = torch.device('cuda')):
    
    #if verbose:
    #print('Loading data info ...\n')
    print(f'\n Dataset: {args.test_mode}, Restore mode: {args.restore_mode}')
    load_path = '../training_set_lists/'
    
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
    
    if args.test_mode == 'test':
        files_source = torch.load(load_path+f'ImageNetTest{args.test_size}_filepaths.pt') 
        #files_source.sort()       
    elif args.test_mode == 'val':
        files_source = torch.load(load_path+f'ImageNetVal{args.val_size}_filepaths.pt')
        #files_source.sort()      
    else:
        files_source = torch.load(load_path+f'{args.test_mode}_filepaths.pt')
    
    
    avreage_metrics_key = ['psnr', 'psnr_delta', 'ssim', 'ssim_delta']
    avg_metrics = {};
    for x in avreage_metrics_key:
        avg_metrics[x] = [];
    
    psnr_list = []
    ssim_list = []
    
    #print(files_source)
    #for f_clean,f_noisy in zip(clean_files,noisy_files):
    for f in files_source:
        # image
        #if args.test_mode == 'test' or args.test_mode == 'val':
        transformT = transforms.ToTensor()
        #print(f_noisy)
        #print(f_clean)
        #print(Image.open(f).size) # note that .size does not show the channels
        #print(Image.open(f).mode) # this shows 'RGB' or 'L' for grayscale
        ISource = torch.unsqueeze(transformT(Image.open(args.path_to_ImageNet_train + f).convert("RGB")),0).to(device)
        
        if args.test_mode == 'val':
            noise_seed = int(f[f.find('train/')+17:-5].replace('_',''))
            gen = gen.manual_seed(noise_seed)

        noise =  torch.randn(ISource.shape,generator = gen) * args.noise_std/255.

        INoisy = noise.to(device) + ISource

        #INoisy = torch.unsqueeze(transformT(Image.open(f_noisy).convert("RGB")),0).to(device)
        
        out = torch.clamp(net(INoisy), 0., 1.)

        ind_metrics = get_all_comparison_metrics(out, ISource, INoisy, return_title_string = False);

        for x in avreage_metrics_key:
            avg_metrics[x].append(ind_metrics[x])

        if(verbose):
            print("%s %s" % (f, convert_dict_to_string(ind_metrics)))
    
    metrics = {}
    for x in avreage_metrics_key:
        metrics[x+'_m'] = np.mean(avg_metrics[x])
        metrics[x+'_s'] = np.std(avg_metrics[x])

    if verbose:
        print("\n Average %s" % (convert_dict_to_string(metrics)))

    #if(not verbose):
    return metrics


def metrics_avg_on_noise_range(net, args, noise_std_array, device = torch.device('cuda')):

    #print(path_to_dataset)

    # For psnr/ssim the dict array_metrics has one key for average/std/delts
    # Each key contains an array of same length as noise levels over which we compute performance
    # Each entry in an array is the average/std/delta at a given noise level over the test set at hand
    array_metrics = {}
    for x in metrics_key:
        array_metrics[x] = np.zeros(len(noise_std_array))

    for j, noise_std in enumerate(noise_std_array):
        metric_list = average_on_folder(args, net, 
                                                noise_std = noise_std,
                                                verbose=False, device=device);

        for x in metrics_key:
            array_metrics[x][j] += metric_list[x]
            print('noise: ', int(noise_std*255), ' ', x, ': ', str(array_metrics[x][j]))

    return array_metrics

