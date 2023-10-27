import os
import os.path
import numpy as np
import torch
import glob
import torchvision.transforms as transforms
from utils.utils_image import *
import scipy.io as sio


def extract_metainfo(path='0151_METADATA_RAW_010.MAT'):
    meta = sio.loadmat(path)['metadata']
    mat_vals = meta[0][0]
    mat_keys = mat_vals.dtype.descr

    keys = []
    for item in mat_keys:
        keys.append(item[0])

    py_dict = {}
    for key in keys:
        py_dict[key] = mat_vals[key]

    device = py_dict['Model'][0].lower()
    bitDepth = py_dict['BitDepth'][0][0]
    if 'iphone' in device or bitDepth != 16:
        noise = py_dict['UnknownTags'][-2][0][-1][0][:2]
        iso = py_dict['DigitalCamera'][0, 0]['ISOSpeedRatings'][0][0]
        pattern = py_dict['SubIFDs'][0][0]['UnknownTags'][0][0][1][0][-1][0]
        time = py_dict['DigitalCamera'][0, 0]['ExposureTime'][0][0]

    else:
        noise = py_dict['UnknownTags'][-1][0][-1][0][:2]
        iso = py_dict['ISOSpeedRatings'][0][0]
        pattern = py_dict['UnknownTags'][1][0][-1][0]
        time = py_dict['ExposureTime'][0][0]  # the 0th row and 0th line item

    rgb = ['R', 'G', 'B']
    pattern = ''.join([rgb[i] for i in pattern])

    asShotNeutral = py_dict['AsShotNeutral'][0]
    b_gain, _, r_gain = asShotNeutral

    # only load ccm1
    ccm = py_dict['ColorMatrix1'][0].astype(float).reshape((3, 3))

    return {'device': device,
            'pattern': pattern,
            'iso': iso,
            'noise': noise,
            'time': time,
            'wb': np.array([r_gain, 1, b_gain]),
            'ccm': ccm, }

def raw2stack(var):
    h, w = var.shape
    if var.is_cuda:
        res = torch.cuda.FloatTensor(4, h // 2, w // 2).fill_(0)
    else:
        res = torch.FloatTensor(4, h // 2, w // 2).fill_(0)
    res[0] = var[0::2, 0::2]
    res[1] = var[0::2, 1::2]
    res[2] = var[1::2, 0::2]
    res[3] = var[1::2, 1::2]
    return res

def transform_to_rggb(img, pattern):
    assert len(img.shape) == 2
    res = img.copy()

    if pattern.lower() == 'bggr':  # same pattern
        res[0::2, 0::2] = img[1::2, 1::2]
        res[1::2, 1::2] = img[0::2, 0::2]
    elif pattern.lower() == 'rggb':
        pass
    elif pattern.lower() == 'grbg':
        res[0::2, 0::2] = img[0::2, 1::2]
        res[0::2, 1::2] = img[0::2, 0::2]
        res[1::2, 0::2] = img[1::2, 1::2]
        res[1::2, 1::2] = img[1::2, 0::2]
    elif pattern.lower() == 'gbrg':
        res[0::2, 0::2] = img[1::2, 0::2]
        res[0::2, 1::2] = img[0::2, 0::2]
        res[1::2, 0::2] = img[1::2, 1::2]
        res[1::2, 1::2] = img[0::2, 1::2]
    else:
        assert 'no support'

    return res


class SIDDSubdataset_fromArray(torch.utils.data.Dataset):
    def __init__(self, size, seed, mode='train', patch_size='128',supervised=True):
        super().__init__()

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        load_transform = transforms.Compose([      
                transforms.ToTensor(),                  
                ])
        
        rng_dataset = np.random.default_rng(seed)
        train_set_indices = rng_dataset.choice(125594, size=size, replace=False, p=None)

        # Path to training data pre-processed with create_training_data.py
        path_to_train = "/tobit/N2N_SIDD_denoising/"
        # Path to "ValidationNoisyBlocksRaw.mat" and "ValidationGtBlocksRaw.mat"
        path_to_validation = "/media/ssd1/SIDD/validation/"
        # Path to SIDD_Benchmark_Data containing the meta data for the validation set (and benchmark set)
        path_to_benchmark_images = "/tobit/N2N_SIDD_denoising/raw_validation_data/SIDD_Benchmark_Data/"

        if mode=='train':
            inputs = np.load(f'{path_to_train}SIDD_train_all_scenes_patchSize_{patch_size}_patches_125594_input_array_raw.npy') 
            if supervised:
                targets = np.load(f'{path_to_train}SIDD_train_all_scenes_patchSize_{patch_size}_patches_125594_gt_array_raw.npy')
            else:
                targets = np.load(f'{path_to_train}SIDD_train_all_scenes_patchSize_{patch_size}_patches_125594_noisy_target_array_raw.npy')

            self.examples = []

            for i in train_set_indices:
                input_image = inputs[i,:,:]

                target_image = targets[i,:,:]

                input_image = load_transform(input_image).to(device)
                target_image = load_transform(target_image).to(device)
                input_image = raw2stack(input_image[0])
                target_image = raw2stack(target_image[0])

                self.examples.append((input_image,target_image))
             
        elif mode=='val' or mode=='test':
            
            # the 40 scenes used for validation and benchmarking
            validation_scenes_list = sorted(glob.glob(path_to_benchmark_images + "0*"))
            
            noisy_images = sio.loadmat(path_to_validation + "ValidationNoisyBlocksRaw.mat")
            gt_images = sio.loadmat(path_to_validation + "ValidationGtBlocksRaw.mat")

            noisy_images = noisy_images['ValidationNoisyBlocksRaw']
            gt_images = gt_images['ValidationGtBlocksRaw']

            transform = transforms.Compose([      
                    transforms.ToTensor(),                  
                    ]) 
            
            self.examples = []
            if mode=='val':
                pick_val_images = np.arange(0,size)
            elif mode=='test':
                pick_val_images = np.arange(40-size,40)
            else:
                raise ValueError('size must be 3 or 37')
            for i in pick_val_images:
                # here we need to implement the correction of the bayer pattern
                scene = validation_scenes_list[i]
                img_name, extension = os.path.splitext(os.path.basename(scene))
                scene_tag = img_name[0:4]
                py_meta = extract_metainfo(scene +"/" + scene_tag + "_METADATA_RAW_010.MAT")
                pattern = py_meta['pattern']

                for j in range(noisy_images.shape[1]):
                    noisy_image = noisy_images[i,j,:,:]
                    gt_image = gt_images[i,j,:,:]

                    noisy_image = transform_to_rggb(noisy_image, pattern)
                    gt_image = transform_to_rggb(gt_image, pattern)

                    noisy_image = transform(noisy_image).to(device)
                    gt_image = transform(gt_image).to(device)

                    noisy_image = raw2stack(noisy_image[0])
                    gt_image = raw2stack(gt_image[0])
                    self.examples.append((noisy_image,gt_image))



    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):

        sample = self.examples[index]

        return sample


class SIDDSubdataset_fromArray_gradDiff(torch.utils.data.Dataset):
    def __init__(self, size, seed, mode='train', patch_size='128'):
        super().__init__()

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        load_transform = transforms.Compose([      
                transforms.ToTensor(),                  
                ])
        
        rng_dataset = np.random.default_rng(seed)
        train_set_indices = rng_dataset.choice(125594, size=size, replace=False, p=None)

        # Path to training data pre-processed with create_training_data.py
        path_to_train = "/tobit/N2N_SIDD_denoising/"


        if mode=='train':
            inputs = np.load(f'{path_to_train}SIDD_train_all_scenes_patchSize_{patch_size}_patches_125594_input_array_raw.npy') 
    
            targets_sup = np.load(f'{path_to_train}SIDD_train_all_scenes_patchSize_{patch_size}_patches_125594_gt_array_raw.npy')
            targets_self = np.load(f'{path_to_train}SIDD_train_all_scenes_patchSize_{patch_size}_patches_125594_noisy_target_array_raw.npy')

            self.examples = []

            for i in train_set_indices:
                input_image = inputs[i,:,:]

                target_image_sup = targets_sup[i,:,:]
                target_image_self = targets_self[i,:,:]

                input_image = load_transform(input_image).to(device)
                target_image_sup = load_transform(target_image_sup).to(device)
                target_image_self = load_transform(target_image_self).to(device)
                input_image = raw2stack(input_image[0])
                target_image_sup = raw2stack(target_image_sup[0])
                target_image_self = raw2stack(target_image_self[0])

                self.examples.append((input_image,target_image_sup,target_image_self))
             




    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):

        sample = self.examples[index]

        return sample

