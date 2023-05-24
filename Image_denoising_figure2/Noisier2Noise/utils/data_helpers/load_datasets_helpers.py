import os
import os.path
import numpy as np
import h5py
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
from utils.utils_image import *

class ImagenetSubdataset(torch.utils.data.Dataset):
    def __init__(self, size, path_to_ImageNet_train, mode='train', patch_size='128', val_crop=True):
        super().__init__()
        load_path = '../training_set_lists/'
        self.path_to_ImageNet_train = path_to_ImageNet_train
        if mode=='train':
            self.files = torch.load(load_path+f'trsize{size}_filepaths.pt') 
            self.transform = transforms.Compose([      
                transforms.CenterCrop(patch_size),
                transforms.ToTensor(),                  
                ])
             
        elif mode=='val':
            self.files = torch.load(load_path+f'ImageNetVal{size}_filepaths.pt') 
            #print(self.files)
            if val_crop:
                self.transform = transforms.Compose([      
                    transforms.CenterCrop(patch_size),
                    transforms.ToTensor(),                  
                    ]) 
            else:
                self.transform = transforms.Compose([      
                    transforms.ToTensor(),                  
                    ]) 

        self.noise_seeds = {}
        for i, file in enumerate(self.files):
            key = file[file.find('train/')+16:-5]
            number = int(file[file.find('train/')+17:-5].replace('_',''))
            self.noise_seeds[key] = number


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        file = self.files[index]
        key = file[file.find('train/')+16:-5]
        noise_seed = self.noise_seeds[key]
        image = Image.open(self.path_to_ImageNet_train + self.files[index]).convert("RGB") #ImageNet contains some grayscale images
        data = self.transform(image)

        return data, noise_seed




