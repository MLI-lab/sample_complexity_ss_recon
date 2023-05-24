import numpy as np
import torch
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torchvision.transforms as transforms
import PIL.Image as Image

from CS_natural_images_functions.log_progress_helpers import save_figure

from CS_natural_images_functions.fftc import fft2c, ifft2c


class CropDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to cropped images from ImageNet.
    """

    def __init__(
        self,
        dataset: List,
        path_to_ImageNet_train: str,
        transform: Callable, 
        experiment_path: str,
        img_size: int,
    ):

        self.transform = transform
        self.experiment_path = experiment_path
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.examples = []

        load_transform = transforms.Compose([      
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),                  
                ]) 


        for datapath in dataset:
            image = Image.open(path_to_ImageNet_train+datapath).convert("L")

            filename = datapath[16:-5]

            self.examples.append((load_transform(image)[0].to(device),filename))
        

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int): 
        # Determine input, target and ground truth
        x,filename = self.examples[i]

        sample = self.transform(x,filename,i)

        return sample


class UnetDataTransform:
    def __init__(
        self,
        acceleration,
        acceleration_total,
        fix_split,
        experiment_path,
        center_fraction,
    ):
        self.acceleration = acceleration
        self.acceleration_total = acceleration_total
        self.fix_split = fix_split
        self.experiment_path = experiment_path
        self.center_fraction = center_fraction

    def __call__(
        self,
        x: np.ndarray,
        fname: str,
        id: int,
    ) -> Tuple[torch.Tensor,torch.Tensor]:
        """
        Args:

        Returns:
            tuple containing:
                x_input: zero-filled coarse reconstruction in image domain
                y_target: the fully sampled kspace
                x: the ground truth image
                input_mask: undersampled input mask
                target_mask: in the case of supervised training all ones
        """
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        n = x.shape[-1]

        # transform x to a tensor with real channel and complex channel. Right now the complex channel is all zerors.
        #x = np.stack((x, np.zeros_like(x)), axis=-1)
        x = torch.stack((x, torch.zeros_like(x)), axis=-1)

        #x = torch.from_numpy(x)

        # obtain kspace
        y = fft2c(x)
        #save_figure(y[:,:,0],"y_real",self.experiment_path) if id==0 else None
        #save_figure(y[:,:,1],"y_imag",self.experiment_path) if id==0 else None

        #######################################
        # sample input mask
        nu = self.center_fraction
        p = 1/self.acceleration
        mu = 1/self.acceleration_total
        q = (mu-p+nu-mu*nu)/(1-p)

        # 1. Determine the set S_low consisting of the indices of the nu*n many center frequencies which are always sampled
        size_low = int(round(n*nu))
        pad = (n - size_low + 1) // 2
        # set of indices of all lines in kspace
        S_all = np.arange(n)
        S_low = S_all[pad : pad + size_low]

        # 1.1 Determine S_mu_high, i.e, S_mu without S_low, so only the random high frequencies
        # set of indices of all high frequencies
        S_high = np.hstack((S_all[: pad],S_all[pad + size_low :]))
        S_mu_size_high = int(round((mu-nu)*n))

        S_p_size_high = int(round((p-nu)*n))

        #### Depending on whether the input/target split is fixed or re-sampled, the order of sampling needs to be adapted
        # This is so that validation during training samples the same input mask as during testing
        # Recall that during testing selfsup=False, hence S_mu_high is not sampled.
        seed = tuple(map(ord, fname))
        rng = np.random.default_rng(seed)
        if self.fix_split:
            # If split is fixed, first sample S_p_high and then additional lines for S_mu_high
            # such that the set S_p_high is the same as if we would sample for selfsup=False
            S_p_high = rng.choice(S_high, size=S_p_size_high, replace=False, p=None) 
            S_mu_size_high_remainding = S_mu_size_high - S_p_size_high

            S_high_remainding = np.array(list(set(S_high)-set(S_p_high)))
            S_q_high = rng.choice(S_high_remainding, size=S_mu_size_high_remainding, replace=False, p=None) 

        else:
            # If split is random, first sample S_mu_high such that this set is always fixed.
            S_mu_high = rng.choice(S_high, size=S_mu_size_high, replace=False, p=None)

            # 2. From S_mu_high sample the set S_p_high of size (p-nu)n
            S_p_high = np.random.choice(S_mu_high, size=S_p_size_high, replace=False, p=None)

            # 3. All other indices in S_mu_high add to the set S_q_high
            S_q_high = np.array(list(set(S_mu_high)-set(S_p_high)))

        # 4. Determine the size of the overlap between S_p_high and S_q_high, sample this many indices from S_p_high and add them to S_q_high
        overlap_size_high = int(round(( (p-nu) / (1-nu) ) * ( (q-nu) / (1-nu) ) *(n-n*nu)))
        S_overlap = S_p_high[0:overlap_size_high]
        S_q_high = np.concatenate([S_q_high,S_overlap])

        # 5. Define the final input and target masks by setting entries to zero or to one for S_p=S_low+S_p_high and S_q=S_low+S_q_high
        input_mask = np.zeros(n)
        input_mask[S_low] = 1.0
        input_mask[S_p_high] = 1.0
        input_mask = torch.from_numpy(input_mask.astype(np.float32)).unsqueeze(0).unsqueeze(-1).to(device)

        # 6. Create a target mask where the random entries are weighted
        weight_on_random_lines = np.sqrt((1-nu)/(q-nu))
        target_mask = np.zeros(n)
        target_mask[S_low] = 1.0
        target_mask[S_q_high] = weight_on_random_lines
        target_mask = torch.from_numpy(target_mask.astype(np.float32)).unsqueeze(0).unsqueeze(-1).to(device)

        #######################################

        # apply mask to kspace
        y_input = y * input_mask + 0.0
        #save_figure(y_input[:,:,0],"y_input_real",self.experiment_path) if id==0 else None
        #save_figure(y_input[:,:,1],"y_input_imag",self.experiment_path) if id==0 else None

        # compute zero-filed coarse reconstruction as input
        x_input = ifft2c(y_input)
        #save_figure(x_input[:,:,0],"x_input_real",self.experiment_path) if id==0 else None
        #save_figure(x_input[:,:,1],"x_input_imag",self.experiment_path) if id==0 else None

        mean = x_input.mean(dim=[0,1],keepdim=True)
        std = x_input.std(dim=[0,1],keepdim=True)
        x_input = (x_input - mean) / (std + 1e-11)

        # training target. target_mask is all ones if supervised training
        y_target = y * target_mask + 0.0

        # training target in image domain
        x_target = ifft2c(y_target)

        return y_input, x_input, y_target, x_target, x, input_mask, target_mask, mean, std, fname
    
def compute_number_of_lines_in_input_target_kspace(p,mu,nu,n=160):
    
    q = (mu-p+nu-mu*nu)/(1-p)

    size_low = int(round(n*nu))

    S_p_size_high = int(round((p-nu)*n))
    S_mu_size_high = int(round((mu-nu)*n))

    S_mu_size_high_remainding = S_mu_size_high - S_p_size_high

    overlap_size_high = int(round(( (p-nu) / (1-nu) ) * ( (q-nu) / (1-nu) ) *(n-n*nu)))

    input_size = size_low + S_p_size_high
    target_size = size_low + S_mu_size_high_remainding + overlap_size_high

    weight_on_random_lines = np.sqrt((1-nu)/(q-nu))

    return input_size, target_size, overlap_size_high, size_low, p, q, mu, nu, weight_on_random_lines


    
