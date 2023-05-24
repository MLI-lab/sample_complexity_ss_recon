"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Dict, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from packaging import version

from functions.coil_combine import rss, rss_complex
from functions.math import complex_abs, complex_conj, complex_mul

from functions.training.debug_helper import print_tensor_stats, save_figure


if version.parse(torch.__version__) >= version.parse("1.7.0"):
    from functions.fftc import fft2c_new as fft2c
    from functions.fftc import ifft2c_new as ifft2c
else:
    from functions.fftc import fft2c_old as fft2c
    from functions.fftc import ifft2c_old as ifft2c



from functions.data.subsample import MaskFunc

def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data: Input numpy array.

    Returns:
        PyTorch version of data.
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)

def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of data.

    Returns:
        The center cropped image.
    """
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]


def complex_center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data: The complex input tensor to be center cropped. It should have at
            least 3 dimensions and the cropping is applied along dimensions -3
            and -2 and the last dimensions should have a size of 2.
        shape: The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        The center cropped image
    """
    if not (0 < shape[0] <= data.shape[-3] and 0 < shape[1] <= data.shape[-2]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to, :]

def center_crop_to_smallest(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a center crop on the larger image to the size of the smaller.

    The minimum is taken over dim=-1 and dim=-2. If x is smaller than y at
    dim=-1 and y is smaller than x at dim=-2, then the returned dimension will
    be a mixture of the two.

    Args:
        x: The first image.
        y: The second image.

    Returns:
        tuple of tensors x and y, each cropped to the minimim size.
    """
    smallest_width = min(x.shape[-1], y.shape[-1])
    smallest_height = min(x.shape[-2], y.shape[-2])
    x = center_crop(x, (smallest_height, smallest_width))
    y = center_crop(y, (smallest_height, smallest_width))

    return x, y

def normalize(
    data: torch.Tensor,
    mean: Union[float, torch.Tensor],
    stddev: Union[float, torch.Tensor],
    eps: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """
    Normalize the given tensor.

    Applies the formula (data - mean) / (stddev + eps).

    Args:
        data: Input data to be normalized.
        mean: Mean value.
        stddev: Standard deviation.
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        Normalized tensor.
    """
    return (data - mean) / (stddev + eps)

def normalize_to_given_mean_std(
    im1: torch.Tensor,
    im2: torch.Tensor
    ) -> torch.Tensor:
    """
    This function computes the mean and std of im1 and normalizes im2 to have this mean and std.
    """
    im2 = (im2-im2.mean()) / im2.std()
    im2 *= im1.std()
    im2 += im1.mean()
    return im1,im2

def normalize_instance(
    data: torch.Tensor, eps: Union[float, torch.Tensor] = 0.0
) -> Tuple[torch.Tensor, Union[torch.Tensor], Union[torch.Tensor]]:
    """
    Normalize the given tensor  with instance norm/

    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed from the data itself.

    Args:
        data: Input data to be normalized
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps), mean, std

def normalize_separate_over_ch(
    x: torch.Tensor,
    mean: Union[float, torch.Tensor] = None,
    std: Union[float, torch.Tensor] = None,
    eps: Union[float, torch.Tensor] = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    If mean and stddev is given x is normalized to have this mean and std.
    If not given x is normalized to have 0 mean and std 1.
    x is supposed to have shape c,h,w and normalization is only over h,w
    Hence mean and std have shape c,1,1
    """
    if x.shape[-1]==2:
        raise ValueError("Group normalize does not expect complex dim at last position.")
    if len(x.shape) != 3:
        raise ValueError("Gourp normalize expects three dimensions in the input tensor.")

    # group norm
    if mean == None and std == None:
        mean = x.mean(dim=[1,2],keepdim=True)
        std = x.std(dim=[1,2],keepdim=True)

    return (x - mean) / (std + eps), mean, std

def apply_mask(
    data: torch.Tensor,
    mask_func: MaskFunc,
    seed: Optional[Union[int, Tuple[int, ...]]] = None,
    fix_selfsup_inputtarget_split: Optional[bool] = True,
    padding: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data: The input k-space data. This should have at least 3 dimensions,
            where dimensions -3 and -2 are the spatial dimensions, and the
            final dimension has size 2 (for complex values).
        mask_func: A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed: Seed for the random number generator.
        fix_selfsup_inputtarget_split: Only important for self-sup training. 
            If it is False the input/target split is random. Always True for validation and testing. 
            Determined by hp_exp['use_mask_seed_for_training'] during self-sup training.
        padding: Padding value to apply for mask.

    Returns:
        tuple containing:
            masked data: Subsampled k-space data
            mask: The generated mask
    """
    shape = np.array(data.shape)
    shape[:-3] = 1
    input_mask, target_mask, target_mask_weighted = mask_func(shape, seed, fix_selfsup_inputtarget_split)
    if padding is not None:
        input_mask[:, :, : padding[0]] = 0
        input_mask[:, :, padding[1] :] = 0  # padding value inclusive on right of zeros
        target_mask[:, :, : padding[0]] = 0
        target_mask[:, :, padding[1] :] = 0  # padding value inclusive on right of zeros

    input_data = data * input_mask + 0.0  # the + 0.0 removes the sign of the zeros
    target_data = data * target_mask + 0.0

    return input_data, input_mask, target_data, target_mask, target_mask_weighted

class UnetDataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
        hp_exp: dict = None,
        mode:str="train",
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
            mode: either train,val or test
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed
        self.hp_exp = hp_exp
        self.mode = mode

    def __call__(
        self,
        kspace: np.ndarray,
        sens_maps: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            sens_maps: Sensitivity maps of shape shape coils,height,width with complex valued entries
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                input_image: Zero-filled input image
                input_kspace: Undersampled input kspace, can be used for data consistency steps
                input_mask: input mask
                target_image: target_image for training in image domain. Can be center cropped, have 1 or 2 channels, be rss or sens combine reconstruction
                target_kspace: target_kspace for training
                target_mask: target_mask
                ground_truth_image: center cropped, real ground truth image for computing val and test scores in image domain.
                sens_maps: sensitivity maps to compute expand operations
                mean: for de-normalization
                std: for de-normalization
                fname: File name for logging
                slice_num: Serial number of the slice for logging

        """
        # Convert sens_maps and kspace to tensors. Stack imaginary parts along the last dimension
        if self.hp_exp['provide_senmaps']:
            sens_maps = to_tensor(sens_maps)
            sens_maps_conj = complex_conj(sens_maps)

            binary_background_mask = torch.round(torch.sum(complex_mul(sens_maps_conj,sens_maps),0)[:,:,0:1])
            binary_background_mask = torch.moveaxis( binary_background_mask , -1, 0 ) 

        else:
            sens_maps = torch.Tensor([0])
            binary_background_mask = torch.Tensor([0])

        kspace = to_tensor(kspace)

        crop_size = (target.shape[-2], target.shape[-1])

        
        #################################
        # Computing the target images that are used for supervised training in the image domain (can be complex or real, cropped or not)
        # and computing the gronud truth images to compute scores in the image domain (always real and center cropped)
        #################################
        target_image = ifft2c(kspace)

        target_image = complex_mul(target_image, sens_maps_conj)
        target_image = target_image.sum(dim=0, keepdim=False)

        ground_truth_image = complex_center_crop(target_image, crop_size)
        ground_truth_image = complex_abs(ground_truth_image)
        ground_truth_image = ground_truth_image.unsqueeze(0)

        if self.hp_exp['two_channel_imag_real']:
            # move complex channels to channel dimension
            target_image = torch.moveaxis( target_image , -1, 0 ) 
        else:
            # absolute value
            target_image = complex_abs(target_image)
            # add channel dimension
            target_image = target_image.unsqueeze(0)
        
        
        #################################
        # Computing input kspace and target kspace
        #################################

        # Get masked input and target kspace from the original kspace self.use_seed during training is determined by hp_exp['use_mask_seed_for_training'], while for validation and testing it is always True
        if self.hp_exp['selfsup']==True or self.use_seed:
            # during self-supervised training we compute the mask seed even if use_seed is False
            # The reason is that if use_seed is False we only want to have the split into input and target to be random, but the overall mask to be fixed
            seed = tuple(map(ord, fname))
        else:
            seed = None

        if seed:
            seed = seed + (slice_num,)
        if self.mode=='train': # the last option only matters for self-supervised training. During training the input/taret split is random if self.hp_exp['use_mask_seed_for_training']=False and fixed otherwise. During validation and testin it is always fixed.
            input_kspace, input_mask, target_kspace, target_mask, target_mask_weighted = apply_mask(kspace, self.mask_func, seed, fix_selfsup_inputtarget_split = self.hp_exp['use_mask_seed_for_training'])
        else:
            # during validation and test we want to always use the same seed for the same slice
            input_kspace, input_mask, target_kspace, target_mask, target_mask_weighted = apply_mask(kspace, self.mask_func, seed, True)

        #################################
        # Computing the coarse input image from the undersampled kspace
        #################################

        # inverse Fourier transform to get zero filled solution
        input_image = ifft2c(input_kspace) #shape: coils,height,width,2

        input_image = complex_mul(input_image, sens_maps_conj)
        input_image = input_image.sum(dim=0, keepdim=False) #shape: height,width,2

        if self.hp_exp['two_channel_imag_real']:
            # move complex channels to channel dimension
            input_image = torch.moveaxis( input_image , -1, 0 ) 
        else:
            # absolute value
            input_image = complex_abs(input_image)
            # add channel dimension
            input_image = input_image.unsqueeze(0)

        # normalize input to have zero mean and std one
        input_image, mean, std = normalize_separate_over_ch(input_image, eps=1e-11)

        return binary_background_mask, input_image, input_kspace, input_mask, target_image, target_kspace, target_mask, target_mask_weighted, ground_truth_image, sens_maps, mean, std, fname, slice_num
