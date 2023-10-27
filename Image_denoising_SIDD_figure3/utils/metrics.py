import numpy as np
import torch

def psnr_gpu(clean, noisy, normalized=True):
    """Use skimage.meamsure.compare_ssim to calculate SSIM
    Args:
        clean (Tensor): (C, H, W)
        noisy (Tensor): (C, H, W)
        normalized (bool): If True, the range of tensors are [0., 1.]
            else [0, 255]
    Returns:
        SSIM per image: (B, )
    """
    if len(clean.shape)!=3 or len(noisy.shape)!=3:
        raise ValueError("psnr expects clean (Tensor): (C, H, W) noisy (Tensor): (C, H, W)")
    if normalized:
        clean = clean.clamp(0, 1)
        noisy = noisy.clamp(0, 1)

    #max_value = x.max().unsqueeze(0)
    mse = torch.mean(torch.abs(noisy-clean)**2)
    psnr = 20 * torch.log10(torch.tensor(1.))- 10 * torch.log10(mse)

    return psnr.item()


