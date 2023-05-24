import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def ssim(clean, noisy, normalized=True):
    """Use skimage.meamsure.compare_ssim to calculate SSIM
    Args:
        clean (Tensor): (B, C, H, W)
        noisy (Tensor): (B, C, H, W)
        normalized (bool): If True, the range of tensors are [0., 1.] else [0, 255]
    Returns:
        SSIM per image: (B, )
    """
    if len(clean.shape)!=4 or len(noisy.shape)!=4:
        raise ValueError("ssim expects clean (Tensor): (B, C, H, W) noisy (Tensor): (B, C, H, W)")
    if normalized:
        clean = clean.mul(255).clamp(0, 255)
        noisy = noisy.mul(255).clamp(0, 255)

    clean = clean.cpu().detach().numpy().astype(np.float32)
    noisy = noisy.cpu().detach().numpy().astype(np.float32)
    clean = np.moveaxis(clean,1,-1)
    noisy = np.moveaxis(noisy,1,-1)

    return np.array([structural_similarity(c, n, data_range=255, multichannel=True) for c, n in zip(clean, noisy)]).mean()


def psnr(clean, noisy, normalized=True):
    """Use skimage.meamsure.compare_ssim to calculate SSIM
    Args:
        clean (Tensor): (B, C, H, W)
        noisy (Tensor): (B, C, H, W)
        normalized (bool): If True, the range of tensors are [0., 1.]
            else [0, 255]
    Returns:
        SSIM per image: (B, )
    """
    if len(clean.shape)!=4 or len(noisy.shape)!=4:
        raise ValueError("psnr expects clean (Tensor): (B, C, H, W) noisy (Tensor): (B, C, H, W)")
    if normalized:
        clean = clean.mul(255).clamp(0, 255)
        noisy = noisy.mul(255).clamp(0, 255)

    clean = clean.cpu().detach().numpy().astype(np.float32)
    noisy = noisy.cpu().detach().numpy().astype(np.float32)
    return np.array([peak_signal_noise_ratio(c, n, data_range=255) for c, n in zip(clean, noisy)]).mean()


