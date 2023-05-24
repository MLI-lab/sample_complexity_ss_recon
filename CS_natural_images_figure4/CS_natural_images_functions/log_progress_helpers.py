import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Sequence, Tuple, Union, List
import os
import torchvision
import io
import torch

from CS_natural_images_functions.losses import SSIMLoss

def complex_abs(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data: A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        Absolute value of data.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data ** 2).sum(dim=-1).sqrt()

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)

    frameTensor = torch.tensor(np.frombuffer(buf.getvalue(), dtype=np.uint8), device='cpu')
    image = torchvision.io.decode_png(frameTensor)
    return image

def get_figure(image,figsize,title):
    """Return a matplotlib figure of a given image."""
    if len(image.shape) != 3:
        raise ValueError("Image dimensions not suitable for logging to tensorboard.")
    if image.shape[0] == 1 or image.shape[0] == 3:
        image = np.rollaxis(image,0,3)
    # Create a figure to contain the plot.
    if figsize:
        figure = plt.figure(figsize=figsize)
    else:
        figure = plt.figure()
    # Start next subplot.
    plt.subplot(1, 1, 1, title=title)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap='gray')
    figure.tight_layout()

    return figure

def plot_figure(
    x: np.array,):
    """"
    x must have dimension height,width
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(x,'gray')
    ax.axis('off')
    #ax.set_title(title,fontsize=10)
    fig.tight_layout()
    plt.show()

def save_figure(
    x: np.array,
    figname: str,
    experiment_path: str,
    save: Optional[bool]=True,):
    """"
    x must have dimension height,width
    """
    if save:
        save_path = experiment_path + 'train_figures/'

        if not os.path.isdir(save_path):
            os.mkdir(save_path)
                    
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111)
        ax.imshow(x,'gray')
        ax.axis('off')
        #ax.set_title(title,fontsize=10)
        fig.tight_layout()
        plt.savefig(save_path + figname + ".png")
        plt.close(fig)


def save_test_image_with_dc(experiment_path, ground_truth_image, input_img, output, output_image_dc, fname, track_meter):

    save_path = experiment_path + 'test_figures/'

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    error = torch.abs(ground_truth_image - output)
    error_dc = torch.abs(ground_truth_image - output_image_dc)
    output = output - output.min() 
    output = output / output.max()
    output_image_dc = output_image_dc - output_image_dc.min() 
    output_image_dc = output_image_dc / output_image_dc.max()
    ground_truth_image = ground_truth_image - ground_truth_image.min()
    ground_truth_image = ground_truth_image / ground_truth_image.max()
    input_img = input_img - input_img.min()
    input_img = input_img / input_img.max()
    error = error - error.min() 
    error_dc = error_dc - error_dc.min() 
    max_norm = torch.stack([error,error_dc]).max()
    error = error / max_norm
    error_dc = error_dc / max_norm

    image = torch.cat([ground_truth_image, input_img, output, output_image_dc, error, error_dc], dim=0)
    image = torchvision.utils.make_grid(image, nrow=2, normalize=False, value_range=(0,1), pad_value=1)
    ssim_score = track_meter["SSIM"].val[-1]
    ssim_score_dc = track_meter["SSIM_dc"].val[-1]
    psnr_score = track_meter["PSNR"].val[-1]
    psnr_score_dc = track_meter["PSNR_dc"].val[-1]
    figure = get_figure(image.cpu().numpy(),figsize=(8,12),title=f"ssim={ssim_score:.4f}, dc={ssim_score_dc:.4f}, psnr={psnr_score:.3f}, dc={psnr_score_dc:.3f}") 

    plt.savefig(experiment_path + 'test_figures/' + f"{fname[0]}.png", dpi='figure')
    plt.close()

def add_img_to_tensorboard(writer, epoch, name, input_img_comp,output_comp,targetcomp,ksp):

    if ksp:
        input_img = torch.log(complex_abs(input_img_comp)[0]+ 1e-9)
        output = torch.log(complex_abs(output_comp)[0]+ 1e-9)
        target = torch.log(complex_abs(targetcomp)[0]+ 1e-9)
    else:
        input_img = complex_abs(input_img_comp)[0]
        output = complex_abs(output_comp)[0]
        target = complex_abs(targetcomp)[0]

    val_ssim_fct = SSIMLoss()
    max_value = target.max().unsqueeze(0)
    ssim_loss = 1-val_ssim_fct(output.unsqueeze(0).unsqueeze(0), target.unsqueeze(0).unsqueeze(0), data_range=max_value)

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

    figure = get_figure(image.cpu().numpy(),figsize=(3,12),title=f"ssim={ssim_loss.item():.6f}")

    writer.add_image(name+"_abs", plot_to_image(figure), epoch)

    if ksp:
        input_img = torch.log(torch.abs(input_img_comp[0,:,:,0])+ 1e-9)
        #input_img_max = torch.max(torch.stack((torch.log(torch.abs(input_img_comp[0,:,:,0])+ 1e-9),torch.log(torch.abs(input_img_comp[0,:,:,1])+ 1e-9))))
        
        output = torch.log(torch.abs(output_comp[0,:,:,0])+ 1e-9)
        #output_max = torch.max(torch.stack((torch.log(torch.abs(output_comp[0,:,:,0])+ 1e-9),torch.log(torch.abs(output_comp[0,:,:,1])+ 1e-9))))

        target = torch.log(torch.abs(targetcomp[0,:,:,0])+ 1e-9)  
        #target_max = torch.max(torch.stack((torch.log(torch.abs(output_comp[0,:,:,0])+ 1e-9),torch.log(torch.abs(output_comp[0,:,:,1])+ 1e-9))))

    else:
        input_img = input_img_comp[0,:,:,0]
        output = output_comp[0,:,:,0]
        target = targetcomp[0,:,:,0]

    val_ssim_fct = SSIMLoss()
    max_value = target.max().unsqueeze(0)
    ssim_loss = 1-val_ssim_fct(output.unsqueeze(0).unsqueeze(0), target.unsqueeze(0).unsqueeze(0), data_range=max_value)

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

    figure = get_figure(image.cpu().numpy(),figsize=(3,12),title=f"ssim={ssim_loss.item():.6f}")

    writer.add_image(name+"_re", plot_to_image(figure), epoch)

    if ksp:
        input_img = torch.log(torch.abs(input_img_comp[0,:,:,1])+ 1e-9)
        output = torch.log(torch.abs(output_comp[0,:,:,1])+ 1e-9)
        target = torch.log(torch.abs(targetcomp[0,:,:,1])+ 1e-9)  
    else:
        input_img = input_img_comp[0,:,:,1]
        output = output_comp[0,:,:,1]
        target = targetcomp[0,:,:,1]


    val_ssim_fct = SSIMLoss()
    max_value = target.max().unsqueeze(0)
    ssim_loss = 1-val_ssim_fct(output.unsqueeze(0).unsqueeze(0), target.unsqueeze(0).unsqueeze(0), data_range=max_value)

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

    figure = get_figure(image.cpu().numpy(),figsize=(3,12),title=f"ssim={ssim_loss.item():.6f}")

    writer.add_image(name+"_im", plot_to_image(figure), epoch)

    plt.close()
