import matplotlib.pyplot as plt
import torchvision
import io
import torch
import numpy as np

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    #img = buf.getvalue()#.to(torch.uint8)
    #img = torch.import_ir_module_from_buffer(img,dtype=torch.uint8)
    frameTensor = torch.tensor(np.frombuffer(buf.getvalue(), dtype=np.uint8), device='cpu')
    image = torchvision.io.decode_png(frameTensor)
    # Add the batch dimension
    #image = torch.unsqueeze(image, 0)
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