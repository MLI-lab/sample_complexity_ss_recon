import torch
import numpy as np
from typing import Dict, Optional, Sequence, Tuple, Union, List
import os
import matplotlib.pyplot as plt

def save_figure(
    x: np.array,
    figname: str,
    hp_exp: Dict,
    save: Optional[bool]=True,):
    """"
    x must have dimension height,width
    """
    if save:
        save_path = hp_exp['log_path'] + 'train_figures/'

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


def print_tensor_stats(
    x: torch.Tensor,
    name: Optional[str]="Tensor",
    dim: Optional[Union[int,List]]=None,
    precision: Optional[float]=6,
    ):
    """
    Prints mean, std and min and max of a realy valued tensor.
    If dim is given stats are computed separatly over this dimension.
    """
    shape = x.shape
    if dim and not isinstance(dim, List):
        dim = [dim]

    if dim:
        for d in dim:
            #print(f"dimension {d}")
            x_reorder = torch.moveaxis(x,d,0)
            for s in range(shape[d]):
                l1norm = torch.sum(torch.abs(x_reorder[s]))
                l2norm = torch.sum(torch.abs(x_reorder[s]**2))
                rss = torch.sqrt(torch.sum(torch.abs(x_reorder[s]**2)))
                print(f"""{name} shape {x.shape} dim {d} {s+1}/{shape[d]}: 
                mean {np.round(x_reorder[s].mean().item(),precision)}, 
                std {np.round(x_reorder[s].std().item(),precision)}, 
                min {np.round(x_reorder[s].min().item(),precision)}, 
                max {np.round(x_reorder[s].max().item(),precision)}, 
                l1norm {np.round(l1norm.item(),precision)},
                l2norm {np.round(l2norm.item(),precision)},
                rss {np.round(rss.item(),precision)}""")
    else:
        l1norm = torch.sum(torch.abs(x))
        l2norm = torch.sum(torch.abs(x**2))
        rss = torch.sqrt(torch.sum(torch.abs(x**2)))
        print(f"""{name} shape {x.shape}: 
        mean {np.round(x.mean().item(),precision)}, 
        std {np.round(x.std().item(),precision)}, 
        min {np.round(x.min().item(),precision)}, 
        max {np.round(x.max().item(),precision)}, 
        l1norm {np.round(l1norm.item(),precision)},
        l2norm {np.round(l2norm.item(),precision)},
        rss {np.round(rss.item(),precision)}""")
