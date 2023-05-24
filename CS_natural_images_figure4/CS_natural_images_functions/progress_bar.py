from collections import OrderedDict
from numbers import Number
from tqdm import tqdm
import torch
import logging
import os
import numpy as np

def init_logging(experiment_path):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    handlers = [logging.StreamHandler()]

    mode = "a" if os.path.exists(experiment_path+"train.log") else "w"
    handlers.append(logging.FileHandler(experiment_path+"train.log", mode=mode))
    logging.basicConfig(handlers=handlers, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.val = val / n
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

class TrackMeter(object):
    def __init__(self, inc_or_dec='decaying'):
        self.inc_or_dec = inc_or_dec
        self.reset()
        

    def reset(self):
        self.val = []
        self.epochs = []
        self.count = 0
        self.best_val = float("inf") if self.inc_or_dec=='decaying' else float("-inf")
        self.best_count = 0
        self.best_epoch = 0

    def update(self, val, epoch):
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.val.append(val)
        self.epochs.append(epoch)
        
        if (self.inc_or_dec=='decaying' and val < self.best_val) or (self.inc_or_dec=='increasing' and val > self.best_val):
            self.best_val = val
            self.best_count = self.count
            self.best_count = epoch
        self.count += 1

class TrackMeter_testing(object):
    def __init__(self,):
        self.reset()  

    def reset(self):
        self.val = []
        self.avg = 0
        self.std = 0

    def update(self, val,):
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.val.append(val)
        self.avg = np.mean(self.val)
        self.std = np.std(self.val)


class ProgressBar:
    def __init__(self, iterable, epoch, quiet=False):
        self.epoch = epoch
        self.quiet = quiet
        self.prefix = f"epoch {epoch:02d}"
        self.iterable = iterable if self.quiet else tqdm(iterable, self.prefix, leave=False)

    def __iter__(self):
        return iter(self.iterable)

    def log(self, stats, verbose=False):
        if not self.quiet:
            self.iterable.set_postfix(self.format_stats(stats, verbose), refresh=True)

    def format_stats(self, stats, verbose=False):
        postfix = OrderedDict(stats) # method set_postfix requires ordered_dict
        for key, value in postfix.items():
            if isinstance(value, Number):
                fmt = "{:.6f}" if value > 0.001 else "{:.3e}"
                postfix[key] = fmt.format(value)
            elif isinstance(value, AverageMeter):
                if verbose:
                    postfix[key] = f"{value.avg:.6f} ({value.val:.6f})"
                else:
                    postfix[key] = f"{value.avg:.6f}"
            elif not isinstance(postfix[key], str):
                postfix[key] = str(value)
        return postfix

    def print(self, stats, verbose=False):
        postfix = " | ".join(key + " " + value.strip() for key, value in self.format_stats(stats, verbose).items())
        return f"{self.prefix + ' | ' if self.epoch is not None else ''}{postfix}"
    