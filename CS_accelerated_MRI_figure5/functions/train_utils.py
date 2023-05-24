
import torch
import numpy as np
import random
import os
import glob
import logging
from torch.serialization import default_restore_location
from tensorboard.backend.event_processing import event_accumulator
import time
import matplotlib.pyplot as plt

def setup_experiment(hp_exp):
    '''
    - Handle seeding
    - Create directories
    - Look for checkpoints to load from
    '''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    torch.manual_seed(hp_exp['seed'])
    torch.cuda.manual_seed(hp_exp['seed'])
    np.random.seed(hp_exp['seed'])
    random.seed(hp_exp['seed'])

    hp_exp['log_path'] = './'+ hp_exp['exp_name'] + '/log_files/'
    os.makedirs(hp_exp['log_path'] + 'checkpoints/', exist_ok=True)
    
    hp_exp['log_file'] = os.path.join(hp_exp['log_path'], "train.log")

    # Look for checkpoints to load from
    available_models = glob.glob(hp_exp['log_path'] + 'checkpoints/*.pt')
    if available_models and hp_exp['resume_from_which_checkpoint']=='last':
        hp_exp['restore_file'] = hp_exp['log_path'] + 'checkpoints/checkpoint_last.pt'
    elif available_models and hp_exp['resume_from_which_checkpoint']=='best':
        hp_exp['restore_file'] = hp_exp['log_path'] + 'checkpoints/checkpoint_best.pt'
    else:
        hp_exp['restore_file'] = None


    # Set attributes of the function save_checkpoint. They will be used to track the validation score and trigger saving a checkpoint
    mode_lookup = {
        'SSIM' : 'max',
        'PSNR' : 'max',
        'L1' : 'min',
        'L2' : 'min',
        'MSE' : 'min',
        'L2_kspace' : 'min',
        'L1_kspace' : 'min',
    }
    
    save_checkpoint.best_epoch = -1
    save_checkpoint.last_epoch = 0
    save_checkpoint.start_epoch = 0
    save_checkpoint.global_step = 0
    save_checkpoint.current_lr = hp_exp['lr']
    save_checkpoint.break_counter = 0
    save_checkpoint.best_val_current_lr_interval = float("inf") if  mode_lookup[hp_exp['decay_metric']] == "min" else float("-inf")
    save_checkpoint.lr_interval_counter = 0
    save_checkpoint.mode = mode_lookup[hp_exp['decay_metric']]
    save_checkpoint.best_score =  float("inf") if save_checkpoint.mode == "min" else float("-inf")

    return hp_exp

def init_logging(hp_exp):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    handlers = [logging.StreamHandler()]

    mode = "a" if hp_exp['restore_file'] else "w"
    handlers.append(logging.FileHandler(hp_exp['log_file'], mode=mode))
    logging.basicConfig(handlers=handlers, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    if hp_exp['mode'] == 'train':
        logging.info("Arguments: {}".format(hp_exp))
        


def save_checkpoint(hp_exp, epoch, model, optimizer=None, scheduler=None, score=None):
    ''''
    This function is used to save a range of parameters related to the training progress. 
    Saving those parameters allows to interrupt and then pick up training later at any point.
    At the beginning of every experiment the parameters are initialized in setup_experiment()
    Parameters:
    - best_score: Holds the best validation score so far
    - best_epoch: Holds the epoch in which the best validation score was achieved
    - last_epoch: Holds the current epoch.
    - break_counter: Count the number of epochs with minimal lr
    - best_val_current_lr_interval: Holds the best val performance for the current lr-inerval
    - lr_interval_counter: Counts for how many lr intervals there was no improvement
    - 
    '''
    save_checkpoint.last_epoch = epoch
    best_score = save_checkpoint.best_score
    
    if (score < best_score and save_checkpoint.mode == "min") or (score > best_score and save_checkpoint.mode == "max"):
        save_checkpoint.best_epoch = epoch
        save_checkpoint.best_score = score

    model = [model] if model is not None and not isinstance(model, list) else model
    optimizer = [optimizer] if optimizer is not None and not isinstance(optimizer, list) else optimizer
    scheduler = [scheduler] if scheduler is not None and not isinstance(scheduler, list) else scheduler
    state_dict = {
        "last_step": save_checkpoint.global_step, #set
        "last_score": score, #set
        "break_counter": save_checkpoint.break_counter,
        "best_val_current_lr_interval": save_checkpoint.best_val_current_lr_interval,
        "lr_interval_counter": save_checkpoint.lr_interval_counter,
        "last_epoch": save_checkpoint.last_epoch, #set
        "best_epoch": save_checkpoint.best_epoch, #set
        "current_lr":save_checkpoint.current_lr, #set
        "mode": save_checkpoint.mode,
        "best_score": getattr(save_checkpoint, "best_score", None), #set
        "model": [m.state_dict() for m in model] if model is not None else None,
        "optimizer": [o.state_dict() for o in optimizer] if optimizer is not None else None,
        "scheduler": [s.state_dict() for s in scheduler] if scheduler is not None else None,
        "args": hp_exp,
    }
    torch.save(state_dict, os.path.join(hp_exp['log_path'] + 'checkpoints/', "checkpoint_last.pt"))

    if hp_exp['epoch_checkpoints']:
        if epoch in hp_exp['epoch_checkpoints']:
            torch.save(state_dict, os.path.join(hp_exp['log_path'] + 'checkpoints/', "checkpoint{}.pt".format(epoch)))
    if (score < best_score and save_checkpoint.mode == "min") or (score > best_score and save_checkpoint.mode == "max"):
        torch.save(state_dict, os.path.join(hp_exp['log_path'] + 'checkpoints/', "checkpoint_best.pt"))

            
            
def load_checkpoint(hp_exp, model=None, optimizer=None, scheduler=None):
    
    print('restoring model..')
    state_dict = torch.load(hp_exp['restore_file'], map_location=lambda s, l: default_restore_location(s, "cpu"))

    save_checkpoint.last_epoch = state_dict["last_epoch"]
    save_checkpoint.start_epoch = state_dict["last_epoch"]+1
    save_checkpoint.global_step = state_dict["last_step"]
    save_checkpoint.best_score = state_dict["best_score"]
    save_checkpoint.best_epoch = state_dict["best_epoch"]
    save_checkpoint.break_counter = state_dict["break_counter"]
    save_checkpoint.best_val_current_lr_interval = state_dict["best_val_current_lr_interval"]
    save_checkpoint.lr_interval_counter = state_dict["lr_interval_counter"]
    save_checkpoint.current_lr = state_dict["current_lr"]
    save_checkpoint.mode = state_dict["mode"]

    

    model = [model] if model is not None and not isinstance(model, list) else model
    optimizer = [optimizer] if optimizer is not None and not isinstance(optimizer, list) else optimizer
    scheduler = [scheduler] if scheduler is not None and not isinstance(scheduler, list) else scheduler
    if model is not None and state_dict.get("model", None) is not None:
        for m, state in zip(model, state_dict["model"]):
            m.load_state_dict(state)
    if optimizer is not None and state_dict.get("optimizer", None) is not None:
        for o, state in zip(optimizer, state_dict["optimizer"]):
            o.load_state_dict(state)
    if scheduler is not None and state_dict.get("scheduler", None) is not None:
        for s, state in zip(scheduler, state_dict["scheduler"]):
            #milestones = s.milestones
            #state['milestones'] = milestones
            s.load_state_dict(state)
            #s.milestones = milestones

    logging.info("Loaded checkpoint {} from best_epoch {} last_epoch {}".format(hp_exp['restore_file'], save_checkpoint.best_epoch, save_checkpoint.last_epoch))


