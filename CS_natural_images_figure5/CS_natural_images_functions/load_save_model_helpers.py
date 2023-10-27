import glob
import torch
import os
from torch.serialization import default_restore_location
import logging

def setup_experiment_or_load_checkpoint(experiment_path, resume_from='best', model=None, optimizer=None, scheduler=None):
    '''
    Args:
        - resume_from: Either 'best' or 'some_number' where some_number could by any epoch at which a checkpoint was saved
    '''

    # Look for checkpoints to load from. If avalable, always load.
    available_models = glob.glob(experiment_path + '*.pt')
    if available_models:

        restore_file = experiment_path + f"checkpoint_{resume_from}.pt"

        print('restoring model..')
        state_dict = torch.load(restore_file, map_location=lambda s, l: default_restore_location(s, "cpu"))

        save_checkpoint.last_epoch = state_dict["best_epoch"] if resume_from=='best' else state_dict["last_epoch"]
        save_checkpoint.start_epoch = state_dict["best_epoch"]+1 if resume_from=='best' else state_dict["last_epoch"]+1
        save_checkpoint.best_score = state_dict["best_score"]
        save_checkpoint.best_epoch = state_dict["best_epoch"]
        

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
                s.load_state_dict(state)

        logging.info("Loaded checkpoint {} with best_epoch {} last_epoch {}".format(restore_file, save_checkpoint.best_epoch, save_checkpoint.last_epoch))

    else:
        print("No checkpoint to load. Start training from scratch.")
   
        save_checkpoint.best_epoch = -1
        save_checkpoint.last_epoch = 0
        save_checkpoint.start_epoch = 0

        save_checkpoint.best_score =  float("-inf")



def save_checkpoint(experiment_path, epoch, model, optimizer=None, scheduler=None, score=None, save_at_epochs=None):
    ''''
    Args:
        - 
    '''
    save_checkpoint.last_epoch = epoch
    best_score = save_checkpoint.best_score
    
    if score > best_score:
        save_checkpoint.best_epoch = epoch
        save_checkpoint.best_score = score

    if score > best_score:
        
        model = [model] if model is not None and not isinstance(model, list) else model
        optimizer = [optimizer] if optimizer is not None and not isinstance(optimizer, list) else optimizer
        scheduler = [scheduler] if scheduler is not None and not isinstance(scheduler, list) else scheduler
        state_dict = {
            "last_epoch": save_checkpoint.last_epoch, 
            "best_epoch": save_checkpoint.best_epoch, 
            "best_score": save_checkpoint.best_score,
            "model": [m.state_dict() for m in model] if model is not None else None,
            "optimizer": [o.state_dict() for o in optimizer] if optimizer is not None else None,
            "scheduler": [s.state_dict() for s in scheduler] if scheduler is not None else None,
        }

        torch.save(state_dict, os.path.join(experiment_path + "checkpoint_best.pt"))

    if save_at_epochs:
        if epoch in save_at_epochs:

            model = [model] if model is not None and not isinstance(model, list) else model
            optimizer = [optimizer] if optimizer is not None and not isinstance(optimizer, list) else optimizer
            scheduler = [scheduler] if scheduler is not None and not isinstance(scheduler, list) else scheduler
            state_dict = {
                "last_epoch": save_checkpoint.last_epoch, #set
                "best_epoch": save_checkpoint.best_epoch, #set
                "best_score": getattr(save_checkpoint, "best_score", None), #set
                "model": [m.state_dict() for m in model] if model is not None else None,
                "optimizer": [o.state_dict() for o in optimizer] if optimizer is not None else None,
                "scheduler": [s.state_dict() for s in scheduler] if scheduler is not None else None,
            }
            
            torch.save(state_dict, os.path.join(experiment_path + f"checkpoint{epoch}.pt"))



    