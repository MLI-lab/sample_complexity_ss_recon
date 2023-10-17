# %%
##################################
# Import python packages
import numpy as np
import os
import traceback

from functions.main_graddiff import main_train


os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# %%

##################################
# Customize hyperparemters

# Specify an ID for each experiment, e.g. 001,002,... 
exp_nums =  ['001']


# Specify an ID for each experiment, e.g. 001,002,... 
exp_nums =  ['001']

## Available datasets for MRI experiments:
#train_49827_selfsup_slice
#train_24900_selfsup_slice
#train_9977_selfsup_slice
#train_4943_selfsup_slice
#train_2466_selfsup_slice
#train_970_selfsup_slice
#train_491_selfsup_slice
#train_223_selfsup_slice
#train_96_selfsup_slice
#train_75_selfsup_slice
#train_48_selfsup_slice
#test_4713_selfsup_slice
#val_313_selfsup_slice


hp_all_exps = {
    #############################################################
    ### SET FALGS AND PICK NUMBER OF RUNS ###
    # Specify the number of runs with different random initialization per experiment, e.g. [0,1,2] means three runs
    'num_runs' : [[0]],
    # Start training or continue training if there exists already a checkpoint.
    'training' : [True],
    # Evaluate best and last checkpoint on validation and test set.
    'testing' : [True],
    # Load model from last or best checkpoint. "last" is for continuing an interrupted training, "best" is automatically set for test mode, Right now "best" is not supported during training.
    'resume_from_which_checkpoint' : ["last"],

    #############################################################
    ### CHOOSE DATASETS ###
    # Path to data directories: 
    'data_path' : ['/media/ssd1/fastMRIdata/brain/'],
    # Path to val data directories. Only specify if different from data_path.
    'val_path' : [None],
    # Path to test data directories. Only specify if different from data_path.
    'test_path' : [None],
    # Path to sensitivity maps directories. 
    'smaps_path' : ['/media/ssd1/fastMRIdata/brain_sensmaps_train_and_val/'],
    # If true sens maps are loaded in the train, validation and test loader.
    'provide_senmaps' : [True],
    # Choose a training set, e.g. train_491_selfsup_slice
    'train_set' : ['./datasets/train_9977_selfsup_slice.yaml'],
    # Specify the size of the training set.
    'train_size' : [10000],
    # Choose a validation set.
    'val_set' : ['./datasets/val_313_selfsup_slice.yaml'],
    # Choose test sets: 
    'test_sets' : [['./datasets/val_313_selfsup_slice.yaml','./datasets/test_4713_selfsup_slice.yaml']],
    # Which challenge to preprocess for.
    'challenge' : ["multicoil"],

    #############################################################
    ### DEFINE SETTINGS FOR MASK FUNCTIONS ###
    # Type of k-space mask. Use n2n.
    'mask_type' : ["n2n"],
    # Number of center lines to use in mask. 0.08 for acceleration 4.
    'center_fraction' : [0.08],
    # Acceleration rates to use for masks, e.g. 4 or 8. This acceleration factor is always applied to create network inputs
    'acceleration' : [4.0],
    # If True: the same input and target masks for both supervised and self-supervised training are used over all epochs.
    # Note that masks still differ over slices in a volume.
    # If False: Supervised training draws a new input mask in every epoch for every training slice
    # Self-supervised still always draws the same set of frequecies defined by 'acceleration_total', but the split into input and target is re-sampled in every epoch.
    'use_mask_seed_for_training' : [True],

    #############################################################
    ### ADDITIONAL FLAGS FOR SELF-SUPERVISED TRAINING
    # If True the training loss is only computed on the masked kspace fraction selfsup_acceleration (plus accelerations if selfsup_compute_loss_on_input_freq is True)
    'selfsup' : [True],
    # Determines the number of frequencies given during self-supervised training. Those are then split into input and target frequencies
    # such that the input has acceleration factor 'acceleration' and the target contains the remaining freqs plus some overlap.
    # Must be larger than 'accelerations'. If selfsup=False this value does not matter.
    'acceleration_total' : [3.0],

    #############################################################
    ### DEFINE THE MODEL ARCHITECTURE ###
    # Number of channels in the first layer.
    'chans' : [64],
    # Number of downsampling/upsampling layers in the UNet (0 or 1 still has one downsampling/upsampling).
    'num_pool_layers' : [4],
    # Use separate channels for real and imaginary part. Must be True.
    'two_channel_imag_real' : [True],

    #############################################################
    ### OPTIMIZER, LOSS FUNCTION, NUMBER OF EPOCHS, BATCH SIZE, INITIAL LEARNING RATE ###
    # Currently available RMSprop or Adam
    'optimizer' : ['RMSprop'],
    # List of loss functions for training. Currently available L1, L2 or SSIM loss or L2_kspace.
    'loss_functions' : [['L2_kspace']],
    # If True, then the network input is not cropped.
    'compute_sup_loss_in_kspace' : [True],
    # Maximal number of epochs.
    'num_epochs' : [2],
    # Initial learning rate.
    'lr' : [1e-3],
    # Batch size for mini batch training.
    'batch_size' : [1],

    #############################################################
    ### LEARNING RATE SCHEME FOR DECAY ###
    # Currently available MultiStepLR or ReduceLROnPlateau
    'lr_scheduler' : ['ReduceLROnPlateau'],
    # If true, lr_convergence_break_counter and lr_min_break_counter are applied 
    'early_stop_lr_deacy' : [True],
    # Terminate training once this lr is reached.
    'lr_min' : [1e-6],
    # Once lr_min is reached training continues for this many epochs before terminated.
    'lr_min_break_counter' : [10],
    # Once for this many consecutive learning rate decays no improvement in val_loss is observed the training is early stopped.
    'lr_convergence_break_counter' : [2],
    # Decay lr by this factor.
    'lr_decay_factor' : [0.1],
    # If scheduler is MultiStepLR then decay lr after these number of epochs.
    'lr_milestones' : [[50,60]],
    # If scheduler is ReduceLROnPlateau, decay lr after these many epochs without improving val loss.
    'lr_patience' : [10],
    # If scheduler is ReduceLROnPlateau, improving val loss by less than this does not count as improvement.
    'lr_threshold' : [0.0001],
    # Metric used to determine a Plateau
    'decay_metric' : ['L2_kspace'],

    #############################################################
    ### CHOOSE OPTIONS FOR ACCELERATING TRAINING AND CHECKPOINTING
    # Number of workers for dataloader.
    'num_workers' : [8],
    # Enable logging to Tensorboard
    'tb_logging' : [True],
    # List of validaiton sample indices to be logged to Tensorboard. Put empty dictionary to not log images
    'log_val_images' : [{'file_brain_AXT2_210_2100189':3, 'file_brain_AXT2_200_2000396': 7, 'file_brain_AXT2_200_6002509': 12}],
    'log_train_images' : [{'file_brain_AXT2_207_2070586':8, 'file_brain_AXT2_210_6001651': 6}],
    # Log images ever log_image_interval epochs
    'log_image_interval' : [8],
    # Optional: A list of epochs at which the model is saved as "checkpoint{}.pt".format(epoch)
    'epoch_checkpoints' : [None],
    # Interval for validation
    'val_interval' : [1],
    # Save images from the testset to the log directory. Dictionary containing filenames and slice numbers. Put empty dictionary to not log images
    'save_test_images': [{'file_brain_AXT2_207_2070041': 7, 'file_brain_AXT2_201_2010484': 12, 'file_brain_AXT2_203_2030348': 3,'file_brain_AXT2_209_6001477':9, 'file_brain_AXT2_210_6001600': 6, 'file_brain_AXT2_205_2050106': 7}],
}  




# Sanity checks
for key in hp_all_exps.keys():
    # expand lists with constant settings to the length of the number of experiments
    if len(hp_all_exps[key]) == 1 and len(hp_all_exps[key]) != len(exp_nums):
        hp_all_exps[key] = [hp_all_exps[key][0] for _ in range(len(exp_nums))]
        #hp_all_exps[key] = hp_all_exps[key]*(len(exp_nums))

    if len(hp_all_exps[key]) != len(exp_nums):
        print(key)
        raise ValueError("Specify hyperparameters for every experiment!") 

# %%
# Print out the names of the new experiments
printouts = []
print('')
for ee in range(len(exp_nums)):
    for rr in hp_all_exps['num_runs'][ee]:
        exp_name =  'E' + str(exp_nums[ee]) + \
                    '_t' + str(hp_all_exps['train_size'][ee]) + \
                    '_l' + str(hp_all_exps['num_pool_layers'][ee]) + \
                    'c' + str(hp_all_exps['chans'][ee]) + \
                    '_bs' + str(hp_all_exps['batch_size'][ee]) +\
                    '_lr' + str(np.round(hp_all_exps['lr'][ee],6))[2:]
        if rr>0:
            exp_name = exp_name + '_run{}'.format(rr+1)
        printouts.append(exp_name)
print(printouts)
# %%
for ee in range(len(exp_nums)):
    hp_exp = {}
    for k in hp_all_exps.keys():
        hp_exp[k] = hp_all_exps[k][ee]#.copy()

    for rr in hp_exp['num_runs']:
        exp_name =  'E' + str(exp_nums[ee]) + \
                    '_t' + str(hp_all_exps['train_size'][ee]) + \
                    '_l' + str(hp_all_exps['num_pool_layers'][ee]) + \
                    'c' + str(hp_all_exps['chans'][ee]) + \
                    '_bs' + str(hp_all_exps['batch_size'][ee]) +\
                    '_lr' + str(np.round(hp_all_exps['lr'][ee],6))[2:]
        if rr>0:
            exp_name = exp_name + '_run{}'.format(rr+1)
        if not os.path.isdir('./'+exp_name):
            os.mkdir('./'+exp_name)
        hp_exp['seed'] = int(42 + 10*rr)
        hp_exp['exp_name'] = exp_name
        if not hp_exp['train_set']:
            hp_exp['train_set'] = f"../datasets/train_{hp_exp['train_size']}_filenames_slice.yaml"
        ########
        # Training
        ########
        try:
            if hp_exp['training']:  
                print('\n{} - Training\n'.format(exp_name))
                hp_exp['mode'] = 'train'
                # Perform training
                train_meters, val_metric_dict = main_train(hp_exp)
                print('\n{} - Training finished\n'.format(exp_name))
        except:
            with open("./"+exp_name+"/errors_train.txt", "a+") as text_file:
                error_str = traceback.format_exc()
                print(error_str, file=text_file)   
            print(error_str)
        
# %%



