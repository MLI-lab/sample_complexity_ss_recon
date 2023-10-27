

# %%
import os
import glob
import traceback

from utils.main_function_helpers import *

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


# %% 


#######################################################
# Adjust the following parameters
#######################################################
training = True
val_testing = True


num_runs = [0,1,2,3,4] 

exp_nums = ['001','002']

# hyperparameter
hp = {
    ###############################
    ### Experimental details
    # Training set sizes
    'train_size' : [1000]*len(exp_nums), 
    # Validation set size 
    'val_size' : [10]*len(exp_nums),
    # Test set size 
    'test_size' : [30]*len(exp_nums),
    'val_on_testset' : [False]*len(exp_nums),
    # Images are cropped to this patch size
    'patch_size' : [128]*len(exp_nums),
    # Supervised
    'supervised' : [True, False],
    ###############################
    ### Model
    # Number of channels in the first layer
    'chans' : [56]*len(exp_nums), 
    # Number of downsampling/upsampling layers in the UNet 
    'num_pool_layers' : [2]*len(exp_nums),
    # Whether to do residual learning or not
    'residual' : [True]*len(exp_nums),
    ###############################
    ### Optimizer
    # Scheduler milestones for MultiStepLR
    'milestones' : [[100, 130, 160, 180, 190, 200, 210, 215, 220, 225]]*len(exp_nums),
    # Initial learning rate
    'lr' : [0.00032]*len(exp_nums),
    # Factor of learning rate decay
    'lr_gamma' : [0.5]*len(exp_nums),
    # Smallest learning rate at which training is terminated
    'lr_min' : [1e-6]*len(exp_nums),
    # If the validation loss in psnr exceeds the previously best loss by this threshold, it counts as an improvement
    'lr_threshold' : [0.001]*len(exp_nums),
    # The number of epochs we train with the smallest learning rate 
    'break_counter' : [5]*len(exp_nums),
    # Maximal number of training epochs
    'num_epochs' : [1000]*len(exp_nums),
    # Batch size
    'batch_size' : [1]*len(exp_nums),
    ###############################
    ### Other
    # Resume training from a saved model or start training from scratch
    'resume_training' : [False]*len(exp_nums),
    # Resume training from the last or the best checkpoint
    'restore_mode' : ['last']*len(exp_nums)
}
#######################################################


# Sanity checks
for key in hp.keys():
    if len(hp[key]) != len(exp_nums):
        print(key)
        raise ValueError("Specify hyperparameters for each experiment") 


# %%


printouts = []
for ee in range(len(exp_nums)):
    
    for rr in num_runs:
        exp_name =  'E' + exp_nums[ee] + '_t' + str(hp['train_size'][ee]) +'_l' + str(hp['num_pool_layers'][ee]) \
            +'c' + str(hp['chans'][ee]) +'_bs' + str(hp['batch_size'][ee]) +'_lr' + str(hp['lr'][ee])[2:]
        if hp['supervised'][ee]:
            exp_name = exp_name + '_sup'
        else:
            exp_name = exp_name + '_selfsup'
        if rr>0:
            exp_name = exp_name + '_run{}'.format(rr+1)
        printouts.append(exp_name)
print(printouts)

# %%
for ee in range(len(exp_nums)):
    
    for rr in num_runs:
        exp_name =  'E' + exp_nums[ee] +'_t' + str(hp['train_size'][ee]) +'_l' + str(hp['num_pool_layers'][ee]) \
            +'c' + str(hp['chans'][ee]) +'_bs' + str(hp['batch_size'][ee]) +'_lr' + str(hp['lr'][ee])[2:]
        
        if hp['supervised'][ee]:
            exp_name = exp_name + '_sup'
        else:
            exp_name = exp_name + '_selfsup'
        if rr>0:
            exp_name = exp_name + '_run{}'.format(rr+1)
        if not os.path.isdir('./'+exp_name):
            os.mkdir('./'+exp_name)
        
        ########
        # Training
        ########
        try:
            if training:  
                print('\n{} - Training\n'.format(exp_name))
                args = get_args(hp,ee,rr)
                args.output_dir = './'+exp_name
                cli_main(args)
                print('\n{} - Training finished\n'.format(exp_name))
        except:
            with open("./"+exp_name+"/errors_train.txt", "a+") as text_file:
                error_str = traceback.format_exc()
                print(error_str, file=text_file)   
            print(error_str)
            
        ########
        # Testing
        ########
        try:
            if val_testing:
                print('\n{} - Testing\n'.format(exp_name))

                test_modes = ["val"]

                for test_mode in test_modes:
                    for restore_mode in ["best"]:

                        args = get_args(hp,ee,rr)
                        args.output_dir = './'+exp_name
                        args.restore_mode = restore_mode
                        args.test_mode = test_mode
                        cli_main_test(args)
                        
                print('\n{} - Testing finished\n'.format(exp_name))
        except:
            with open("./"+exp_name+"/errors_test.txt", "a+") as text_file:
                error_str = traceback.format_exc()
                print(error_str, file=text_file)   
            print(error_str)      