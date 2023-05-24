'''
Create MRI training and test sets and potentially also with predefined masks
Each dataset is defined by a list of dictionaries. 
Each dictionary has entries of the form 

{'path': 'multicoil_train/file_brain_AXT2_209_2090116.h5', 
'slice': 8, 
'filename': 'file_brain_AXT2_209_2090116',
'predefined_mask': 'predefined_masks/ttt_brain_mask'}

If the entry 'slice' is empty all slices of the volume are added to the dataset.
If the entry 'predefined_mask' is empty a random or equispaced mask is generated based on the seed computed from filename.
If the entry 'predefined_mask' contains a path as string the mask is loaded from the path. (not implemented yet)
If the entry 'predefined_mask' contains a 1-D list of lenght equal number of columns, load list to create mask.
    In the fastMRI training code masks are 1-D numpy arrays of dimensions [coils,rows,columns,real/imag], i.e. [1,1,columns,1]
'''

#%%
import glob
import yaml
import pickle
import h5py
import random

def save_filename_list(filename_list, dataset_name):
    with open(dataset_name + ".yaml", "w") as fp:
        yaml.dump(filename_list,fp)

# Get a list of all files in the multicoil_train and multicoil_val directory
# Then we are able to match files to the correct directory
brain_multicoil_train_path_list = glob.glob('../../../../media/ssd1/fastMRIdata/brain/multicoil_train/*.h5')

brain_multicoil_train_file_list = []
for file in brain_multicoil_train_path_list:
    brain_multicoil_train_file_list.append(file[file.find('file'):-3])

brain_multicoil_val_path_list = glob.glob('../../../../media/ssd1/fastMRIdata/brain/multicoil_val/*.h5')

brain_multicoil_val_file_list = []
for file in brain_multicoil_val_path_list:
    brain_multicoil_val_file_list.append(file[file.find('file'):-3])

#%%
'''
Data sets for scaling law simulations
'''
# Write the training sets used for MRI scaling laws into the new format that allows to select slice numbers
trainset_lists = glob.glob('train*')

print(trainset_lists)

slice_trainset_list = []

for trainset_list in trainset_lists:
    slice_trainset_list = []

    with open(trainset_list, 'r') as stream:
        filenames = yaml.safe_load(stream)
        #print(filenames)
    for filename in filenames:
        #print(filename)
        if filename in brain_multicoil_train_file_list:
            slice_trainset_list.append({'path': 'multicoil_train/'+filename+'.h5', 'slice': None, 'filename': filename, 'predefined_mask': None})
        elif filename in brain_multicoil_val_file_list:
            slice_trainset_list.append({'path': 'multicoil_val/'+filename+'.h5', 'slice': None, 'filename': filename, 'predefined_mask': None})
        else:
            print('file not found')

    #print(slice_trainset_list)
    save_filename_list(slice_trainset_list, trainset_list[:-5]+'_slice')

# %%
# Write validation and test set in new format
for list in ['test_filenames.yaml','val_filenames.yaml']:
    slice_list = []

    with open(list, 'r') as stream:
        filenames = yaml.safe_load(stream)
    for filename in filenames:
        if filename in brain_multicoil_train_file_list:
            slice_list.append({'path': 'multicoil_train/'+filename+'.h5', 'slice': None, 'filename': filename, 'predefined_mask': None})
        elif filename in brain_multicoil_val_file_list:
            slice_list.append({'path': 'multicoil_val/'+filename+'.h5', 'slice': None, 'filename': filename, 'predefined_mask': None})
        else:
            print('file not found')

    save_filename_list(slice_list, list[:-5]+'_slice')

# %%
'''
Datasets from TTT simulations
'''
# Load training and test sets from Mohammad's TTT paper and save in our format
# Also load and save the masks used for training

# mask that was used during ttt experiments on T2 brain. For training and testing
with open('../../ttt_for_deep_learning_cs/unet/train_data/' + 'brain' + '_mask','rb') as fn:
    mask2d = pickle.load(fn)
mask1d = mask2d[0,:].tolist()

# brain AXT2 310 mid slices training set from multicoil_train
with open('../../ttt_for_deep_learning_cs/unet/train_data/' + 'brain' + '_train','rb') as fn:
    dataset = pickle.load(fn)

slice_list = []
T2_train_slice_list_filenames = []
for dataslice in dataset:
    filename = dataslice['filename'][:-3]
    if filename in brain_multicoil_train_file_list:
        slice_list.append({'path': 'multicoil_train/'+filename+'.h5', 'slice': dataslice['slice'], 'filename': filename, 'predefined_mask': mask1d})      
    elif filename in brain_multicoil_val_file_list:
        slice_list.append({'path': 'multicoil_val/'+filename+'.h5', 'slice': dataslice['slice'], 'filename': filename, 'predefined_mask': mask1d})
    T2_train_slice_list_filenames.append(filename)

print(len(slice_list))
save_filename_list(slice_list, 'ttt_brain_train')

# brain AXT2 100 mid slices test set from multicoil_val
with open('../../ttt_for_deep_learning_cs/unet/train_data/' + 'brain' + '_val','rb') as fn:
    dataset = pickle.load(fn)

slice_list = []
T2_test_slice_list_filenames = []
for dataslice in dataset:
    filename = dataslice['filename'][:-3]
    if filename in brain_multicoil_train_file_list:
        slice_list.append({'path': 'multicoil_train/'+filename+'.h5', 'slice': dataslice['slice'], 'filename': filename, 'predefined_mask': mask1d})
    elif filename in brain_multicoil_val_file_list:
        slice_list.append({'path': 'multicoil_val/'+filename+'.h5', 'slice': dataslice['slice'], 'filename': filename, 'predefined_mask': mask1d})
    T2_test_slice_list_filenames.append(filename)

print(len(slice_list))
save_filename_list(slice_list, 'ttt_brain_test')

# brain AXT2 100 mid slices validation set from multicoil_val. This one I have to create from scratch
dataset_size = 100
counter = 0
slice_list = []

tmp = list(zip(brain_multicoil_val_file_list,brain_multicoil_val_path_list))
random.shuffle(tmp) # randomly shuffle the file list

for filename,path in tmp:
    if 'AXT2' in filename:
        if filename not in T2_train_slice_list_filenames:
            if filename not in T2_test_slice_list_filenames:
                f = h5py.File(path, 'r')
                center_slice = f['reconstruction_rss'][()].shape[0]//2
                slice_list.append({'path': 'multicoil_val/'+filename+'.h5', 'slice': center_slice, 'filename': filename, 'predefined_mask': mask1d})
                counter+=1
                if counter==dataset_size:
                    break

print(len(slice_list))
save_filename_list(slice_list, 'ttt_brain_val')

#%%
# Sanity check if everything worked well
with open('ttt_brain_val.yaml', 'r') as stream:
    filenames = yaml.safe_load(stream)

print(filenames[10])
print(len(filenames))

# %%

# %%
