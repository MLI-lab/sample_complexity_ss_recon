#

# %%
import glob
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import h5py
import scipy.io as sio

from multiprocessing import Pool

def extract_metainfo(path='0151_METADATA_RAW_010.MAT'):
    meta = sio.loadmat(path)['metadata']
    mat_vals = meta[0][0]
    mat_keys = mat_vals.dtype.descr

    keys = []
    for item in mat_keys:
        keys.append(item[0])

    py_dict = {}
    for key in keys:
        py_dict[key] = mat_vals[key]

    device = py_dict['Model'][0].lower()
    bitDepth = py_dict['BitDepth'][0][0]
    if 'iphone' in device or bitDepth != 16:
        noise = py_dict['UnknownTags'][-2][0][-1][0][:2]
        iso = py_dict['DigitalCamera'][0, 0]['ISOSpeedRatings'][0][0]
        pattern = py_dict['SubIFDs'][0][0]['UnknownTags'][0][0][1][0][-1][0]
        time = py_dict['DigitalCamera'][0, 0]['ExposureTime'][0][0]

    else:
        noise = py_dict['UnknownTags'][-1][0][-1][0][:2]
        iso = py_dict['ISOSpeedRatings'][0][0]
        pattern = py_dict['UnknownTags'][1][0][-1][0]
        time = py_dict['ExposureTime'][0][0]  # the 0th row and 0th line item

    rgb = ['R', 'G', 'B']
    pattern = ''.join([rgb[i] for i in pattern])

    asShotNeutral = py_dict['AsShotNeutral'][0]
    b_gain, _, r_gain = asShotNeutral

    # only load ccm1
    ccm = py_dict['ColorMatrix1'][0].astype(float).reshape((3, 3))

    return {'device': device,
            'pattern': pattern,
            'iso': iso,
            'noise': noise,
            'time': time,
            'wb': np.array([r_gain, 1, b_gain]),
            'ccm': ccm, }


def transform_to_rggb(img, pattern):
    assert len(img.shape) == 2
    res = img.copy()

    if pattern.lower() == 'bggr':  # same pattern
        res[0::2, 0::2] = img[1::2, 1::2] #r
        res[1::2, 1::2] = img[0::2, 0::2] #b
    elif pattern.lower() == 'rggb':
        pass
    elif pattern.lower() == 'grbg':
        res[0::2, 0::2] = img[0::2, 1::2] #r
        res[0::2, 1::2] = img[0::2, 0::2] #g
        res[1::2, 0::2] = img[1::2, 1::2] #g
        res[1::2, 1::2] = img[1::2, 0::2] #b
    elif pattern.lower() == 'gbrg':
        res[0::2, 0::2] = img[1::2, 0::2] #r
        res[0::2, 1::2] = img[0::2, 0::2] #g
        res[1::2, 0::2] = img[1::2, 1::2] #g
        res[1::2, 1::2] = img[0::2, 1::2] #b
    else:
        assert 'no support'

    return res


# %%

# path to  /SIDD_Medium_Raw/Data/ containing all 160 scenes except scenes 0199 and 0200
path_to_train = '/tobit/N2N_SIDD_denoising/raw_training_data/SIDD_Medium_Raw/Data/'
patch_size = 128
scenes_list = glob.glob(path_to_train + "0*")
print(len(scenes_list))
print(scenes_list[0:5])

# this folder can be deleted after the training data is created
tmp_save_path = './raw_train_patches_per_scene/'
if not os.path.exists(tmp_save_path):
    os.makedirs(tmp_save_path)

# %%
# %%
def get_cropped_patches(scene):
    train_set_gt_array = None
    train_set_input_array = None
    train_set_noisy_target_array = None
    
    img_name, extension = os.path.splitext(os.path.basename(scene))
    scene_tag = img_name[0:4]
    print(scene_tag)

    gt_full = h5py.File(scene + "/" + scene_tag + "_GT_RAW_010.MAT", 'r')
    gt_full = gt_full['x'][()]
    input_full = h5py.File(scene + "/" + scene_tag  + "_NOISY_RAW_010.MAT", 'r')
    input_full = input_full['x'][()]
    noisy_target_full = h5py.File(scene + "/" + scene_tag  + "_NOISY_RAW_011.MAT", 'r')
    noisy_target_full = noisy_target_full['x'][()]

    # Bayer pattern transformation. 
    py_meta = extract_metainfo(scene +"/" + scene_tag + "_METADATA_RAW_010.MAT")
    pattern = py_meta['pattern']
    gt_full = transform_to_rggb(gt_full, pattern)
    input_full = transform_to_rggb(input_full, pattern)
    noisy_target_full = transform_to_rggb(noisy_target_full, pattern)

    
    img_shape_x = gt_full.shape[0]
    img_shape_y = gt_full.shape[1]

    possible_x_crop = list(np.arange(0, img_shape_x - patch_size, patch_size))
    possible_y_crop = list(np.arange(0, img_shape_y - patch_size, patch_size))

    for x_crop in possible_x_crop:
        for y_crop in possible_y_crop:
                    
            gt_crop = gt_full[x_crop : x_crop + patch_size, y_crop : y_crop + patch_size]
            input_crop = input_full[x_crop : x_crop + patch_size, y_crop : y_crop + patch_size]
            noisy_target_crop = noisy_target_full[x_crop : x_crop + patch_size, y_crop : y_crop + patch_size]

            if train_set_noisy_target_array is not None:
                train_set_noisy_target_array = np.concatenate((train_set_noisy_target_array, np.array(noisy_target_crop)[np.newaxis,:,:]),axis=0)
            else:
                train_set_noisy_target_array = np.array(noisy_target_crop)[np.newaxis,:,:]

            if train_set_gt_array is not None:
                train_set_gt_array = np.concatenate((train_set_gt_array, np.array(gt_crop)[np.newaxis,:,:]),axis=0)
            else:
                train_set_gt_array = np.array(gt_crop)[np.newaxis,:,:]

            if train_set_input_array is not None:
                train_set_input_array = np.concatenate((train_set_input_array, np.array(input_crop)[np.newaxis,:,:]),axis=0)
            else:
                train_set_input_array = np.array(input_crop)[np.newaxis,:,:]

    num_patches = train_set_gt_array.shape[0]    
    np.save('./{}/SIDD_train_scene_{}_patchSize_{}_patches_{}_gt_array.npy'.format(tmp_save_path,scene_tag,patch_size,num_patches),train_set_gt_array)
    np.save('./{}/SIDD_train_scene_{}_patchSize_{}_patches_{}_input_array.npy'.format(tmp_save_path,scene_tag,patch_size,num_patches),train_set_input_array)
    np.save('./{}/SIDD_train_scene_{}_patchSize_{}_patches_{}_noisy_target_array.npy'.format(tmp_save_path,scene_tag,patch_size,num_patches),train_set_noisy_target_array)


# %%
with Pool(10) as pool:
    _ = pool.map(get_cropped_patches,scenes_list)

# %%
# Add the patches from the different scenes together in one file

gt_files_list = sorted(glob.glob('./{}/*gt_array.npy'.format(tmp_save_path)))
input_files_list = sorted(glob.glob('./{}/*input_array.npy'.format(tmp_save_path)))
noisy_target_files_list = sorted(glob.glob('./{}/*noisy_target_array.npy'.format(tmp_save_path)))

print(len(gt_files_list),len(input_files_list),len(noisy_target_files_list))
print(gt_files_list[0:2],input_files_list[0:2],noisy_target_files_list[0:2])

train_set_gt_array = None
train_set_input_array = None
train_set_noisy_target_array = None
idx=0
for gt_file,input_file,noisy_target_file in zip(gt_files_list,input_files_list,noisy_target_files_list):
    print(idx)
    idx+=1

    img_name, extension = os.path.splitext(os.path.basename(gt_file))
    scene_id_gt = img_name[17:21]
    img_name, extension = os.path.splitext(os.path.basename(input_file))
    scene_id_input = img_name[17:21]
    img_name, extension = os.path.splitext(os.path.basename(noisy_target_file))
    scene_id_noisy_target = img_name[17:21]

    if scene_id_gt != scene_id_input or scene_id_gt != scene_id_noisy_target:
        raise Exception("scene ids do not match")

    gt_array = np.load(gt_file)
    input_array = np.load(input_file)
    noisy_target_array = np.load(noisy_target_file)

    if train_set_noisy_target_array is not None:
        train_set_noisy_target_array = np.concatenate((train_set_noisy_target_array, noisy_target_array),axis=0)
    else:
        train_set_noisy_target_array = noisy_target_array

    if train_set_gt_array is not None:
        train_set_gt_array = np.concatenate((train_set_gt_array, gt_array),axis=0)
    else:
        train_set_gt_array = gt_array

    if train_set_input_array is not None:
        train_set_input_array = np.concatenate((train_set_input_array, input_array),axis=0)
    else:
        train_set_input_array = input_array

num_patches = train_set_gt_array.shape[0]  
 
np.save('SIDD_train_all_scenes_patchSize_{}_patches_{}_gt_array_raw.npy'.format(patch_size,num_patches),train_set_gt_array)
np.save('SIDD_train_all_scenes_patchSize_{}_patches_{}_input_array_raw.npy'.format(patch_size,num_patches),train_set_input_array)
np.save('SIDD_train_all_scenes_patchSize_{}_patches_{}_noisy_target_array_raw.npy'.format(patch_size,num_patches),train_set_noisy_target_array)

# %%
