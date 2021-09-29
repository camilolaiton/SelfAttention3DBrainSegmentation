"""
    Made by:
        - Camilo Laiton
        Universidad Nacional de Colombia, Colombia
        2021-1
        GitHub: https://github.com/camilolaiton/

        This file belongs to the private repository "master_thesis" where
        I save all the files that are related to my thesis which is called
        "Método para la segmentación de imágenes de resonancia magnética 
        cerebrales usando una arquitectura de red neuronal basada en modelos
        de atención".
"""

from utils import utils
# import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
# from tifffile import imsave
from sklearn.preprocessing import MinMaxScaler
# import time
import random
import os
import nilearn
import tensorflow as tf
from model.config import *
from volumentations import *

def structure_validation():

    train_img_dir = "dataset_3D/images/"
    train_mask_dir = "dataset_3D/masks/"
    train_img_list=os.listdir(train_img_dir)
    train_mask_list = os.listdir(train_mask_dir)

    batch_size = 2

    train_img_datagen = utils.mri_generator(train_img_dir, train_img_list, 
                                    train_mask_dir, train_mask_list, batch_size)

    #Verify generator.... In python 3 next() is renamed as __next__()
    img, msk = train_img_datagen.__next__()


    img_num = random.randint(0,img.shape[0]-1)
    print("img num: ", img_num)
    test_img=img[img_num]
    test_mask=msk[img_num]
    test_mask=np.argmax(test_mask, axis=3)

    n_slice=95#random.randint(0, test_mask.shape[2])
    print("n_silce: ", n_slice)
    plt.figure(figsize=(12, 6))

    plt.subplot(221)
    plt.imshow(test_img[:,:,n_slice], cmap='gray')
    plt.title('Image')
    plt.subplot(222)
    plt.imshow(test_mask[:,:,n_slice])
    plt.title('Mask')
    plt.show()

def helper_anat_structure(msk, data_seg, lut_structure, new_id):
    roi_data = (data_seg==lut_structure['id'])*lut_structure['id']
    return np.where(roi_data == lut_structure['id'], new_id, msk)

def test_tf_function():
    """
        Function to check the patches inside the MRI
    """
    n = 10
    # images is a 1 x 10 x 10 x 1 array that contains the numbers 1 through 100
    images = [
        [
            [
                [x * n + y + 1] for y in range(n)
            ] for x in range(n)
        ] 
    ]
    
    n = 32
    volume = [
        [
            [
                [
                    [x * n + y + 1] for y in range(n)
                ] for x in range(n)
            ] for w in range(n)
        ]
    ]

    print("images: ", np.array(images).shape)

    print("volume: ", np.array(volume).shape)

    # We generate two outputs as follows:
    # 1. 3x3 patches with stride length 5
    # 2. Same as above, but the rate is increased to 2
    
    # patch_size = 5
    # resp = tf.image.extract_patches(images=images,
    #                         sizes=[1, patch_size, patch_size, 1],
    #                         strides=[1, patch_size, patch_size, 1],
    #                         rates=[1, 1, 1, 1],
    #                         padding='VALID')

    # print("resp: \n", resp)
    # print("R1: ", resp[0][1][1])
    # patch_dims = resp.shape[-1]
    # resp = tf.reshape(resp, [1, -1, patch_dims])
    # print("resp: \n", resp)


    patch_size = 8
    patches = tf.extract_volume_patches(
        input=volume,
        ksizes=[1, patch_size, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, patch_size, 1],
        padding='VALID',
    )
    print("resp: \n", patches)
    patch_dims = patches.shape[-1]
    print("patchs dims: ", patch_dims)
    patches = tf.reshape(patches, [1, -1, patch_dims])
    print("resp: \n", patches)

import glob

def load_files_py(img_path, msk_path):
    img = np.load(img_path).astype(np.float32)
    msk = np.load(msk_path).astype(np.uint8)
    return img, msk

def load_files(img_path, msk_path):
    return tf.numpy_function(
        load_files_py,
        inp=[img_path, msk_path],
        Tout=[tf.float32, tf.uint8]
    )

def get_augmentation():
    return Compose([
        # Rotate((-15, 15), (0, 0), (0, 0), p=0.5),
        # RandomCropFromBorders(crop_value=0.1, p=0.5),
        ElasticTransform((0, 0.25), interpolation=2, p=0.1),
        # Resize(patch_size, interpolation=1, always_apply=True, p=1.0),
        # Flip(0, p=0.5),
        # Flip(1, p=0.5),
        # Flip(2, p=0.5),
        # RandomRotate90((1, 2), p=0.5),
        # GaussianNoise(var_limit=(0, 5), p=0.2),
        # RandomGamma(gamma_limit=(0.5, 1.5), p=0.2),
    ], p=1.0)

def augmentor_py(img, msk):
    aug = get_augmentation()
    data = {'image': img, 'msk': msk}
    aug_data = aug(**data)
    img = aug_data['image']
    msk = aug_data['msk']
    return np.ndarray.astype(img, np.float32), np.ndarray.astype(msk, np.float32)

def augmentor(img, msk):
    return tf.numpy_function(
        augmentor_py,
        inp=[img, msk],
        Tout=[tf.float32, tf.float32]
    )

def read_files_from_directory(files_path):
    files = []
    for file_path in files_path:
        files.append(np.load(file_path))
    return np.array(files)

def main():
    scaler = MinMaxScaler()
    LUT_PATH = './data/FreeSurferColorLUT.txt'
    lut_file = utils.load_lut(LUT_PATH)
    DATASET_PATH = '/home/camilo/Programacion/master_thesis/dataset_test/'
    class_info = utils.get_classes_same_id()
    config_orig = {
        'RAS': True, 
        'normalize': False
    }
    config_msk = {
        'RAS': True, 
        'normalize': False
    }

    config = get_config_patchified()

    image_list_train = sorted(glob.glob(
        config.dataset_path + 'train/images/*'))
    mask_list_train = sorted(glob.glob(
        config.dataset_path + 'train/masks/*'))
    print(len(image_list_train), " ", len(mask_list_train))
    image_list_test = sorted(glob.glob(
        config.dataset_path + 'test/images/*'))
    mask_list_test = sorted(glob.glob(
        config.dataset_path + 'test/masks/*'))

    train_imgs = read_files_from_directory(image_list_train)
    train_msks = read_files_from_directory(mask_list_train)

    print(train_imgs.shape, "  ", train_msks.shape)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_imgs, train_msks)
        #(image_list_train, 
        #mask_list_train)
    )

    dataset = {
        "train" : train_dataset,
        # "val" : val_dataset
    }

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    BATCH_SIZE = 1

    dataset['train'] = dataset['train'].map(load_files).map(augmentor, num_parallel_calls=AUTOTUNE) #.
    dataset['train'] = dataset['train'].repeat()
    dataset['train'] = dataset['train'].batch(BATCH_SIZE)
    dataset['train'] = dataset['train'].prefetch(AUTOTUNE)
    see = next(iter(dataset['train']))
    print("SEE: ", see, " ", len(see))
    # dataset['val'] = dataset['val'].from_generator(val_datagen)
    # dataset['val'] = dataset['val'].repeat()
    # dataset['val'] = dataset['val'].batch(BATCH_SIZE)
    # dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)

    # print(dataset['train'])
    # print(dataset['val'])
    # test_tf_function()
    exit()
    STRUCTURES = utils.read_test_to_list('data/common_anatomical_structures.txt')
    mri_paths = utils.read_test_to_list('data/common_mri_images.txt')
    
    for idx in range(len(mri_paths)):
        tmp = mri_paths[idx].split('-')
        mri_paths[idx] = f"data/{'-'.join(tmp[:-1])}/{mri_paths[idx]}"

    # data_img, data = utils.readMRI(mri_paths[0] + '/001.mgz', config_orig)
    # data_msk_img, data_msk = utils.readMRI(mri_paths[0] + '/aparcNMMjt+aseg.mgz', config_msk)

    # print(data.shape, "  ", data_msk.shape)
    # print(np.unique(data))
    # if data.shape != data_msk.shape:
    #     print("NO ", mri_paths[0])
    #     data_img = nilearn.image.resample_to_img(data_img, data_msk_img)
    #     data = data_img.get_fdata()
    #     print(data_img.shape, "  ", np.unique(data))

    #     # utils.plotting_superposition(161, data, roi_data, 'coronal')
    #     data = scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
    #     print(np.unique(data))
    #     # print(data_img.shape, "  ", np.unique(data))

    # utils.create_folder('dataset_3D/images')
    # utils.create_folder('dataset_3D/masks')

    # # start_time = time.time()
    # # MMRR-21-3

    # mri_paths = mri_paths[mri_paths.index('data/HLN-12/HLN-12-6'):mri_paths.index('data/HLN-12/HLN-12-6')+1]
    # print(mri_paths)
    # for mri_path in mri_paths:

    #     name = mri_path.split('/')[-1]
        
    #     data_img, data = utils.readMRI(mri_path + '/001.mgz', config_orig)
    #     data_msk_img, data_msk = utils.readMRI(mri_path + '/aparcNMMjt+aseg.mgz', config_msk)
    #     if data.shape != data_msk.shape:
    #         print("Fixing shapes in: ", name)
    #         data_img = nilearn.image.resample_to_img(data_img, data_msk_img)
    #         data = data_img.get_fdata()
    #         data = scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)

    #     msk = np.zeros((256, 256, 256), dtype=np.uint8)
    #     for structure in STRUCTURES:
    #         msk = helper_anat_structure(msk, data_msk, lut_file[structure], class_info[structure]['new_id'])

    #     msk = to_categorical(msk, num_classes=4)
    #     # Saving normalized mri
    #     np.save(f'dataset_3D/test/images/{name}.npy', data)

    #     # Saving msk
    #     np.save(f'dataset_3D/test/masks/{name}.npy', msk)

    # # seconds = (time.time() - start_time)
    # # print("Processing time: ", seconds)

    # test_image = np.load('dataset_3D/images/HLN-12-1.npy')
    # test_msk = np.load('dataset_3D/masks/HLN-12-1.npy')

    # print(np.unique(test_image), " ", test_image.shape)
    # print(np.unique(test_msk), " ", test_msk.shape)

    # print(np.unique(msk))

if __name__ == "__main__":
    main()