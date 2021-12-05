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
from model.model import *
from model.model_2 import *
from model.config import *
import segmentation_models as sm
sm.set_framework('tf.keras')
# from tifffile import imsave
from sklearn.preprocessing import MinMaxScaler
# import time
import random
import os
import nilearn
import tensorflow as tf
from model.config import *
from volumentations import *
from model.blocks import Keras3DAugmentation
import nibabel as nib
from patchify import patchify, unpatchify
from models_comparative.unet_3D import build_unet3D_model, build_unet3D_model_2

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
    msk = np.load(msk_path).astype(np.float32)
    return img, msk

def load_files(img_path, msk_path):
    return tf.numpy_function(
        load_files_py,
        inp=[img_path, msk_path],
        Tout=[tf.float32, tf.float32]
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
    DATASET_PATH = 'dataset_3D/'
    DATASET_PATH_MASKS = 'dataset_3D_p64/train/masks'
    class_info = utils.get_classes_same_id()
    config_orig = {
        'RAS': True, 
        'normalize': False
    }
    config_msk = {
        'RAS': True, 
        'normalize': False
    }

    # config = get_config_patchified()
    # model = build_model_patchified_patchsize8(config)
    # image_files = [os.path.join(DATASET_PATH_MASKS, file) for file in os.listdir(DATASET_PATH_MASKS) if file.endswith('.npy')]
    # # print("Executing median frequency balancing in with ", len(image_files), " files")
    # results_list, label_to_frequency_dict = utils.median_frequency_balancing(image_files, num_classes=4)
    # print(results_list)
    # print(label_to_frequency_dict)
    # exit()
    # print("List: ", results_list)
    
    config = get_config_local_path()
    model = build_unet3D_model(config)# build_model(config) # build_unet3D_model(config)

    # sample_weight = np.ones(4)
    # sample_weight[0] = 2.0

    # print(sample_weight)

    optimizer = tf.optimizers.SGD(
        learning_rate=config.learning_rate, 
        momentum=config.momentum,
        name='optimizer_SGD_0'
    )

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",#loss,#tversky_loss,
        metrics=[
            # 'accuracy',
            sm.metrics.IOUScore(threshold=0.5),
            sm.metrics.FScore(threshold=0.5),
        ],
    )
    print(f"[+] Building model with config {config}")
    model.summary()
    tf.keras.utils.plot_model(
        model,
        to_file="test_model.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=96,
    )
    exit()
    
    # image_list_train = sorted(glob.glob(
    #     config.dataset_path + 'train/images/*'))
    # mask_list_train = sorted(glob.glob(
    #     config.dataset_path + 'train/masks/*'))
    # print(config.dataset_path, " ", len(image_list_train), " ", len(mask_list_train))
    # image_list_test = sorted(glob.glob(
    #     config.dataset_path + 'test/images/*'))
    # mask_list_test = sorted(glob.glob(
    #     config.dataset_path + 'test/masks/*'))

    # # train_imgs = read_files_from_directory(image_list_train)
    # # train_msks = read_files_from_directory(mask_list_train)

    # # print(train_imgs.shape, "  ", train_msks.shape)

    # train_dataset = tf.data.Dataset.from_tensor_slices(
    #     # (train_imgs, train_msks)
    #     (image_list_train, 
    #     mask_list_train)
    # )

    # dataset = {
    #     "train" : train_dataset,
    #     # "val" : val_dataset
    # }

    # AUTOTUNE = tf.data.experimental.AUTOTUNE
    # BATCH_SIZE = 1

    # options = tf.data.Options()
    # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    # aug_layer = Keras3DAugmentation(
    #     12, 
    #     config.image_width, 
    #     config.image_height, 
    #     config.image_channels, 
    #     name='data_aug'
    # )

    # dataset['train'] = dataset['train'].map(load_files)
    # if (config.unbatch):
    #     dataset['train'] = dataset['train'].unbatch()
    # # dataset['train'] = dataset['train'].map(augmentor, num_parallel_calls=AUTOTUNE)
    # dataset['train'] = dataset['train'].repeat()
    # dataset['train'] = dataset['train'].batch(BATCH_SIZE)
    # dataset['train'] = dataset['train'].prefetch(AUTOTUNE)
    # dataset['train'] = dataset['train'].with_options(options)
    # dataset['train'] = dataset['train'].map(lambda x, y: (aug_layer(x),y), num_parallel_calls=AUTOTUNE)
    # x, y = next(iter(dataset['train']))
    
    # print(x.shape, " ", y.shape, " ", np.unique(np.argmax(y, axis=4)))

    # # plt.figure(figsize=(30, 20))
    
    # # for j in range(config.image_height):
    # # plt.subplot(8, 8, j + 1)
    # watch = x[0, 32, :, :, 0]
    # orig = np.load('dataset_3D/train/images/HLN-12-1.npy')
    # print(watch.shape)
    # plt.imshow(watch, cmap='gray')
    # plt.axis("off")
    # plt.title('test')
    # plt.show()

    # print(orig[:128,128,:128].shape)
    # plt.imshow(orig[:128,128,:128], cmap='gray')
    # plt.axis("off")
    # plt.title('test')
    # plt.show()

    # exit()
    # dataset['val'] = dataset['val'].from_generator(val_datagen)
    # dataset['val'] = dataset['val'].repeat()
    # dataset['val'] = dataset['val'].batch(BATCH_SIZE)
    # dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)

    # print(dataset['train'])
    # print(dataset['val'])
    # test_tf_function()


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

    # # start_time = time.time()
    # # MMRR-21-3
    """
    STRUCTURES = utils.read_test_to_list('data/common_anatomical_structures.txt')
    mri_paths = utils.read_test_to_list('data/common_mri_images.txt')
    
    for idx in range(len(mri_paths)):
        tmp = mri_paths[idx].split('-')
        mri_paths[idx] = f"data/{'-'.join(tmp[:-1])}/{mri_paths[idx]}"

    print(mri_paths[0], len(mri_paths))

    for folder in ['train', 'test']:
        utils.create_folder(f"dataset_3D/{folder}/images")
        utils.create_folder(f"dataset_3D/{folder}/masks")

    mri_paths = mri_paths[:2] + [mri_paths[mri_paths.index('data/HLN-12/HLN-12-1')]]
    # print(mri_paths)
    for mri_path in mri_paths:

        name = mri_path.split('/')[-1]
        
        data_img, data = utils.readMRI(mri_path + '/001.mgz', config_orig)
        data_msk_img, data_msk = utils.readMRI(mri_path + '/aparcNMMjt+aseg.mgz', config_msk)
        print("after read: ", nib.aff2axcodes(data_img.affine))
        if data.shape != data_msk.shape:
            print("Fixing shapes in: ", name)
            data_img = nilearn.image.resample_to_img(data_img, data_msk_img)
            data = data_img.get_fdata()
            data = scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)

        msk = np.zeros((256, 256, 256), dtype=np.uint8)
        for structure in STRUCTURES:
            msk = helper_anat_structure(msk, data_msk, lut_file[structure], class_info[structure]['new_id'])

        msk = msk[45:215, 40:210, 30:200]
        msk = to_categorical(msk, num_classes=4)
        # Saving normalized mri
        np.save(f'dataset_3D/test/images/{name}.npy', data[45:215, 40:210, 30:200])

        # Saving msk
        np.save(f'dataset_3D/test/masks/{name}.npy', msk)
    # exit()
    """
    def plot_np_file(np_file, np_msk, view, name):
        shapes = np_file.shape        
        
        for xj in range(0, 64, 16):
            plt.figure(figsize=(30, 20))
            plt.suptitle(f"{name} - {view}")
            for w in [0, 1]:
                for x in range(0, 16):
                    # print(start, " ", x, " ", xj-start)
                    plt.subplot(4, 4, x + 1)
                    if (view == 'saggital' ):
                        plt.imshow(np_file[w, x+xj, :, :, :], cmap='gray')
                        plt.imshow(np_msk[w, x+xj, :, :], alpha=0.5)
                    elif (view == 'coronal' ):
                        plt.imshow(np_file[w, :, x+xj, :, :], cmap='gray')
                        plt.imshow(np_msk[w, :, x+xj, :], alpha=0.5)
                    else:
                        plt.imshow(np_file[w, :, :, x+xj, :], cmap='gray')
                        plt.imshow(np_msk[w, :, :, x+xj], alpha=0.5)

                    plt.axis("off")
                    plt.title(x+xj)
            plt.show()

    palette = np.array([[  0,   0,   0],   # black
                [255,   0,   0],   # red
                [  0, 255,   0],   # green
                [  0,   0, 255]])   # blue
                # [255, 255, 255]])  # white    
    
    patch_size = 64
    for path in glob.glob('dataset_3D_p64/test/images/*'):
        file = np.load(path)
        print(file.shape)
        # np_file_patches = patchify(file, (patch_size, patch_size, patch_size), step=patch_size)
        # np_file_patches = np.reshape(np_file_patches, (-1, np_file_patches.shape[3], np_file_patches.shape[4], np_file_patches.shape[5]))
        # print(np_file_patches.shape)

        file_msk = np.argmax(np.load(path.replace('images', 'masks')), axis=4)
        # print("Before msk: ", np.unique(file_msk))
        # np_file_msk_patches = patchify(file_msk, (patch_size, patch_size, patch_size), step=patch_size)
        # np_file_msk_patches = np.reshape(np_file_msk_patches, (-1, np_file_msk_patches.shape[3], np_file_msk_patches.shape[4], np_file_msk_patches.shape[5]))
        # print("After msk: ", np.unique(np_file_msk_patches))
        # np_file_msk_patches = palette[np_file_msk_patches]
        print(np.unique(file_msk))
        file_msk = palette[file_msk]
        # plt.imshow(file[2, :, :, 35, :], cmap='gray')
        # plt.imshow(file_msk[2, :, :, 35], alpha=0.5)
        plt.axis("off")
        plt.show()
        name = path.split('/')[-1]
        
    #     # Adding colors
    #     file_msk = palette[file_msk]

        for view in ['saggital', 'coronal', 'axial']:
            plot_np_file(file, file_msk, view, name)
    exit()
    # # seconds = (time.time() - start_time)
    # # print("Processing time: ", seconds)

    # test_image = np.load('dataset_3D/images/HLN-12-1.npy')
    # test_msk = np.load('dataset_3D/masks/HLN-12-1.npy')

    # print(np.unique(test_image), " ", test_image.shape)
    # print(np.unique(test_msk), " ", test_msk.shape)

    # print(np.unique(msk))

if __name__ == "__main__":
    main()