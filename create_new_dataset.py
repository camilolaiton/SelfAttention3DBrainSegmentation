import glob
from utils import utils
import numpy as np
from patchify import patchify, unpatchify
from tensorflow.keras.utils import to_categorical

def generating_images(patch_size, ori_path, dest_path):
    ij = 0
    for filename in glob.glob(ori_path):
        if (ij == 1):
            print("Exits with ", ij)
            break
        np_file = np.load(filename)
        print(np_file.shape)
        np_file_patches = patchify(np_file, (patch_size, patch_size, patch_size), step=patch_size)
        np_file_patches = np.reshape(np_file_patches, (-1, np_file_patches.shape[3], np_file_patches.shape[4], np_file_patches.shape[5]))
        np_file_patches = np.expand_dims(np_file_patches, axis=4)
        print(np_file_patches.shape)
        name = filename.split('/')[-1] # .split('.')[0]
        name = name.split('\\')[-1].split('.')[0]
        print("Saving imgs patches for ", name)
        np.save(dest_path + f"/{name}-patchified.npy", np_file_patches)
        ij+=1
        # for idx in range(np_file_patches.shape[0]):
        #     np.save(dest_path + f"/{name}-patch-{idx}.npy", np_file_patches[idx, :, :, :, :])

def generating_msks(patch_size, ori_path, dest_path, num_classes):
    i = 0
    for filename in glob.glob(ori_path):
        if (i == 1):
            print("Exits with ", i)
            break
        np_file = np.argmax(np.load(filename), axis=3)
        np_file_patches = patchify(np_file, (patch_size, patch_size, patch_size), step=patch_size)
        np_file_patches = np.reshape(np_file_patches, (-1, np_file_patches.shape[3], np_file_patches.shape[4], np_file_patches.shape[5]))
        np_file_patches = np.expand_dims(np_file_patches, axis=4)
        np_file_patches = to_categorical(np_file_patches, num_classes=num_classes)
        name = filename.split('/')[-1] # .split('.')[0]
        name = name.split('\\')[-1].split('.')[0]
        print("Saving masks patches for ", name)
        np.save(dest_path + f"/{name}-patchified.npy", np_file_patches)
        i+=1
        # for idx in range(np_file_patches.shape[0]):
        #     np.save(dest_path + f"/{name}-patch-{idx}.npy", np_file_patches[idx, :, :, :, :])

def main():
    patch_size = 64
    num_classes = 4

    dataset_name_folder = 'dataset_3D_3'

    train_dir_imgs = 'dataset_3D/train/images/*'
    train_dir_msks = 'dataset_3D/train/masks/*'
    
    test_dir_imgs = 'dataset_3D/test/images/*'
    test_dir_msks = 'dataset_3D/test/masks/*'

    dest_train_dir_imgs = dataset_name_folder + '/train/images'
    dest_train_dir_msks = dataset_name_folder + '/train/masks'

    dest_test_dir_imgs = dataset_name_folder + '/test/images'
    dest_test_dir_msks = dataset_name_folder + '/test/masks'

    for folder in [
        dest_train_dir_imgs, 
        dest_train_dir_msks, 
        dest_test_dir_imgs,
        dest_test_dir_msks,
    ]:
        utils.create_folder(folder)

    # Generating images and masks for training
    print("\n[+] Generating patched images for training")
    generating_images(patch_size, train_dir_imgs, dest_train_dir_imgs)
    print("\n[+] Generating patched masks for training")
    generating_msks(patch_size, train_dir_msks, dest_train_dir_msks, num_classes)


    # Generating images and masks for test
    print("\n[+] Generating patched images for test")
    generating_images(patch_size, test_dir_imgs, dest_test_dir_imgs)
    print("\n[+] Generating patched masks for test")
    generating_msks(patch_size, test_dir_msks, dest_test_dir_msks, num_classes)

if __name__ == '__main__':
    main()