import glob
from utils import utils
import numpy as np
from patchify import patchify, unpatchify
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import nilearn
import nibabel as nib

def generating_images(patch_size, ori_path, dest_path):
    for filename in glob.glob(ori_path):
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
        # for idx in range(np_file_patches.shape[0]):
        #     np.save(dest_path + f"/{name}-patch-{idx}.npy", np_file_patches[idx, :, :, :, :])

def generating_msks(patch_size, ori_path, dest_path, num_classes):
    for filename in glob.glob(ori_path):
        np_file = np.argmax(np.load(filename), axis=3)
        np_file_patches = patchify(np_file, (patch_size, patch_size, patch_size), step=patch_size)
        np_file_patches = np.reshape(np_file_patches, (-1, np_file_patches.shape[3], np_file_patches.shape[4], np_file_patches.shape[5]))
        np_file_patches = np.expand_dims(np_file_patches, axis=4)
        np_file_patches = to_categorical(np_file_patches, num_classes=num_classes)
        name = filename.split('/')[-1] # .split('.')[0]
        name = name.split('\\')[-1].split('.')[0]
        print("Saving masks patches for ", name)
        np.save(dest_path + f"/{name}-patchified.npy", np_file_patches)
        # for idx in range(np_file_patches.shape[0]):
        #     np.save(dest_path + f"/{name}-patch-{idx}.npy", np_file_patches[idx, :, :, :, :])

def helper_anat_structure(msk, data_seg, lut_structure, new_id):
    roi_data = (data_seg==lut_structure['id'])*lut_structure['id']
    return np.where(roi_data == lut_structure['id'], new_id, msk)

def main():
    config_orig = {
        'RAS': True, 
        'normalize': False
    }
    config_msk = {
        'RAS': True, 
        'normalize': False
    }
    LUT_PATH = './data/FreeSurferColorLUT.txt'
    lut_file = utils.load_lut(LUT_PATH)
    scaler = MinMaxScaler()
    # class_info = utils.get_classes_same_id()
    class_info = utils.get_classes_different_id()
    STRUCTURES = utils.read_test_to_list('data/common_anatomical_structures.txt')
    mri_paths = utils.read_test_to_list('data/common_mri_images.txt')
    patch_size = 64
    # num_classes = 4
    num_classes = len(STRUCTURES) + 1

    dataset_name_folder = 'dataset_3D_p64'

    for idx in range(len(mri_paths)):
        tmp = mri_paths[idx].split('-')
        mri_paths[idx] = f"data/{'-'.join(tmp[:-1])}/{mri_paths[idx]}"

    # print(mri_paths[0], len(mri_paths))

    for folder in ['train', 'test']:
        utils.create_folder(f"{dataset_name_folder}/{folder}/images")
        utils.create_folder(f"{dataset_name_folder}/{folder}/masks")

    test_mri = [
        'HLN-12-6',
        'HLN-12-12',
        'MMRR-21-1',
        'MMRR-21-5',
        'MMRR-21-10',
        'MMRR-21-15',
        'MMRR-21-20',
        'NKI-RS-22-1',
        'NKI-RS-22-5',
        'NKI-RS-22-10',
        'NKI-RS-22-15',
        'NKI-RS-22-20',
        'NKI-TRT-20-1',
        'NKI-TRT-20-10',
        'NKI-TRT-20-20',
        'OASIS-TRT-20-5',
        'OASIS-TRT-20-10',
        'OASIS-TRT-20-15',
        'OASIS-TRT-20-20',
    ]

    # mri_paths = [
    #     # mri_paths[mri_paths.index('data/HLN-12/HLN-12-1')],
    #     mri_paths[mri_paths.index('data/MMRR-21/MMRR-21-20')]
    # ]
    
    for mri_path in mri_paths:

        name = mri_path.split('/')[-1]
        data_img, data = utils.readMRI(mri_path + '/001.mgz', config_orig)
        data_msk_img, data_msk = utils.readMRI(mri_path + '/aparcNMMjt+aseg.mgz', config_msk)
        # print("after read: ", nib.aff2axcodes(data_img.affine))
        if data.shape != data_msk.shape:
            print("Fixing shapes in: ", name)
            data_img = nilearn.image.resample_to_img(data_img, data_msk_img)
            data = data_img.get_fdata()
            data = scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)

        msk = np.zeros((256, 256, 256), dtype=np.uint8)
        for structure in STRUCTURES:
            # print("Structure ", structure, ": ", class_info[structure])
            msk = helper_anat_structure(msk, data_msk, lut_file[structure], class_info[structure]['new_id'])
            # print(np.unique(msk))
        data = data[45:237, 38:230, 30:222]
        msk = msk[45:237, 38:230, 30:222]
        # print(data.shape)
        # Saving normalized mri
        end_folder = 'train'

        if name in test_mri:
            end_folder = 'test'

        # Patches for images
        data = patchify(data, (patch_size, patch_size, patch_size), step=patch_size)
        data = np.reshape(data, (-1, data.shape[3], data.shape[4], data.shape[5]))
        data = np.expand_dims(data, axis=4)
        data = data.astype(np.float16)
        # print("Data shape: ", data.shape)
        # Patches for test
        msk = patchify(msk, (patch_size, patch_size, patch_size), step=patch_size)
        msk = np.reshape(msk, (-1, msk.shape[3], msk.shape[4], msk.shape[5]))
        msk = np.expand_dims(msk, axis=4)
        # print(np.unique(msk), " ", msk.shape)
        msk = to_categorical(msk, num_classes=num_classes)
        # print(np.unique(msk), " ", msk.shape)
        msk = msk.astype(np.uint8)
        # print("msk shape: ", msk.shape)
        np.save(f'{dataset_name_folder}/{end_folder}/images/{name}_patched.npy', data)

        # Saving msk
        np.save(f'{dataset_name_folder}/{end_folder}/masks/{name}_patched.npy', msk)
        exit()

    # train_dir_imgs = 'dataset_3D/train/images/*'
    # train_dir_msks = 'dataset_3D/train/masks/*'
    
    # test_dir_imgs = 'dataset_3D/test/images/*'
    # test_dir_msks = 'dataset_3D/test/masks/*'

    # dest_train_dir_imgs = dataset_name_folder + '/train/images'
    # dest_train_dir_msks = dataset_name_folder + '/train/masks'

    # dest_test_dir_imgs = dataset_name_folder + '/test/images'
    # dest_test_dir_msks = dataset_name_folder + '/test/masks'

    # for folder in [
    #     dest_train_dir_imgs, 
    #     dest_train_dir_msks, 
    #     dest_test_dir_imgs,
    #     dest_test_dir_msks,
    # ]:
    #     utils.create_folder(folder)

    # # Generating images and masks for training
    # print("\n[+] Generating patched images for training")
    # generating_images(patch_size, train_dir_imgs, dest_train_dir_imgs)
    # print("\n[+] Generating patched masks for training")
    # generating_msks(patch_size, train_dir_msks, dest_train_dir_msks, num_classes)


    # # Generating images and masks for test
    # print("\n[+] Generating patched images for test")
    # generating_images(patch_size, test_dir_imgs, dest_test_dir_imgs)
    # print("\n[+] Generating patched masks for test")
    # generating_msks(patch_size, test_dir_msks, dest_test_dir_msks, num_classes)

if __name__ == '__main__':
    main()
