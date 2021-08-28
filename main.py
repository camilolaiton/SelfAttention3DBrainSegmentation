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
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
# from tifffile import imsave
from sklearn.preprocessing import MinMaxScaler
import time
import nilearn

def helper_anat_structure(msk, data_seg, lut_structure, new_id):
    roi_data = (data_seg==lut_structure['id'])*lut_structure['id']
    return np.where(roi_data == lut_structure['id'], new_id, msk)

def main():
    scaler = MinMaxScaler()
    LUT_PATH = './data/FreeSurferColorLUT.txt'
    lut_file = utils.load_lut(LUT_PATH)
    DATASET_PATH = '/home/camilo/Programacion/master_thesis/dataset_test/'
    class_info = utils.get_classes()
    config_orig = {
        'RAS': True, 
        'normalize': False
    }
    config_msk = {
        'RAS': True, 
        'normalize': False
    }

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

    utils.create_folder('dataset_3D/images')
    utils.create_folder('dataset_3D/masks')

    start_time = time.time()

    for mri_path in mri_paths:

        name = mri_path.split('/')[-1]

        data_img, data = utils.readMRI(mri_path + '/001.mgz', config_orig)
        data_msk_img, data_msk = utils.readMRI(mri_path + '/aparcNMMjt+aseg.mgz', config_msk)
        if data.shape != data_msk.shape:
            print("Correcting shapes in: ", name)
            data_img = nilearn.image.resample_to_img(data_img, data_msk_img)
            data = data_img.get_fdata()
            data = scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)

        msk = np.zeros((256, 256, 256), dtype=np.uint8)
        for structure in STRUCTURES:
            msk = helper_anat_structure(msk, data_msk, lut_file[structure], class_info[structure]['new_id'])

        msk = to_categorical(msk, num_classes=59)
        # Saving normalized mri
        np.save(f'dataset_3D/images/{name}.npy', data)

        # Saving msk
        np.save(f'dataset_3D/masks/{name}.npy', msk)

    seconds = (time.time() - start_time)
    print("Processing time: ", seconds)

    # print(np.unique(msk))

if __name__ == "__main__":
    main()