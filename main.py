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

def helper_anat_structure(msk, data_seg, lut_structure, new_id):
    roi_data = (data_seg==lut_structure['id'])*lut_structure['id']
    return np.where(roi_data == lut_structure['id'], new_id, msk)

def main():
    LUT_PATH = './data/FreeSurferColorLUT.txt'
    lut_file = utils.load_lut(LUT_PATH)
    DATASET_PATH = '/home/camilo/Programacion/master_thesis/dataset_test/'
    class_info = utils.get_classes()
    config_orig = {
        'RAS': True, 
        'normalize': True
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

    _, data = utils.readMRI(mri_paths[0] + '/001.mgz', config_orig)
    _, data_msk = utils.readMRI(mri_paths[0] + '/aparcNMMjt+aseg.mgz', config_msk)

    start_time = time.time()

    msk = np.zeros((256, 256, 256), dtype=np.uint8)
    for structure in STRUCTURES:
        msk = helper_anat_structure(msk, data_msk, lut_file[structure], class_info[structure]['new_id'])

    seconds = (time.time() - start_time)
    print("Processing time: ", seconds)

    print(np.unique(msk))

if __name__ == "__main__":
    main()