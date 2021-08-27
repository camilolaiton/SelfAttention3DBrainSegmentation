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
import nilearn
import time
import multiprocessing

# find . -type d -name segSlices -exec rm -r {} \;     -> To remove specific folders

def helper_anat(msk, orig_msk, lut_file, structure):
    
    k = 0
    msk[:, :, k] = np.where(orig_msk == lut_file[structure]['id'], lut_file[structure]['rgba'][k], msk[:, :, k])

    k = 1
    msk[:, :, k] = np.where(orig_msk == lut_file[structure]['id'], lut_file[structure]['rgba'][k], msk[:, :, k])

    k = 2
    msk[:, :, k] = np.where(orig_msk == lut_file[structure]['id'], lut_file[structure]['rgba'][k], msk[:, :, k])
    
    return msk

def helper_anat_integer(msk, orig_msk, lut_file_structure):
    msk[:, :, 0] = np.where(orig_msk == lut_file_structure['id'], lut_file_structure['id'], msk[:, :, 0])
    return msk

def show_anat_slide(ns, lut_file, canonical_data, STRUCTURES, dest, filename, view):
  
    for n in ns:
        # start_time = time.time()
        # msk = np.zeros((256, 256, 3), dtype=np.uint8)
        msk = np.zeros((256, 256, 1), dtype=np.uint8)

        for structure in STRUCTURES:
            roi_data = (canonical_data==lut_file[structure]['id'])*lut_file[structure]['id']
            rot = None
            
            if (view == 'saggital'):
                rot = roi_data[n, :, :]

            elif (view == 'coronal'):
                rot = roi_data[:, n, :]

            else:
                rot = roi_data[:, :, n]

            # msk = helper_anat(msk, rot, lut_file, structure)
            msk = helper_anat_integer(msk, rot, lut_file[structure])

        if (view == 'saggital'):
            utils.saveSlice(np.rot90(msk), f"{filename}_{255 - n}", dest)
        elif (view == 'coronal'):
            utils.saveSlice(np.fliplr(np.rot90(msk)), f"{filename}_{n}", dest)
        else:
            utils.saveSlice(np.fliplr(np.rot90(msk)), f"{filename}_{255-n}", dest)

def main():
    LUT_PATH = './data/FreeSurferColorLUT.txt'
    brainPath = './data/NKI-TRT-20/NKI-TRT-20-1/001.mgz'#'./data/sub01/001.mgz' #brain.mgz'
    image_path = "./data/NKI-TRT-20/NKI-TRT-20-1/aparcNMMjt+aseg.mgz"#"./data/sub01/aparcNMMjt+aseg.mgz"
    DATASET_PATH = '/home/camilo/Programacion/master_thesis/dataset_test/' 
    PREFIX_PATH = '/home/camilo/Programacion/master_thesis/data/'

    scaler = MinMaxScaler()

    mri_slides = {
        'HLN-12-1': {
            'saggital:': [59, 202],
            'coronal': [46, 204],
            'axial': [69, 225]
        },
        'HLN-12-2': {
            'saggital:': [69, 204],
            'coronal': [34, 209],
            'axial': [48, 221]
        },
        'HLN-12-10': {
            'saggital:': [59, 196],
            'coronal': [40, 200],
            'axial': [51, 187]
        },
    }

    roots = [
        'HLN-12',
        # 'Colin27',
        'MMRR-3T7T-2',
        'NKI-RS-22',
        'NKI-TRT-20',
        'MMRR-21',
        'OASIS-TRT-20',
        'Twins-2',
        'Afterthought'
    ]


    roots = [PREFIX_PATH + root for root in roots]
    lut_file = utils.load_lut(LUT_PATH)

    config = {
        'RAS': True, 
        'normalize': True
    }

    # image_obj = nib.load(image_path)

    canonical_img, canonical_data = utils.readMRI(imagePath=image_path, config=config)
    canonical_nifti = nib.Nifti1Image(canonical_data, affine=canonical_img.affine)

    brain_img, brain_data = utils.readMRI(imagePath=brainPath, config=config)
    brain_nifti = nib.Nifti1Image(brain_data, affine=brain_img.affine)

    # STRUCTURES = utils.read_test_to_list('data/common_anatomical_structures.txt')
    # mri_paths = utils.read_test_to_list('data/common_mri_images.txt')

    # for idx in range(len(mri_paths)):
    #     tmp = mri_paths[idx].split('-')
    #     mri_paths[idx] = f"data/{'-'.join(tmp[:-1])}/{mri_paths[idx]}"

    # print(mri_paths[0])

    # utils.create_folder('dataset_test_3/segmented/saggital')
    # utils.create_folder('dataset_test_3/segmented/coronal')
    # utils.create_folder('dataset_test_3/segmented/axial')

    # slides_n = [x for x in range(256)]

    # show_anat_slide(slides_n, lut_file, canonical_data, 
    #     STRUCTURES,
    #     f"dataset_test_3/segmented/axial",
    #     'MMRR-21-21',
    #     'axial'   
    # )


    # utils.create_folder('dataset_test_2/segmented/saggital')
    # utils.create_folder('dataset_test_2/segmented/coronal')
    # utils.create_folder('dataset_test_2/segmented/axial')

    # for mri_path in mri_paths:
    #     canonical_img, canonical_data = utils.readMRI(imagePath=mri_path + '/aparcNMMjt+aseg.mgz', config=config)
    #     mri_name = mri_path.split('/')[-1]
    #     print("Saving axial, saggital and coronal images for ", mri_name)

    #     for ori in ['axial', 'saggital', 'coronal']:
    #         show_anat_slide(slides_n, lut_file, canonical_data, 
    #             [
    #                 'left-cerebral-white-matter',
    #                 'right-cerebral-white-matter',
    #                 'left-cerebellum-white-matter',
    #                 'right-cerebellum-white-matter'
    #             ],
    #             f"dataset_test_2/segmented/{ori}",
    #             mri_name,
    #             ori   
    #         )

    # utils.show_all_slices_per_view('coronal', brain_data, counter=70)
    # utils.plot_roi_modified(lut_file, 'right-cerebral-white-matter', brain_nifti, canonical_data, canonical_img)
    
    # show_anat_slide([54,84,120], lut_file, canonical_data, canonical_img, brain_nifti)
    
    # show_anat_slide([54,84,120], lut_file, canonical_data, STRUCTURES, './')

    # start_time = time.time()

    # shared_data = multiprocessing.Manager().dict()
    # p1 = multiprocessing.Process(target=show_anat_slide, args=([54,84,120], lut_file, canonical_data, canonical_img, brain_nifti))
    # p2 = multiprocessing.Process(target=show_anat_slide, args=([55,89,140], lut_file, canonical_data, canonical_img, brain_nifti))
    # p3 = multiprocessing.Process(target=show_anat_slide, args=([31,70,142], lut_file, canonical_data, canonical_img, brain_nifti))

    # p1.start()
    # p2.start()
    # p3.start()
    # r1 = p1.join()
    # r2 = p2.join()
    # r3 = p3.join()
    
    # seconds = (time.time() - start_time)
    # print("Processing time: ", seconds)
    # print(shared_data.values())

    
    # print(r1)
    # plt.imshow(r1)
    # plt.show()

    # start_time = time.time()

    # r1 = show_anat_slide(54, lut_file, canonical_data, canonical_img, brain_nifti)
    # r2 = show_anat_slide(84, lut_file, canonical_data, canonical_img, brain_nifti)
    # r3 = show_anat_slide(124, lut_file, canonical_data, canonical_img, brain_nifti)

    # seconds = (time.time() - start_time)
    # print("Processing time: ", seconds)
    # plt.imshow(r1)
    # plt.show()

    # show_anat_slide(120, lut_file, canonical_data, canonical_img, brain_nifti)
    
    # print(np.expand_dims(canonical_data[:, 120, :], axis=2).shape)
    
    # utils.show_slices([canonical_data[:, 148, :], canonical_data[148, :, :], canonical_data[:, :, 148]])

    # if (roi_nifti):
        # utils.plotting_superposition(142, canonical_data, roi_nifti.get_fdata(), 'coronal')
        # brain_nifti = nilearn.image.resample_to_img(brain_nifti, roi_nifti)
        # n = 150
        # for i in range(3):
        #     utils.elastic_deform_2(brain_nifti.get_fdata()[n, :, :], roi_nifti.get_fdata()[n, :, :])
        #     n += 1
        # utils.plotting_superposition(161, brain_nifti.get_fdata(), roi_nifti.get_fdata(), colors, 'axial')
        # utils.plotting_superposition(127, brain_nifti.get_fdata(), roi_nifti.get_fdata(), colors, 'saggital')
        # utils.plotting_superposition(127, brain_nifti.get_fdata(), roi_nifti.get_fdata(), colors, 'coronal')
    
    # utils.create_file_anat_structures(roots=roots, lut_file=lut_file, readConfig=config)

    """
    STRUCTURES, lut_res = utils.get_common_anatomical_structures(roots=roots, lut_file=lut_file.copy(), common_number=100)
    lut_res_2 = {}
    total_list = []
    for key, val in lut_res.items():
        if (val['count'] >= 99):
            lut_res_2[key] = val
            total_list.append(set(lut_res_2[key]['folders']))


    res = []

    for idx in range(len(total_list)):
        if not idx:
            res = set(total_list[idx])
        else:
            res = (res & set(total_list[idx]))

    STRUCTURES = list(lut_res_2.keys())
    utils.save_list_to_txt(STRUCTURES, 'data/common_anatomical_structures.txt')
    utils.save_list_to_txt(list(res), 'data/common_mri_images.txt')
    """

    # print("Structures: ", STRUCTURES, " Number: ", len(STRUCTURES))
    # utils.show_all_slices_per_view('coronal', brain_data, counter=70)
    # utils.show_all_slices_per_view('coronal', canonical_data, counter=70)
    # utils.show_all_slices_per_view('coronal', roi_nifti.get_fdata(), counter=70)

    # utils.saveSegSlicesPerRoot(roots, config, lut_file, saveSeg=True, segLabels=['left-cerebellum-white-matter'], origSlices=True)

    # for view in ['axial', 'coronal', 'saggital']:
    #     utils.creating_symlinks_to_dataset(roots=roots, dataset_root=DATASET_PATH, structures=['left-cerebellum-white-matter'], view=view, copy_files=True)

    # utils.elastic_deform(brain_data)
    # utils.elastic_deform(roi_nifti.get_fdata())
    
    # Check all images loop
    # roots = ['test', 'train']
    # labels = ['orig', 'left-cerebellum-white-matter']
    # for root in roots:
    #     for label in labels:
    #         path = 'dataset/'+ root + '/axial/' + label + '/img'
    #         print(f"Checking imgs in path: ", path)
    #         utils.check_imgs(path, 'png')
    
    # limit = 145

    # for n in range(limit-1, limit+100):
    #     name = f'data/HLN-12/HLN-12-1/segSlices/left-cerebellum-white-matter/axial/HLN-12-1_{n}.png'
    #     mask = utils.check_mask_img(name)

if __name__ == "__main__":
    main()