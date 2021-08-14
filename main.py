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
import nilearn
import time

# find . -type d -name segSlices -exec rm -r {} \;     -> To remove specific folders

# Improve parameters and function later, but it works!
def show_anat_slide(n, lut_file, canonical_data, canonical_img, brain_nifti):
  
    STRUCTURES = utils.read_test_to_list('data/common_anatomical_structures.txt') + [
        'left-cerebral-white-matter', 
        'right-cerebral-white-matter',
        'left-lateral-ventricle',
        'right-lateral-ventricle',
        'right-hippocampus',
        'left-hippocampus',
    ]
    # STRUCTURES = ['left-cerebellum-white-matter', 'right-cerebellum-white-matter']
    structure = STRUCTURES[0]

    roi_nifti, colors = utils.get_roi_data(lut_file, structure, canonical_data, canonical_img)
    brain_nifti = nilearn.image.resample_to_img(brain_nifti, roi_nifti)
    brain_data = brain_nifti.get_fdata()
    roi_data = roi_nifti.get_fdata()
    
    # print(canonical_data[roi_data.nonzero()])
    
    # for i in range(255):
    #     res = roi_data[:, i, :].nonzero()
    #     if (res[0].shape[0] or res[1].shape[0]):
    #         print("slide: ", i)
    #         print(res)
    #         break
    
    start_time = time.time()

    # print(lut_file[structure]['rgba'])
    msk = np.zeros((256, 256, 3), dtype=int)
    # n = 54
    rot = roi_data[:, n, :]#np.fliplr(np.rot90()) #np.fliplr(rotate(roi_data[:, n, :], angle=90))
    # print()
    # rot[np.where(check_msk == 0)] = 0
    msk[:, :, 0] = rot
    msk[:, :, 1] = rot
    msk[:, :, 2] = rot

    k = 0
    msk[:, :, k] = np.where(msk[:, :, k] == lut_file[structure]['id'], lut_file[structure]['rgba'][k], msk[:, :, k])

    k = 1
    msk[:, :, k] = np.where(msk[:, :, k] == lut_file[structure]['id'], lut_file[structure]['rgba'][k], msk[:, :, k])

    k = 2
    msk[:, :, k] = np.where(msk[:, :, k] == lut_file[structure]['id'], lut_file[structure]['rgba'][k], msk[:, :, k])

    for structure in STRUCTURES[1:]:
        # print("Unique: ", np.unique(msk))
        
        # roi_nifti, colors = utils.get_roi_data(lut_file, structure, canonical_data, canonical_img)
        # roi_data = roi_nifti.get_fdata()
        roi_data = (canonical_data==lut_file[structure]['id'])*lut_file[structure]['id']
        rot = roi_data[:, n, :] # np.fliplr(np.rot90(roi_data[:, n, :])) #np.fliplr(rotate(roi_data[:, n, :], angle=90))

        k = 0
        msk[:, :, k] = np.where(rot == lut_file[structure]['id'], lut_file[structure]['rgba'][k], msk[:, :, k])

        k = 1
        msk[:, :, k] = np.where(rot == lut_file[structure]['id'], lut_file[structure]['rgba'][k], msk[:, :, k])

        k = 2
        msk[:, :, k] = np.where(rot == lut_file[structure]['id'], lut_file[structure]['rgba'][k], msk[:, :, k])
        
    # test_msk = roi_data[:, 58, :]
    seconds = (time.time() - start_time)
    print("Processing time: ", seconds)
    plt.imshow(np.fliplr(np.rot90(msk)))
    plt.imshow(np.fliplr(np.rot90(brain_data[:, n, :])), cmap='bone', alpha=0.4) # np.fliplr(rotate(, angle=90))
    plt.show()

def main():
    LUT_PATH = './data/FreeSurferColorLUT.txt'
    brainPath = './data/NKI-TRT-20/NKI-TRT-20-1/001.mgz'#'./data/sub01/001.mgz' #brain.mgz'
    image_path = "./data/NKI-TRT-20/NKI-TRT-20-1/aparcNMMjt+aseg.mgz"#"./data/sub01/aparcNMMjt+aseg.mgz"
    DATASET_PATH = '/home/camilo/Programacion/master_thesis/dataset_test/' 
    PREFIX_PATH = '/home/camilo/Programacion/master_thesis/data/'
    roots = [
        'HLN-12',
        'Colin27',
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
        'normalize': False
    }

    # image_obj = nib.load(image_path)

    canonical_img, canonical_data = utils.readMRI(imagePath=image_path, config=config)
    print(canonical_data.shape)
    canonical_nifti = nib.Nifti1Image(canonical_data, affine=canonical_img.affine)

    brain_img, brain_data = utils.readMRI(imagePath=brainPath, config=config)
    brain_nifti = nib.Nifti1Image(brain_data, affine=brain_img.affine)

    # utils.show_all_slices_per_view('coronal', brain_data, counter=70)
    # utils.plot_roi_modified(lut_file, 'right-cerebral-white-matter', brain_nifti, canonical_data, canonical_img)
    
    show_anat_slide(54, lut_file, canonical_data, canonical_img, brain_nifti)
    show_anat_slide(84, lut_file, canonical_data, canonical_img, brain_nifti)
    show_anat_slide(120, lut_file, canonical_data, canonical_img, brain_nifti)
    
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

    # STRUCTURES, lut_res = utils.get_common_anatomical_structures(roots=roots, lut_file=lut_file.copy(), common_number=101)
    # utils.save_list_to_txt(STRUCTURES, 'data/common_anatomical_structures.txt')
    # print(lut_res)
        
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