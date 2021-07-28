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

import nilearn

# find . -type d -name segSlices -exec rm -r {} \;     -> To remove specific folders

def main():
    LUT_PATH = './data/FreeSurferColorLUT.txt'
    brainPath = './data/HLN-12/HLN-12-1/001.mgz'#'./data/sub01/001.mgz' #brain.mgz'
    image_path = "./data/HLN-12/HLN-12-1/aparcNMMjt+aseg.mgz"#"./data/sub01/aparcNMMjt+aseg.mgz"
    DATASET_PATH = '/home/camilo/Programacion/master_thesis/dataset/' 
    PREFIX_PATH = '/home/camilo/Programacion/master_thesis/data/'
    roots = [
        'HLN-12',
        # 'Colin27',
        # 'MMRR-3T7T-2',
        # 'NKI-RS-22',
        # 'NKI-TRT-20',
        # 'MMRR-21',
        # 'OASIS-TRT-20',
        # 'Twins-2',
        # 'Afterthought'
    ]
    roots = [PREFIX_PATH + root for root in roots]
    lut_file = utils.load_lut(LUT_PATH)

    config = {
        'RAS': True, 
        'normalize': False
    }

    # image_obj = nib.load(image_path)

    canonical_img, canonical_data = utils.readMRI(imagePath=image_path, config=config)
    canonical_nifti = nib.Nifti1Image(canonical_data, affine=canonical_img.affine)
    brain_img, brain_data = utils.readMRI(imagePath=brainPath, config=config)
    brain_nifti = nib.Nifti1Image(brain_data, affine=brain_img.affine)

    # utils.show_all_slices_per_view('coronal', brain_data, counter=70)
    # utils.plot_roi_modified(lut_file, 'right-cerebral-white-matter', brain_nifti, canonical_data, canonical_img)
    
    roi_nifti, colors = utils.get_roi_data(lut_file, 'right-cerebral-white-matter', canonical_data, canonical_img)

    if (roi_nifti):
        # utils.plotting_superposition(142, canonical_data, roi_nifti.get_fdata(), 'coronal')
        brain_nifti = nilearn.image.resample_to_img(brain_nifti, roi_nifti)
        # n = 150
        # for i in range(3):
        #     utils.elastic_deform_2(brain_nifti.get_fdata()[n, :, :], roi_nifti.get_fdata()[n, :, :])
        #     n += 1
        # utils.plotting_superposition(161, brain_nifti.get_fdata(), roi_nifti.get_fdata(), colors, 'axial')
        # utils.plotting_superposition(127, brain_nifti.get_fdata(), roi_nifti.get_fdata(), colors, 'saggital')
        # utils.plotting_superposition(127, brain_nifti.get_fdata(), roi_nifti.get_fdata(), colors, 'coronal')
    
    # utils.create_file_anat_structures(roots=roots, lut_file=lut_file, readConfig=config)

    STRUCTURES, lut_res = utils.get_common_anatomical_structures(roots=roots, lut_file=lut_file.copy(), common_number=101)
    # utils.save_list_to_txt(STRUCTURES, 'data/common_anatomical_structures.txt')
    # print(lut_res)
    
    # STRUCTURES = utils.read_test_to_list('data/common_anatomical_structures.txt')
    
    print("Structures: ", STRUCTURES, " Number: ", len(STRUCTURES))
    # utils.show_all_slices_per_view('coronal', brain_data, counter=70)
    # utils.show_all_slices_per_view('coronal', canonical_data, counter=70)
    # utils.show_all_slices_per_view('coronal', roi_nifti.get_fdata(), counter=70)

    # utils.saveSegSlicesPerRoot(roots, config, lut_file, saveSeg=True, segLabels=['left-cerebellum-white-matter'], origSlices=True)

    # for view in ['axial', 'coronal', 'saggital']:
    #     utils.creating_symlinks_to_dataset(roots=roots, dataset_root=DATASET_PATH, structures=['left-cerebellum-white-matter'], view=view)

    # utils.elastic_deform(brain_data)
    # utils.elastic_deform(roi_nifti.get_fdata())

    roots = ['test', 'train']
    labels = ['orig', 'left-cerebellum-white-matter']

    for root in roots:
        for label in labels:
            path = 'dataset/'+ root + '/axial/' + label + '/img'
            print(f"Checking imgs in path: ", path)
            utils.check_imgs(path, 'png')

if __name__ == "__main__":
    main()