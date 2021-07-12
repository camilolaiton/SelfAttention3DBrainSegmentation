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

import outils
import nibabel as nib

def main():
    LUT_PATH = './data/FreeSurferColorLUT.txt'
    brainPath = './data/sub01/001.mgz' #brain.mgz'
    image_path = "./data/sub01/aparcNMMjt+aseg.mgz"
    cut_coords_limit = 400
    lut_file = outils.load_lut(LUT_PATH)

    config = {
        'RAS': True, 
        'normalize': False
    }

    # image_obj = nib.load(image_path)

    canonical_img, canonical_data = outils.readMRI(imagePath=image_path, config=config)

    canonical_nifti = nib.Nifti1Image(canonical_data, affine=canonical_img.affine)

    brain_img, brain_data = outils.readMRI(imagePath=brainPath, config=config)

    # brain_img = nib.load(brainPath)
    # brain_data = brain_img.get_fdata()
    brain_nifti = nib.Nifti1Image(brain_data, affine=brain_img.affine)

    #print(f'Type of the image {type(canonical_img)}')
    #print(canonical_img)

    # Obtengo el shape

    # height, width, depth = canonical_data.shape
    # print(f"The image object has the following dimensions: height: {height}, width:{width}, depth:{depth}")

    # outils.show_all_slices_per_view('z', brain_data, counter=70)
    # outils.plot_roi_modified(lut_file, 'left-pallidum', brain_nifti, canonical_data, canonical_img)
    
    roi_nifti, colors = outils.get_roi_data(lut_file, 'right-cerebral-white-matter', canonical_data, canonical_img)
    
    # if (roi_nifti):
    #     # outils.plotting_superposition(142, canonical_data, roi_nifti.get_fdata(), 'z')
    #     outils.plotting_superposition(85, brain_nifti.get_fdata(), roi_nifti.get_fdata(), 'x')
    #     outils.plotting_superposition(127, brain_nifti.get_fdata(), roi_nifti.get_fdata(), 'y')
    #     outils.plotting_superposition(127, brain_nifti.get_fdata(), roi_nifti.get_fdata(), 'z')



    prefix_path = 'data/'
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
    roots = [prefix_path + root for root in roots]
    
    # outils.create_file_anat_structures(roots=roots, lut_file=lut_file, readConfig=config)
    # STRUCTURES, lut_res = outils.get_common_anatomical_structures(roots=roots, lut_file=lut_file.copy(), common_number=101)
    # outils.save_list_to_txt(STRUCTURES, 'data/common_anatomical_structures.txt')
    # print(lut_res)
    STRUCTURES = outils.read_test_to_list('data/common_anatomical_structures.txt')
    # print("Structures: ", STRUCTURES, " Number: ", len(STRUCTURES))
    # outils.show_all_slices_per_view('z', brain_data, counter=70)
    # outils.show_all_slices_per_view('z', canonical_data, counter=70)
    # outils.show_all_slices_per_view('z', roi_nifti.get_fdata(), counter=70)

    outils.saveSlicesPerRoot(roots, config)

if __name__ == "__main__":
    main()