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
import elasticdeform
from augmend import Augmend, Elastic, FlipRot90, AdditiveNoise, GaussianBlur, IntensityScaleShift, Identity, CutOut, IsotropicScale, Rotate

# find . -type d -name segSlices -exec rm -r {} \;     -> To remove specific folders
def augmentation(img, msk):
    img = np.squeeze(img)
    msk = np.argmax(msk, axis=3)

    aug = Augmend()
    aug.add([
        FlipRot90(axis=(0, 1, 2)),
        FlipRot90(axis=(0, 1, 2)),
    ], probability=1)

    aug.add([
        Elastic(axis=(0, 1, 2), amount=5, order=1),
        Elastic(axis=(0, 1, 2), amount=5, order=0),
    ], probability=1)

    # aug.add([
    #     GaussianBlur(),
    #     GaussianBlur()
    # ], probability=0.9)

    # aug.add([
    #     IntensityScaleShift(scale = (0.4,2)),
    #     IntensityScaleShift(scale = (0.4,2))
    # ], probability=0.9)

    # aug.add([
    #     Identity(),
    #     Identity()
    # ], probability=0.9)

    # aug.add([
    #     AdditiveNoise(sigma=.2),
    #     AdditiveNoise(sigma=.2)
    # ], probability=0.9)

    # aug.add([
    #     CutOut(width = (40,41)),
    #     CutOut(width = (40,41))
    # ], probability=1)

    # aug.add([
    #     IsotropicScale(),
    #     IsotropicScale()
    # ], probability=1)

    # aug.add([
    #     Rotate(),
    #     Rotate()
    # ], probability=1)

    return aug([np.expand_dims(img, axis=-1), to_categorical(msk)])

def helper_anat(msk, orig_msk, lut_file, structure):
    
    k = 0
    msk[:, :, k] = np.where(orig_msk == lut_file[structure]['id'], lut_file[structure]['rgba'][k], msk[:, :, k])

    k = 1
    msk[:, :, k] = np.where(orig_msk == lut_file[structure]['id'], lut_file[structure]['rgba'][k], msk[:, :, k])

    k = 2
    msk[:, :, k] = np.where(orig_msk == lut_file[structure]['id'], lut_file[structure]['rgba'][k], msk[:, :, k])
    
    return msk

def helper_anat_integer(msk, orig_msk, lut_file_structure, new_id):
    msk[:, :, 0] = np.where(orig_msk == lut_file_structure['id'], new_id, msk[:, :, 0])
    return msk

def save_orig_slide(ns, data, view, dest, filename):
    for n in ns:
        rot = None

        name = f"{dest}/{filename}_{255 - n}"

        if (view == 'saggital'):
            rot = data[n, :, :]

        elif (view == 'coronal'):
            name = f"{dest}/{filename}_{n}"
            rot = data[:, n, :]

        else:
            rot = data[:, :, n]
        
        np.save(name, rot)

def show_anat_slide(ns, lut_file, canonical_data, STRUCTURES, dest, filename, view, class_info):

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
            msk = helper_anat_integer(msk, rot, lut_file[structure], class_info[structure]['new_id'])
            print(n, " ", np.unique(msk), " ", np.unique(rot), " ", np.unique(roi_data))

        print(n, "  OUT ", np.unique(msk))

        name = f"{dest}/{filename}_{255 - n}"
        if (view == 'saggital'):
            # utils.saveSlice(np.rot90(msk), f"{filename}_{255 - n}", dest)
            msk = np.rot90(msk)

        elif (view == 'coronal'):
            # utils.saveSlice(np.fliplr(np.rot90(msk)), f"{filename}_{n}", dest)
            msk = np.fliplr(np.rot90(msk))
            name = f"{dest}/{filename}_{n}"
        else:
            msk = np.fliplr(np.rot90(msk))
            # utils.saveSlice(msk, f"{filename}_{255-n}", dest)

        print(n, " AFTER IFS ", np.unique(msk))
        msk = to_categorical(msk, num_classes=105)
        print(msk.shape)
        # np.save(name, msk)

def main():
    dataset_path = 'dataset_3D_p64/'
    # Getting images
    test_filename = 'MMRR-21-20'

    img = np.load(dataset_path + f"test/images/{test_filename}.npy")
    msk = np.load(dataset_path + f"test/masks/{test_filename}.npy")

    print(msk.shape)

    img_d, msk_d = augmentation(img[0, :, :, :], msk[0, :, :, :,])
    img_d2, msk_d2 = augmentation(img[1, :, :, :], msk[1, :, :, :])
    
    palette = np.array([[  0,   0,   0],   # black
                    [255,   0,   0],   # red
                    [  0, 255,   0],   # green
                    [  0,   0, 255]])   # blue

    print(img_d.shape, " ", msk_d.shape)
    print(img.shape, "  ", msk.shape)
    print(img_d.shape, "  ", msk_d.shape)
    print(np.unique(img_d))
    exit()
    msk_d = palette[msk_d]
    msk_d2 = palette[msk_d2]
    msk = palette[np.argmax(msk, axis=4)]
    
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))

    idx=0
    idx2=58

    axs[0][0].imshow(img[idx, idx2, :, :], cmap='bone')
    axs[0][1].imshow(img_d[idx2, :, :], cmap='bone')

    axs[1][0].imshow(msk[idx, idx2, :, :])#, cmap='bone')
    axs[1][1].imshow(msk_d[idx2, :, :])#, cmap='bone')

    plt.show()

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))

    idx=1
    idx2=58

    axs[0][0].imshow(img[idx, idx2, :, :], cmap='bone')
    axs[0][1].imshow(img_d2[idx2, :, :], cmap='bone')

    axs[1][0].imshow(msk[idx, idx2, :, :])#, cmap='bone')
    axs[1][1].imshow(msk_d2[idx2, :, :])#, cmap='bone')
    plt.show()
    # elastic_deform_3D(img, msk, 0, 50)
    exit()
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

    class_info = utils.get_classes_2()

    roots = [PREFIX_PATH + root for root in roots]
    lut_file = utils.load_lut(LUT_PATH)

    config = {
        'RAS': True, 
        'normalize': False
    }

    # image_obj = nib.load(image_path)

    canonical_img, canonical_data = utils.readMRI(imagePath=image_path, config=config)
    # canonical_nifti = nib.Nifti1Image(canonical_data, affine=canonical_img.affine)

    # brain_img, brain_data = utils.readMRI(imagePath=brainPath, config=config)
    # brain_nifti = nib.Nifti1Image(brain_data, affine=brain_img.affine)

    STRUCTURES = utils.read_test_to_list('data/common_anatomical_structures.txt')
    mri_paths = utils.read_test_to_list('data/common_mri_images.txt')

    for idx in range(len(mri_paths)):
        tmp = mri_paths[idx].split('-')
        mri_paths[idx] = f"data/{'-'.join(tmp[:-1])}/{mri_paths[idx]}"

    # print(mri_paths[0])

    # utils.create_folder('dataset_test_3/segmented/saggital')
    # utils.create_folder('dataset_test_3/segmented/coronal')
    # utils.create_folder('dataset_test_3/segmented/axial')

    slides_n = [x for x in range(256)]

    # show_anat_slide(slides_n, lut_file, canonical_data, 
    #     STRUCTURES,
    #     f"dataset_test_3/segmented/axial",
    #     'MMRR-21-21',
    #     'axial'   
    # )




    # for folder in ['images', 'segmented']:
    #     utils.create_folder(f"dataset_test_5/{folder}/saggital")
    #     utils.create_folder(f"dataset_test_5/{folder}/coronal")
    #     utils.create_folder(f"dataset_test_5/{folder}/axial")

    # start_time = time.time()

    # for mri_path in mri_paths:

    #     mri_name = mri_path.split('/')[-1]

    #     data_img, data = utils.readMRI(mri_path + '/001.mgz', config)
    #     data_msk_img, data_msk = utils.readMRI(imagePath=mri_path + '/aparcNMMjt+aseg.mgz', config=config)
        
    #     if data.shape != data_msk.shape:
    #         print("Correcting shapes in: ", mri_name)
    #         data_img = nilearn.image.resample_to_img(data_img, data_msk_img)
    #         data = data_img.get_fdata()
    #         data = scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
    #     else:
    #         data = scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)

    #     print("Saving axial, saggital and coronal images for ", mri_name)

    #     for ori in ['axial', 'saggital', 'coronal']:
    #         # save_orig_slide(slides_n, data, ori, f"dataset_test_4/images/{ori}", mri_name)
    #         show_anat_slide(slides_n, lut_file, data_msk, 
    #             # [
    #             #     'left-cerebral-white-matter',
    #             #     'right-cerebral-white-matter',
    #             #     'left-cerebellum-white-matter',
    #             #     'right-cerebellum-white-matter'
    #             # ],
    #             STRUCTURES,
    #             f"dataset_test_5/segmented/{ori}",
    #             mri_name,
    #             ori,
    #             class_info
    #         )
    #     break
    
    # seconds = (time.time() - start_time)
    # print("Processing time: ", seconds)

    # utils.show_all_slices_per_view('coronal', brain_data, counter=70)
    # utils.plot_roi_modified(lut_file, 'right-cerebral-white-matter', brain_nifti, canonical_data, canonical_img)
    

    show_anat_slide([120], lut_file, canonical_data, STRUCTURES, f"dataset_test_5/segmented/coronal", 
    "NKI-TRT-20-1", "coronal", class_info)
    
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