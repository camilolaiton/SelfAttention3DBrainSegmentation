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

from PIL.Image import NEAREST
import numpy as np
import nibabel as nib

import PIL
from PIL import Image, ImageOps
from pathlib import Path
from PIL import UnidentifiedImageError

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patheffects as path_effects

from nilearn.plotting import plot_roi
from nilearn.image import resample_to_img
from scipy.ndimage import rotate

import os, errno
import time
from sklearn.utils import shuffle
import math
from glob import glob
import shutil
import elasticdeform

def normalizeIntensityImage(img_data:np, min_value:float, max_value:float):
  """
    Normalizing intensity values for MRI

    Parameters:
      - img_data: numpy array containing MRI's data
      - min_value: minimum intensity value from img_data array and could be gotten with np.min()
      - max_value: maximum intensity value from img_data array and could be gotten with np.max()

    Returns:
      - numpy.Array
  """

  img_data[img_data < min_value] = min_value
  img_data[img_data > max_value] = max_value
  return (img_data - min_value) / (max_value - min_value)

def readMRI(imagePath:str, config:dict, nifti_format:bool=False):
  """
    Reading MRI with different preprocessing steps

    Parameters:
      - imagePath: String that contains the path where the MRI is located 
      - config: Dictionary that contains configuration values for processing the MRI when it's read
      - nifti_format: Boolean that says if the image should return in the nifti format or not.

    Returns:
      - nib.Nifti1Image, numpy.Array
  """
  
  imageObj = nib.load(imagePath)
  
  if (config['RAS']):
    imageObj = nib.as_closest_canonical(imageObj)
  
  imageData = imageObj.get_fdata()

  if (config['normalize']):
    imageData = normalizeIntensityImage(imageData, np.min(imageData), np.max(imageData))

  if (nifti_format):
    imageObj = nib.Nifti1Image(imageData, affine=imageObj.affine)
    imageData = imageObj.get_fdata()

  return imageObj, imageData

def load_lut(lut_path):
    """
      Loads FreeSurferColorLut.txt file into a dictionary

      Parameters:
        - lut_path: Path where the FreeSurferColorLut.txt is located

      Returns:
        - Dictionary: { label:{'id':int, 'rgba':list}, ...}
    """

    with open(lut_path) as f:
        content= f.read().split('\n')

    lut_colors={}
    for line in content:
        if line and '#' not in line:
          tmp = line.split()
          lut_colors[tmp[1].lower()] = {
                'id' : int(tmp[0]),
                'rgba': [int(tmp[2]), int(tmp[3]), int(tmp[4]), int(tmp[5])],
                'count':0
            }

    return lut_colors

def get_label_data(lut:dict, label:str):
  """
    Gets the data associated to a label

    Parameters:
      - lut: Dictionary that contains FreeSurferColorLut.txt file.
      - label: String that contains the brain's section name

    Returns:
      - Dictionary OR None
  """
  if (type(lut) == dict):
    try:
      return lut[label.lower()]
    except KeyError as err:
      print(f"Error getting {label} data. Error: {err}")
  return None

def get_roi_data(lut:dict, label:str, segBrain_data:np, segBrain_img:nib):
  roi_n, colors = None, None
  label_data = get_label_data(lut, label)
  if (label_data):
    roi = (segBrain_data==label_data['id'])*label_data['id']
    res = roi.nonzero()

    if (res[0].shape[0] or res[1].shape[0] or res[2].shape[0]):
      
      roi_n = nib.Nifti1Image(roi, affine = segBrain_img.affine)
      colors = ListedColormap([value/255 for value in label_data['rgba'][:-1]])

  return roi_n, colors


def plot_roi_modified(lut:dict, label:str, brain:nib, segBrain_data:np, segBrain_img:nib, orientations:list=None):
  """
    plots the ROI in a brain

    Parameters:
      - lut: Dictionary that contains FreeSurferColorLut.txt file.
      - label: String that contains the brain's section name
      - brain: nib object that containes the brain where the ROI is going to be plotted
      - segBrain_data: numpy array that contains the data for the MRI segmented brain
      - segBrain_img: nib object that contains the ROI data
      - orientations: list that specifies which views will be plotted. ['axial', 'saggital', 'coronal', ...]

    Returns:
      - None
  """

  # label_data = get_label_data(lut, label)
  # if (label_data):
  #   roi = (segBrain_data==label_data['id'])*label_data['id']
  #   res = roi.nonzero()

  #   if (res[0].shape[0] or res[1].shape[0] or res[2].shape[0]):
      
  #     roi_n = nib.Nifti1Image(roi, affine = segBrain_img.affine)
      
  #     colors = ListedColormap([value/255 for value in label_data['rgba'][:-1]])

  roi_n, colors = get_roi_data(lut, label, segBrain_data, segBrain_img)
  if (roi_n and colors):      
    if (orientations):
      for ori in orientations:
        plot_roi(roi_n, bg_img=brain, cmap=colors, title=label + f" orientation: {ori}", display_mode=ori, black_bg=True)
    else:
      plot_roi(roi_n, bg_img=brain, cmap=colors, title=label, black_bg=True, draw_cross=False)#, cut_coords=256)
    plt.show()
  else:
    print("This label is not segmented in the aseg file.")

def show_slices(slices):

  """ 
    Function to display row of image slices 

    Parameters:
      - Slices: Array that containes the slices that will be displayed
    
    Returns:
      - None
  """

  fig, axes = plt.subplots(1, len(slices))

  for i, slice in enumerate(slices):
    axes[i].imshow(slice.T, origin="lower", vmin=10, vmax=110)#, cmap="gray")
  plt.show()

def show_all_slices_per_view(view:str, data:np, counter:int=100, angle=270):
  """ 
    Function to display multiple slices consequently 

    Parameters:
      - view: String that contains the plotted view ['axial', 'saggital', 'coronal']
      - data: MRI data
      - counter: Integer that dictates from where the slices with be plotted
    
    Returns:
      - None
   """

  if view in ['axial', 'saggital', 'coronal']:
    y, z, x = data.shape

    view_limit = x
    if view == 'saggital':
      view_limit = y
    elif view == 'coronal':
      view_limit = z

    columns = 10
    rows = 10
    fig, axs = plt.subplots(columns, rows, figsize=(20, 20))

    for row in range(rows):
      for column in range(columns):
        if (view == 'saggital'):
          axs[row, column].imshow(data[counter, :, :], cmap='bone')
        elif (view == 'coronal'):
          axs[row, column].imshow(data[:, counter, :], cmap='bone')
        else:
          axs[row, column].imshow(data[:, :, counter], cmap='bone')

        axs[row, column].grid(False)
        axs[row, column].axis('off')
        axs[row, column].set_title(f"Slide {counter}", fontsize='small', loc='center')

        if counter < view_limit:
          counter += 1

    plt.tight_layout()
    plt.show()

def imshow_mri(data, img, vox, xyz, suptitle):
    """
      Show an MRI slice with a voxel annotated.
    """
    
    i, j, k = vox
    fig, ax = plt.subplots(1, figsize=(6, 6))
    codes = nib.orientations.aff2axcodes(img.affine)
    # Figure out the title based on the code of this axis
    ori_slice = dict(P='Coronal', A='Coronal',
                     I='Axial', S='Axial',
                     L='Sagittal', R='Saggital')
    ori_names = dict(P='posterior', A='anterior',
                     I='inferior', S='superior',
                     L='left', R='right')
    title = ori_slice[codes[0]]
    ax.imshow(data[i], vmin=10, vmax=120, cmap='gray', origin='lower')
    ax.axvline(k, color='y')
    ax.axhline(j, color='y')
    for kind, coords in xyz.items():
        annotation = ('{}: {}, {}, {} mm'
                      .format(kind, *np.round(coords).astype(int)))
        text = ax.text(k, j, annotation, va='baseline', ha='right',
                       color=(1, 1, 0.7))
        text.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground='black'),
            path_effects.Normal()])
    # reorient view so that RAS is always rightward and upward
    x_order = -1 if codes[2] in 'LIP' else 1
    y_order = -1 if codes[1] in 'LIP' else 1
    ax.set(xlim=[0, data.shape[2] - 1][::x_order],
           ylim=[0, data.shape[1] - 1][::y_order],
           xlabel=f'k ({ori_names[codes[2]]}+)',
           ylabel=f'j ({ori_names[codes[1]]}+)',
           title=f'{title} view: i={i} ({ori_names[codes[0]]}+)')
    fig.suptitle(suptitle)
    fig.subplots_adjust(0.1, 0.1, 0.95, 0.85)
    return fig

def create_folder(dest_dir:str):
    """
      This function creates a folder. If it exists already I doesn't do anything.

      Parameters:
        - dest_dir: string that has the path where the folder(s) are going to be created
      
      Returns:
        - None
    """
    if not (os.path.exists(dest_dir)):
        try:
            print(f"Creating new directory: {dest_dir}")
            os.makedirs(dest_dir)
        except OSError as e:
            if (e.errno != e.errno.EEXIST):
                raise

def saveSlice(image_data:np, filename:str, destination:str):
  """
    Function that saves a slice in a specific path

    Parameters:
      - image_data: numpy array that contains the MRI data
      - filename: string that has the filename
      - destination: string that has the path where the slice will be saved
    
    Returns:
      - None
  """

  path = os.path.join(destination, f"{filename}.png")
  plt.grid(False)
  plt.axis('off')
  plt.imsave(path, image_data, cmap='bone')
  # plt.imshow(image_data, cmap='bone')
  #cv2.imwrite(path, image_data)
  #print(f"[+] Slice saved: {path}", end='\r')

def saveSliceView(filename:str, image_data:np, view:str, destination:str):
  """
    Function that saves multiple slices from a view in a specific path

    Parameters:
      - filename: string that has the filename
      - image_data: numpy array that contains the MRI data
      - view: string that has the view from where the images will be taken. View could be ['axial', 'saggital', 'coronal']
      - destination: string that has the path where the slice will be saved
    
    Returns:
      - None
  """
  saggital, coronal, axial = image_data.shape
  
  if (view == 'saggital'):
    # saggital
    create_folder(f"{destination}/saggital")
    for idx in range(saggital):#-1, -1, -1):
      saveSlice(image_data=rotate(image_data[idx, :, :], angle=90), filename=f"{filename}_{255-idx}", destination=f"{destination}/saggital")

  elif (view == 'coronal'):
    # coronal
    create_folder(f"{destination}/coronal")
    for idx in range(coronal):#-1, -1, -1):
      saveSlice(image_data=np.fliplr(rotate(image_data[:, idx, :], angle=90)), filename=f"{filename}_{idx}", destination=f"{destination}/coronal")

  else:
    # For x axial
    create_folder(f"{destination}/axial")
    for idx in range(axial):#-1, -1, -1):
      saveSlice(image_data=np.fliplr(rotate(image_data[:, :, idx], angle=90)), filename=f"{filename}_{255-idx}", destination=f"{destination}/axial")

def saveAllSlices(filename:str, brain_data:np, destination:str):
    """
        Function that saves multiple slices from multiple views in a specific path

        Parameters:
        - filename: string that has the filename
        - brain_data: numpy array that contains the MRI data
        - destination: string that has the path where the slice will be saved

        Returns:
        - None
    """
    for ori in ['axial', 'saggital', 'coronal']:
        saveSliceView(filename, brain_data, ori, destination)

def get_segmented_structures(lut, segBrain_data):
  """
    Gets the names of the segmented structures

    Parameters:
      - lut: Dictionary that contains FreeSurferColorLut.txt file.
      - segBrain_data: numpy array that contains the data for the MRI segmented brain

    Returns:
      - Array: Array that contains the names of the anatomical structures that exist in that segmented brain.
      - Float: Time that the process lasted
  """
  segmented_structures = []
  start_time = time.time()

  for key, label_data in lut.items():
    roi = (segBrain_data==label_data['id'])*label_data['id']
    res = roi.nonzero()

    if (res[0].shape[0] or res[1].shape[0] or res[2].shape[0]):
      segmented_structures.append(f"{key}")

  return segmented_structures, (time.time() - start_time)

def create_file_anat_structures(roots:list, lut_file:dict, readConfig:dict):
  """
    Creates a file with the list of the existing anatomical structures per individual.

    Parameters:
      - root: String that contains the root path of the individuals.
      - lut_file: Dictionary that contains FreeSurferColorLut.txt file.
      - readConfig: dictionary that contains the configuration for reading a MRI.

    Returns:
      - None
  """

  mri_to_process = {}

  for root in roots:
    # Reading folders
    i = 0
    for folder in os.walk(root):
      if not i:
        i += 1
      else:
        if ('anatomical_structures.txt' not in folder[2]):
          mri_to_process[folder[0].split('/')[-1]] = {
            'root': folder[0],  
            'mri': folder[0] + '/' + 'aparcNMMjt+aseg.mgz', 
          }

  # Writing anatomical structures in each folder
  for key, items in mri_to_process.items():
    print(f"[+] Processing brain: {key}")
    
    _, canonical_data = readMRI(imagePath=items['mri'], config=readConfig)
    structures, seconds = get_segmented_structures(lut_file, canonical_data)
    print(f"[+] Structures read. Time {seconds} seconds.")

    with open(items['root'] + '/anatomical_structures.txt', 'w') as ana_file:
      for structure in structures:
        ana_file.write(structure + '\n')

def helper_extract_common_structures(lut_dict:dict, common_number:int):
  """
    Helper function to extract the common anatomical structures from
    a dictionary. It works with a counter and extracts just the anatomical structures
    with counter == common_number.

    Parameters:
      - common_number: Integer that determines the amount of common anatomical structures in
      a number n of MRIs.
      - lut_dict: Dictionary that contains FreeSurferColorLut.txt file.

    Returns:
      - list
  """
  common_structures = []

  for key, items in lut_dict.items():
    if (int(items['count']) == common_number):
      common_structures.append(key)

  return common_structures

def get_common_anatomical_structures(roots:list, lut_file:dict, common_number:int=101):
  """
    Function to extract the common anatomical sstructures.

    Parameters:
      - root: String that contains the root path of the individuals.
      - lut_file: Dictionary that contains FreeSurferColorLut.txt file.
      - common_number: Integer that determines the amount of common anatomical structures in
      a number n of MRIs.

    Returns:
      - list, dict
  """

  # Getting paths of anat_structures files
  anat_structures = {}
  for root in roots:
    # Reading folders
    i = 0
    for folder in os.walk(root):
      if not i:
        i += 1
      else:
        if ('anatomical_structures.txt' in folder[2]):
          anat_structures[folder[0].split('/')[-1]] = {
            'root': folder[0],  
            'anat_file': folder[0] + '/' + 'anatomical_structures.txt', 
          }
  
  # Getting the common anatomical structures
  count = 0
  for key, items in anat_structures.items():
    count += 1
    # print(f"[+] Processing anatomical structures file from {key}")
    with open(items['anat_file'], 'r') as anat_file:
      while True:
        anat_structure = anat_file.readline().rstrip('\r\n')
        
        if not anat_structure:
          break
        else:
          try:
              lut_file[anat_structure]['count'] = lut_file[anat_structure]['count'] + 1
          except KeyError as err:
            pass

  print("\n[+] Anatomical structures files processed: ", count)
  
  return helper_extract_common_structures(lut_file, common_number), lut_file

def save_list_to_txt(list_text:list, dest:str):
  """
    Function to save the information saved in a list.

    Parameters:
      - list_text: List that contains information.
      - dest: Destination path where the file will be saved.

    Returns:
      - None
  """
  with open(dest, 'w') as file:
    for text in list_text:
      file.write(str(text) + '\n')

def read_test_to_list(path:str):
  """
    Function to get the information saved in a file.

    Parameters:
      - path: Path where the file is located.

    Returns:
      - list
  """
  list_text = []

  with open(path, 'r') as anat_file:
    while True:
      anat_structure = anat_file.readline().rstrip('\r\n')

      if not anat_structure:
        break
      else:
        list_text.append(anat_structure)

  return list_text

def helperPlottingOverlay(img, msk):
  # plt.imshow(first_roi_img, cmap='bone')
  plt.imshow(img, cmap='bone', interpolation='none')
  plt.imshow(msk, alpha=0.5, interpolation='none')#, cmap='Oranges')
  # plt.imshow(slice_roi, cmap='bone')
  plt.show()

def plotting_superposition(n_slice, brain_data, roi_data, roi_color, orientation='axial'):
  """
  
  """
  
  slice_orig, slice_roi = None, None

  if (orientation == 'saggital'):
    slice_orig = rotate(brain_data[n_slice, :, :], angle=90)
    slice_roi = rotate(roi_data[n_slice, :, :], angle=90)
  
  elif (orientation == 'coronal'):
    slice_orig = np.fliplr(rotate(brain_data[:, n_slice, :], angle=90))
    slice_roi = np.fliplr(rotate(roi_data[:, n_slice, :], angle=90))
  
  else:
    slice_orig = np.fliplr(rotate(brain_data[:, :, n_slice], angle=90))
    slice_roi = np.fliplr(rotate(roi_data[:, :, n_slice], angle=90))

  # print(slice_orig.shape)
  # print(slice_roi.shape)
  
  slice_roi = np.ma.masked_where(slice_roi == 0, slice_roi)
  helperPlottingOverlay(slice_orig, slice_roi)

def helperGetRootFolders(roots:list, max_depth:int=1):
  """
  
  """
  print(roots)

  mri_files = {}

  for root in roots:
    # Reading folders
    i = 0
    for root, dirs, files in os.walk(root):
      if not i:
        i += 1
      else:
        mri_files[root.split('/')[-1]] = {
          'root': root,  
          'orig': root + '/' + '001.mgz',
          'segmented': root + '/' + 'aparcNMMjt+aseg.mgz', 
        }
        if root.count(os.sep) - root.count(os.sep) == max_depth - 1:
          del dirs[:]

  return mri_files

def saveSegSlicesPerRoot(roots:list, configMRI:dict, lut_file:dict, saveSeg:bool=False, segLabels:list=[], origSlices=False):
  """
  
  """
  mri_files = helperGetRootFolders(roots)
  # mri_files = {'HLN-12-1': {'root': 'data/HLN-12/HLN-12-1', 'orig': 'data/HLN-12/HLN-12-1/001.mgz', 'segmented': 'data/HLN-12/HLN-12-1/aparcNMMjt+aseg.mgz'}}

  if (saveSeg and len(segLabels)):
    coronal_count, axial_count, saggital_count = 0, 0, 0
    coronal_seg_count, axial_seg_count, saggital_seg_count = 0, 0, 0

    for key, items in mri_files.items():
      canonical_img, canonical_data = readMRI(imagePath=items['segmented'], config=configMRI)
      orig_img_nifti, orig_data_nifti = readMRI(imagePath=items['orig'], config=configMRI, nifti_format=True)

      for segLabel in segLabels:
        roi_nifti, colors = get_roi_data(lut_file, segLabel, canonical_data, canonical_img)

        # canonical_img_nifti = nib.Nifti1Image(canonical_data, affine=canonical_img.affine)
        # canonical_data_nifti = canonical_img_nifti.get_fdata()

        orig_img_nifti = resample_to_img(orig_img_nifti, roi_nifti)

        orig_data_nifti = orig_img_nifti.get_fdata()
        roi_nifti_data = roi_nifti.get_fdata()

        shape = canonical_data.shape
        saggital_count += shape[0]
        coronal_count += shape[1]
        axial_count += shape[2]

        if (origSlices):
          print(f"\n[+] Saving orig slices for {key} shape: {shape}")
          saveAllSlices(key, orig_data_nifti, items['root']+'/slices')

        shape = roi_nifti_data.shape
        saggital_seg_count += shape[0]
        coronal_seg_count += shape[1]
        axial_seg_count += shape[2]

        print(f"\n[+] Saving segmented slices for {key} shape: {shape}")
        saveAllSlices(key, roi_nifti_data, items['root']+'/segSlices/' + segLabel)

    print(f"Total seg slices per coronal view: ", coronal_seg_count)
    print(f"Total seg slices per axial view: ", axial_seg_count)
    print(f"Total seg slices per saggital view: ", saggital_seg_count)

    print(f"Total slices per coronal view: ", coronal_count)
    print(f"Total slices per axial view: ", axial_count)
    print(f"Total slices per saggital view: ", saggital_count)
    
    print(f"\n Total orig images: {coronal_count+axial_count+saggital_count}")
    print(f"\n Total seg images: {coronal_seg_count+axial_seg_count+saggital_seg_count}")

  # Coronal images: 25856
  # Axial images:   25856
  # Saggital images: 25856

  # Total: 77568

def saveSlicesPerRoot(roots:list, configMRI:dict, saveOrig=False):
  """
    [Deprecated]
  """
  # This function has an error with the amount of slices
  mri_files = helperGetRootFolders(roots)
  
  if (saveOrig):
    coronal_count, axial_count, saggital_count = 0, 0, 0
    for key, items in mri_files.items():
      _, canonical_data = readMRI(imagePath=items['orig'], config=configMRI)
      shape = canonical_data.shape
      coronal_count += shape[0]
      axial_count += shape[1]
      saggital_count += shape[2]

      print(f"\n[+] Saving slices for {key} shape: {shape}")
      
      saveAllSlices(key, canonical_data, items['root']+'/slices')

    print(f"Total slices per coronal view: ", coronal_count)
    print(f"Total slices per axial view: ", axial_count)
    print(f"Total slices per saggital view: ", saggital_count)
    
    print(f"\n Total images: {coronal_count+axial_count+saggital_count}")

  # Coronal images: 18094
  # Axial images:   25705
  # Saggital images: 25781

  # Total: 69580

def get_train_test_dirs(roots:list, view:str, structure, prefix_path:str='data/', train_percentage:float=.8) -> tuple:
  """
  
  """
  
  final_roots = []

  for root in roots:
      for dir in os.listdir(root):
          final_roots.append((root + '/' + dir + '/slices/' + view, root + '/' + dir + '/segSlices/' + structure + '/' + view))
  # images = glob(DATA_PATH)
  final_roots = shuffle(final_roots, random_state=12)

  data_size = len(final_roots)
  train_size = math.ceil(data_size*train_percentage)
  test_size = data_size - train_size

  train_dirs = final_roots[:train_size]
  test_dirs = final_roots[train_size:]

  return train_dirs, test_dirs

def helper_create_symlinks(list_dir:list, type_folder:str, dataset_root:str, view:str, copy_files:bool):
  """
  
  """
  
  def create_symlink(src, dest):
    try:
        os.symlink(src, dest)
    except OSError as e:
      # print(err)
      if (e.errno == errno.EEXIST):
        os.remove(dest)
        os.symlink(src, dest)
      else:
        raise e

  def copy_file(src, dest):
    shutil.copyfile(src, dest)

  print("[+] Creating dataset, please wait.")
  
  for origView, segView in list_dir:
    # 'data/NKI-RS-22/NKI-RS-22-13/slices/axial'
    # origImages = glob(origView + '/*')

    # 'data/NKI-RS-22/NKI-RS-22-13/segSlices/left-cerebellum-white-matter/axial'
    segImages = glob(segView + '/*')

    for mask_src in segImages:
      isMask = check_mask_img(mask_src)

      if (isMask):
        segImageSplitted = mask_src.split('/')
        orig_src = '' + mask_src
        orig_src = orig_src.replace('segSlices/', 'slices')
        orig_src = orig_src.replace('left-cerebellum-white-matter', '')

        mask_dest = f"{dataset_root}{type_folder}/{view}/{segImageSplitted[-3]}/img/" + segImageSplitted[-1]
        orig_dest = f"{dataset_root}{type_folder}/{view}/orig/img/" + segImageSplitted[-1]

        # print(mask_src, " is mask!\n Mask destination: ", mask_dest, "\n\nOrig src: ", orig_src, '\nOrig dest: ', orig_dest)
        # link_name = dataset_root + f"{type_folder}/{view}/{segImageSplitted[-3]}/img/" + segImageSplitted[-1]
      
        if (copy_files):
          copy_file(mask_src, mask_dest)
          copy_file(orig_src, orig_dest)
        else:
          create_symlink(mask_src, mask_dest)
          create_symlink(orig_src, orig_dest)

  message = 'Symlinks'

  if (copy_files):
    message = 'Files'

  print(f"[+] {message} created for {type_folder} folder.")

def creating_symlinks_to_dataset(roots:list, dataset_root:str, structures:list, view:str, copy_files:bool=False) -> None:
  """
  
  """
  
  # Creating train and test directories with their structures
  
  for folder in ['train', 'test']:
    create_folder(dataset_root + folder + '/' + view + '/orig/img')
    
    for structure in structures:
      create_folder(dataset_root + folder + '/' + view + '/' + structure + '/img')

  for structure in structures:
    train_dirs, test_dirs = get_train_test_dirs(roots=roots, view=view, structure=structure)
    # print(train_dirs)
    helper_create_symlinks(train_dirs, 'train', dataset_root, view, copy_files)
    helper_create_symlinks(test_dirs, 'test', dataset_root, view, copy_files)

def elastic_deform_2(img, msk):
  """
  
  """
  # img = rotate(img, angle=90)
  # msk = rotate(msk, angle=90)
  img = np.reshape(img, (256, 256))
  msk = np.reshape(msk, (256, 256))
  print(img.shape, "  ", msk.shape)
  fig, axs = plt.subplots(3, 2, figsize=(20, 20))
  # img_deformed, msk_deformed = elasticdeform.deform_random_grid([img, msk], sigma=7, points=3)

  displacement = np.random.randn(2, 3, 3) * 7
  img_deformed = elasticdeform.deform_grid(img, displacement=displacement)
  msk_deformed = elasticdeform.deform_grid(msk, displacement=displacement)

  axs[0][0].imshow(img, cmap='bone')
  axs[0][1].imshow(img_deformed, cmap='bone')

  axs[1][0].imshow(msk, cmap='bone')
  axs[1][1].imshow(msk_deformed, cmap='bone')

  slice_roi = np.ma.masked_where(msk == 0, msk)
  slice_roi_deformed = np.ma.masked_where(msk_deformed == 0, msk_deformed)
  
  # plt.imshow(first_roi_img, cmap='bone')
  axs[2][0].imshow(img, cmap='bone', interpolation='none')
  axs[2][0].imshow(slice_roi, alpha=0.5, interpolation='none')#, cmap='Oranges')

  axs[2][1].imshow(img_deformed, cmap='bone', interpolation='none')
  axs[2][1].imshow(slice_roi_deformed, alpha=0.5, interpolation='none')#, cmap='Oranges')

  fig.suptitle('Original imgs & masks VS Deformed imgs & masks', fontsize=16)
  print(np.expand_dims(img, axis=2).shape, "  ", np.expand_dims(msk, axis=2).shape)
  plt.show()

def elastic_deform(image_data):
  height, width, channels = image_data.shape
  depth = 0
  print("Ver: ", image_data.shape)

  nimages = 3
  print(f"The image object has the following dimensions: height: {height}, width:{width}, depth:{depth}, channels:{channels}")
  fig, axs = plt.subplots(nimages, 2, figsize=(20, 10))
  channel = 120

  # elastic deformation
  X = np.zeros((200, 300))
  X[::10, ::10] = 1

  for row in range(nimages):
      # apply deformation with a random 2 x 2 grid
      X_deformed = elasticdeform.deform_random_grid(image_data[:, channel, :], sigma=7, points=3)
      
      axs[row, 0].imshow(rotate(image_data[:, channel, :], angle=90), cmap='bone')
      axs[row, 1].imshow(rotate(X_deformed, angle=90), cmap='bone')
      channel = channel + 1


  # axs[0].imshow(image_data[:, :, channel], cmap='bone')
  # axs[1].imshow(image_data[:, :, channel + 1], cmap='bone')

  # axs[0].imshow(image_data[:, :, channel], cmap='bone')
  # axs[1].imshow(X_deformed, cmap='bone')

  plt.tight_layout()
  plt.grid(False)
  plt.axis('off')
  plt.show()

def check_imgs(path, ext):
  path = Path(path).rglob("*." + ext)
  for img_p in path:
      try:
          img = PIL.Image.open(img_p)
      except PIL.UnidentifiedImageError:
              print(img_p)

def check_mask_img(path):
  if os.path.exists(path):
    img = ImageOps.grayscale(Image.open(path))
    arr = np.asarray(img)
    res = arr.nonzero()

    if (res[0].shape[0] or res[1].shape[0]):
      print(arr[res]/255)
      plt.imshow(img)
      plt.show()
      return True

  return False