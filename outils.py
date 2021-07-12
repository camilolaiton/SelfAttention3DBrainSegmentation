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

import numpy as np
import nibabel as nib

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patheffects as path_effects

from nilearn.plotting import plot_roi
from scipy.ndimage import rotate

import os
import time

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

def readMRI(imagePath:str, config:dict):
  """
    Reading MRI with different preprocessing steps

    Parameters:
      - imagePath: String that contains the path where the MRI is located 
      - config: Dictionary that contains configuration values for processing the MRI when it's read

    Returns:
      - nib.Nifti1Image, numpy.Array
  """
  
  imageObj = nib.load(imagePath)
  
  if (config['RAS']):
    imageObj = nib.as_closest_canonical(imageObj)
  
  imageData = imageObj.get_fdata()

  if (config['normalize']):
    imageData = normalizeIntensityImage(imageData, np.min(imageData), np.max(imageData))

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
      - orientations: list that specifies which views will be plotted. ['x', 'y', 'z', ...]

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

       axes[i].imshow(slice.T, cmap="gray", origin="lower", vmin=10, vmax=110)

def show_all_slices_per_view(view:str, data:np, counter:int=100, angle=270):
  """ 
    Function to display multiple slices consequently 

    Parameters:
      - view: String that contains the plotted view ['x', 'y', 'z']
      - data: MRI data
      - counter: Integer that dictates from where the slices with be plotted
    
    Returns:
      - None
   """

  if view in ['x', 'y', 'z']:
    x, y, z = data.shape

    view_limit = x
    if view == 'y':
      view_limit = y
    elif view == 'z':
      view_limit = z

    columns = 10
    rows = 10
    fig, axs = plt.subplots(columns, rows, figsize=(20, 20))

    for row in range(rows):
      for column in range(columns):
        if (view == 'x'):
          axs[row, column].imshow(data[counter, :, :], cmap='bone')
        elif (view == 'y'):
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
      - view: string that has the view from where the images will be taken. View could be ['x', 'y', 'z']
      - destination: string that has the path where the slice will be saved
    
    Returns:
      - None
  """
  x, y, z = image_data.shape
  
  if (view == 'y'):
    # coronal
    create_folder(f"{destination}/y")
    for idx in range(x):
      saveSlice(image_data=rotate(image_data[idx, :, :], angle=90), filename=f"{filename}_{idx}", destination=f"{destination}/y")

  elif (view == 'z'):
    # axial
    create_folder(f"{destination}/z")
    for idx in range(y):
      saveSlice(image_data=np.fliplr(rotate(image_data[:, idx, :], angle=90)), filename=f"{filename}_{idx}", destination=f"{destination}/z")

  else:
    # For x sagittal
    create_folder(f"{destination}/x")
    for idx in range(z):
      saveSlice(image_data=np.fliplr(rotate(image_data[:, :, idx], angle=90)), filename=f"{filename}_{idx}", destination=f"{destination}/x")

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
    for ori in ['x', 'y', 'z']:
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

def plotting_superposition(n_slice, brain_data, roi_data, orientation='x'):
  slice_orig, slice_roi = None, None
  
  if (orientation == 'x'):
    slice_orig = rotate(brain_data[n_slice, :, :], angle=90)
    slice_roi = roi_data[n_slice, :, :]
  
  elif (orientation == 'y'):
    slice_orig = np.fliplr(rotate(brain_data[:, n_slice, :], angle=90))
    slice_roi = roi_data[:, n_slice, :]
  
  else:
    slice_orig = np.fliplr(rotate(brain_data[:, :, n_slice], angle=90))
    slice_roi = roi_data[:, :, n_slice]

  plt.imshow(slice_orig, cmap='bone')
  # plt.imshow(slice_roi, alpha=0.5)
  plt.grid(False)
  plt.axis('off')
  plt.show()

def saveSlicesPerRoot(roots:list, configMRI:dict):
  print(roots)

  mri_files = {}
  MAX_DEPTH = 1

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
          # 'segmented': folder[0] + '/' + 'aparcNMMjt+aseg.mgz', 
        }
        if root.count(os.sep) - root.count(os.sep) == MAX_DEPTH - 1:
          del dirs[:]
          
  # print(mri_files)
  # print(len(mri_files))
  
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