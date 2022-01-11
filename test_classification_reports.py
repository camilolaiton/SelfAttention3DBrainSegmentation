from model.config import *
from model.model_2 import build_model
import tensorflow as tf
from glob import glob
from utils import utils
import os
import numpy as np
import argparse
from sklearn.metrics import classification_report
import time
import pandas as pd

def read_patches_filename(filename, path):
    patches = []
    for file in glob(path):
        if (filename in file):
            patches.append(np.load(file))

            if (len(patches) >= 64):
                break
    
    patches = np.array(patches)
    return patches

def main():

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--folder_name', metavar='folder', type=str,
                        help='Insert the folder for insights')
    args = vars(parser.parse_args())

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

    STRUCTURES = utils.read_test_to_list('data/common_anatomical_structures.txt')
    STRUCTURES.insert(0, 'background')
    print(STRUCTURES, " ", len(STRUCTURES))

    training_folder = 'trainings/' + args['folder_name']
    utils.create_folder(f"{training_folder}/reports")
    deep_folder = '/reports'

    config = get_config_local_path()#get_config_test()
    
    # Getting images
    image_list_test = sorted(glob(
        config.dataset_path + 'test/images/*'))
    mask_list_test = sorted(glob(
        config.dataset_path + 'test/masks/*'))
    
    model = build_model(config)
    model_path = f"{training_folder}/model_trained_architecture_2.hdf5"
    model.load_weights(model_path)
    times = {}

    writer = pd.ExcelWriter(training_folder + deep_folder + f"/report_test.xlsx", engine='xlsxwriter')

    for idx in range(len(image_list_test)):
        print(f"[{idx}] Image path: {image_list_test[idx]} test path: {mask_list_test[idx]}")
        filename = image_list_test[idx].split('/')[-1].split('.')[0]
        filename = filename.replace('images\\', '')
        img_patches = np.load(image_list_test[idx])
        msk_patches = np.argmax(np.load(mask_list_test[idx]), axis=4)

        start_time = time.time()
        
        print(f"[{idx}] Starting prediction for {filename}")
        prediction = model.predict(img_patches)
        end_time = time.time()
        final_time = (end_time-start_time)/60
        times[filename] = final_time
        print(f"[{idx}] Finished prediction for {filename} in {final_time} minutes")

        prediction = np.argmax(prediction, axis=4)
        print(f"[{idx}] Unique: ", np.unique(prediction))

        
        report = classification_report(msk_patches.flatten(), prediction.flatten(), target_names=STRUCTURES)
        utils.classification_report_csv(report, training_folder + deep_folder, filename, sheets=True, writer=writer)

    writer.save()
    utils.write_dict_to_txt(times, training_folder + deep_folder + '/times.txt')

if __name__ == "__main__":
    main()