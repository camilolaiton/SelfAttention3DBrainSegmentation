from model.config import *
from model.model_2 import build_model
from glob import glob
from utils import utils
import os
import numpy as np
import argparse
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
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

# def create_excel(src_folder, dest_folder):
#     dict_users = {"class": []}

#     writer = pd.ExcelWriter(dest_folder + '.xlsx', engine='xlsxwriter')

#     for folder in glob.glob(src_folder + '/*'):
#         name = folder.split('/')[-1]
#         aseg_file = [file_ for file_ in glob.glob(f"{folder}/mri/*") if 'aseg.stats' in file_]

#         if (len(aseg_file)):
#             print(f"Creating excel - Reading {name}")
#             aseg_file = aseg_file[0]

#             dict_users["users"].append(name)
#             dictionary = read_aseg_file(aseg_file, save_local=True)
#             if (not read_struct_cols):
#                 for structname in dictionary['StructName']:
#                     dict_users[structname] = []
#                 read_struct_cols = 1

#             for i in range(len(dictionary['StructName'])):
#                 dict_users[dictionary['StructName'][i]].append(dictionary['Volume_mm3'][i])

#     pd.DataFrame(dict_users).to_csv(dest_folder + '.csv', header=True,index=False)
#     pd.DataFrame(dict_users).to_excel(dest_folder + '.xlsx', header=True,index=False)

def helper_fnc(structure, dictionary, column, info):
    if structure in dictionary:
        dictionary[structure].append(info[column][structure])
    else:
        dictionary[structure] = [info[column][structure]]

    return dictionary

def creating_excel(path):
    STRUCTURES = utils.read_test_to_list('data/common_anatomical_structures.txt')
    STRUCTURES.insert(0, 'background')
    print(STRUCTURES, " ", len(STRUCTURES))

    precision = {'name': []}
    recall = {'name': []}
    f1_score = {'name': []}
    support = {'name': []}

    for folder in glob(path + '/*structures*'):
        info = pd.read_excel(folder, index_col='class')
        
        name = folder.split('/')[-1].split('.')[0].split('_')[0]
        
        precision['name'].append(name)
        recall['name'].append(name)
        f1_score['name'].append(name)
        support['name'].append(name)


        for structure in STRUCTURES:
            precision = helper_fnc(structure, precision, 'precision', info)
            recall = helper_fnc(structure, recall, 'recall', info)
            f1_score = helper_fnc(structure, f1_score, 'f1_score', info)
            support = helper_fnc(structure, support, 'support', info)

    writer = pd.ExcelWriter(path + '/collection_pre_re_f1_spport.xlsx', engine='xlsxwriter')
    pd.DataFrame(precision).to_excel(
        writer, 
        header=True,
        index=False,
        sheet_name='precision'
    )
    pd.DataFrame(recall).to_excel(
        writer, 
        header=True,
        index=False,
        sheet_name='recall'
    )
    pd.DataFrame(f1_score).to_excel(
        writer, 
        header=True,
        index=False,
        sheet_name='f1_score'
    )
    pd.DataFrame(support).to_excel(
        writer, 
        header=True,
        index=False,
        sheet_name='support'
    )
    writer.save()

def main():

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--folder_name', metavar='folder', type=str,
                        help='Insert the folder for insights')
    args = vars(parser.parse_args())

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

    # STRUCTURES = utils.read_test_to_list('data/common_anatomical_structures.txt')
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
    model_path = f"{training_folder}/model_trained_architecture.hdf5"
    model.load_weights(model_path)
    # times = {}

    # writer = pd.ExcelWriter(training_folder + deep_folder + f"/report_test.xlsx", engine='xlsxwriter')
    msk_imgs = []
    pred_imgs = []
    idx_limit = 2
    for idx in range(len(image_list_test)):
        if idx_limit == idx:
            break

        print(f"[{idx}] Image path: {image_list_test[idx]} test path: {mask_list_test[idx]}")
        filename = image_list_test[idx].split('/')[-1].split('.')[0]
        filename = filename.replace('images\\', '')
        img_patches = np.load(image_list_test[idx])
        msk_patches = np.load(mask_list_test[idx])
        msk_imgs.append(msk_patches[:,:,:,:,1])

        # start_time = time.time()
        
        print(f"[{idx}] Starting prediction for {filename}")
        prediction = model.predict(img_patches)
        # end_time = time.time()
        # final_time = (end_time-start_time)/60
        # times[filename] = final_time
        # print(f"[{idx}] Finished prediction for {filename} in {final_time} minutes")

        # prediction = np.argmax(prediction, axis=4)
        print(f"[{idx}] Unique: ", np.unique(prediction))
        pred_imgs.append(prediction[:,:,:,:,1])
        
        # report = classification_report(msk_patches.flatten(), prediction.flatten(), target_names=STRUCTURES)
        # utils.classification_report_csv(report, training_folder + deep_folder, filename, sheets=True, writer=writer)

    msk_imgs = np.asarray(msk_imgs).flatten()
    pred_imgs = np.asarray(pred_imgs).flatten()
    fpr, tpr, thresholds = roc_curve(msk_imgs, pred_imgs)

    fig, c_ax = plt.subplots(1,1, figsize = (12, 8))
    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % ("class_1", auc(fpr, tpr)))
    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    plt.show()
    plt.savefig(training_folder + deep_folder + f"roc_curve_.png", dpi=100, bbox_inches='tight')
    # writer.save()
    # utils.write_dict_to_txt(times, training_folder + deep_folder + '/times.txt')

if __name__ == "__main__":
    main()
    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--folder_name', metavar='folder', type=str,
    #                     help='Insert the folder for insights')
    # args = vars(parser.parse_args())

    # training_folder = 'trainings/' + args['folder_name']

    # creating_excel(training_folder + '/reports')