from model.config import *
from model.model import *
# import segmentation_models as sm
# sm.set_framework('tf.keras')
import pickle
import matplotlib.pyplot as plt
# from utils import utils
from glob import glob
from utils import utils
import os
import numpy as np
from matplotlib.colors import ListedColormap

def plot_model_training_info(model_history, dest_path):
    loss = model_history['loss']
    val_loss = model_history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{dest_path}/loss.png")
    plt.show()

    iou = model_history['iou_score']
    val_iou = model_history['val_iou_score']

    plt.plot(epochs, iou, 'y', label='Training IOU')
    plt.plot(epochs, val_iou, 'r', label='Validation IOU')
    plt.title('Training and validation IOU')
    plt.xlabel('Epochs')
    plt.ylabel('IOU')
    plt.legend()
    plt.savefig(f"{dest_path}/iou_score.png")
    plt.show()

    f1 = model_history['f1-score']
    val_f1 = model_history['val_f1-score']

    plt.plot(epochs, f1, 'y', label='Training F1-Score')
    plt.plot(epochs, val_f1, 'r', label='Validation F1-Score')
    plt.title('Training and validation F1-Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.savefig(f"{dest_path}/f1_score.png")
    plt.show()

def read_history(model_history_path):
    model_history = None
    with open(model_history_path, "rb") as file:
        model_history = pickle.load(file)
    
    return model_history

def read_patches_filename(filename, path):
    patches = []
    for file in glob(path):
        if (filename in file):
            patches.append(np.load(file))

            if (len(patches) >= 64):
                break
    
    patches = np.array(patches)
    print("Size: ", patches.shape)
    return patches

def plot_examples(msk_patches, prediction, idx, dest_path, name):
    palette = np.array([[  0,   0,   0],   # black
                    [255,   0,   0],   # red
                    [  0, 255,   0],   # green
                    [  0,   0, 255]])   # blue
                    # [255, 255, 255]])  # white
    RGB_ground = palette[msk_patches[idx, :, 45, :]]
    RGB_prediction = palette[prediction[idx, :, 45, :]]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Original mask VS Predicted")
    ax1.imshow(RGB_ground)
    ax1.set_title(f"msk_patch[{idx}, :, 45, :]")
    ax2.imshow(RGB_prediction)
    ax2.set_title(f"prediction[{idx}, :, 45, :]")
    print(dest_path)
    plt.savefig(f"{dest_path}/example_prediction_{idx}_{name}.png")
    # plt.show()

def test_models(training_folder):
    prediction = np.load(training_folder+'/prediction.npy')
    msk_patches = np.load(training_folder+'/ground_truth.npy')
    
    for idx in range(64):
        print(idx)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle("Original mask VS Predicted")
        ax1.imshow(msk_patches[idx, :, 45, :])
        ax1.set_title(f"msk_patch[{idx}, :, 45, :]")
        ax2.imshow(prediction[idx, :, 45, :])
        ax2.set_title(f"prediction[{idx}, :, 45, :]")
        # plt.imshow(prediction[idx, :, 45, :])
        plt.show()

def plot_predicted(msk_patches, prediction, idx, idx2, dest_path, name):
    palette = np.array([[  0,   0,   0],   # black
                    [255,   0,   0],   # red
                    [  0, 255,   0],   # green
                    [  0,   0, 255]])   # blue
                    # [255, 255, 255]])  # white
    RGB_ground = palette[msk_patches[idx, :, idx2, :]]
    RGB_prediction = palette[prediction[idx, :, idx2, :]]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Original mask VS Predicted")
    # ax1.imshow(msk_patches[idx, :, idx2, :], cmap=cmap)
    ax1.imshow(RGB_ground)
    ax1.set_title(f"msk_patch[{idx}, :, {idx2}, :]")
    print("unique values here: ", np.unique(prediction[idx, :, idx2, :]))
    # ax2.imshow(prediction[idx, :, idx2, :], cmap=cmap)
    ax2.imshow(RGB_prediction)
    ax2.set_title(f"prediction[{idx}, :, {idx2}, :]")
    print(dest_path)
    plt.savefig(f"{dest_path}/test_example_prediction_{idx}_{idx2}_{name}.png")
    plt.show()

def main():

    # path_prediction = 'trainings/version_8_0_2paths/insights/prediction_0.npy'
    # truth_path = 'trainings/version_8_0_2paths/ground_truth.npy'
    # dest_path = 'trainings/version_8_0_2paths'

    # arr_predicted = np.load(path_prediction)
    # ground_truth = np.load(truth_path)

    # plot_predicted(ground_truth, arr_predicted, 14, 36, dest_path, 'test_label')
    # exit()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

    training_folder = 'trainings/version_31'
    utils.create_folder(f"{training_folder}/insights")
    
    model_path = f"{training_folder}/model_trained_architecture.hdf5"
    model_history_path = f"{training_folder}/history.obj"
    config = get_config_test()
    config.dataset_path = 'dataset_3D_p64/'
    # Getting images
    test_filename = 'HLN-12-12'
    # Use 5 and 6 for idx
    
    img_patches = read_patches_filename(
        test_filename, 
        f"{config.dataset_path}test/images/*"
    )
    # img_patches = np.load(f"{config.dataset_path}test/images/{test_filename}")

    msk_patches = np.argmax(read_patches_filename(
        test_filename, 
        f"{config.dataset_path}test/masks/*"
    ), axis=4)
    # msk_patches = np.argmax(np.load(f"{config.dataset_path}test/masks/{test_filename}"), axis=4)

    model = test_model_3(config)
    # print(f"[+] Building model with config {config}")
    np.save(training_folder+"/ground_truth.npy", msk_patches)

    x = 0
    # for i in [
    #     '/model_trained_architecture.hdf5',
    #     '/checkpoints/model_trained_10_0.70.hdf5',
    #     '/checkpoints_2/model_trained_10_0.73.hdf5',
    #     '/checkpoints_3/model_trained_20_0.74.hdf5',
    # ]:
    deep_folder = '/insights'
    model_paths = [
        f"{training_folder}/model_trained_architecture.hdf5",
    ] + glob(training_folder + '/checkpoints/*')

    for i in model_paths:
        print("Loading model in ", i)
        model.load_weights(i)
        # model_history = read_history(model_history_path)
        # plot_model_training_info(model_history, training_folder)

        # if (x != 0):
        #     deep_folder = '/' + i.split('/')[1]

        prediction = model.predict(img_patches)
        prediction = np.argmax(prediction, axis=4)
        
        name = training_folder + deep_folder + f"/prediction_{x}.npy"
        print("Saving prediction ", name)
        np.save(name, prediction)
        
        for id in [31, 19, 15, 14, 5]:
            plot_examples(msk_patches, prediction, id, training_folder + deep_folder, x)
        
        x += 1

if __name__ == "__main__":
    main()