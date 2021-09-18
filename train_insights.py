from model.config import *
from model.model import *
# import segmentation_models as sm
# sm.set_framework('tf.keras')
import pickle
import matplotlib.pyplot as plt
# from utils import utils
from glob import glob
import os

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

def plot_examples(msk_patches, prediction, idx, dest_path):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Original mask VS Predicted")
    ax1.imshow(msk_patches[idx, :, 45, :])
    ax1.set_title(f"msk_patch[{idx}, :, 45, :]")
    ax2.imshow(prediction[idx, :, 45, :])
    ax2.set_title(f"prediction[{idx}, :, 45, :]")
    plt.savefig(f"{dest_path}/example_prediction_{idx}.png")
    # plt.show()

def main():
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

    training_folder = 'trainings/version_1_0'
    model_path = f"{training_folder}/trained_architecture.hdf5"
    model_history_path = f"{training_folder}/history.obj"

    config = get_config_patchified()
    model = build_model_patchified(config)
    # print(f"[+] Building model with config {config}")
    
    model.load_weights(model_path)
    # model_history = read_history(model_history_path)
    # plot_model_training_info(model_history, training_folder)
    
    test_filename = 'HLN-12-1'
    # Use 5 and 6 for idx
    
    img_patches = read_patches_filename(
        test_filename, 
        f"{config.dataset_path}train/images/*"
    )

    msk_patches = np.argmax(read_patches_filename(
        test_filename, 
        f"{config.dataset_path}train/masks/*"
    ), axis=4)

    # print(msk_patches.shape)
    # for i in range(64):
    #     print(i)
    #     plt.imshow(msk_patches[i, :, 45, :])
    #     plt.show()

    prediction = model.predict(img_patches)
    prediction = np.argmax(prediction, axis=4)

    plot_examples(msk_patches, prediction, 5, training_folder)

if __name__ == "__main__":
    main()