from model.network import BrainSegmentationNetwork
from model.config import get_config
import torch
from model.dataset import Mindboggle_101
import pickle
import matplotlib.pyplot as plt
# from utils import utils
from glob import glob
from utils import utils
import os
import numpy as np
import argparse
from sklearn.metrics import classification_report
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1
from time import sleep

def plot_examples(msk_patches, prediction, idx, dest_path, name):
    palette = np.array([
                        [  0,   0,   0],   # black
                        [245,   245,   245],   # cerebral-white-matter
                        [245,   245,   245],
                        [220, 248,   164],   # cerebellum-white-matter
                        [220, 248,   164],   # cerebellum-white-matter
                        [230, 148, 34],      # Cerebellum-cortex,
                        [230, 148, 34],      # Cerebellum-cortex,
                        [120, 18,  134], # lateral-ventricle
                        [120, 18,  134], # lateral-ventricle
                        [196, 58,  250], #int-lat-vent
                        [196, 58,  250], #int-lat-vent
                        [0,   118, 14], # thalamus
                        [0,   118, 14], # thalamus
                        [122, 186, 220], # caudate
                        [122, 186, 220], # caudate
                        [236, 13,  176], # putamen
                        [236, 13,  176], # putamen
                        [12,  48,  255], # pallidum
                        [12,  48,  255], # pallidum
                        [204, 182, 142], # 3rd-ventricle
                        [42,  204, 164], # 4th-ventricle
                        [119, 159, 176], # brain-stem
                        [220, 216, 20], # hippocampus
                        [220, 216, 20], # hippocampus
                        [103, 255, 255], # amygdala
                        [103, 255, 255], # amygdala
                        [60,  60,  60], # csf
                        [255, 165, 0], # accumbens-area
                        [255, 165, 0], # accumbens-area
                        [165, 42,  42], # ventraldc
                        [165, 42,  42], # ventraldc
                        [0,   200, 200], # choroid plexus
                        [0,   200, 200], # choroid plexus
                        [0,   0,   64], # cc_posterior
                        [0,   0,   112], # cc_mid_posterior
                        [0,   0,  160], # cc_central,
                        [0,  0,   208], # cc_mid_anterior
                        [0,   0,   255], # cc_anterior
                    ])
    # print(msk_patches.shape, " ", prediction.shape)
    for w in [0, 1]:
        RGB_ground = palette[msk_patches[w, idx, :, :]]
        RGB_prediction = palette[prediction[w, idx, :, :]]

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle("Original mask VS Predicted")
        ax1.imshow(RGB_ground)
        ax1.set_title(f"msk_patch[{idx}, :, :]")
        ax2.imshow(RGB_prediction)
        ax2.set_title(f"prediction[{idx}, :, :]")
        # print(dest_path)
        plt.savefig(f"{dest_path}/example_prediction_{w}_{idx}_{name}.png")
    # plt.show()

def main():

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--folder_name', metavar='folder', type=str,
                        help='Insert the folder for insights')
    args = vars(parser.parse_args())

    STRUCTURES = utils.read_test_to_list('../data/common_anatomical_structures.txt')
    STRUCTURES.insert(0, 'background')
    print(STRUCTURES, " ", len(STRUCTURES))

    training_folder = 'trainings/' + args['folder_name']
    utils.create_folder(f"{training_folder}/insights")
    
    model_path = f"{training_folder}/model_trained_architecture.pt"
    config = get_config()#get_config_test()
    
    model = BrainSegmentationNetwork()
    model.load_state_dict(torch.load(model_path))
    device = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(model)

    model.eval()
    
    # Loading testing dataset
    test_image_path = 'test/images/'
    test_mask_path = 'test/masks/'

    num_workers = 2 # os.cpu_count()

    mindboggle_101_test = Mindboggle_101(
        dataset_path=config.dataset_path,
        image_path=test_image_path,
        mask_path=test_mask_path,
        limit=27,
        transform=None
    )

    test_dataloader = DataLoader(mindboggle_101_test, batch_size=27, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # Metrics
    average = 'weighted'
    metric_collection = MetricCollection([
        Accuracy().to(device),
        F1(num_classes=config.n_classes, average=average, mdmc_average='global').to(device),
        Precision(num_classes=config.n_classes, average=average, mdmc_average='global').to(device),
        Recall(num_classes=config.n_classes, average=average, mdmc_average='global').to(device),
    ])

    f1 = []
    accuracy = []
    precision = []
    recall = []

    with torch.no_grad():
        with tqdm(test_dataloader, unit='batch', position=0, leave=True) as tbatch:
            for i, data in enumerate(tbatch):
                image, mask = data['image'].to(device), data['mask'].to(device)
                pred = model(image)

                pred_argmax = torch.argmax(pred, dim=1)
                mask_argmax = torch.argmax(mask, dim=1)
                
                metrics = metric_collection(
                    pred_argmax, 
                    mask_argmax
                )

                accuracy.append(metrics['Accuracy'].item())
                f1.append(metrics['F1'].item())
                recall.append(metrics['Recall'].item())
                precision.append(metrics['Precision'].item())

                if i == 0:
                    pred_numpy = pred_argmax.numpy()
                    mask_numpy = mask_argmax.numpy()
                    print("Unique: ", np.unique(pred_numpy), " ", pred_numpy.shape)
                    name = training_folder + f"/prediction_{i}.npy"
                    print("Saving prediction ", name)
                    np.save(name, pred_numpy)
                    report = classification_report(mask_numpy.flatten(), pred_numpy.flatten(), target_names=STRUCTURES)
                    utils.classification_report_csv(report, training_folder, f"/{len(STRUCTURES)}_structures_{i}")
                    for id in [50, 58]:
                        plot_examples(mask_numpy, pred_numpy, id, training_folder, i)

                tbatch.set_description("Training")
                tbatch.set_postfix({
                    'MRI Idx': f"{i+1}",
                    'Accuracy': np.mean(accuracy),
                    'F1': np.mean(f1),
                    'Recall': np.mean(recall),
                    'Precision': np.mean(precision),
                })
                tbatch.update()
                sleep(0.01)

if __name__ == "__main__":
    main()