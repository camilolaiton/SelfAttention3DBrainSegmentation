import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../')

import argparse
from utils import utils
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from augmend import Augmend, Elastic, FlipRot90
from model.config import get_config
from model.dataset import Mindboggle_101
# from torch.utils.tensorboard import SummaryWriter
# from datetime import datetime
import torch.optim as optim
from model.network import BrainSegmentationNetwork

def defining_augmentations():
    aug = Augmend()
    aug.add([
        Elastic(axis=(0, 1, 2), amount=5, order=1, use_gpu=False),
        Elastic(axis=(0, 1, 2), amount=5, order=0, use_gpu=False),
    ], probability=1)
    
    return aug

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters())

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--retrain', metavar='retr', type=int,
                        help='Retrain architecture', default=0)

    parser.add_argument('--folder_name', metavar='folder', type=str,
                        help='Insert the folder for insights')

    parser.add_argument('--lr_epoch_start', metavar='lr_decrease', type=int,
                        help='Start epoch lr decrease', default=10)
    args = vars(parser.parse_args())

    retrain = args['retrain']
    training_folder = 'trainings/' + args['folder_name']

    model_path = f"{training_folder}/model_trained_architecture.pt"

    # Creating folders for saving trainings
    utils.create_folder(f"{training_folder}/checkpoints")

    # getting training config
    config = get_config()

    transform = transforms.Compose([
        defining_augmentations()
    ])

    # Loading train dataset -> Original + augmented
    train_image_path = 'train/images/'
    train_mask_path = 'train/masks/'

    mindboggle_101_train_original = Mindboggle_101(
        dataset_path=config.dataset_path,
        image_path=train_image_path,
        mask_path=train_mask_path,
        limit=27,
        transform=None
    )

    mindboggle_101_train_transformed = Mindboggle_101(
        dataset_path=config.dataset_path,
        image_path=train_image_path,
        mask_path=train_mask_path,
        limit=27,
        transform=transform
    )

    mindboggle_101_aug = torch.utils.data.ConcatDataset([
        mindboggle_101_train_original,
        mindboggle_101_train_transformed,
    ])

    # Loading testing dataset
    test_image_path = 'test/images/'
    test_mask_path = 'test/masks/'

    mindboggle_101_test = Mindboggle_101(
        dataset_path=config.dataset_path,
        image_path=test_image_path,
        mask_path=test_mask_path,
        limit=27,
        transform=None
    )

    # Creating dataloaders
    num_workers = 2 # os.cpu_count()
    train_dataloader = DataLoader(mindboggle_101_aug, batch_size=8, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(mindboggle_101_test, batch_size=8, shuffle=False, num_workers=num_workers)
    
    # Loading the model
    model = BrainSegmentationNetwork()

    # Loading device
    device = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.device_count() > 1:
        print("[INFO] Using {} GPUs!" % torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    print("[INFO] Device: ", device)

    model.to(device)
    
    print(model)

    trainable_params, total_params = count_params(model)
    print("[INFO] Trainable params: ", trainable_params, " total params: ", total_params)

    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=config.weight_decay, 
        amsgrad=False 
    )

    for epoch in range(0, config.num_epochs):
        running_loss = 0.0

        for i, data in enumerate(train_dataloader):
            # Getting the data
            image, mask = data['image'].to(device), torch.tensor(data['mask'].detach().clone(), dtype=torch.long).to(device)

            # forward + backward + optimize
            output = model(image)
            loss = loss_fn(output, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % 10 == 0:
                print("Loss: ", (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0



if __name__ == "__main__":
    main()