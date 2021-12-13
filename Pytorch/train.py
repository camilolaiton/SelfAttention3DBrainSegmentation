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
from torch.utils.tensorboard import SummaryWriter
from model.losses import FocalDiceLoss#Dice_and_Focal_loss
from model.metrics import dice_coefficient
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1
import numpy as np
import glob
import time
from tqdm import tqdm
from time import sleep

# https://discuss.pytorch.org/t/combining-two-loss-functions-with-trainable-paramers/23343/3

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
    writer = SummaryWriter(training_folder)

    # getting training config
    config = get_config()

    # Setting up weights 
    weights = utils.read_test_to_list(config.dataset_path + 'weights.txt')
    
    div_factor = 100

    if (weights == False):
        end_path = '/train/masks'
        image_files = [file for file in glob.glob(config.dataset_path + end_path + '/*') if file.endswith('.npy')]
        print(f"Calculating weights for {config.n_classes}")
        weights, label_to_frequency_dict = utils.median_frequency_balancing(image_files=image_files, num_classes=config.n_classes)
        print("Resulting weights: ", weights)
        if (weights == False):
            print("Please check the path")
            exit()
        utils.write_list_to_txt(weights, config.dataset_path + 'weights.txt')
        print("Weights calculated")
    else:
        # weights = [0.0, 1, 2.7, 3]
        # weights = [0.0, 2.3499980585022096, 6.680915101433645, 7.439929426050408]
        print("Weights read!")

    weights = [float(weight)/div_factor for weight in weights]

    # For parallel data pytorch
    # torch.distributed.init_process_group(backend='nccl')

    gpu = 0

    torch.manual_seed(12)
    torch.cuda.manual_seed(12)
    # torch.cuda.set_device(gpu)

    # Loading the model
    model = BrainSegmentationNetwork()

    # Loading device
    device = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.device_count() > 1:
        print(f"[INFO] Using {torch.cuda.device_count()} GPUs!")
        # model = torch.nn.DistributedDataParallel(model, device_ids=[gpu], output_device=gpu, find_unused_parameters=True)
        model = torch.nn.DataParallel(model)

    print("[INFO] Device: ", device)

    model.to(device)
    torch_weights = torch.as_tensor(np.array(weights, dtype=np.float16)).to(device)

    print(model)

    trainable_params, total_params = count_params(model)
    print("[INFO] Trainable params: ", trainable_params, " total params: ", total_params)

    print(torch_weights)

    # Model training mode
    model.train()

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

    # mindboggle_101_test = Mindboggle_101(
    #     dataset_path=config.dataset_path,
    #     image_path=test_image_path,
    #     mask_path=test_mask_path,
    #     limit=27,
    #     transform=None
    # )

    # Creating dataloaders
    num_workers = 2 # os.cpu_count()
    train_dataloader = DataLoader(mindboggle_101_aug, batch_size=8, shuffle=True, num_workers=num_workers, pin_memory=True)
    # test_dataloader = DataLoader(mindboggle_101_test, batch_size=8, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    model.cuda(device)

    # Metrics
    metric_collection = MetricCollection([
        Accuracy().to(device),
        F1(num_classes=config.n_classes, average='macro').to(device),
        Precision(num_classes=config.n_classes, average='macro', mdmc_reduce='global').to(device),
        Recall(num_classes=config.n_classes, average='macro', mdmc_reduce='global').to(device),
    ])

    # Loss function
    loss_fn = FocalDiceLoss()#torch.nn.CrossEntropyLoss()#.cuda(gpu)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=config.weight_decay, 
        amsgrad=False 
    )

    best_loss = 100
    best_epoch = -1
    
    # For mixed precision
    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print("[INFO] Starting training!")
    
    for epoch in range(0, config.num_epochs):
        running_loss = 0.0
        
        end_i = 0
        start_time = time.time()
        
        with tqdm(train_dataloader, unit='batch', position=0, leave=True) as tbatch:
        
        # for i, data in enumerate(train_dataloader):
            for i, data in enumerate(tbatch):
                # Getting the data
                metrics = {}
                image, mask = data['image'].to(device), data['mask'].to(device)
                # image.cuda(device)
                # mask.cuda(device)

                with torch.cuda.amp.autocast():
                    # forward + backward + optimize
                    pred = model(image)
                    loss = loss_fn(pred, mask)#, torch_weights)
                    # loss_1 = dice_loss(pred, mask)
                    # loss_2 = focal_loss(pred, mask)
                    running_loss += loss
                    print("pred: ", pred.shape, " mask ", mask.shape)
                    metrics = metric_collection(
                        torch.argmax(pred, dim=1), 
                        torch.argmax(mask, dim=1)
                    )
                
                scaler.scale(loss).backward()
                # loss.backward(loss)
                # scaler.scale(loss_1).backward()
                # scaler.scale(loss_2).backward()

                # optimizer.step()
                scaler.step(optimizer)

                # Updates the scale for next iteration.
                scaler.update()

                # this reduces the number of memory operations.
                optimizer.zero_grad(set_to_none=True)
                
                # print(f"[Epoch {epoch}-{i}]: loss {loss}")
                
                tbatch.set_description("Training model")
                tbatch.set_postfix({
                    'Epoch': epoch, 
                    'Inner batch': i, 
                    'Loss': running_loss/i, 
                    'F1': metrics['F1'],
                    'Accuracy': metrics['Accuracy'],
                    'Precision': metrics['Precision'],
                    'Recall': metrics['Recall'],
                })
                tbatch.update()
                sleep(0.01)
                end_i = i
        
        end_time = time.time()
        epoch_time = (end_time-start_time)/60

        running_loss = running_loss / end_i
        print(f"{epoch} Loss: {running_loss} in {epoch_time} minutes")

        if (best_loss > running_loss):
            best_loss = running_loss
            best_epoch = epoch
            print(f"Saving best model in epoch {best_epoch} with loss {best_loss}")
            torch.save(model.state_dict(), training_folder+'best-model-parameters.pt')

        writer.add_scalar('LearningRate/train', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Loss/train', running_loss, epoch)

    writer.close()
        

if __name__ == "__main__":
    main()