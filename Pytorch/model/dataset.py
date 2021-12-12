import os
from torch.utils.data import Dataset
import torch
import numpy as np
from glob import glob
from tensorflow.keras.utils import to_categorical

class Mindboggle_101(Dataset):
    def __init__(self, dataset_path, image_path, mask_path, limit, transform=None):
        super(Mindboggle_101, self).__init__()

        # Dataset paths
        self.dataset_path = dataset_path
        self.image_path = image_path
        self.mask_path = mask_path

        # Transformations over data
        self.transform = transform
        
        # Filenames for dataset
        self.image_filenames = None
        self.mask_filenames = None

        # Attributes for generating dataset
        self.next_file = True
        # internat len until LIMIT
        self.limit = limit
        self.internal_idx = 0
        self.batch_len = 0

        # Current image and mask for memmap numpy array
        self.current_img = None
        self.current_msk = None

        self.__set_filepaths()

    def __getitem__(self, index):
        # print("Batch len: ", index, " batch len own ", self.batch_len, " internal len: ", self.internal_idx)

        if (self.next_file):
            # print("Loaded ", self.image_filenames[self.batch_len])
            img_path = self.image_filenames[self.batch_len]
            msk_path = self.mask_filenames[self.batch_len]
            self.next_file = False
            
            if self.transform:
                self.current_img = np.squeeze(np.load(img_path, mmap_mode='r').astype(np.float32), axis=-1)
                self.current_msk = np.argmax(np.load(msk_path, mmap_mode='r'), axis=4).astype(np.uint8)
            else:
                self.current_img = np.load(img_path, mmap_mode='r').astype(np.float32)
                self.current_msk = np.load(msk_path, mmap_mode='r').astype(np.uint8)
            # print("image: ", self.current_img.shape)
            # print("msk: ", self.current_msk.shape)

        sample = {
            'image': self.current_img[self.internal_idx, :],#np.expand_dims, axis=0), 
            'mask': self.current_msk[self.internal_idx, :]#np.expand_dims, axis=0)
        }

        self.internal_idx += 1

        if self.internal_idx == self.limit:
            # print("Restart")
            self.internal_idx = 0
            self.batch_len += 1
            self.next_file = True

        if self.transform:
            transformed_samples = self.transform([sample['image'], sample['mask']])

            # print("TRANS: ", transformed_samples[0].shape, " msk ", transformed_samples[1].shape)
            sample['image'] = np.expand_dims(transformed_samples[0], axis=-1)
            sample['mask'] = to_categorical(transformed_samples[1], num_classes=38)#np.squeeze(sample['mask'], axis=0)
            # print("AFTER: ", sample['image'].shape, " msk ", sample['mask'].shape)
        
        sample['image'] = torch.from_numpy(sample['image'].copy().astype(np.float32))
        sample['mask'] = torch.from_numpy(sample['mask'].copy().astype(np.uint8))

        return sample
        
    def __set_filepaths(self):
        self.image_filenames = sorted(glob(self.dataset_path + self.image_path + '*'))
        self.mask_filenames = sorted(glob(self.dataset_path + self.mask_path + '*'))

    def __len__(self):
        return len(self.image_filenames)*self.limit

# def collate_function(batch):

#     images = [item for item in batch[0]['image']]
#     masks = [item for item in batch[0]['label']]
#     # for data in batch:

#     # for idx in range(data['image'].shape[0]):
#     #     print("idx: ", idx, " ", data['image'][idx].shape, " ", data['label'][idx].shape)
#     #     return data['image'][idx], data['label'][idx]

#     # batch = {'image': data['image'], 'label': data['label']}
#     return [images, masks]

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from torchvision import transforms
    # import torchio as tio
    from augmend import Augmend, Elastic, FlipRot90

    dataset_path = '../../dataset_3D_37/'
    image_path = 'train/images/'
    mask_path = 'train/masks/'

    aug = Augmend()
    aug.add([
        Elastic(axis=(0, 1, 2), amount=5, order=1, use_gpu=False),
        Elastic(axis=(0, 1, 2), amount=5, order=0, use_gpu=False),
    ], probability=1)

    # file = np.load(dataset_path + image_path + 'MMRR-21-20_patched.npy', mmap_mode='r').astype(np.uint8)

    # n = 5
    # max_displacement = n, n, n
    # num_control_points = 7

    # transform = tio.Compose([
    #     tio.RandomElasticDeformation(
    #         # max_displacement=max_displacement,
    #         # num_control_points=num_control_points,
    #         include=['image', 'mask']
    #     ),
    #     # tio.OneOf({                                # either
    #     #     tio.RandomAffine(): 0.8,               # random affine
    #     #     tio.RandomElasticDeformation(): 0.2,   # or random elastic deformation
    #     # }, p=0.8)
    # ])

    transform = transforms.Compose([
        aug
    ])

    mindboggle_101_train_original = Mindboggle_101(
        dataset_path=dataset_path,
        image_path=image_path,
        mask_path=mask_path,
        limit=27,
        transform=None
    )

    mindboggle_101_train_transformed = Mindboggle_101(
        dataset_path=dataset_path,
        image_path=image_path,
        mask_path=mask_path,
        limit=27,
        transform=transform
    )

    mindboggle_101_aug = torch.utils.data.ConcatDataset([
        mindboggle_101_train_original,
        mindboggle_101_train_transformed,
    ])

    dataloader = DataLoader(mindboggle_101_aug, batch_size=8, shuffle=True, num_workers=1)
    
    for i, sample in enumerate(dataloader):
        # print(len(sample[0]), " ", len(sample[1]))
        image, mask = sample['image'], sample['mask']
        print(i, " image shape: ", image.shape, " mask shape: ", mask.shape)
        plt.imshow(image[1][0], cmap='gray')
        plt.show()
        # exit()