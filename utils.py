import cv2
import tqdm
import glob
import torch
import shutil
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import settings

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def data_preparation(data_folder):
    """Read data from folder and create Pandas DataFrames for train/valid"""
    # Create DataFrame
    images_list = sorted(glob.glob(f'{data_folder}/train/*'))
    masks_list = sorted(glob.glob(f'{data_folder}/masks/*'))
    assert len(images_list) == len(masks_list)
    df = pd.DataFrame()
    df['images'] = images_list
    df['masks'] = masks_list
    # Split data on train/valid
    train, valid = train_test_split(
        df,
        train_size=.8,
        random_state=42,
        shuffle=True
    )
    return train, valid


class LocalDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.images_files = data['images'].tolist()
        self.masks_files = data['masks'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.images_files)

    def __getitem__(self, index):
        # Select on image-mask couple
        image_path = self.images_files[index]
        mask_path = self.masks_files[index]
        # Image processing
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image = image.astype(np.uint8)
        # Maks processing
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)[None]
        mask = mask.astype(np.uint8)
        # Augmentation
        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']
        return image, mask


def batch_image_mask_show(dataloader, number_of_images=5, initial_index=0):
    """Plot samples after augmentation"""
    images, masks = next(iter(dataloader))
    for tensor in [images, masks]:
        if tensor is masks:
            tensor = tensor.numpy().transpose(0, 2, 3, 1)
            tensor = tensor * 255
        else:
            tensor = tensor.numpy().transpose(0, 2, 3, 1)
            tensor = settings.STD * tensor + settings.MEAN
        fig = plt.figure(figsize=(12, 7))
        for i in range(number_of_images):
            fig.add_subplot(1, number_of_images + 1, i + 1)
            plt.imshow(tensor[i + initial_index], cmap='viridis')
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
    plt.show()