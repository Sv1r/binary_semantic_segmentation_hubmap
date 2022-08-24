import cv2
import tqdm
import time
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import segmentation_models_pytorch
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
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, axis=-1)
        mask = mask.astype(np.uint8)
        # Augmentation
        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']
        return image, mask


def batch_image_mask_show(dataloader, number_of_images=5):
    """Plot samples after augmentation"""
    images, masks = next(iter(dataloader))
    images = images.numpy().transpose(0, 2, 3, 1)
    masks = masks.numpy()

    fig = plt.figure(figsize=(20, 5))
    for i in range(number_of_images):
        image = settings.STD * images[i] + settings.MEAN
        image = image * 255
        image = image.astype(np.uint8)
        mask = masks[i][0]
        mask = mask.astype(np.uint8)

        fig.add_subplot(1, number_of_images + 1, i + 1)
        plt.imshow(image)
        plt.imshow(mask, alpha=.3, cmap='gnuplot')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
    plt.show()


def train_model(model, train_dataloader, valid_dataloader, loss, optimizer, num_epochs, threshold, avg_results):
    """Train and Validate Model"""
    train_loss, valid_loss = [], []
    train_accuracy, valid_accuracy = [], []
    train_dice, valid_dice = [], []
    train_iou, valid_iou = [], []
    train_accuracy, valid_accuracy = [], []
    model = model.to(device)
    min_loss = 1e3
    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['Train', 'Valid']:
            if phase == 'Train':
                dataloader = train_dataloader
                model.train()  # Set model to training mode
            else:
                dataloader = valid_dataloader
                model.eval()  # Set model to evaluate mode
            running_loss = 0.
            running_accuracy = 0.
            running_dice = 0.
            running_iou = 0.
            # Iterate over data.
            with tqdm.tqdm(dataloader, unit='batch') as tqdm_loader:
                for image, mask in tqdm_loader:
                    tqdm_loader.set_description(f'Epoch {epoch} - {phase}')
                    # Image/Mask to device
                    image = image.to(device, dtype=torch.float32)
                    mask = mask.to(device, dtype=torch.float32)
                    optimizer.zero_grad()
                    # forward and backward
                    with torch.set_grad_enabled(phase == 'Train'):
                        preds = model(image)
                        loss_value = loss(preds, mask)
                        # Compute metrics
                        tp, fp, fn, tn = segmentation_models_pytorch.metrics.functional.get_stats(
                            preds, mask.int(), mode='binary', threshold=threshold
                        )
                        accuracy = segmentation_models_pytorch.metrics.functional.accuracy(
                            tp, fp, fn, tn, reduction='micro', zero_division=1.
                        ).cpu().detach().numpy() * 100.
                        dice = segmentation_models_pytorch.metrics.functional.f1_score(
                            tp, fp, fn, tn, reduction='micro', zero_division=1.
                        ).cpu().detach().numpy() * 100.
                        iou = segmentation_models_pytorch.metrics.functional.iou_score(
                            tp, fp, fn, tn, reduction='micro', zero_division=1.
                        ).cpu().detach().numpy() * 100.
                        # backward + optimize only if in training phase
                        if phase == 'Train':
                            loss_value.backward()
                            optimizer.step()
                        # Current statistics
                        tqdm_loader.set_postfix(
                            Accuracy=accuracy, Dice=dice, IOU=iou, Loss=loss_value.item()
                        )
                        time.sleep(.1)
                    # Statistics
                    running_loss += loss_value.item()
                    running_accuracy += accuracy
                    running_dice += dice
                    running_iou += iou
            # Average values along one epoch
            epoch_loss = running_loss / len(dataloader)
            epoch_dice = running_dice / len(dataloader)
            epoch_iou = running_iou / len(dataloader)
            epoch_accuracy = running_accuracy / len(dataloader)
            # Checkpoint
            if epoch_loss < min_loss and phase != 'Train':
                min_loss = epoch_loss
                model = model.cpu()
                torch.save(model, rf'checkpoint\best.pth')
                model = model.to(device)
            # Epoch final metric
            if phase == 'Train':
                train_loss.append(epoch_loss)
                train_accuracy.append(epoch_accuracy)
                train_dice.append(epoch_dice)
                train_iou.append(epoch_iou)
            else:
                valid_loss.append(epoch_loss)
                valid_accuracy.append(epoch_accuracy)
                valid_dice.append(epoch_dice)
                valid_iou.append(epoch_iou)
            # Show mean results on current epoch
            if avg_results:
                print(f'Average along Epoch {epoch} for {phase}:')
                print('| Loss   | Accuracy | Dice  | IOU   |')
                print(f'| {epoch_loss:.4f} | {epoch_accuracy:.2f}    | {epoch_dice:.2f} | {epoch_iou:.2f} |')
                time.sleep(.1)
    # Save model on last epoch
    torch.save(model, rf'checkpoint\last.pth')
    # Loss and Metrics dataframe
    df = pd.DataFrame.from_dict({
        'Train_Loss': train_loss,
        'Valid_Loss': valid_loss,
        'Train_Accuracy': train_accuracy,
        'Valid_Accuracy': valid_accuracy,
        'Train_Dice': train_dice,
        'Valid_Dice': valid_dice,
        'Train_IOU': train_iou,
        'Valid_IOU': valid_iou,
    })
    return model, df


def result_plot(loss_and_metrics):
    """Plot loss function and Metrics"""
    stage_list = np.unique(list(map(lambda x: x.split(sep='_')[0], loss_and_metrics.columns)))
    variable_list = np.unique(list(map(lambda x: x.split(sep='_')[1], loss_and_metrics.columns)))
    sub_plot_len = len(variable_list) // 2
    fig, axs = plt.subplots(sub_plot_len, sub_plot_len, figsize=(16, 16))
    for stage in stage_list:
        for variable, ax in zip(variable_list, axs.ravel()):
            loss_and_metrics[f'{stage}_{variable}'].plot(ax=ax)
            ax.set_title(f'{variable} Plot', fontsize=10)
            ax.set_xlabel('Epoch', fontsize=8)
            ax.set_ylabel(f'{variable} Value', fontsize=8)
            ax.legend()
    fig.suptitle('Result of Model Training', fontsize=12)
    fig.tight_layout()
    plt.show()
