import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

import settings

train_aug = A.Compose([
    A.OneOf([
        A.HorizontalFlip(p=1.),
        A.VerticalFlip(p=1.)
    ], p=.2),
    A.OneOf([
        A.ElasticTransform(p=1.),
        A.GridDistortion(p=1.),
        A.OpticalDistortion(p=1.)
    ], p=.2),
    A.OneOf([
        A.MotionBlur(p=1.),
        A.MedianBlur(p=1.),
        A.Blur(p=1.),
        A.GaussianBlur(p=1.)
    ], p=.2),
    A.OneOf([
        A.ISONoise(p=1.),
        A.GaussNoise(p=1.),
        A.MultiplicativeNoise(p=1.)
    ], p=.2),
    A.OneOf([
        A.RandomBrightnessContrast(p=1.),
        A.RandomGamma(p=1.),
        A.CLAHE(p=1.)
    ], p=.2),
    A.OneOf([
        A.RGBShift(p=1.),
        A.InvertImg(p=1.),
        A.FancyPCA(p=1.),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=1)
    ], p=.2),
    A.OneOf([
        A.CropAndPad(
            px=(-20, 20),
            pad_mode=cv2.BORDER_REPLICATE,
            p=1.
        ),
        A.Affine(
            translate_px=(-20, 20),
            rotate=(-45, 45),
            shear=(-10, 10),
            interpolation=cv2.INTER_CUBIC,
            mode=cv2.BORDER_REPLICATE,
            p=1.
        ),
        A.ShiftScaleRotate(
            shift_limit=.2,
            scale_limit=(-.2, .2),
            rotate_limit=(-30, 30),
            interpolation=cv2.INTER_CUBIC,
            border_mode=cv2.BORDER_REPLICATE,
            p=1.
        )
    ], p=.2),
    # A.Resize(height=settings.IMAGE_SIZE, width=settings.IMAGE_SIZE, interpolation=cv2.INTER_CUBIC, p=1.),
    A.Normalize(mean=settings.MEAN, std=settings.STD, p=1.),
    ToTensorV2()
])
valid_aug = A.Compose([
    # A.Resize(height=settings.IMAGE_SIZE, width=settings.IMAGE_SIZE, interpolation=cv2.INTER_CUBIC, p=1.),
    A.Normalize(mean=settings.MEAN, std=settings.STD, p=1.),
    ToTensorV2()
])
