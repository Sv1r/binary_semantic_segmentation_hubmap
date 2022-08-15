import albumentations as A
from albumentations.pytorch import ToTensorV2

import settings

train_aug = A.Compose([
    A.OneOf([
        A.HorizontalFlip(p=1.),
        A.VerticalFlip(p=1.),
        A.RandomRotate90(p=1.)
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
    A.Normalize(mean=settings.MEAN, std=settings.STD, p=1.),
    ToTensorV2(transpose_mask=True)
])
valid_aug = A.Compose([
    A.Normalize(mean=settings.MEAN, std=settings.STD, p=1.),
    ToTensorV2(transpose_mask=True)
])
