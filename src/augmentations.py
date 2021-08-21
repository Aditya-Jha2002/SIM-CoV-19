import config

import albumentations as A
from albumentations import (
    HorizontalFlip, VerticalFlip, ShiftScaleRotate,
    Transpose, ShiftScaleRotate,  HueSaturationValue,
     RandomResizedCrop,
     RandomBrightnessContrast, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2

img_size = config.CFG.img_size


def get_train_transforms():
        return Compose([
            RandomResizedCrop( img_size, img_size),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2,sat_shift_limit=0.2,val_shift_limit=0.2,p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1),contrast_limit=(-0.1,0.1),p=0.5),
            Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.255],max_pixel_value=255.0,p=1.0),
            CoarseDropout(p=0.5),
            Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ],p=1.0)

def get_valid_transforms():
        return Compose([
            CenterCrop(img_size,img_size,p=1.0),
            Resize(img_size,img_size),
            Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.255],max_pixel_value=255.0,p=1.0),
            ToTensorV2(p=1.0),
        ],p=1)