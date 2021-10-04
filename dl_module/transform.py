from typing import Dict

import torchvision.transforms as tfms


def get_train_val_transforms(d: Dict) -> (tfms.Compose, tfms.Compose, tfms.Compose, tfms.Compose):
    scale = int(d["scale"])

    d = d["loaders"]
    train_crop = int(d["train_crop"])
    validation_crop = int(d["validation_crop"])

    train_tfms = tfms.Compose([
        tfms.ToTensor(),
        tfms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        tfms.RandomHorizontalFlip(),
        tfms.RandomVerticalFlip(),
        tfms.RandomCrop(train_crop, train_crop)
    ])

    train_noise_tfms = tfms.Compose([
        tfms.ToTensor(),
        tfms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        tfms.RandomCrop(train_crop // scale, train_crop // scale)
    ])

    val_tfms = tfms.Compose([
        tfms.ToTensor(),
        tfms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        tfms.CenterCrop(validation_crop)
    ])

    val_noise_tfms = tfms.Compose([
        tfms.ToTensor(),
        tfms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        tfms.RandomCrop(validation_crop // scale, validation_crop // scale)
    ])

    return train_tfms, train_noise_tfms, val_tfms, val_noise_tfms
