from typing import Dict

import torchvision.transforms as tfms


def get_train_val_tfms(d: Dict) -> (tfms.Compose, tfms.Compose, tfms.Compose, tfms.Compose):
    scale = int(d["scale"])
    validate = d["validate"]

    d = d["loaders"]

    train_crop = int(d["train_crop"])
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

    val_tfms, val_noise_tfms = None, None
    if validate:
        validation_crop = int(d["validation_crop"])
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


def get_test_tfms(d: Dict) -> tfms.Compose:
    test_tfms = tfms.Compose([
        tfms.ToTensor(),
        tfms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    return test_tfms
