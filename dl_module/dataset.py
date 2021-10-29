import os
import os.path
import random

import PIL
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
import torchvision.transforms as tfms
from torch.utils.data import Dataset

from utils.config import Config


def get_train_val_ds(cfg: Config,
                     train_tfms: tfms.Compose,
                     train_noise_tfms: tfms.Compose,
                     val_tfms: tfms.Compose,
                     val_noise_tfms: tfms.Compose) -> (Dataset, Dataset):
    scale = cfg.general.scale

    train_dir = cfg.training.image_dir
    train_kernels_dir = getattr(cfg, "training.kernel_dir", None)
    train_noises_dir = getattr(cfg, "training.noise_dir", None)
    train_ds = SuperResolutionDataset(scale=scale,
                                      image_dir=train_dir,
                                      noises_dir=train_noises_dir,
                                      kernels_dir=train_kernels_dir,
                                      image_transforms=train_tfms,
                                      noise_transforms=train_noise_tfms)

    val_ds = None
    if hasattr(cfg, "validation"):
        validation_dir = cfg.validation.image_dir
        validation_kernels_dir = getattr(cfg, "validation.kernel_dir", None)
        validation_noises_dir = getattr(cfg, "validation.noise_dir", None)
        val_ds = SuperResolutionDataset(scale=scale,
                                        image_dir=validation_dir,
                                        noises_dir=validation_noises_dir,
                                        kernels_dir=validation_kernels_dir,
                                        image_transforms=val_tfms,
                                        noise_transforms=val_noise_tfms)

    return train_ds, val_ds


def get_test_ds(cfg: Config, test_tfms: tfms.Compose) -> Dataset:
    test_dir = cfg.test.image_dir

    test_ds = ImageFolderDataset(image_dir=test_dir, transform=test_tfms)

    return test_ds


def get_inference_ds(cfg: Config, test_tfms: tfms.Compose) -> Dataset:
    inference_dir = cfg.inference.in_dir

    test_ds = ImageFolderDataset(image_dir=inference_dir, transform=test_tfms)

    return test_ds


def _load_kernels(kernels_dir: str):
    kernels = []
    for filename in os.listdir(kernels_dir):
        mat = sio.loadmat(os.path.join(kernels_dir, filename))["Kernel"]
        mat = torch.tensor([[mat], [mat], [mat]]).type(torch.FloatTensor)
        mat.requires_grad = False
        kernels.append(mat)
    return kernels


class ImageFolderDataset(Dataset):
    def __init__(self, image_dir: str, transform=None):
        self.main_dir = image_dir
        self.transform = transform
        self.do_transforms = transform is not None
        self.total_images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.total_images)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_images[idx])
        image = PIL.Image.open(img_loc).convert("RGB")
        if self.do_transforms:
            image = self.transform(image)
        return image


class SuperResolutionDataset(Dataset):
    def __init__(self,
                 scale: int,
                 image_dir: str,
                 noises_dir: str = None,
                 kernels_dir: str = None,
                 image_transforms=None,
                 noise_transforms=None,
                 downscale_mode: str = "bicubic"
                 ):
        self.scale = scale
        self.downscale_mode = downscale_mode

        self.apply_noise = noises_dir is not None
        self.apply_kernel = kernels_dir is not None

        self.image_dataset = ImageFolderDataset(image_dir, image_transforms)

        if self.apply_noise:
            self.noise_dataset = ImageFolderDataset(noises_dir, noise_transforms)

        if self.apply_kernel:
            self.kernels = _load_kernels(kernels_dir)

    def __getitem__(self, idx):
        gt = self.image_dataset[idx]
        lr_image = gt

        if self.apply_kernel:
            kernel = random.choice(self.kernels)
            lr_image = torch.unsqueeze(lr_image, dim=0)
            padding = (kernel.shape[-1] - 1) // 2
            lr_image = F.pad(lr_image, [padding] * 4, mode="reflect")
            lr_image = torch.conv2d(lr_image, kernel, stride=self.scale, groups=3)
            lr_image = torch.squeeze(lr_image)
        else:
            lr_image = F.interpolate(torch.unsqueeze(lr_image, dim=0),
                                     size=lr_image.shape[-1] // self.scale,
                                     mode=self.downscale_mode)
            lr_image = torch.clamp(lr_image, -1, 1)
            lr_image = torch.squeeze(lr_image)

        if self.apply_noise:
            noise_idx = np.random.randint(0, len(self.noise_dataset))
            noise_patch = self.noise_dataset[noise_idx]
            lr_image += noise_patch
            lr_image = torch.clamp(lr_image, -1, 1)

        return lr_image, gt

    def __len__(self):
        return len(self.image_dataset)
