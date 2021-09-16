import os
import os.path
import random

import PIL
import torch
import torchvision.transforms as tfms
import torchvision.utils
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import numpy as np
import torch.nn.functional as F


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
        self.total_imgs = sorted(os.listdir(image_dir))
        print(self.total_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
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
        lowres_image = gt

        if self.apply_kernel:
            kernel = random.choice(self.kernels)
            lowres_image = torch.unsqueeze(lowres_image, dim=0)
            padding = (kernel.shape[-1] - 1) // 2
            lowres_image = F.pad(lowres_image, [padding] * 4, mode="reflect")
            lowres_image = torch.conv2d(lowres_image, kernel, stride=self.scale, groups=3)
            lowres_image = torch.squeeze(lowres_image)
        else:
            lowres_image = F.interpolate(torch.unsqueeze(lowres_image, dim=0),
                                         size=lowres_image.shape[-1] // self.scale,
                                         mode=self.downscale_mode)
            lowres_image = torch.clamp(lowres_image, -1, 1)
            lowres_image = torch.squeeze(lowres_image)
        if self.apply_noise:
            noise_idx = np.random.randint(0, len(self.noise_dataset))
            noise_patch = self.noise_dataset[noise_idx]
            lowres_image += noise_patch
            lowres_image = torch.clamp(lowres_image, -1, 1)
        return lowres_image, gt

    def __len__(self):
        return len(self.image_dataset)


img_tfms = tfms.Compose([
    tfms.ToTensor(),
    tfms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    tfms.RandomHorizontalFlip(),
    tfms.RandomVerticalFlip(),
    tfms.CenterCrop(512)
])
noise_tfms = tfms.Compose([
    tfms.ToTensor(),
    tfms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    tfms.RandomCrop(256, 256)
])
path = "/home/artermiloff/PycharmProjects/TmpSR/"
srds = SuperResolutionDataset(scale=2, image_dir=path + "/Datasets/DIV2K/Valid/Valid/",
                              noises_dir=path + "Noises/noises_s3w7k0/noises_s3w7k0",
                              kernels_dir=None, image_transforms=img_tfms,
                              noise_transforms=noise_tfms, downscale_mode="nearest")
lr, gt = srds[0]
gt = torch.clamp((gt + 1) / 2, 0, 1)
lr = torch.clamp((lr + 1) / 2, 0, 1)
dl = DataLoader(srds, batch_size=4)

torchvision.utils.save_image(gt, "gt.png")
torchvision.utils.save_image(lr, "lr.png")
