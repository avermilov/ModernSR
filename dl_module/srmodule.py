import warnings

import pytorch_lightning
import torchmetrics

from dl_module.loss import VGGPerceptual

if True:
    warnings.filterwarnings("ignore")

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tfms
from pytorch_lightning import loggers as pl_loggers
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from dl_module.model import get_model
from dl_module.dataset import SuperResolutionDataset
import lpips

pytorch_lightning.seed_everything(seed=42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LitSuperResolutionModule(pl.LightningModule):
    def __init__(self,
                 scale: int,
                 sr_model: nn.Module,
                 loss_fn,
                 optimizer,
                 scheduler,
                 log_metrics: bool = True,
                 log_images: bool = True,
                 val_img_log_count: int = 10):
        super().__init__()
        self.scale = scale
        self.sr_model = sr_model
        self.loss_fn = loss_fn
        self.sr_optimizer = optimizer
        self.sr_scheduler = scheduler

        self.log_images = log_images
        self.val_img_log_count = val_img_log_count

        self.running_train_loss = self.running_train_count = 0
        self.running_valid_loss = self.running_valid_count = 0
        self.logged_images = 0

        self.log_metrics = log_metrics
        if log_metrics:
            self.lpips_alex = lpips.LPIPS(net="alex").to(DEVICE)
            self.lpips_vgg = lpips.LPIPS(net="vgg").to(DEVICE)
            # self.ssim = torchmetrics.SSIM()
            self.psnr = torchmetrics.PSNR()

    def forward(self, x):
        y_hat = self.sr_model(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        lr, gt = batch
        sr = self.sr_model(lr)
        loss = self.loss_fn(sr, gt)

        # logging and tracking
        self.log("train_loss", loss, on_epoch=True, on_step=True)
        self.running_train_loss += loss.item()
        self.running_train_count += 1

        return loss

    def validation_step(self, batch, batch_idx):
        lr, gt = batch
        sr = self.sr_model(lr)
        loss = self.loss_fn(sr, gt)

        # logging and tracking
        if self.log_images and self.logged_images < self.val_img_log_count:
            self._log_images(gt, lr, sr)

        if self.log_metrics:
            self._log_metrics(sr, gt)

        self.running_valid_loss += loss.item()
        self.running_valid_count += 1

        return loss

    def training_epoch_end(self, outputs):
        logger = self.logger.experiment
        logger.add_scalar("train_epoch_loss", self.running_train_loss / self.running_train_count, self.current_epoch)
        self.running_train_loss = self.running_train_count = 0

    def validation_epoch_end(self, outputs):
        logger = self.logger.experiment
        logger.add_scalar("valid_epoch_loss", self.running_valid_loss / self.running_valid_count, self.current_epoch)
        self.running_valid_count = self.running_valid_loss = self.logged_images = 0

    def configure_optimizers(self):
        return [self.sr_optimizer], [self.sr_scheduler]

    def _log_images(self, gt, lr, sr) -> None:
        logged_so_far_count = self.logged_images
        gt = torch.clamp((gt + 1) / 2, min=0, max=1)
        sr = torch.clamp((sr + 1) / 2, min=0, max=1)
        for i in range(min(gt.shape[0], self.val_img_log_count - logged_so_far_count)):
            # save gt image only twice (second is for tensorboard slider matching), since it will not change
            if self.current_epoch == 0:
                self.logger.experiment.add_image(f"Val Images/Image {self.logged_images:04}/GT",
                                                 gt[i], 0)
                self.logger.experiment.add_image(f"Val Images/Image {self.logged_images:04}/GT",
                                                 gt[i], 1)

            # save bicubically upscaled lr image
            upscaled_lr = torch.squeeze(F.interpolate(torch.unsqueeze(lr[i], dim=0),
                                                      size=lr.shape[-1] * self.scale,
                                                      mode="bicubic"))
            upscaled_lr = torch.clamp((upscaled_lr + 1) / 2, min=0, max=1)
            self.logger.experiment.add_image(f"Val Images/Image {self.logged_images:04}/LR BI",
                                             upscaled_lr,
                                             self.current_epoch)

            # save model output upscaled image
            self.logger.experiment.add_image(f"Val Images/Image {self.logged_images:04}/SR",
                                             sr[i],
                                             self.current_epoch)

            self.logged_images += 1

    def _log_metrics(self, sr, gt) -> None:
        psnr_score = self.psnr(sr, gt)
        # ssim_score = self.ssim(sr, gt)
        lpips_alex_score = self.lpips_alex(sr, gt)
        lpips_vgg_score = self.lpips_vgg(sr, gt)

        self.log("Metrics/PSNR", psnr_score)
        # self.log("Metrics/SSIM", ssim_score)
        self.log("Metrics/LPIPS Alex", lpips_alex_score)
        self.log("Metrics/LPIPS VGG", lpips_vgg_score)

# img_tfms = tfms.Compose([
#     tfms.ToTensor(),
#     tfms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     tfms.RandomHorizontalFlip(),
#     tfms.RandomVerticalFlip(),
#     tfms.RandomCrop(64, 64)
# ])
# noise_tfms = tfms.Compose([
#     tfms.ToTensor(),
#     tfms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     tfms.RandomCrop(32, 32)
# ])
# val_tfms = tfms.Compose([
#     tfms.ToTensor(),
#     tfms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     tfms.CenterCrop(256)
# ])
# val_noise_tfms = tfms.Compose([
#     tfms.ToTensor(),
#     tfms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     tfms.RandomCrop(128, 128)
# ])
# path = "/home/artermiloff/PycharmProjects/TmpSR/"
# srval = SuperResolutionDataset(scale=2, image_dir=path + "/Datasets/DIV2K/Valid/Valid/",
#                                noises_dir=path + "Noises/noises_s3w7k0/noises_s3w7k0",
#                                kernels_dir=path + "Kernels/kernels_x2", image_transforms=val_tfms,
#                                noise_transforms=val_noise_tfms)
# srtrain = SuperResolutionDataset(scale=2, image_dir=path + "/Datasets/DIV2K/Train/Train/",
#                                  noises_dir=path + "Noises/noises_s3w7k0/noises_s3w7k0",
#                                  kernels_dir=path + "Kernels/kernels_x2", image_transforms=img_tfms,
#                                  noise_transforms=noise_tfms, downscale_mode="nearest")
# train_loader = DataLoader(srtrain, num_workers=8, batch_size=32)
# valid_loader = DataLoader(srval, num_workers=8, batch_size=32)
# litmodel = LitSuperResolutionModule(scale=2, sr_model=get_model(scale=2, model_type=input("MODEL:")),
#                                     log_images=True,
#                                     val_img_log_count=40, log_metrics=True)
# trainer = pl.Trainer(gpus=[0], max_epochs=20, logger=pl_loggers.TensorBoardLogger("../logs", default_hp_metric=False),
#                      deterministic=True)
# trainer.fit(litmodel, train_loader, valid_loader)
