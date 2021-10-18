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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LitSuperResolutionModule(pl.LightningModule):
    def __init__(self,
                 scale: int,
                 sr_model,
                 criterion,
                 optimizer,
                 scheduler,
                 log_metrics: bool = True,
                 log_images: bool = True,

                 val_img_log_count: int = 10):
        super().__init__()
        self.scale = scale
        self.sr_model = sr_model
        self.criterion = criterion
        self.sr_optimizer = optimizer
        self.sr_scheduler = scheduler

        self.log_images = log_images
        self.val_img_log_count = val_img_log_count

        self.logged_val_images = self.logged_test_images = 0

        self.log_metrics = log_metrics
        if log_metrics:
            self.lpips_alex = lpips.LPIPS(net="alex", verbose=False).to(DEVICE)
            self.lpips_vgg = lpips.LPIPS(net="vgg", verbose=False).to(DEVICE)
            # self.ssim = torchmetrics.SSIM()
            self.psnr = torchmetrics.PSNR()

    def forward(self, x):
        y_hat = self.sr_model(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        lr, gt = batch
        sr = self.sr_model(lr)
        loss = self.criterion(sr, gt)

        # logging and tracking
        self.log("Train Loss/Step Wise Loss", loss, on_epoch=False, on_step=True)
        self.log("Train Loss/Epoch Wise Loss", loss, on_epoch=True, on_step=False)

        return loss

    def on_validation_start(self) -> None:
        self.logged_val_images = 0

    def validation_step(self, batch, batch_idx):
        lr, gt = batch
        sr = self.sr_model(lr)
        loss = self.criterion(sr, gt)

        # logging and tracking
        if self.log_images and self.logged_val_images < self.val_img_log_count:
            self._log_images(gt, lr, sr)

        if self.log_metrics:
            self._log_metrics(sr, gt)

        return loss

    def on_test_start(self) -> None:
        self.logged_test_images = 0

    def test_step(self, batch, batch_idx):
        sr = self.sr_model(batch)

        logger = self.logger.experiment
        for img in sr:
            img = torch.clamp((img + 1) / 2, min=0, max=1)
            logger.add_image(f"Test Images/Image {self.logged_test_images}", img)

    def configure_optimizers(self):
        return [self.sr_optimizer], [self.sr_scheduler]

    def _log_images(self, gt, lr, sr) -> None:
        logged_so_far_count = self.logged_val_images
        gt = torch.clamp((gt + 1) / 2, min=0, max=1)
        sr = torch.clamp((sr + 1) / 2, min=0, max=1)

        for i in range(min(gt.shape[0], self.val_img_log_count - logged_so_far_count)):
            # save gt image only twice (second is for tensorboard slider matching), since it will not change
            if self.current_epoch == 0:
                self.logger.experiment.add_image(f"Val Images/Image {self.logged_val_images:04}/GT",
                                                 gt[i], 0)
                self.logger.experiment.add_image(f"Val Images/Image {self.logged_val_images:04}/GT",
                                                 gt[i], 1)

            # save bicubically upscaled lr image
            upscaled_lr = torch.squeeze(F.interpolate(torch.unsqueeze(lr[i], dim=0),
                                                      size=lr.shape[-1] * self.scale,
                                                      mode="bicubic"))
            upscaled_lr = torch.clamp((upscaled_lr + 1) / 2, min=0, max=1)
            self.logger.experiment.add_image(f"Val Images/Image {self.logged_val_images:04}/LR BI",
                                             upscaled_lr,
                                             self.current_epoch)

            # save model output upscaled image
            self.logger.experiment.add_image(f"Val Images/Image {self.logged_val_images:04}/SR",
                                             sr[i],
                                             self.current_epoch)

            self.logged_val_images += 1

    def _log_metrics(self, sr, gt) -> None:
        psnr_score = self.psnr(sr, gt)
        # ssim_score = self.ssim(sr, gt)
        lpips_alex_score = torch.mean(self.lpips_alex(sr, gt))
        lpips_vgg_score = torch.mean(self.lpips_vgg(sr, gt))

        self.log("Metrics/Step Wise/PSNR", psnr_score, on_step=True, on_epoch=False)
        # self.log("Metrics/SSIM", ssim_score)
        self.log("Metrics/Step Wise/LPIPS Alex", lpips_alex_score, on_step=True, on_epoch=False)
        self.log("Metrics/Step Wise/LPIPS VGG", lpips_vgg_score, on_step=True, on_epoch=False)

        self.log("Metrics/Epoch Wise/PSNR", psnr_score, on_step=False, on_epoch=True)
        # self.log("Metrics/SSIM", ssim_score)
        self.log("Metrics/Epoch Wise/LPIPS Alex", lpips_alex_score, on_step=False, on_epoch=True)
        self.log("Metrics/Epoch Wise/LPIPS VGG", lpips_vgg_score, on_step=False, on_epoch=True)
