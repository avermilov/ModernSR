import warnings

warnings.filterwarnings("ignore")

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import lpips
from settings import DEVICE

LPIPS_ALEX_FN = lpips.LPIPS(net="alex", verbose=False).to(DEVICE)
LPIPS_VGG_FN = lpips.LPIPS(net="vgg", verbose=False).to(DEVICE)
PSNR_FN = torchmetrics.PSNR().to(DEVICE)


class LitSuperResolutionModule(pl.LightningModule):
    def __init__(self,
                 scale: int,
                 sr_model,
                 criterion,
                 optimizer,
                 scheduler,
                 log_metrics: bool = True,
                 log_images: bool = True,
                 val_img_log_count: int = 10,
                 test_loader=None,
                 log_frequency=None):
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

        self.test_loader = test_loader
        self.log_frequency = log_frequency
        if test_loader is not None and log_frequency is None:
            raise ValueError("Test loader passed but no log frequency specified.")

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
        self.log("Validation Loss/Step Wise Loss", loss, on_epoch=False, on_step=True)
        self.log("Validation Loss/Epoch Wise Loss", loss, on_epoch=True, on_step=False)

        # logging and tracking
        if self.log_images and self.logged_val_images < self.val_img_log_count:
            self._log_images(gt, lr, sr)

        if self.log_metrics:
            self._log_metrics(sr, gt)

        return loss

    def on_validation_epoch_end(self) -> None:
        if self.test_loader is None or self.current_epoch % self.log_frequency != 0:
            return

        logged_test_images = 0
        for imgs in self.test_loader:
            imgs = imgs.to(DEVICE)
            sr_imgs = self.sr_model(imgs)
            sr_imgs = torch.clamp((sr_imgs + 1) / 2, min=0, max=1)

            logger = self.logger.experiment
            for img in sr_imgs:
                logger.add_image(f"Test Images/Image {logged_test_images}",
                                 img, global_step=self.current_epoch)
                logged_test_images += 1

    def configure_optimizers(self):
        return [self.sr_optimizer], [self.sr_scheduler]

    def _log_images(self, gt, lr, sr) -> None:
        logged_so_far_count = self.logged_val_images
        gt = torch.clamp((gt + 1) / 2, min=0, max=1)
        sr = torch.clamp((sr + 1) / 2, min=0, max=1)

        for i in range(min(gt.shape[0], self.val_img_log_count - logged_so_far_count)):
            # save gt image only twice (second is for tensorboard slider matching), since it will not change
            if self.current_epoch == 0:
                self.logger.experiment.add_image(f"Validation Images/Image {self.logged_val_images:04}/GT",
                                                 gt[i], 0)
                self.logger.experiment.add_image(f"Validation Images/Image {self.logged_val_images:04}/GT",
                                                 gt[i], 1)

            # save bicubically upscaled lr image
            upscaled_lr = torch.squeeze(F.interpolate(torch.unsqueeze(lr[i], dim=0),
                                                      size=lr.shape[-1] * self.scale,
                                                      mode="bicubic"))
            upscaled_lr = torch.clamp((upscaled_lr + 1) / 2, min=0, max=1)
            self.logger.experiment.add_image(f"Validation Images/Image {self.logged_val_images:04}/LR BI",
                                             upscaled_lr,
                                             self.current_epoch)

            # save model output upscaled image
            self.logger.experiment.add_image(f"Validation Images/Image {self.logged_val_images:04}/SR",
                                             sr[i],
                                             self.current_epoch)

            self.logged_val_images += 1

    def _log_metrics(self, sr, gt) -> None:
        psnr_score = PSNR_FN(sr, gt)
        # ssim_score = self.ssim(sr, gt)
        lpips_alex_score = torch.mean(LPIPS_ALEX_FN(sr, gt))
        lpips_vgg_score = torch.mean(LPIPS_VGG_FN(sr, gt))

        self.log("Validation Metrics/Step Wise/PSNR", psnr_score, on_step=True, on_epoch=False)
        # self.log("Metrics/SSIM", ssim_score)
        self.log("Validation Metrics/Step Wise/LPIPS Alex", lpips_alex_score, on_step=True, on_epoch=False)
        self.log("Validation Metrics/Step Wise/LPIPS VGG", lpips_vgg_score, on_step=True, on_epoch=False)

        self.log("Validation Metrics/Epoch Wise/PSNR", psnr_score, on_step=False, on_epoch=True)
        # self.log("Metrics/SSIM", ssim_score)
        self.log("Validation Metrics/Epoch Wise/LPIPS Alex", lpips_alex_score, on_step=False, on_epoch=True)
        self.log("Validation Metrics/Epoch Wise/LPIPS VGG", lpips_vgg_score, on_step=False, on_epoch=True)
