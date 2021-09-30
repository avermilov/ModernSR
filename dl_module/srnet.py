import warnings

from dl_module.model import get_model

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

from dl_module.dataset import SuperResolutionDataset
from model_zoo.rdn import RDN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class VGGPerceptual(nn.Module):
    def __init__(self, l1_coeff: float = 1.0, vgg_coeff: float = 1.0):
        super(VGGPerceptual, self).__init__()
        # set params
        self.l1_coeff = l1_coeff
        self.vgg_coeff = vgg_coeff
        # get pretrained VGG model
        self.validation_model = torchvision.models.vgg16(pretrained=True).to(DEVICE)
        # remove classifier
        self.validation_model.classifier = nn.Identity()
        # remove layers with deep features
        self.validation_model.features = nn.Sequential(*self.validation_model.features[:22])
        # freeze model
        self.validation_model.eval()
        for param in self.validation_model.parameters():
            param.requires_grad = False
        # create L1 loss for measuring distance
        self.l1_loss = nn.L1Loss()

    def forward(self, prediction: torch.tensor, target: torch.tensor) -> torch.tensor:
        l1 = self.l1_loss(prediction, target)
        l1_features = self.l1_loss(self.validation_model(prediction), self.validation_model(target))
        return self.l1_coeff * l1 + self.vgg_coeff * l1_features


class LitSuperResolutionModule(pl.LightningModule):
    def __init__(self,
                 scale: int,
                 sr_model: nn.Module,
                 log_images: bool = True,
                 val_img_log_count: int = 10):
        super().__init__()
        self.scale = scale
        self.sr_model = sr_model
        self.loss_fn = VGGPerceptual()

        self.log_images = log_images
        self.val_img_log_count = val_img_log_count

        self.running_train_loss = self.running_train_count = 0
        self.running_valid_loss = self.running_valid_count = 0
        self.logged_images = 0

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
        gen_optimizer = torch.optim.Adam(self.sr_model.parameters(), lr=3e-4, betas=(0.5, 0.999))
        return [gen_optimizer], [ExponentialLR(gen_optimizer, gamma=0.995)]


img_tfms = tfms.Compose([
    tfms.ToTensor(),
    tfms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    tfms.RandomHorizontalFlip(),
    tfms.RandomVerticalFlip(),
    tfms.RandomCrop(64, 64)
])
noise_tfms = tfms.Compose([
    tfms.ToTensor(),
    tfms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    tfms.RandomCrop(32, 32)
])
val_tfms = tfms.Compose([
    tfms.ToTensor(),
    tfms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    tfms.CenterCrop(256)
])
val_noise_tfms = tfms.Compose([
    tfms.ToTensor(),
    tfms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    tfms.RandomCrop(128, 128)
])
path = "/home/artermiloff/PycharmProjects/TmpSR/"
srval = SuperResolutionDataset(scale=2, image_dir=path + "/Datasets/DIV2K/Valid/Valid/",
                               noises_dir=path + "Noises/noises_s3w7k0/noises_s3w7k0",
                               kernels_dir=path + "Kernels/kernels_x2", image_transforms=val_tfms,
                               noise_transforms=val_noise_tfms)
srtrain = SuperResolutionDataset(scale=2, image_dir=path + "/Datasets/DIV2K/Train/Train/",
                                 noises_dir=path + "Noises/noises_s3w7k0/noises_s3w7k0",
                                 kernels_dir=path + "Kernels/kernels_x2", image_transforms=img_tfms,
                                 noise_transforms=noise_tfms, downscale_mode="nearest")
train_loader = DataLoader(srtrain, num_workers=8, batch_size=32)
valid_loader = DataLoader(srval, num_workers=8, batch_size=32)
litmodel = LitSuperResolutionModule(scale=2, sr_model=get_model(scale=2, model_type=input("MODEL:")),
                                    log_images=True,
                                    val_img_log_count=40)
trainer = pl.Trainer(gpus=[0], max_epochs=20, logger=pl_loggers.TensorBoardLogger("../logs", default_hp_metric=False),
                     deterministic=True)
trainer.fit(litmodel, train_loader, valid_loader)
