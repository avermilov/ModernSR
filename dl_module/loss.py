from typing import Dict

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_loss(d: Dict) -> nn.Module:
    loss_type = d["loss_type"].lower()
    if loss_type == "l1":
        return nn.L1Loss()
    elif loss_type == "vgg":
        return VGGPerceptual(l1_coeff=0, vgg_coeff=1)
    elif loss_type == "perceptual":
        return VGGPerceptual(l1_coeff=float(d["l1_coeff"]), vgg_coeff=float(d["vgg_coeff"]))
    else:
        raise ValueError(f"Unknown loss type: {loss_type}.")


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
        l1 = l1_features = 0
        if self.l1_coeff > 0:
            l1 = self.l1_loss(prediction, target)
        if self.vgg_coeff > 0:
            l1_features = self.l1_loss(self.validation_model(prediction), self.validation_model(target))
        return self.l1_coeff * l1 + self.vgg_coeff * l1_features
