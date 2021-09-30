import torch
import torchvision
from torch import nn

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
