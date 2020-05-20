import torch
from torch import nn
from torchvision import models


class FeatureAndStyleLoss(nn.Module):

    def __init__(self, feature_loss_weight: float):
        super(FeatureAndStyleLoss, self).__init__()
        self.feature_loss_weight = feature_loss_weight
        self._resnet = models.resnet50(pretrained=True).eval()._modules
        for param in self._resnet.parameters():
            param.requires_grad = False

    def forward(self, ground_truth, prediction):
        pass
