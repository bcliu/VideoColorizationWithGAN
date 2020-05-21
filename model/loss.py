import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F


class FeatureAndStyleLoss(nn.Module):

    def __init__(self, device, feature_loss_weight: float = 1.0, style_loss_weight: float = 1.0):
        super(FeatureAndStyleLoss, self).__init__()
        self.feature_loss_weight = feature_loss_weight
        self.style_loss_weight = style_loss_weight
        resnet = models.resnet50(pretrained=True).to(device).eval()
        for param in resnet.parameters():
            param.requires_grad = False

        all_layers = list(resnet.children())
        self._resnet_layers = nn.Sequential(*list(filter(FeatureAndStyleLoss._filter_layer, all_layers)))

    @staticmethod
    def _filter_layer(layer):
        return not isinstance(layer, nn.AdaptiveAvgPool2d) and not isinstance(layer, nn.Linear)

    @staticmethod
    def _gram_matrix(tensor):
        b, c, h, w = tensor.size()
        features = tensor.view(b * c, h * w)
        gram = torch.mm(features, features.t())
        return gram / (b * c * h * w)

    def forward(self, ground_truth, prediction):
        ground_truth_features = self._resnet_layers(ground_truth)
        prediction_features = self._resnet_layers(prediction)
        feature_loss = F.mse_loss(prediction_features, ground_truth_features)
        style_loss = F.mse_loss(FeatureAndStyleLoss._gram_matrix(ground_truth_features),
                                FeatureAndStyleLoss._gram_matrix(prediction_features))
        return feature_loss * self.feature_loss_weight + style_loss * self.style_loss_weight
