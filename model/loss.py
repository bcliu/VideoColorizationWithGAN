import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F


class FeatureAndStyleLoss(nn.Module):

    def __init__(self,
                 device,
                 feature_loss_weight: float = 1.0,
                 style_loss_weight: float = 1.0,
                 pixel_loss_weight: float = 1.0):
        super(FeatureAndStyleLoss, self).__init__()
        self.feature_loss_weight = feature_loss_weight
        self.style_loss_weight = style_loss_weight
        self.pixel_loss_weight = pixel_loss_weight
        resnet = models.resnet50(pretrained=True).to(device).eval()
        for param in resnet.parameters():
            param.requires_grad = False

        all_layers = list(resnet.children())
        self._resnet_layers = list(filter(FeatureAndStyleLoss._filter_layer, all_layers))
        self._feature_layer_indices = [2, 4, 5, 6, 7]
        self._layer_weights = [10, 4, 2, 2, 1]

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
        layer_losses = []

        ground_truth_output = ground_truth
        prediction_output = prediction
        for layer_idx, layer in enumerate(self._resnet_layers):
            ground_truth_output = layer(ground_truth_output).detach()
            prediction_output = layer(prediction_output)
            if layer_idx in self._feature_layer_indices:
                feature_loss = F.l1_loss(prediction_output, ground_truth_output)
                style_loss = F.l1_loss(FeatureAndStyleLoss._gram_matrix(ground_truth_output),
                                       FeatureAndStyleLoss._gram_matrix(prediction_output))
                layer_losses.append(feature_loss * self.feature_loss_weight + style_loss * self.style_loss_weight)

        total_style_loss = sum([loss * weight for loss, weight in zip(layer_losses, self._layer_weights)])
        pixel_loss = F.l1_loss(prediction, ground_truth) * self.pixel_loss_weight

        return total_style_loss + pixel_loss
