from collections import OrderedDict

import torch.nn as nn
from torchvision import models

from model.unet import UNetDecoderBlock


class ResNetBasedUNet(nn.Module):
    """UNet with a pretrained ResNet as encoder"""

    def __init__(self):
        super(ResNetBasedUNet, self).__init__()
        resnet_layers = models.resnet34(pretrained=True)._modules

        # Averaging first layer weights so that it can take in grayscale input
        conv1_weight = resnet_layers['conv1'].state_dict()['weight']
        averaged_state_dict = OrderedDict([('weight', conv1_weight.mean(dim=1, keepdim=True))])
        grayscale_conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        grayscale_conv1.load_state_dict(averaged_state_dict)

        self._encoders = nn.ModuleList([
            nn.Sequential(
                grayscale_conv1,
                resnet_layers['bn1'],
                resnet_layers['relu']
            ),
            nn.Sequential(
                resnet_layers['maxpool'],
                resnet_layers['layer1']
            ),
            resnet_layers['layer2'],
            resnet_layers['layer3'],
            resnet_layers['layer4'],
        ])

        self._decoders = nn.ModuleList([
            UNetDecoderBlock(512),
            UNetDecoderBlock(256),
            UNetDecoderBlock(128),
            UNetDecoderBlock(64, conv_input_channels=96),
            UNetDecoderBlock(32, conv_input_channels=16),
        ])

        self._conv_final = nn.Conv2d(16, 2, kernel_size=1, padding=0)

    def forward(self, x):
        encoder_outputs = []
        for i in range(len(self._encoders)):
            encoder_outputs.append(self._encoders[i](x if len(encoder_outputs) == 0 else encoder_outputs[-1]))

        decoder_output = None

        for i in range(len(self._decoders) - 1):
            decoder_output = self._decoders[i](
                encoder_outputs[-1] if decoder_output is None else decoder_output,
                encoder_outputs[-(i + 2)]
            )

        # TODO: concatenate input x?
        decoder_output = self._decoders[4](decoder_output)

        return self._conv_final(decoder_output)

    def set_encoders_requires_grad(self, requires_grad: bool):
        for param in self._encoders.parameters():
            param.requires_grad = requires_grad
