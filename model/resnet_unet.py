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

        self._encoder_layer1 = nn.Sequential(
            grayscale_conv1,
            resnet_layers['bn1'],
            resnet_layers['relu']
        )
        self._encoder_layer2 = nn.Sequential(
            resnet_layers['maxpool'],
            resnet_layers['layer1']
        )
        self._encoder_layer3 = resnet_layers['layer2']
        self._encoder_layer4 = resnet_layers['layer3']
        self._encoder_layer5 = resnet_layers['layer4']

        self._decoders = nn.ModuleList([
            UNetDecoderBlock(512),
            UNetDecoderBlock(256),
            UNetDecoderBlock(128),
            UNetDecoderBlock(64, conv_input_channels=96),
            UNetDecoderBlock(32, conv_input_channels=16),
        ])

        self._conv_final = nn.Conv2d(16, 2, kernel_size=1, padding=0)

    def forward(self, x):
        encoder_output1 = self._encoder_layer1(x)
        encoder_output2 = self._encoder_layer2(encoder_output1)
        encoder_output3 = self._encoder_layer3(encoder_output2)
        encoder_output4 = self._encoder_layer4(encoder_output3)
        encoder_output5 = self._encoder_layer5(encoder_output4)

        decoder_output1 = self._decoders[0](encoder_output5, encoder_output4)
        decoder_output2 = self._decoders[1](decoder_output1, encoder_output3)
        decoder_output3 = self._decoders[2](decoder_output2, encoder_output2)
        decoder_output4 = self._decoders[3](decoder_output3, encoder_output1)
        # TODO: concatenate input x?
        decoder_output5 = self._decoders[4](decoder_output4)

        output = self._conv_final(decoder_output5)

        return output
