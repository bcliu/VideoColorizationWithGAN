import torch
import torch.nn as nn
from torchvision import models

from model.unet import UNetDecoderBlock


class UserGuidedUNet(nn.Module):

    def __init__(self):
        super(UserGuidedUNet, self).__init__()
        resnet_layers = models.resnet34(pretrained=False)._modules

        self._encoders = nn.ModuleDict({
            'layer1': nn.Sequential(
                nn.Conv2d(4, 64, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(64)
            ),
            'layer2': nn.Sequential(
                resnet_layers['maxpool'],
                resnet_layers['layer1']
            ),
            'layer3': resnet_layers['layer2'],
            'layer4': resnet_layers['layer3'],
            'layer5': resnet_layers['layer4'],
        })

        self._decoders = nn.ModuleDict({
            'layer1': UNetDecoderBlock(512),
            'layer2': UNetDecoderBlock(256),
            'layer3': UNetDecoderBlock(128),
            'layer4': UNetDecoderBlock(64, conv_input_channels=96),
        })

        self._conv_final = nn.Sequential(
            nn.Conv2d(32, 2, kernel_size=1, padding=0),
            nn.Tanh()
        )

    def forward(self, L_channel, ab_hint, ab_mask):
        assert L_channel.shape[1] == 1
        assert ab_hint.shape[1] == 2
        assert ab_mask.shape[1] == 1

        x = torch.cat((L_channel, ab_hint, ab_mask), dim=1)
        encoder_outputs = []
        for _, encoder in self._encoders.items():
            encoder_outputs.append(encoder(x if len(encoder_outputs) == 0 else encoder_outputs[-1]))

        decoder_output = None

        decoder_list = list(self._decoders.values())
        for i in range(len(self._decoders)):
            decoder_output = decoder_list[i](
                encoder_outputs[-1] if decoder_output is None else decoder_output,
                encoder_outputs[-(i + 2)]
            )

        return self._conv_final(decoder_output)

    def set_encoders_requires_grad(self, requires_grad: bool):
        for param in self._encoders.parameters():
            param.requires_grad = requires_grad
