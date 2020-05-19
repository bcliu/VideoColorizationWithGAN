import torch
import torch.nn as nn


class UNetEncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, max_pool: bool = True):
        super(UNetEncoderBlock, self).__init__()
        self._convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv_output = None

        if max_pool:
            self._max_pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        else:
            self._max_pool = None

    def forward(self, x):
        self.conv_output = self._convs(x)
        return self._max_pool(self.conv_output) if self._max_pool else self.conv_output


class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, conv_input_channels: int = None):
        super(UNetDecoderBlock, self).__init__()
        out_channels = in_channels // 2
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels if conv_input_channels is None else conv_input_channels,
                      out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.tensor, encoder_output: torch.tensor = None):
        upsampled = self.deconv(x)

        if encoder_output is None:
            return self.convs(upsampled)

        concatenated = torch.cat([encoder_output, upsampled], dim=1)
        return self.convs(concatenated)


class UNet(nn.Module):
    """Implementation of the U-Net architecture"""

    def __init__(self):
        super(UNet, self).__init__()
        self._encoders = nn.ModuleList([
            UNetEncoderBlock(1, 64),
            UNetEncoderBlock(64, 128),
            UNetEncoderBlock(128, 256),
            UNetEncoderBlock(256, 512),
            UNetEncoderBlock(512, 1024, max_pool=False),
        ])

        self._decoders = nn.ModuleList([
            UNetDecoderBlock(1024),
            UNetDecoderBlock(512),
            UNetDecoderBlock(256),
            UNetDecoderBlock(128),
        ])

        self._conv_final = nn.Conv2d(64, 2, kernel_size=1, padding=0)

    def forward(self, x):
        output = x
        for encoder in self._encoders:
            output = encoder(output)

        for i in range(len(self._decoders)):
            output = self._decoders[i](output, self._encoders[-i - 2].conv_output)

        return self._conv_final(output)
