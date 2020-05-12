import torch
import torch.nn as nn

class UNetEncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, max_pool: bool = True):
        super(UNetEncoderBlock, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv_output = None

        if max_pool:
            self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        else:
            self.max_pool = None

    def forward(self, x):
        self.conv_output = self.convs(x)
        return self.max_pool(self.conv_output) if self.max_pool else self.conv_output


class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels: int):
        super(UNetDecoderBlock, self).__init__()
        out_channels = in_channels // 2
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.tensor, encoder_output: torch.tensor):
        # In case the upsampled dimensions are larger than those of encoder output, truncate the tensor
        upsampled = self.deconv(x)[:, :, :encoder_output.shape[2], :encoder_output.shape[3]]
        concatenated = torch.cat([encoder_output, upsampled], dim=1)
        return self.convs(concatenated)


class UNet(nn.Module):
    """Implementation of the U-Net architecture"""

    def __init__(self):
        super(UNet, self).__init__()
        self.encoders = nn.ModuleList([
            UNetEncoderBlock(1, 64),  # 480x360x1 => 240x180x64
            UNetEncoderBlock(64, 128),  # 240x180x64 => 120x90x128
            UNetEncoderBlock(128, 256),  # 120x90x128 => 60x45x256
            UNetEncoderBlock(256, 512),  # 60x45x256 => 30x23x512
            UNetEncoderBlock(512, 1024, max_pool=False),  # 30x23x512 => 30x23x1024
        ])

        self.decoders = nn.ModuleList([
            UNetDecoderBlock(1024),  # 30x23x1024 => 60x46x512 (throw out last row)
            UNetDecoderBlock(512),  # 60x45x512 => 120x90x256
            UNetDecoderBlock(256),  # 120x90x256 => 240x180x128
            UNetDecoderBlock(128),  # 240x180x128 => 480x360x64
        ])

        self.conv_final = nn.Conv2d(64, 2, kernel_size=1, padding=0)  # 480x360x64 => 480x360x2

    def forward(self, x):
        output = x
        for encoder in self.encoders:
            output = encoder(output)

        for i in range(len(self.decoders)):
            output = self.decoders[i](output, self.encoders[-i - 2].conv_output)

        return self.conv_final(output)
