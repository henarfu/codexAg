"""UNetLeon architecture (encoder-decoder with skip connections)."""

from __future__ import annotations

import torch
import torch.nn as nn


def double_conv_leon(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class UNetLeon(nn.Module):
    def __init__(self, n_channels: int = 3, base_channel: int = 64):
        super().__init__()
        self.dconv_down1 = double_conv_leon(n_channels, base_channel)
        self.dconv_down2 = double_conv_leon(base_channel, base_channel * 2)
        self.dconv_down3 = double_conv_leon(base_channel * 2, base_channel * 4)
        self.dconv_down4 = double_conv_leon(base_channel * 4, base_channel * 8)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.dconv_up3 = double_conv_leon(base_channel * 12, base_channel * 4)
        self.dconv_up2 = double_conv_leon(base_channel * 6, base_channel * 2)
        self.dconv_up1 = double_conv_leon(base_channel * 3, base_channel)

        self.conv_last = nn.Conv2d(base_channel, n_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        bottleneck = self.dconv_down4(x)

        x = self.upsample(bottleneck)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)

        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)

        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)

        out = self.conv_last(x)
        return out


def build_unet_leon(device: torch.device, n_channels: int = 3, base_channel: int = 64) -> UNetLeon:
    net = UNetLeon(n_channels=n_channels, base_channel=base_channel)
    return net.to(device)
