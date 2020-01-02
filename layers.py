import torch
from torch import nn
import numpy as np


class GatedConv2D(nn.Module):
    def __init__(self, out_channels, kernel_size=3, stride=1, dilation=1, activation=nn.ELU):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(out_channels, kernel_size, stride, pad, dilation)
        self.activation = activation

    def forward(self, x):
        self.conv.in_channels = x.shape[1]
        x = self.conv(x)
        if x.shape[1] == 3:
            return x
        x, gate = torch.split(x, 2, 1)
        x = self.activation(x)
        gate = torch.sigmoid(gate)
        x = x * gate
        return x


class GatedDeconv2D(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.conv = GatedConv2D(out_channels)

    def forward(self, x):
        self.conv.in_channels = x.shape[1]
        x = nn.functional.interpolate(x, scale_factor=2)
        x = self.conv(x)
        return x


class SpectralConv2D(nn.Module):
    def __init__(self, out_channels, kernel_size=5, stride=2):
        super().__init__()
        self.conv = nn.utils.spectral_norm(nn.Conv2d(out_channels, kernel_size, stride))
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        self.conv.in_channels = x.shape[1]
        self.conv.padding = ((x.shape[2] - 1) * (self.stride - 1) + (self.kernel_size - 1)) // 2
        x = F.leaky_relu(self.conv(x))
        return x
