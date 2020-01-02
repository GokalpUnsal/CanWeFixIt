import torch
from torch import nn
import numpy as np


class GatedConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, activation=nn.ELU):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, dilation)
        self.sigmoid = nn.Sigmoid()
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if x.shape[1] == 3:
            return x
        x, gate = torch.split(x, 2, 1)
        x = self.activation(x)
        gate = self.sigmoid(gate)
        x = x * gate
        return x


class GatedDeconv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GatedConv2D(in_channels, out_channels)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2)
        x = self.conv(x)
        return x


class SpectralConv2D(nn.Module):
    def __init__(self, input_size, in_channels, out_channels, kernel_size=5, stride=2):
        super().__init__()
        pad = ((input_size - 1) * (self.stride - 1) + (self.kernel_size - 1)) // 2
        self.conv = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = nn.LeakyReLU(self.conv(x))
        return x
