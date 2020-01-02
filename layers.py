import torch
import torch.nn.functional as F
import numpy as np
from torch import nn


class GatedConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, activation=F.elu):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, dilation)
        self.activation = activation

    def forward(self, xin):
        xout = self.conv(xin)
        if xout.shape[1] == 3:
            return xout
        xout, gate = xout.split(xout.shape[1]//2, 1)
        xout = self.activation(xout)
        gate = torch.sigmoid(gate)
        xout = xout * gate
        return xout


class GatedDeconv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GatedConv2D(in_channels, out_channels)

    def forward(self, xin):
        xin = nn.functional.interpolate(xin, scale_factor=2)
        xout = self.conv(xin)
        return xout


class SpectralConv2D(nn.Module):
    def __init__(self, input_size, in_channels, out_channels, kernel_size=5, stride=2):
        super().__init__()
        pad = ((input_size - 1) * (self.stride - 1) + (self.kernel_size - 1)) // 2
        self.conv = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, xin):
        xout = F.leaky_relu(self.conv(xin))
        return xout
