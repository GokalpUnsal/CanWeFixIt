import math

import torch.nn.functional as fun
from torch import nn

from ops_util import *


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        cnum = params.ch_dis
        self.conv1 = SpectralConv2D(4, cnum)
        self.conv2 = SpectralConv2D(cnum, 2 * cnum)
        self.conv3 = SpectralConv2D(2 * cnum, 4 * cnum)
        self.conv4 = SpectralConv2D(4 * cnum, 4 * cnum)
        self.conv5 = SpectralConv2D(4 * cnum, 4 * cnum)
        self.conv6 = SpectralConv2D(4 * cnum, 4 * cnum)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = torch.flatten(x)
        return x


class GatedConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, activation=fun.elu):
        super().__init__()
        p = dilation * (kernel_size - 1) / 2
        ph = math.ceil(p)
        pl = math.floor(p)
        self.pad = nn.ZeroPad2d((pl, ph, pl, ph))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation)
        self.conv_mask = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation)
        self.activation = activation

    def forward(self, x):
        x = self.pad(x)
        mask = self.conv_mask(x)
        gate = torch.sigmoid(mask)
        x = self.conv(x)
        if x.shape[1] == 3 or self.activation is None:
            return x
        x = self.activation(x)
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
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2):
        super().__init__()
        p = (kernel_size - 1) / 2
        ph = math.ceil(p)
        pl = math.floor(p)
        self.pad = nn.ZeroPad2d((pl, ph, pl, ph))
        self.conv = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride))

    def forward(self, x):
        x = self.pad(x)
        x = fun.leaky_relu(self.conv(x))
        return x


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, K=3, S=1, P=1, D=1, activation=nn.ELU()):
        super(Conv, self).__init__()
        if activation:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=K, stride=S, padding=P, dilation=D),
                activation
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=K, stride=S, padding=P, dilation=D)
            )

    def forward(self, x):
        x = self.conv(x)
        return x



