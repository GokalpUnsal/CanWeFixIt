import math
import torch
import torch.nn.functional as F
from torch import nn


class GatedConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, activation=F.elu):
        super().__init__()
        p = dilation * (kernel_size - 1) / 2
        ph = math.ceil(p)
        pl = math.floor(p)
        self.pad = nn.ZeroPad2d((pl, ph, pl, ph))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation)
        self.conv_mask = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation)
        self.activation = activation
        nn.init.normal_(self.conv.weight.data, 0.0, 0.02)
        nn.init.constant_(self.conv.bias.data, 0)
        nn.init.normal_(self.conv_mask.weight.data, 0.0, 0.02)
        nn.init.constant_(self.conv_mask.bias.data, 0)

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
        nn.init.normal_(self.conv.weight.data, 0.0, 0.02)
        nn.init.constant_(self.conv.bias.data, 0)

    def forward(self, x):
        x = self.pad(x)
        x = F.leaky_relu(self.conv(x))
        return x


class ContextualAttention(nn.Module):
    def __init__(self, f, b, mask=None, ksize=3, stride=1, rate=1, fuse_k=3, softmax_scale=10., fuse=True):
        super().__init__()
        self.f_shape = f.shape
        self.b_shape = b.shape

    def forward(self, xin):

        return 0
