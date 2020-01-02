import torch
from torch import nn
import torch.nn.functional as F


class GatedConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, dilation)

    def forward(self, x):
        x = self.conv(x)
        x, gate = torch.split(x, 2, 1)
        x = F.elu(x)
        gate = torch.sigmoid(gate)
        x = x * gate
        return x

class GatedDeconv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GatedConv2D(in_channels, out_channels)

    def forward(self, x):
        #x = resize(x, func=tf.image.resize_nearest_neighbor)
        x = self.conv(x)
        return x
