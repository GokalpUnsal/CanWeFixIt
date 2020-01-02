import torch
from torch import nn
import torch.nn.functional as F


class GatedConv2D(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        pad = dilation * (kernel_size - 1) / 2
        self.conv = nn.Conv2d(input_size, output_size, kernel_size, stride, pad, dilation)

    def forward(self, x):
        x = self.conv(x)
        x, gate = torch.split(x, 2, 1)
        x = F.elu(x)
        gate = torch.sigmoid(gate)
        x = x * gate
        return x
