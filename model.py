import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, in_channel):
        super(Model, self).__init__()

        #stage_1
        self.conv1 = nn.Conv2d(in_channel, 2*in_channel, kernel_size=5, stride=1)
        self.conv2_downsample = nn.Conv2d(2*in_channel, 2*in_channel, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(2*in_channel, 4*in_channel, kernel_size=3, stride=1)
        self.conv4_downsample = nn.Conv2d(4*in_channel, 4*in_channel, kernel_size=3, stride=2)
        self.conv5 = nn.Conv2d(4*in_channel, 4*in_channel, kernel_size=3, stride=1)
        self.conv6 = nn.Conv2d(4*in_channel, 4*in_channel, kernel_size=3, stride=1)
        #mask s but its rezied in ours
        self.conv7_atrous = nn.Conv2d(4*in_channel, 4*in_channel, kernel_size=3, stride=1, dilation=2)
        self.conv8_atrous = nn.Conv2d(4*in_channel, 4*in_channel, kernel_size=3, stride=1, dilation=4)
        self.conv9_atrous = nn.Conv2d(4*in_channel, 4*in_channel, kernel_size=3, stride=1, dilation=8)
        self.conv10_atrous = nn.Conv2d(4*in_channel, 4*in_channel, kernel_size=3, stride=1, dilation=16)
        self.conv11 = nn.Conv2d(4*in_channel, 4*in_channel, kernel_size=3, stride=1)
        self.conv12 = nn.Conv2d(4*in_channel, 4*in_channel, kernel_size=3, stride=1)
        self.conv13_upsample = nn.ConvTranspose2d(4*in_channel, 2*in_channel, kernel_size=5, stride=2)
        self.conv14 = nn.Conv2d(2*in_channel, in_channel, kernel_size=3, stride=1)
        self.conv15_upsample = nn.ConvTranspose2d(in_channel, in_channel//2, kernel_size=5, stride=2)
        self.conv16 = nn.Conv2d(in_channel//2, 3, kernel_size=3, stride=1)
        self.conv17 = nn.Conv2d(3, 3, kernel_size=3, stride=1)

    def forward(self, x, mask):
        #TODO: 3, turn relus to gates
        ones_x = torch.ones_like(x)[:, 0:1, :, :]
        x = torch.cat([x, ones_x, ones_x * mask], axis=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2_downsample(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4_downsample(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7_atrous(x))
        x = F.relu(self.conv8_atrous(x))
        x = F.relu(self.conv9_atrous(x))
        x = F.relu(self.conv10_atrous(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13_upsample(x))
        x = F.relu(self.conv14(x))
        x = F.relu(self.conv15_upsample(x))
        x = F.relu(self.conv16(x))
        x = self.conv17(x)
        x = torch.tanh(x)
        x_stage_1 = x

        return x

    def gate(self,x_in):
        x,gate = torch.split(x_in,2,1)
        x = F.elu(x)
        gate = torch.sigmoid(gate)
        x = x * gate
        return x

    def gated_conv(self, in_c, out_c, ksize, stride,dil):
        p = int(dil * (ksize - 1) / 2)
        x = nn.Conv2d(in_c, out_c, ksize, stride)
        x = self.gate(x)

    def gated_t_conv(self, in_c, out_c, ksize, stride,dil):
        p = int(dil * (ksize - 1) / 2)
        x = nn.Conv2d(in_c, out_c, ksize, stride)
        x = self.gate(x)


def main():
    dtype = torch.float32
    #number * depth * width * height
    x = torch.rand((64, 3, 512, 512), dtype=dtype)
    #mask is  (number of input) * binary matrix
    mask = torch.zeros((64, 1, 512, 512), dtype=dtype)
    model = Model(5)
    model.forward(x, mask)

if __name__ == '__main__':
    main()