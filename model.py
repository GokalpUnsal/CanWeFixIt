from .layers import *


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        ch = 4

        # stage_1
        self.conv1 = GatedConv2D(ch, 2 * ch, kernel_size=5)
        self.conv2_downsample = GatedConv2D(2 * ch, 2 * ch, stride=2)
        self.conv3 = GatedConv2D(2 * ch, 4 * ch)
        self.conv4_downsample = GatedConv2D(4 * ch, 4 * ch, stride=2)
        self.conv5 = GatedConv2D(4 * ch, 4 * ch)
        self.conv6 = GatedConv2D(4 * ch, 4 * ch)
        # mask s but its resized in ours
        self.conv7_atrous = GatedConv2D(4 * ch, 4 * ch, dilation=2)
        self.conv8_atrous = GatedConv2D(4 * ch, 4 * ch, dilation=4)
        self.conv9_atrous = GatedConv2D(4 * ch, 4 * ch, dilation=8)
        self.conv10_atrous = GatedConv2D(4 * ch, 4 * ch, dilation=16)
        self.conv11 = GatedConv2D(4 * ch, 4 * ch)
        self.conv12 = GatedConv2D(4 * ch, 4 * ch)
        self.conv13_upsample = nn.ConvTranspose2d(4 * ch, 2 * ch, kernel_size=5, stride=2)
        self.conv14 = GatedConv2D(2 * ch, ch)
        self.conv15_upsample = nn.ConvTranspose2d(ch, ch // 2, kernel_size=5, stride=2)
        self.conv16 = GatedConv2D(ch // 2, 3)
        self.conv17 = GatedConv2D(3, 3)

    def forward(self, x, mask):
        # TODO: 3, turn relus to gates
        ones_x = torch.ones_like(x)[:, 0:1, :, :]
        x = torch.cat([x, ones_x, ones_x * mask], axis=1)
        x = self.conv1(x)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13_upsample(x)
        x = self.conv14(x)
        x = self.conv15_upsample(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = torch.tanh(x)
        x_stage_1 = x

        return x

    def gate(self, x_in):
        x, gate = torch.split(x_in, 2, 1)
        x = F.elu(x)
        gate = torch.sigmoid(gate)
        x = x * gate
        return x

    def gated_conv(self, in_c, out_c, ksize, stride, dil):
        p = int(dil * (ksize - 1) / 2)
        x = nn.Conv2d(in_c, out_c, ksize, stride)
        x = self.gate(x)

    def gated_t_conv(self, in_c, out_c, ksize, stride, dil):
        p = int(dil * (ksize - 1) / 2)
        x = nn.Conv2d(in_c, out_c, ksize, stride)
        x = self.gate(x)


def main():
    dtype = torch.float32
    # number * depth * width * height
    x = torch.rand((64, 3, 512, 512), dtype=dtype)
    # mask is  (number of input) * binary matrix
    mask = torch.zeros((64, 1, 512, 512), dtype=dtype)
    model = Model(5)
    model.forward(x, mask)


if __name__ == '__main__':
    main()
