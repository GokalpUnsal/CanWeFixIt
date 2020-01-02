from .layers import *
import cv2


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        ch = 48
        ch_input = 5

        # stage_1
        self.conv1 = GatedConv2D(ch_input, ch, kernel_size=5)
        self.conv2_downsample = GatedConv2D(ch // 2, 2 * ch, stride=2)
        self.conv3 = GatedConv2D(ch, 2 * ch)
        self.conv4_downsample = GatedConv2D(ch, 4 * ch, stride=2)
        self.conv5 = GatedConv2D(2 * ch, 4 * ch)
        self.conv6 = GatedConv2D(2 * ch, 4 * ch)
        # mask s but its resized in ours
        self.conv7_atrous = GatedConv2D(2 * ch, 4 * ch, dilation=2)
        self.conv8_atrous = GatedConv2D(2 * ch, 4 * ch, dilation=4)
        self.conv9_atrous = GatedConv2D(2 * ch, 4 * ch, dilation=8)
        self.conv10_atrous = GatedConv2D(2 * ch, 4 * ch, dilation=16)
        self.conv11 = GatedConv2D(2 * ch, 4 * ch)
        self.conv12 = GatedConv2D(2 * ch, 4 * ch)
        self.conv13_upsample = GatedDeconv2D(2 * ch, 2 * ch)
        self.conv14 = GatedConv2D(ch, 2 * ch)
        self.conv15_upsample = GatedDeconv2D(ch, ch)
        self.conv16 = GatedConv2D(ch // 2, ch // 2)
        self.conv17 = GatedConv2D(ch // 4, 3)

        # stage 2
        # TODO: reverse mask in here or
        self.xconv1 = GatedConv2D(3, ch, kernel_size=5)
        self.xconv2_downsample = GatedConv2D(ch // 2, ch, stride=2)
        self.xconv3 = GatedConv2D(ch // 2, 2 * ch)
        self.xconv4_downsample = GatedConv2D(ch, 2 * ch, stride=2)
        self.xconv5 = GatedConv2D(ch, 4 * ch)
        self.xconv6 = GatedConv2D(2 * ch, 4 * ch)
        self.xconv7_atrous = GatedConv2D(2 * ch, 4 * ch, dilation=2)
        self.xconv8_atrous = GatedConv2D(2 * ch, 4 * ch, dilation=4)
        self.xconv9_atrous = GatedConv2D(2 * ch, 4 * ch, dilation=8)
        self.xconv10_atrous = GatedConv2D(2 * ch, 4 * ch, dilation=16)
        # x halu = x in forward

        # attention branch
        self.pmconv1 = GatedConv2D(3, ch, kernel_size=5)
        self.pmconv2_downsample = GatedConv2D(ch // 2, ch, stride=2)
        self.pmconv3 = GatedConv2D(ch // 2, 2 * ch)
        self.pmconv4_downsample = GatedConv2D(ch, 4 * ch, stride=2)
        self.pmconv5 = GatedConv2D(2 * ch, 4 * ch)
        self.pmconv6 = GatedConv2D(2 * ch, 4 * ch, activation=F.relu)
        # TODO: contextual attention
        self.pmconv9 = GatedConv2D(2 * ch, 4 * ch)
        self.pmconv10 = GatedConv2D(2 * ch, 4 * ch)
        # pm = x

        # concat xhalu and pm

        self.allconv11 = GatedConv2D(4 * ch, 4 * ch, 3, 1)
        self.allconv12 = GatedConv2D(2 * ch, 4 * ch, 3, 1)
        self.allconv13_upsample = GatedDeconv2D(2 * ch, 2 * ch)
        self.allconv14 = GatedConv2D(ch, 2 * ch, 3, 1)
        self.allconv15_upsample = GatedDeconv2D(ch, ch)
        self.allconv16 = GatedConv2D(ch // 2, ch // 2)
        self.allconv17 = GatedConv2D(ch // 4, 3)

    def forward(self, x, mask):
        xin = x
        ones_x = torch.ones_like(x)[:, 0:1, :, :]
        x = torch.cat([x, ones_x, ones_x * mask], dim=1)
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

        x = x * mask + xin[:, 0:3, :, :] * (1 - mask)
        x.reshape(xin[:, 0:3, :, :].shape)

        xnow = x
        x = self.xconv1(xnow)
        x = self.xconv2_downsample(x)
        x = self.xconv3(x)
        x = self.xconv4_downsample(x)
        x = self.xconv5(x)
        x = self.xconv6(x)
        x = self.xconv7_atrous(x)
        x = self.xconv8_atrous(x)
        x = self.xconv9_atrous(x)
        x = self.xconv10_atrous(x)
        x_halu = x

        x = self.pmconv1(xnow)
        x = self.pmconv2_downsample(x)
        x = self.pmconv3(x)
        x = self.pmconv4_downsample(x)
        x = self.pmconv5(x)
        x = self.pmconv6(x)
        # TODO: contextual attention
        x = self.pmconv9(x)
        x = self.pmconv10(x)
        pm = x

        x = torch.cat([x_halu, pm], dim=1)
        x = self.allconv11(x)
        x = self.allconv12(x)
        x = self.allconv13_upsample(x)
        x = self.allconv14(x)
        x = self.allconv15_upsample(x)
        x = self.allconv16(x)
        x = self.allconv17(x)
        x = torch.tanh(x)
        x_stage_2 = x
        # return x_stage_1, x_stage_2, offset_flow
        return x


def main():
    device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    # number * depth * width * height
    x = torch.randn((8, 3, 256, 256), dtype=dtype, device=device)
    print(x.requires_grad)
    # mask is  (number of input) * binary matrix
    mask = torch.zeros((8, 1, 256, 256), dtype=dtype, device=device)
    generator = Generator()

    generator = generator.to(device)
    print(device)
    out = generator(x, mask)
    print(x.requires_grad)
    # img = out[0].permute(1,2,0)
    # img = img.cpu()
    # img = img.detach().numpy()
    # img = (img /2) + 0.5
    # plt.imshow(img)
    # plt.show()
    pass


if __name__ == '__main__':
    main()
