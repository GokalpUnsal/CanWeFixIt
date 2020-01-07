import math
import torch.nn.functional as F
from torch import nn
from util_ops import *
from util_ops import resize_mask_like


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        ch = 48
        ch_input = 5

        # stage_1
        self.conv1 = GatedConv2D(ch_input, ch, kernel_size=5)
        self.conv2_downsample = GatedConv2D(ch, 2 * ch, stride=2)
        self.conv3 = GatedConv2D(2 * ch, 2 * ch)
        self.conv4_downsample = GatedConv2D(2 * ch, 4 * ch, stride=2)
        self.conv5 = GatedConv2D(4 * ch, 4 * ch)
        self.conv6 = GatedConv2D(4 * ch, 4 * ch)
        self.conv7_atrous = GatedConv2D(4 * ch, 4 * ch, dilation=2)
        self.conv8_atrous = GatedConv2D(4 * ch, 4 * ch, dilation=4)
        self.conv9_atrous = GatedConv2D(4 * ch, 4 * ch, dilation=8)
        self.conv10_atrous = GatedConv2D(4 * ch, 4 * ch, dilation=16)
        self.conv11 = GatedConv2D(4 * ch, 4 * ch)
        self.conv12 = GatedConv2D(4 * ch, 4 * ch)
        self.conv13_upsample = GatedDeconv2D(4 * ch, 2 * ch)
        self.conv14 = GatedConv2D(2 * ch, 2 * ch)
        self.conv15_upsample = GatedDeconv2D(2 * ch, ch)
        self.conv16 = GatedConv2D(ch, ch // 2)
        self.conv17 = GatedConv2D(ch // 2, 3, activation=None)

        # stage 2
        self.xconv1 = GatedConv2D(3, ch, kernel_size=5)
        self.xconv2_downsample = GatedConv2D(ch, ch, stride=2)
        self.xconv3 = GatedConv2D(ch, 2 * ch)
        self.xconv4_downsample = GatedConv2D(2 * ch, 2 * ch, stride=2)
        self.xconv5 = GatedConv2D(2 * ch, 4 * ch)
        self.xconv6 = GatedConv2D(4 * ch, 4 * ch)
        self.xconv7_atrous = GatedConv2D(4 * ch, 4 * ch, dilation=2)
        self.xconv8_atrous = GatedConv2D(4 * ch, 4 * ch, dilation=4)
        self.xconv9_atrous = GatedConv2D(4 * ch, 4 * ch, dilation=8)
        self.xconv10_atrous = GatedConv2D(4 * ch, 4 * ch, dilation=16)

        # attention branch
        self.pmconv1 = GatedConv2D(3, ch, kernel_size=5)
        self.pmconv2_downsample = GatedConv2D(ch, ch, stride=2)
        self.pmconv3 = GatedConv2D(ch, 2 * ch)
        self.pmconv4_downsample = GatedConv2D(2 * ch, 4 * ch, stride=2)
        self.pmconv5 = GatedConv2D(4 * ch, 4 * ch)
        self.pmconv6 = GatedConv2D(4 * ch, 4 * ch, activation=F.relu)
        # TODO: contextual attention
        self.pmconv9 = GatedConv2D(4 * ch, 4 * ch)
        self.pmconv10 = GatedConv2D(4 * ch, 4 * ch)

        # concat xhalu and pm

        self.allconv11 = GatedConv2D(8 * ch, 4 * ch, 3, 1)
        self.allconv12 = GatedConv2D(4 * ch, 4 * ch, 3, 1)
        self.allconv13_upsample = GatedDeconv2D(4 * ch, 2 * ch)
        self.allconv14 = GatedConv2D(2 * ch, 2 * ch, 3, 1)
        self.allconv15_upsample = GatedDeconv2D(2 * ch, ch)
        self.allconv16 = GatedConv2D(ch, ch // 2)
        self.allconv17 = GatedConv2D(ch // 2, 3, activation=None)

    def forward(self, x, mask):
        # x: input image with erased parts
        # mask: binary tensor that shows erased parts

        # prepare input channels
        xin = x
        ones_x = torch.ones_like(x)[:, 0:1, :, :]
        x = torch.cat([x, ones_x, ones_x * mask], dim=1)

        # stage_1
        x = self.conv1(x)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        mask_s = resize_mask_like(mask, x)
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

        # prepare coarse result for stage 2
        # put generated patch into input image without patch
        x = x * mask + xin[:, 0:3, :, :] * (1 - mask)
        x.reshape(xin[:, 0:3, :, :].shape)
        x_branch = x

        # convolution branch
        x = self.xconv1(x_branch)
        x = self.xconv2_downsample(x)
        x = self.xconv3(x)
        x = self.xconv4_downsample(x)
        x = self.xconv5(x)
        x = self.xconv6(x)
        x = self.xconv7_atrous(x)
        x = self.xconv8_atrous(x)
        x = self.xconv9_atrous(x)
        x = self.xconv10_atrous(x)
        x_conv = x

        # attention branch
        x = self.pmconv1(x_branch)
        x = self.pmconv2_downsample(x)
        x = self.pmconv3(x)
        x = self.pmconv4_downsample(x)
        x = self.pmconv5(x)
        x = self.pmconv6(x)
        # TODO: contextual attention
        # x, offset_flow = contextual_attention(x, x, mask_s, 3, 1, rate=2)
        x = self.pmconv9(x)
        x = self.pmconv10(x)
        x_att = x

        # concatenate results from two branches and do the last convolutions
        x = torch.cat([x_conv, x_att], dim=1)
        x = self.allconv11(x)
        x = self.allconv12(x)
        x = self.allconv13_upsample(x)
        x = self.allconv14(x)
        x = self.allconv15_upsample(x)
        x = self.allconv16(x)
        x = self.allconv17(x)
        x = torch.tanh(x)
        x_stage_2 = x

        # return stage 1, stage 2 and offset flow results
        return x_stage_1, x_stage_2, None  # TODO: Set offset_flow instead of None


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        cnum = 64
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
        x = x.data * gate.data
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
    def __init__(self, mask=None, ksize=3, stride=1, rate=1, fuse_k=3, softmax_scale=10., fuse=True):
        super().__init__()
        # self.f_shape = f.shape
        # self.b_shape = b.shape
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse

    def forward(self, f, b, mask):
        """

        :param f: foreground
        :param b: background
        :param mask: input mask
        :return: torch.tensor
        """
        """ Contextual attention layer implementation.
        Contextual attention is first introduced in publication:
            Generative Image Inpainting with Contextual Attention, Yu et al.
        Args:
            f: Input feature to match (foreground).
            b: Input feature for match (background).
            mask: Input mask for b, indicating patches not available.
            ksize: Kernel size for contextual attention.
            stride: Stride for extracting patches from b.
            rate: Dilation for matching.
            softmax_scale: Scaled softmax for attention.
        Returns:
            torch.tensor: output
        """
        # get shapes
        raw_int_fs = list(f.size())  # b*c*h*w
        raw_int_bs = list(b.size())  # b*c*h*w

        # extract patches from background with stride and rate
        kernel = 2 * self.rate
        # raw_w is extracted for reconstruction
        raw_w = extract_image_patches(b, ksizes=[kernel, kernel],
                                      strides=[self.rate * self.stride,
                                               self.rate * self.stride],
                                      rates=[1, 1],
                                      padding='same')  # [N, C*k*k, L]
        # raw_shape: [N, C, k, k, L]
        raw_w = raw_w.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)  # raw_shape: [N, L, C, k, k]
        raw_w_groups = torch.split(raw_w, 1, dim=0)

        # downscaling foreground option: downscaling both foreground and
        # background for matching and use original background for reconstruction.
        f = F.interpolate(f, scale_factor=1. / self.rate, mode='nearest')
        b = F.interpolate(b, scale_factor=1. / self.rate, mode='nearest')
        int_fs = list(f.size())  # b*c*h*w
        int_bs = list(b.size())
        f_groups = torch.split(f, 1, dim=0)  # split tensors along the batch dimension
        # w shape: [N, C*k*k, L]
        w = extract_image_patches(b, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        # w shape: [N, C, k, k, L]
        w = w.view(int_bs[0], int_bs[1], self.ksize, self.ksize, -1)
        w = w.permute(0, 4, 1, 2, 3)  # w shape: [N, L, C, k, k]
        w_groups = torch.split(w, 1, dim=0)

        # process mask
        if mask is None:
            mask = torch.zeros([int_bs[0], 1, int_bs[2], int_bs[3]])
            if self.use_cuda:
                mask = mask.cuda()
        else:
            mask = F.interpolate(mask, scale_factor=1. / (4 * self.rate), mode='nearest')
        int_ms = list(mask.size())
        # m shape: [N, C*k*k, L]
        m = extract_image_patches(mask, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        # m shape: [N, C, k, k, L]
        m = m.view(int_ms[0], int_ms[1], self.ksize, self.ksize, -1)
        m = m.permute(0, 4, 1, 2, 3)  # m shape: [N, L, C, k, k]
        m = m[0]  # m shape: [L, C, k, k]
        # mm shape: [L, 1, 1, 1]
        mm = (reduce_mean(m, axis=[1, 2, 3], keepdim=True) == 0.).to(torch.float32)
        mm = mm.permute(1, 0, 2, 3)  # mm shape: [1, L, 1, 1]

        y = []
        offsets = []
        k = self.fuse_k
        scale = self.softmax_scale  # to fit the PyTorch tensor image value range
        fuse_weight = torch.eye(k).view(1, 1, k, k)  # 1*1*k*k
        if 1 if torch.cuda.is_available() else 0:
            fuse_weight = fuse_weight.cuda()

        for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
            '''
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front; (B=1, C=128, H=32, W=32)
            wi : separated patch tensor along batch dimension of back; (B=1, O=32*32, I=128, KH=3, KW=3)
            raw_wi : separated tensor along batch dimension of back; (B=1, I=32*32, O=128, KH=4, KW=4)
            '''
            # conv for compare
            escape_NaN = torch.FloatTensor([1e-4])
            if 1 if torch.cuda.is_available() else 0:
                escape_NaN = escape_NaN.cuda()
            wi = wi[0]  # [L, C, k, k]
            max_wi = torch.max(torch.sqrt(reduce_sum(torch.pow(wi, 2),
                                                     axis=[1, 2, 3],
                                                     keepdim=True)),
                               escape_NaN)
            wi_normed = wi / max_wi
            # xi shape: [1, C, H, W], yi shape: [1, L, H, W]
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            yi = F.conv2d(xi, wi_normed, stride=1)  # [1, L, H, W]
            # conv implementation for fuse scores to encourage large patches
            if self.fuse:
                # make all of depth to spatial resolution
                yi = yi.view(1, 1, int_bs[2] * int_bs[3], int_fs[2] * int_fs[3])  # (B=1, I=1, H=32*32, W=32*32)
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)  # (B=1, C=1, H=32*32, W=32*32)
                yi = yi.contiguous().view(1, int_bs[2], int_bs[3], int_fs[2], int_fs[3])  # (B=1, 32, 32, 32, 32)
                yi = yi.permute(0, 2, 1, 4, 3)
                yi = yi.contiguous().view(1, 1, int_bs[2] * int_bs[3], int_fs[2] * int_fs[3])
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)
                yi = yi.contiguous().view(1, int_bs[3], int_bs[2], int_fs[3], int_fs[2])
                yi = yi.permute(0, 2, 1, 4, 3).contiguous()
            yi = yi.view(1, int_bs[2] * int_bs[3], int_fs[2], int_fs[3])  # (B=1, C=32*32, H=32, W=32)
            # softmax to match
            yi = yi * mm
            yi = F.softmax(yi * scale, dim=1)
            yi = yi * mm  # [1, L, H, W]

            offset = torch.argmax(yi, dim=1, keepdim=True)  # 1*1*H*W

            if int_bs != int_fs:
                # Normalize the offset value to match foreground dimension
                times = float(int_fs[2] * int_fs[3]) / float(int_bs[2] * int_bs[3])
                offset = ((offset + 1).float() * times - 1).to(torch.int64)
            offset = torch.cat([offset // int_fs[3], offset % int_fs[3]], dim=1)  # 1*2*H*W

            # deconv for patch pasting
            wi_center = raw_wi[0]
            # yi = F.pad(yi, [0, 1, 0, 1])    # here may need conv_transpose same padding
            yi = F.conv_transpose2d(yi, wi_center, stride=self.rate, padding=1) / 4.  # (B=1, C=128, H=64, W=64)
            y.append(yi)
            offsets.append(offset)

        y = torch.cat(y, dim=0)  # back to the mini-batch
        y.contiguous().view(raw_int_fs)

        offsets = torch.cat(offsets, dim=0)
        offsets = offsets.view(int_fs[0], 2, *int_fs[2:])

        # case1: visualize optical flow: minus current position
        h_add = torch.arange(int_fs[2]).view([1, 1, int_fs[2], 1]).expand(int_fs[0], -1, -1, int_fs[3])
        w_add = torch.arange(int_fs[3]).view([1, 1, 1, int_fs[3]]).expand(int_fs[0], -1, int_fs[2], -1)
        ref_coordinate = torch.cat([h_add, w_add], dim=1)
        if 1 if torch.cuda.is_available() else 0:
            ref_coordinate = ref_coordinate.cuda()

        offsets = offsets - ref_coordinate
        # flow = pt_flow_to_image(offsets)

        flow = torch.from_numpy(flow_to_image(offsets.permute(0, 2, 3, 1).cpu().data.numpy())) / 255.
        flow = flow.permute(0, 3, 1, 2)
        if 1 if torch.cuda.is_available() else 0:
            flow = flow.cuda()
        # case2: visualize which pixels are attended
        # flow = torch.from_numpy(highlight_flow((offsets * mask.long()).cpu().data.numpy()))

        if self.rate != 1:
            flow = F.interpolate(flow, scale_factor=self.rate * 4, mode='nearest')

        return y, flow
