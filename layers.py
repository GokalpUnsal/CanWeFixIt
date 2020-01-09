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


class ContextualAttention(nn.Module):
    def __init__(self, in_ch, out_ch, rate=2, stride=1):
        super(ContextualAttention, self).__init__()
        self.rate = rate
        self.padding = nn.ZeroPad2d(1)
        self.up_sample = nn.Upsample(scale_factor=self.rate, mode='nearest')
        layers = []
        for i in range(2):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1))
        self.out = nn.Sequential(*layers)

    def forward(self, f, b, mask=None, ksize=3, stride=1,
                fuse_k=3, softmax_scale=10., training=True, fuse=True):

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
            training: Indicating if current graph is training or inference.
        Returns:
            tf.Tensor: output
        """

        # get shapes
        raw_fs = f.size()  # B x 128 x 64 x 64
        raw_int_fs = list(f.size())
        raw_int_bs = list(b.size())

        # extract patches from background with stride and rate
        kernel = 2 * self.rate
        raw_w = self.extract_patches(b, kernel=kernel, stride=self.rate)
        raw_w = raw_w.contiguous().view(raw_int_bs[0], -1, raw_int_bs[1], kernel,
                                        kernel)  # B*HW*C*K*K (B, 32*32, 128, 4, 4)

        # downscaling foreground option: downscaling both foreground and
        # background for matching and use original background for reconstruction.
        f = down_sample(f, scale_factor=1 / self.rate, mode='nearest')
        b = down_sample(b, scale_factor=1 / self.rate, mode='nearest')
        fs = f.size()  # B x 128 x 32 x 32
        int_fs = list(f.size())
        f_groups = torch.split(f, 1, dim=0)  # Split tensors by batch dimension; tuple is returned

        # from b(B*H*W*C) to w(b*k*k*c*h*w)
        bs = b.size()  # B x 128 x 32 x 32
        int_bs = list(b.size())
        w = self.extract_patches(b)

        w = w.contiguous().view(int_fs[0], -1, int_fs[1], ksize, ksize)  # B*HW*C*K*K (B, 32*32, 128, 3, 3)

        # process mask
        if mask is not None:
            mask = down_sample(mask, scale_factor=1. / self.rate, mode='nearest')
        else:
            mask = torch.zeros([1, 1, bs[2], bs[3]])

        m = self.extract_patches(mask)

        m = m.contiguous().view(1, 1, -1, ksize, ksize)  # B*C*HW*K*K
        m = m[0]  # (1, 32*32, 3, 3)
        m = reduce_mean(m)  # smoothing, maybe
        mm = m.eq(0.).float()  # (1, 32*32, 1, 1)

        w_groups = torch.split(w, 1, dim=0)  # Split tensors by batch dimension; tuple is returned
        raw_w_groups = torch.split(raw_w, 1, dim=0)  # Split tensors by batch dimension; tuple is returned
        y = []
        offsets = []
        k = fuse_k
        scale = softmax_scale
        fuse_weight = torch.eye(k).view(1, 1, k, k).cuda()  # 1 x 1 x K x K

        for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
            '''
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front; (B=1, C=128, H=32, W=32)
            wi : separated patch tensor along batch dimension of back; (B=1, O=32*32, I=128, KH=3, KW=3)
            raw_wi : separated tensor along batch dimension of back; (B=1, I=32*32, O=128, KH=4, KW=4)
            '''
            # conv for compare
            wi = wi[0]
            escape_NaN = torch.FloatTensor([1e-4]).cuda()
            wi_normed = wi / torch.max(l2_norm(wi), escape_NaN)
            yi = fun.conv2d(xi, wi_normed, stride=1, padding=1)  # yi => (B=1, C=32*32, H=32, W=32)

            # conv implementation for fuse scores to encourage large patches
            if fuse:
                yi = yi.view(1, 1, fs[2] * fs[3],
                             bs[2] * bs[3])  # make all of depth to spatial resolution, (B=1, I=1, H=32*32, W=32*32)
                yi = fun.conv2d(yi, fuse_weight, stride=1, padding=1)  # (B=1, C=1, H=32*32, W=32*32)

                yi = yi.contiguous().view(1, fs[2], fs[3], bs[2], bs[3])  # (B=1, 32, 32, 32, 32)
                yi = yi.permute(0, 2, 1, 4, 3)
                yi = yi.contiguous().view(1, 1, fs[2] * fs[3], bs[2] * bs[3])

                yi = fun.conv2d(yi, fuse_weight, stride=1, padding=1)
                yi = yi.contiguous().view(1, fs[3], fs[2], bs[3], bs[2])
                yi = yi.permute(0, 2, 1, 4, 3)

            yi = yi.contiguous().view(1, bs[2] * bs[3], fs[2], fs[3])  # (B=1, C=32*32, H=32, W=32)

            # softmax to match
            yi = yi * mm  # mm => (1, 32*32, 1, 1)
            yi = fun.softmax(yi * scale, dim=1)
            yi = yi * mm  # mask

            _, offset = torch.max(yi, dim=1)  # argmax; index
            division = torch.div(offset, fs[3]).long()
            offset = torch.stack([division, torch.div(offset, fs[3]) - division], dim=-1)

            # deconv for patch pasting
            # 3.1 paste center
            wi_center = raw_wi[0]
            yi = fun.conv_transpose2d(yi, wi_center, stride=self.rate, padding=1) / 4.  # (B=1, C=128, H=64, W=64)
            y.append(yi)
            offsets.append(offset)

        y = torch.cat(y, dim=0)  # back to the mini-batch
        y.contiguous().view(raw_int_fs)
        offsets = torch.cat(offsets, dim=0)
        offsets = offsets.view([int_bs[0]] + [2] + int_bs[2:])

        # case1: visualize optical flow: minus current position
        h_add = torch.arange(0, float(bs[2])).cuda().view([1, 1, bs[2], 1])
        h_add = h_add.expand(bs[0], 1, bs[2], bs[3])
        w_add = torch.arange(0, float(bs[3])).cuda().view([1, 1, 1, bs[3]])
        w_add = w_add.expand(bs[0], 1, bs[2], bs[3])

        offsets = offsets - torch.cat([h_add, w_add], dim=1).long()

        # to flow image
        flow = torch.from_numpy(flow_to_image(offsets.permute(0, 2, 3, 1).cpu().data.numpy()))
        flow = flow.permute(0, 3, 1, 2)

        # # case2: visualize which pixels are attended
        # flow = torch.from_numpy(highlight_flow((offsets * mask.int()).numpy()))
        if self.rate != 1:
            flow = self.up_sample(flow)
        return self.out(y), flow

    # padding1(16 x 128 x 64 x 64) => (16 x 128 x 64 x 64 x 3 x 3)
    def extract_patches(self, x, kernel=3, stride=1):
        x = self.padding(x)
        all_patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
        return all_patches
