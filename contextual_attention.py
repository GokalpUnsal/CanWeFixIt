"""
All code in file taken from https://github.com/DAA233/generative-inpainting-pytorch
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


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
            yi = F.conv2d(xi, wi_normed, stride=1, padding=1)  # yi => (B=1, C=32*32, H=32, W=32)

            # conv implementation for fuse scores to encourage large patches
            if fuse:
                yi = yi.view(1, 1, fs[2] * fs[3],
                             bs[2] * bs[3])  # make all of depth to spatial resolution, (B=1, I=1, H=32*32, W=32*32)
                yi = F.conv2d(yi, fuse_weight, stride=1, padding=1)  # (B=1, C=1, H=32*32, W=32*32)

                yi = yi.contiguous().view(1, fs[2], fs[3], bs[2], bs[3])  # (B=1, 32, 32, 32, 32)
                yi = yi.permute(0, 2, 1, 4, 3)
                yi = yi.contiguous().view(1, 1, fs[2] * fs[3], bs[2] * bs[3])

                yi = F.conv2d(yi, fuse_weight, stride=1, padding=1)
                yi = yi.contiguous().view(1, fs[3], fs[2], bs[3], bs[2])
                yi = yi.permute(0, 2, 1, 4, 3)

            yi = yi.contiguous().view(1, bs[2] * bs[3], fs[2], fs[3])  # (B=1, C=32*32, H=32, W=32)

            # softmax to match
            yi = yi * mm  # mm => (1, 32*32, 1, 1)
            yi = F.softmax(yi * scale, dim=1)
            yi = yi * mm  # mask

            _, offset = torch.max(yi, dim=1)  # argmax; index
            division = torch.div(offset, fs[3]).long()
            offset = torch.stack([division, torch.div(offset, fs[3]) - division], dim=-1)

            # deconv for patch pasting
            # 3.1 paste center
            wi_center = raw_wi[0]
            yi = F.conv_transpose2d(yi, wi_center, stride=self.rate, padding=1) / 4.  # (B=1, C=128, H=64, W=64)
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


# def reduce_mean(x, axis=None, keepdim=False):
#     if not axis:
#         axis = range(len(x.shape))
#     for i in sorted(axis, reverse=True):
#         x = torch.mean(x, dim=i, keepdim=keepdim)
#     return x
#
#
# def reduce_sum(x, axis=None, keepdim=False):
#     if not axis:
#         axis = range(len(x.shape))
#     for i in sorted(axis, reverse=True):
#         x = torch.sum(x, dim=i, keepdim=keepdim)
#     return x


def l2_norm(x):
    def reduce_sum(x):
        for i in range(4):
            if i==1: continue
            x = torch.sum(x, dim=i, keepdim=True)
        return x

    x = x**2
    x = reduce_sum(x)
    return torch.sqrt(x)


def down_sample(x, size=None, scale_factor=None, mode='nearest'):
    # define size if user has specified scale_factor
    if size is None: size = (int(scale_factor * x.size(2)), int(scale_factor * x.size(3)))
    # create coordinates
    h = torch.arange(0, size[0]) / (size[0] - 1) * 2 - 1
    w = torch.arange(0, size[1]) / (size[1] - 1) * 2 - 1
    # create grid
    grid = torch.zeros(size[0], size[1], 2)
    grid[:, :, 0] = w.unsqueeze(0).repeat(size[0], 1)
    grid[:, :, 1] = h.unsqueeze(0).repeat(size[1], 1).transpose(0, 1)
    # expand to match batch size
    grid = grid.unsqueeze(0).repeat(x.size(0), 1, 1, 1)
    if x.is_cuda: grid = grid.cuda()
    # do sampling
    return F.grid_sample(x, grid, mode=mode)


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    batch_s, channel, height, width = images.size()
    if padding == 'same':
        # TODO: same padding
        images = same_padding(images, ksizes, strides, rates)
        pass
    else:
        pass
    unfold = torch.nn.Unfold(kernel_size=ksizes, dilation=rates, padding=0, stride=strides)
    patches = unfold(images)
    return patches


def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x


def flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u / (maxrad + np.finfo(float).eps)
        v = v / (maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))


def compute_color(u, v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    # colorwheel = COLORWHEEL
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))
    return img


def make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255
    return colorwheel
