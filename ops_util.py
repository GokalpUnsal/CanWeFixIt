import cv2
import numpy as np
import torch

import params


def random_bbox(width=128, height=128, vertical_margin=0, horizontal_margin=0, img_shape=(256, 256, 3)):
    """Generate a random tlhw.
    Returns:
        tuple: (top, left, height, width)
    """
    img_shape = img_shape
    img_height = img_shape[0]
    img_width = img_shape[1]
    maxt = img_height - vertical_margin - height
    maxl = img_width - horizontal_margin - width
    t = torch.tensor(0, dtype=torch.int32).random_(vertical_margin, maxt)
    l = torch.tensor(0, dtype=torch.int32).random_(horizontal_margin, maxl)
    h = torch.tensor(height, dtype=torch.int32)
    w = torch.tensor(width, dtype=torch.int32)
    return t, l, h, w


def brush_stroke_mask(dimensions=(256, 256), min_num_vertex=4, max_num_vertex=12, min_line_width=12, max_line_width=40):
    """ Generate random mask
        Args:
            :param dimensions: tuple, (width, height)
            :param min_num_vertex: int
            :param max_num_vertex: int
            :param min_line_width: int
            :param max_line_width: int
        Returns:
            torch.tensor: output with shape [1, 1, H, W]
    """
    mean_angle = 2 * np.math.pi / 5
    angle_range = 2 * np.math.pi / 15
    W, H = dimensions

    average_radius = np.math.sqrt(H * H + W * W) / 8
    mask = np.zeros((H, W, 1), np.uint8)

    for _ in range(np.random.randint(1, 4)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(2 * np.math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.shape[:2]
        vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        width = int(np.random.uniform(min_line_width, max_line_width))
        for i in range(1, num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius // 2),
                0, 2 * average_radius)
            new_x = np.clip(vertex[-1][0] + r * np.math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * np.math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))
            cv2.line(mask, vertex[i - 1], vertex[i], 1, thickness=width)
            cv2.circle(mask, vertex[i], width // 2, 1, thickness=-1)
    if np.random.normal() > 0:
        cv2.flip(mask, 0)
    if np.random.normal() > 0:
        cv2.flip(mask, 1)

    mask = np.asarray(mask, np.float32)
    mask = np.reshape(mask, (1, 1, H, W))
    tensor = torch.tensor(mask)
    return tensor


def bbox2mask(bbox, max_delta_height=32, max_delta_width=32, img_shape=(256, 256, 3)):
    """Generate mask tensor from bbox.
    Args:
        :param bbox: tuple, (top, left, height, width)
        :param max_delta_width: int
        :param max_delta_height: int
        :param img_shape: tuple, (height, width, depth)
    Returns:
        torch.tensor: output with shape [1, 1, H, W]
    """
    height = img_shape[0]
    width = img_shape[1]
    mask = np.zeros((1, 1, height, width), np.float32)

    h = np.random.randint(max_delta_height // 2 + 1)
    w = np.random.randint(max_delta_width // 2 + 1)

    mask[:, :, bbox[0] + h:bbox[0] + bbox[2] - h, bbox[1] + w:bbox[1] + bbox[3] - w] = 1.

    tensor = torch.tensor(mask)
    return tensor


def resize_mask_like(mask, x):
    """Resize mask like shape of x.
    Args:
        mask: Original mask.
        x: To shape of x.
    Returns:
        torch.tensor: resized mask
    """
    return torch.nn.functional.interpolate(mask, size=x.shape[2:])


def gan_hinge_loss(pos, neg):
    """
    gan with hinge loss:
    https://github.com/JiahuiYu/neuralgym/blob/master/neuralgym/ops/gan_ops.py
    """
    hinge_pos = torch.mean(torch.nn.functional.relu(1 - pos)).to(params.device)
    hinge_neg = torch.mean(torch.nn.functional.relu(1 + neg)).to(params.device)
    d_loss = (torch.tensor(.5, device=params.device) * hinge_pos + torch.tensor(.5, device=params.device) * hinge_neg)
    g_loss = (-torch.mean(neg))
    return g_loss, d_loss


def normalize_tensor(t, input_range=(0, 1), output_range=(-1, 1)):
    # t: tensor to be normalized
    # input_range: tuple of min and max values of t's current representation
    #              for example, if t is an image tensor with 8-bit integer values, input_range=(0, 255)
    # output_range: tuple of min and max values in the new representation
    # returns: normalized tensor
    min1 = input_range[0]
    diff1 = input_range[1] - input_range[0]
    min2 = output_range[0]
    diff2 = output_range[1] - output_range[0]
    assert diff1 != 0
    tn = (t - min1) * (diff2 / diff1) + min2
    return tn


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


def pt_flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = torch.tensor(-999)
    maxv = torch.tensor(-999)
    minu = torch.tensor(999)
    minv = torch.tensor(999)
    maxrad = torch.tensor(-1)
    if torch.cuda.is_available():
        maxu = maxu.cuda()
        maxv = maxv.cuda()
        minu = minu.cuda()
        minv = minv.cuda()
        maxrad = maxrad.cuda()
    for i in range(flow.shape[0]):
        u = flow[i, 0, :, :]
        v = flow[i, 1, :, :]
        idxunknow = (torch.abs(u) > 1e7) + (torch.abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = torch.max(maxu, torch.max(u))
        minu = torch.min(minu, torch.min(u))
        maxv = torch.max(maxv, torch.max(v))
        minv = torch.min(minv, torch.min(v))
        rad = torch.sqrt((u ** 2 + v ** 2).float()).to(torch.int64)
        maxrad = torch.max(maxrad, torch.max(rad))
        u = u / (maxrad + torch.finfo(torch.float32).eps)
        v = v / (maxrad + torch.finfo(torch.float32).eps)
        # TODO: change the following to pytorch
        img = pt_compute_color(u, v)
        out.append(img)

    return torch.stack(out, dim=0)


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
