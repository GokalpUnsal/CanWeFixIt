import matplotlib.pyplot as plt
import torch

import params


def plot_losses(g_losses, d_losses, l_losses):
    plt.figure(figsize=(10, 5))
    plt.title("Losses During Training")
    plt.plot(g_losses, label="G")
    plt.plot(d_losses, label="D")
    plt.plot(l_losses, label="L1")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_gan_loss(g_losses, d_losses_real, d_losses_fake):
    plt.figure(figsize=(10, 5))
    plt.title("GAN Loss During Training")
    plt.plot(g_losses, label="G")
    plt.plot(d_losses_real, label="D_real")
    plt.plot(d_losses_fake, label="D_fake")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_images(images):
    columns = 10
    rows = 10
    fig, axarr = plt.subplots(rows, columns, figsize=(rows, columns))
    fig.suptitle("Example Generated Images")
    for i in range(rows * columns):
        row = i // columns
        col = i % columns
        if i < len(images):
            if col == 0:
                axarr[row, col].set_ylabel("{:d}".format(i * params.iter_print))
            img = display_tensor_image(images[i], inplace=False)
            axarr[row, col].imshow(img)
        else:
            axarr[row, col].axis('off')
        axarr[row, col].set_xticks([])
        axarr[row, col].set_yticks([])
    plt.show()


def plot_masks(masks):
    columns = 10
    rows = 10
    fig, axarr = plt.subplots(rows, columns, figsize=(rows, columns))
    fig.suptitle("Example Generated Masks")
    for i in range(rows * columns):
        row = i // columns
        col = i % columns
        if i < len(masks):
            mask = display_tensor_mask(masks[i], inplace=False)
            axarr[row, col].imshow(mask)
        else:
            axarr[row, col].axis('off')
        axarr[row, col].set_xticks([])
        axarr[row, col].set_yticks([])
    plt.show()


def display_tensor_image(img, inplace=True):
    # img: Tensor with shape (1, 3, im, im) or (3, im, im)
    # Output image shape = (im, im, 3)
    out = img
    if len(out.shape) == 4:
        if out.shape[0] == 1:
            out = out.squeeze(0)
        else:
            print("Input is not a single image!")
            return
    assert out.shape == (3, params.image_size, params.image_size)
    out = out.permute(1, 2, 0)
    out = out.cpu()
    out = out.detach().numpy()
    if not inplace:
        return out
    plt.imshow(out)
    plt.show()


def display_tensor_mask(mask, inplace=True):
    # mask: Tensor with shape (1, 1, im, im) or (1, im, im)
    # Output image shape = (im, im, 3)
    out = mask
    if len(out.shape) == 4:
        if out.shape[0] == 1:
            out = out.squeeze(0)
        else:
            print("Input is not a single mask!")
            return
    assert out.shape == (1, params.image_size, params.image_size)
    out = torch.cat((out, out, out), dim=0)
    out = out.permute(1, 2, 0)
    out = out.cpu()
    out = out.detach().numpy()
    if not inplace:
        return out
    plt.imshow(out)
    plt.show()
