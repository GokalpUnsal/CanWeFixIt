import matplotlib.pyplot as plt
import torch


def plot_losses(g_losses, d_losses, l_losses):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="G")
    plt.plot(d_losses, label="D")
    plt.plot(l_losses, label="L1")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def display_tensor_image(img):
    # x: Tensor with values from -1 to 1. Shape = (1, 3, 256, 256)
    # Output image shape = (256, 256, 3)
    out = img / 2 + 0.5     # Normalize between 0 and 1
    out = out.permute(0, 2, 3, 1).squeeze(0)
    out = out.cpu()
    plt.imshow(out)
    plt.show()


def display_tensor_mask(mask):
    # mask: Tensor with value from 0 to 1. Shape = (1, 1, 256, 256)
    # Output image shape = (256, 256, 3)
    out = torch.cat((mask, mask, mask), dim=1)
    out = out.permute(0, 2, 3, 1).squeeze(0)
    out = out.cpu()
    plt.imshow(out)
    plt.show()
