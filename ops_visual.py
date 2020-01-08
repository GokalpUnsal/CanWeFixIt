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


def display_tensor_image(x):
    # x: Tensor with value from -1 to 1. Shape = (1, 3, 256, 256)
    # Output image shape = (256, 256, 3)
    x = x.permute(0, 2, 3, 1).squeeze(0)
    x = x.cpu()
    plt.imshow(x)
    plt.show()


def display_tensor_mask(x):
    # x: Tensor with value from 0 to 1. Shape = (1, 1, 256, 256)
    # Output image shape = (256, 256, 3)
    x = torch.cat((x, x, x), dim=1)
    x = x.permute(0, 2, 3, 1).squeeze(0)
    x = x.cpu()
    plt.imshow(x)
    plt.show()
