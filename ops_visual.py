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
    # img: Tensor with shape (1, 3, 256, 256) or (3, 256, 256)
    # Output image shape = (256, 256, 3)
    out = img
    if len(out.shape) == 4:
        if out.shape[0] == 1:
            out = out.squeeze(0)
        else:
            print("Input is not a single image!")
            return
    assert out.shape == (3, 256, 256)
    out = out.permute(1, 2, 0)
    out = out.cpu()
    out = out.detach().numpy()
    plt.imshow(out)
    plt.show()


def display_tensor_mask(mask):
    # mask: Tensor with shape (1, 1, 256, 256) or (1, 256, 256)
    # Output image shape = (256, 256, 3)
    out = mask
    if len(out.shape) == 4:
        if out.shape[0] == 1:
            out = out.squeeze(0)
        else:
            print("Input is not a single mask!")
            return
    assert out.shape == (1, 256, 256)
    out = torch.cat((out, out, out), dim=0)
    out = out.permute(1, 2, 0)
    out = out.cpu()
    out = out.detach().numpy()
    plt.imshow(out)
    plt.show()
