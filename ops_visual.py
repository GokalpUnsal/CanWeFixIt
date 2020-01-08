import matplotlib.pyplot as plt


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
    x = x.permute(0, 2, 3, 1).squeeze(0)
    x = x.cpu()
    plt.imshow(x)
    plt.show()
