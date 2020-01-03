import numpy as np
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from .gan import *


def main():
    dataroot = "./"
    batch_size = 1
    image_size = 256

    # Decide which device we want to run on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    # Create the dataset
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    # real_batch = next(iter(dataloader))
    # plt.figure(figsize=(8, 8))
    # plt.axis("off")
    # plt.title("Training Images")
    # a = np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0))
    # plt.imshow(a
    #     )
    # plt.show()

    # Create GAN
    network = GAN(device)
    network.train(dataset)


if __name__ == '__main__':
    main()
