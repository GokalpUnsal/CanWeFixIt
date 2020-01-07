import torch.utils.data
from torchvision import datasets
import torchvision.transforms as transforms

from gan import *


def main():
    dataroot = "./"
    batch_size = 1
    image_size = 256

    # Decide which device we want to run on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create the dataset
    dataset = datasets.ImageFolder(root=dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))

    # Create GAN
    network = GAN(device)
    network.train(dataset)
    pass


if __name__ == '__main__':
    main()
