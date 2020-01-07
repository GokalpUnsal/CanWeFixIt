import torch.utils.data
from torchvision import datasets
import torchvision.transforms as transforms
import pickle
from gan import GAN
import os

def main():
    dataroot = "./places2/"
    batch_size = 1
    image_size = 256

    # Decide which device we want to run on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device is " + str(device))

    # Create the dataset
    dataset = datasets.ImageFolder(root=dataroot,
                                   transform=transforms.Compose([
                                       transforms.RandomCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))

    # Create GAN
    network = GAN(device)
    network.train_gan(dataset)
    model_path = "./model.pth"

    if not os.path.isfile(model_path):
        with open(model_path, 'wb') as f:
            pickle.dump(network, f)
            f.close()

    if os.path.isfile(model_path):
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
            f.close()
    pass

if __name__ == '__main__':
    main()
