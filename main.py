import torch.utils.data


import pickle
from data_ops import import_data, preprocess
from gan import GAN
import os

def main():
    dataroot = "./places2/"
    image_size = 256

    # Decide which device we want to run on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device is " + str(device))

    # Create the dataset
    dataset = import_data(dataroot)
    preprocess(dataset, image_size)

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
