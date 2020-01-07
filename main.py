import torch.utils.data

from data_ops import import_data, preprocess
import torch.utils.data as tud
from model import GAN
import params


def main():
    # Decide which device we want to run on
    params.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device is " + str(params.device))

    # Create the dataset
    dataset = import_data(params.data_root)
    preprocess(dataset, params.image_size)
    dataloader = tud.DataLoader(dataset, batch_size=params.batch_size, shuffle=True)

    # Create GAN
    network = GAN()
    network.train_gan(dataloader)


if __name__ == '__main__':
    main()
