import params
import torch.utils.data as tud
from ops_data import import_data, preprocess, export_model
from model import GAN


def main():
    # Decide which device we want to run on
    print("Device is " + str(params.device))

    # Create the dataset
    dataset = import_data(params.data_root)
    preprocess(dataset, params.image_size)
    dataloader = tud.DataLoader(dataset, batch_size=params.batch_size, shuffle=True)

    # Create GAN
    network = GAN()
    network.train_gan(dataloader)

    export_model(network.gen, params.model_path)


if __name__ == '__main__':
    main()
