import torch

import params
import torch.utils.data as tud
from ops_data import import_data, preprocess, export_model, import_model
from model import GAN


def main():
    # Decide which device we want to run on
    print("Device is " + str(params.device))

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Create the dataset
    dataset = import_data(params.data_root)
    preprocess(dataset, params.image_size)
    dataloader = tud.DataLoader(dataset, batch_size=params.batch_size, shuffle=True)

    # Create GAN
    network = GAN()
    if params.pretrained:
        gen = import_model(params.gen_model_path, "G")
        network.gen.load_state_dict(gen.state_dict())
        dis = import_model(params.dis_model_path, "D")
        network.dis.load_state_dict(dis.state_dict())
    network.train_gan(dataloader)

    export_model(network.gen, params.gen_model_path)
    export_model(network.dis, params.dis_model_path)


if __name__ == '__main__':
    main()
