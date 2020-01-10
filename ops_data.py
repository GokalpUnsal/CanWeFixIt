import torch
from torchvision import datasets, transforms

import params
from generator import Generator
from layers import Discriminator


def import_data(data_root):
    return datasets.ImageFolder(root=data_root)


def preprocess(dataset, image_size):
    dataset.transform = transforms.Compose([
        transforms.RandomCrop(image_size),
        transforms.ToTensor(),
    ])


def export_model(model, model_path):
    torch.save(model.state_dict(), model_path)


def import_model(model_path, model_name):
    if model_name == "G":
        model = Generator().to(params.device)
    elif model_name == "D":
        model = Discriminator().to(params.device)
    else:
        return None
    model.load_state_dict(torch.load(model_path))
    return model


def export_tensors(t, path):
    torch.save(t, path)


def export_losses(g_losses, d_losses, d_losses_real, d_losses_fake, l_losses):
    with open('g_losses.txt', 'w') as f:
        for item in g_losses:
            f.write("%s\n" % item)
    with open('d_losses.txt', 'w') as f:
        for item in d_losses:
            f.write("%s\n" % item)
    with open('d_losses_real.txt', 'w') as f:
        for item in d_losses_real:
            f.write("%s\n" % item)
    with open('d_losses_fake.txt', 'w') as f:
        for item in d_losses_fake:
            f.write("%s\n" % item)
    with open('l_losses.txt', 'w') as f:
        for item in l_losses:
            f.write("%s\n" % item)
