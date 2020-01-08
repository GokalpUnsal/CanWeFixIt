import os
import pickle
import torch
from generator import Generator
import params

from torchvision import datasets, transforms

from layers import Discriminator


def import_data(data_root):
    return datasets.ImageFolder(root=data_root)


def preprocess(dataset, image_size):
    dataset.transform = transforms.Compose([
        transforms.RandomCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def export_model(model, model_path):
    torch.save(model.state_dict(), model_path)


def import_model(model_path, model_name):
    if model_name == "G":
        model = Generator().to(params.device)
    if model_name == "D":
        model = Discriminator().to(params.device)
    model.load_state_dict(torch.load(model_path))
    return model
