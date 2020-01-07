import os
import pickle

from torchvision import datasets, transforms


def import_data(dataroot):
    return datasets.ImageFolder(root=dataroot)


def preprocess(dataset, image_size):
    dataset.transform = transforms.Compose([
        transforms.RandomCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def export_model(model, model_path):
    if not os.path.isfile(model_path):
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            f.close()


def import_model(model_path):
    if os.path.isfile(model_path):
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
            f.close()
