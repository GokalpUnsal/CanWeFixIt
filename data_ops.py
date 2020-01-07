from torchvision import datasets, transforms


def import_data(dataroot):
    return datasets.ImageFolder(root=dataroot)


def preprocess(dataset, image_size):
    dataset.transform = transforms.Compose([
        transforms.RandomCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
