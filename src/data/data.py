import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from pathlib import Path


class NPZLoader(Dataset):
    def __init__(self, path, train=True, transform=None):
        super().__init__()
        self.path = path
        self.transform = transform
        if train:
            self.images = np.empty(shape=(0, 28, 28))
            self.labels = np.empty(shape=(0, ))
            for file in Path(path).iterdir():
                if "train" in str(file):
                    self.images = np.append(self.images, np.load(file)["images"], axis=0)
                    self.labels = np.append(self.labels, np.load(file)["labels"], axis=0)

        else:
            self.images = np.load(path+"test.npz")["images"]
            self.labels = np.load(path+"test.npz")["labels"]

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, item):
        image = self.images[item]
        label = self.labels[item]
        if self.transform is not None:
            image = self.transform(image)

        return image, label

def mnist():
    path = "data/raw/"
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
    train = NPZLoader(path=path, train=True, transform=transform)
    test = NPZLoader(path=path, train=False, transform=transform)
    return train, test
