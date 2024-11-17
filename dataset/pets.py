import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import urllib.request
import zipfile
import tarfile
import h5py
import numpy as np

from torchvision.datasets.utils import download_url

class PetsDataset(Dataset):
    base_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/"
    files = {
        "images": "images.tar.gz",
        "annotations": "annotations.tar.gz"
    }

    def __init__(self, root, transform=None, download=False, train=True):
        self.root = os.path.join(root, "pets")
        self.transform = transform
        self.train = train

        if download:
            self.download()

        self.images, self.labels = self.load_data()

    def download(self):
        os.makedirs(self.root, exist_ok=True)
        for key, filename in self.files.items():
            url = self.base_url + filename
            download_url(url, self.root, filename)
            self.extract_file(os.path.join(self.root, filename))

    def extract_file(self, file_path):
        if file_path.endswith("tar.gz"):
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(path=self.root)
        elif file_path.endswith("zip"):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(self.root)

    def load_data(self):
        if self.train:
            annotations_path = os.path.join(self.root, "annotations", "trainval.txt")
        else:
            annotations_path = os.path.join(self.root, "annotations", "test.txt")
        
        images = []
        labels = []
        with open(annotations_path, "r") as file:
            for line in file.readlines():
                parts = line.strip().split()
                image_name = parts[0]
                label = int(parts[1]) - 1
                images.append(os.path.join(self.root, "images", image_name + ".jpg"))
                labels.append(label)
                
        # set number of classes
        self.classes = sorted(set(labels))
        self.num_classes = len(set(labels))
        
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label = self.labels[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "index": index,
            "image": image,
            "label": torch.tensor(label),
            "caption": str(label),
            "path": image_path
        }

class PetsDatasetLMDB(Dataset):
    def __init__(self, root, transform=None, train=True, download=False):
        self.hdf5_file = root
        self.transform = transform
        self.train = train
        self.split = 'train' if train else 'test'

        self.images, self.labels, self.real_paths = self.load_data()

    def load_data(self):
        with h5py.File(self.hdf5_file, 'r') as hdf:
            if self.train:
                group = hdf['train']
            else:
                group = hdf['test']
            
            images = []
            real_paths = []
            labels = []
            for class_key in group.keys():
                for img_key in group[class_key].keys():
                    for sample_key in group[class_key][img_key].keys():
                        images.append(group[class_key][img_key][sample_key][()])
                        real_paths.append(group[class_key][img_key][sample_key].attrs['path_real'])
                        labels.append(int(class_key))
            
            self.classes = sorted(set(labels))
            self.num_classes = len(self.classes)
        
        return images, labels, real_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        path_real = self.real_paths[index]

        image = Image.fromarray(image, 'RGB')

        if self.transform:
            image = self.transform(image)

        return {
            "index": index,
            "image": image,
            "label": torch.tensor(label),
            "caption": str(label),
            "path": str(path_real)
        }

if __name__ == "__main__":
    dataset = PetsDataset(root="data", download=True)
    print(len(dataset))
    print(dataset[0]["image"].size)