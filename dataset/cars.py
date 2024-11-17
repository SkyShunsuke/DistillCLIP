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

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import kaggle
import torchvision
from tqdm import tqdm

class StanfordCarsDataset(Dataset):
    def __init__(self, root, transform=None, download=False, train=True):
        self.root = root
        self.transform = transform
        self.train = train

        if download:
            self.download()

        # Use torchvision's StanfordCars dataset class
        split = 'train' if self.train else 'test'
        self.dataset = torchvision.datasets.StanfordCars(root=self.root, split=split, transform=self.transform, download=False)
        print(f"Number of samples in {split} set: {len(self.dataset)}")
        
        # Access the data and labels from the torchvision dataset
        # self.images = []
        # self.labels = []
        # for i in tqdm(range(len(self.dataset))):
        #     self.images.append(self.dataset[i][0])
        #     self.labels.append(self.dataset[i][1])
        self.classes = self.dataset.classes
        self.num_classes = len(self.classes)

    def download(self):
        if os.path.exists(os.path.join(self.root, "stanford_cars", "devkit")):
            print("Dataset already downloaded")
            return
        # Download and unzip the dataset using Kaggle API:
        kaggle.api.dataset_download_files('rickyyyyyyy/torchvision-stanford-cars', path=self.root, unzip=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return {
            "index": index,
            "image": image,
            "label": torch.tensor(label),
            "caption": str(label),
            "path": self.dataset._samples[index][0]  # Get the image path
        }


class StanfordCarsDatasetLMDB(Dataset):
    def __init__(self, root, transform=None, train=True, download=False):
        self.hdf5_file = root
        self.transform = transform
        self.train = train

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
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    dataset = StanfordCarsDataset(root="data", transform=transform, download=False, train=True)
    print(len(dataset))
    print(dataset[0]["image"].size())
    print(dataset[0]["label"])
    print(dataset[0]["path"])
