import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
import h5py
from torchvision.datasets.utils import download_url

LABEL_TO_CLASSNAME_SVHN = {
    0: "digit 0",
    1: "digit 1",
    2: "digit 2",
    3: "digit 3",
    4: "digit 4",
    5: "digit 5",
    6: "digit 6",
    7: "digit 7",
    8: "digit 8",
    9: "digit 9",
}
SVHN_CLASSES = [f"digit {i}" for i in range(10)]

class SVHNDataset(Dataset):
    def __init__(self, root, transform=None, download=False, train=True, base_prompt="A photo of a ", caption_file=None):
        self.root = root
        self.transform = transform
        self.train = train
        self.base_prompt = base_prompt
        self.caption_file = caption_file

        if download:
            self.download()

        if self.train:
            self.split = 'train'
        else:
            self.split = 'test'

        self.dataset = datasets.SVHN(root=self.root, split=self.split, transform=self.transform, download=True)
        print(f"Number of samples in {self.split} set: {len(self.dataset)}")

        if self.caption_file is not None:
            with open(self.caption_file, "r") as f:
                self.captions = json.load(f)

        self.classes = [f"digit {i}" for i in range(10)]
        self.num_classes = len(self.classes)

    def download(self):
        if os.path.exists(os.path.join(self.root, "SVHN")):
            print("Dataset already downloaded")
            return
        datasets.SVHN(root=self.root, split='train', download=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        pseudo_path = f"svhn_{self.split}_{index}.png"

        if self.caption_file is not None:
            caption = self.captions[pseudo_path]
        else:
            caption = f"{self.base_prompt}{LABEL_TO_CLASSNAME_SVHN[label]}"
        return {
            "index": index,
            "image": image,
            "label": torch.tensor(label),
            "caption": caption,
            "path": pseudo_path,
        }
        
class SVHNDatasetLMDB(Dataset):
    def __init__(self, root, transform=None, train=True, download=False):
        self.hdf5_file = root
        self.transform = transform
        if train:
            self.split = "train"
        else:
            self.split = "test"
        
        self.images, self.labels, self.real_paths, self.captions = self.load_data()
        self.classes = SVHN_CLASSES
        self.num_classes = len(self.classes)
    
    def load_data(self):
        with h5py.File(self.hdf5_file, 'r') as hdf:
            if self.split == 'train':
                group = hdf['train']
            elif self.split == 'test':
                group = hdf['test']
            
            images = []
            real_paths = []
            labels = []
            captions = []
            for class_key in group.keys():
                for img_key in group[class_key].keys():
                    for sample_key in group[class_key][img_key].keys():
                        images.append(group[class_key][img_key][sample_key][()])
                        real_paths.append(group[class_key][img_key][sample_key].attrs['path_real'])
                        labels.append(int(class_key))
                        captions.append(group[class_key][img_key][sample_key].attrs['caption'])
            
            self.classes = sorted(set(labels))
            self.num_classes = len(self.classes)
        
        self.classes = sorted(set(labels))
        self.num_classes = len(self.classes)
        return images, labels, real_paths, captions

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        path_real = self.real_paths[index]
        caption = self.captions[index]

        image = Image.fromarray(image, 'RGB')

        if self.transform:
            image = self.transform(image)

        return {
            "index": index,
            "image": image,
            "label": torch.tensor(label),
            "caption": caption,
            "path": path_real
        }

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # SVHN images are 32x32
        transforms.ToTensor(),
    ])

    dataset = SVHNDataset(root="data", transform=transform, download=True, train=False)
    print(len(dataset))
    print(dataset[-1]["image"].size())
    print(dataset[-1]["caption"])
    print(dataset[-1]["label"])
    print(dataset[-1]["path"])
