import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from torchvision.datasets.utils import download_url
from PIL import Image
import h5py
import json

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from datasets import load_dataset
import os

LABEL_TO_CLASSNAME = {
    0: "AnnualCrop",
    1: "Forest",
    2: "HerbaceousVegetation",
    3: "Highway",
    4: "Industrial",
    5: "Pasture",
    6: "PermanentCrop",
    7: "Residential",
    8: "River",
    9: "SeaLake",
}

EUROSAT_CLASSES = list(LABEL_TO_CLASSNAME.values())

class EuroSATDataset(Dataset):
    def __init__(self, root, transform=None, download=False, train=True, base_prompt="An satellite image of ", caption_file=None):
        self.root = root
        self.transform = transform
        self.train = train
        self.base_prompt = base_prompt
        self.caption_file = caption_file

        full_dataset = load_dataset("blanchon/EuroSAT_RGB", cache_dir=root)
        # Get the training and validation splits directly
        train_data = full_dataset["train"]
        val_data = full_dataset["validation"]

        if self.train:
            self.dataset = train_data
        else:
            self.dataset = val_data

        if self.caption_file is not None:
            with open(self.caption_file, "r") as f:
                self.captions = json.load(f)

        self.classes = list(LABEL_TO_CLASSNAME.values())
        self.num_classes = len(self.classes)

        print(f"Number of samples: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]["image"], self.dataset[index]["label"]
        
        image = image.convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        pseudo_path = f"eurosat_{index}.png"
        if self.caption_file is not None:
            caption = self.captions[pseudo_path]
        else:
            caption = f"{self.base_prompt}{EUROSAT_CLASSES[label]}"

        return {
            "index": index,
            "image": image,
            "label": torch.tensor(label),
            "caption": caption,
            "path": pseudo_path,
        }

class EuroSATDatasetLMDB(Dataset):
    def __init__(self, root, transform=None, train=True, download=False):
        self.hdf5_file = root
        self.transform = transform
        self.split = "train" if train else "test"

        self.images, self.labels, self.real_paths = self.load_data()

    def load_data(self):
        with h5py.File(self.hdf5_file, 'r') as hdf:
            if self.split == 'train':
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
        transforms.Resize((64, 64)),  # Resize images to 64x64
        transforms.ToTensor(),
    ])

    # Initialize EuroSAT dataset
    dataset = EuroSATDataset(root="data/eurosat", transform=transform, download=True)

    print(len(dataset))
    sample = dataset[0]
    print(sample["image"].size())
    print(sample["caption"])
    print(sample["label"])
    print(sample["path"])
