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
import json

import torchvision
from tqdm import tqdm

LABEL_TO_CLASSNAME = {
    0: "airplane",
    1: "bird",
    2: "car",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "horse",
    7: "monkey",
    8: "ship",
    9: "truck",
    -1: "unlabeled",
}

class STL10Dataset(Dataset):
    def __init__(self, root, transform=None, download=False, train=True, with_unlabeled=False):
        self.root = root
        self.transform = transform
        self.train = train
        self.with_unlabeled = with_unlabeled

        if download:
            self.download()

        if self.train:
            if self.with_unlabeled:
                split = "train+unlabeled"
            else:
                split = "train"
        else:
            split = "test"
        self.dataset = torchvision.datasets.STL10(root=self.root, split=split, transform=self.transform, download=False)

        print(f"Number of samples in {split} set: {len(self.dataset)}")

        self.classes = self.dataset.classes
        self.num_classes = len(self.classes)

    def download(self):
        if os.path.exists(os.path.join(self.root, "stl10_binary")):
            print("Dataset already downloaded")
            return
        # Download the dataset using torchvision's download functionality
        torchvision.datasets.STL10(root=self.root, split='train', download=True)
        # One of {‘train’, ‘test’, ‘unlabeled’, ‘train+unlabeled’}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        pseudo_path = f"stl10_{self.dataset.split}_{index}.png"
        return {
            "index": index,
            "image": image,
            "label": torch.tensor(label),
            "caption": f"A photo of {LABEL_TO_CLASSNAME[label]}",
            "path": pseudo_path,
        }

class STL10DatasetLMDB(Dataset):
    def __init__(self, root, transform=None, train=True, download=False, with_unlabeled=False):
        self.hdf5_file = root
        self.transform = transform
        if train:
            if with_unlabeled:
                self.split = "train+unlabeled"
            else:
                self.split = "train"
        else:
            self.split = "test"
        
        self.images, self.labels, self.real_paths, self.captions = self.load_data()
    
    def load_data(self):
        with h5py.File(self.hdf5_file, 'r') as hdf:
            if self.split == 'train':
                group = hdf['train']
            elif self.split == 'train+unlabeled':
                group = hdf['train+unlabeled']
            else:
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

class STL10DatasetFolder(Dataset):
    def __init__(self, root, transform=None, train=True, download=False, with_unlabeled=False, caption_file=None):
        """
        STL10DatasetFolder with support for 'train', 'test', and 'train+unlabeled' splits.

        Args:
            root (str): Path to the root directory containing images and labels CSV.
            transform (callable, optional): A function/transform to apply to images.
            train (bool, optional): Whether to use training or test data.
            download (bool, optional): Currently unused, included for compatibility.
            with_unlabeled (bool, optional): Whether to include unlabeled data in the training set.
            caption_file (str, optional): Path to the captions json file.
        """
        self.root = root
        self.transform = transform
        self.caption_file = caption_file

        # Determine split type
        if train:
            if with_unlabeled:
                self.split = "train+unlabeled"
            else:
                self.split = "train"
        else:
            self.split = "test"

        # Load appropriate label file based on the split
        label_file_name = f"labels_{self.split}.csv"
        label_file_path = os.path.join(root, label_file_name)
        if not os.path.exists(label_file_path):
            raise FileNotFoundError(f"Label file not found: {label_file_path}")

        # Load the label file into a DataFrame
        self.labels_df = pd.read_csv(label_file_path)

        # Ensure the label file contains necessary columns
        if not all(col in self.labels_df.columns for col in ["index", "path", "label"]):
            raise ValueError("The label file must contain 'index', 'path', and 'label' columns.")
        
        if self.caption_file is not None:
            with open(self.caption_file, "r") as f:
                self.captions = json.load(f)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary containing:
                - "index": Index of the image.
                - "image": Transformed image tensor.
                - "label": Label tensor (or -1 for unlabeled data).
                - "path": Path to the image.
        """
        row = self.labels_df.iloc[idx]
        root_parent = os.path.dirname(self.root)
        image_path = os.path.join(root_parent, row["path"])
        label = row["label"]

        # Load the image
        image = Image.open(image_path).convert("RGB")

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
            
        if self.caption_file is not None:
            caption = self.captions[image_path]
        else:
            caption = f"A photo of {LABEL_TO_CLASSNAME[label]}"
            
        return {
            "index": row["index"],
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "path": image_path,
            "caption": caption,   
        }

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((96, 96)),  # STL10 images are 96x96
        transforms.ToTensor(),
    ])

    ## For real
    # dataset = STL10Dataset(root="data", transform=transform, download=True, train=True, with_unlabeled=True)
    # print(len(dataset))
    # print(dataset[-1]["image"].size())
    # print(dataset[-1]["caption"])
    # print(dataset[-1]["label"])
    # print(dataset[-1]["path"])
    
    # # save image to file
    # img = dataset[0]["image"]
    # torchvision.utils.save_image(img, "stl10_example.png")
    
    ## For synthetic
    # dataset = STL10DatasetLMDB(root="/home/haselab/projects/sakai/DistillCLIP/syn_data/txt2img_stl10_classtxi_s9_n20_x16_rev.hdf5", transform=transform, train=True, download=False, with_unlabeled=False)
    # # save image of the first sample in each class
    # for i in range(10):
    #     idx = np.where(np.array(dataset.labels) == i)[0][0]
    #     img = dataset[idx]["image"]
    #     torchvision.utils.save_image(img, f"stl10_example_{i}.png")
    
    
    ## For folder dataset
    # caption_path = "/home/haselab/projects/sakai/DistillCLIP/caption_data/train_stl10_train_labeled_gpt4omini_low.json"
    # data_path = "data/stl10_images"
    # dataset = STL10DatasetFolder(root=data_path, transform=transform, train=True, download=False, with_unlabeled=False, caption_file=caption_path)
    # print(len(dataset))
    # idx = 500
    # print(dataset[idx]["image"].size())
    # print(dataset[idx]["caption"])
    # print(dataset[idx]["path"])
    # print(dataset[idx]["label"])
    
    
    
