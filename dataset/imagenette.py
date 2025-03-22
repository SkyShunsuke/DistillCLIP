import os
import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from PIL import Image
import json

# Define label-to-classname mapping for ImageNette
LABEL_TO_CLASSNAME_IMAGENETTE = {
    0: "tench",
    1: "English springer",
    2: "cassette player",
    3: "chain saw",
    4: "church",
    5: "French horn",
    6: "garbage truck",
    7: "gas pump",
    8: "golf ball",
    9: "parachute",
    -1: "unlabeled",
}

IMAGENETTE_CLASSES = [
    "tench",
    "English springer",
    "cassette player",
    "chain saw",
    "church",
    "French horn",
    "garbage truck",
    "gas pump",
    "golf ball",
    "parachute",
]

class ImageNetteDataset(Dataset):
    def __init__(self, root, transform=None, train=True, download=False, base_prompt="A photo of a ", caption_file=None):
        self.root = root
        self.transform = transform
        self.train = train
        self.base_prompt = base_prompt
        self.caption_file = caption_file

        if download:
            self.download()

        split = "train" if self.train else "val"
        self.dataset = datasets.ImageFolder(root=os.path.join(self.root, "imagenette2-320", split), transform=self.transform)

        print(f"Number of samples in {split} set: {len(self.dataset)}")

        if self.caption_file is not None:
            with open(self.caption_file, "r") as f:
                self.captions = json.load(f)

        self.classes = IMAGENETTE_CLASSES
        self.num_classes = len(self.classes)

    def download(self):
        if os.path.exists(os.path.join(self.root, "train")) and os.path.exists(os.path.join(self.root, "val")):
            print("Dataset already downloaded")
            return

        url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
        download_path = os.path.join(self.root, "imagenette2-320.tgz")
        os.makedirs(self.root, exist_ok=True)

        if not os.path.exists(download_path):
            print("Downloading ImageNette dataset...")
            torch.hub.download_url_to_file(url, download_path)

        print("Extracting dataset...")
        import tarfile
        with tarfile.open(download_path, "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        pseudo_path = f"imagenette_{'train' if self.train else 'val'}_{index}.png"
        
        if self.caption_file is not None:
            caption = self.captions[pseudo_path]
        else:
            caption = f"{self.base_prompt}{LABEL_TO_CLASSNAME_IMAGENETTE[label]}"

        return {
            "index": index,
            "image": image,
            "label": torch.tensor(label),
            "caption": caption,
            "path": pseudo_path,
        }

class ImageNetteDatasetLMDB(Dataset):
    def __init__(self, root, transform=None, train=True, download=False):
        self.hdf5_file = root
        self.transform = transform
        if train:
            self.split = "train"
        else:
            self.split = "test"
        
        self.images, self.labels, self.real_paths, self.captions = self.load_data()
        self.classes = IMAGENETTE_CLASSES
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
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize using ImageNet stats
    ])

    # Example usage
    dataset = ImageNetteDataset(root="data/imagenette", transform=transform, train=True, download=True)
    print(len(dataset))
    sample = dataset[0]
    print(sample["caption"])
    print(sample["label"])
    print(sample["path"])
