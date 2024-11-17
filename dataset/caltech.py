import h5py
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

class Caltech101Dataset(Dataset):
    def __init__(self, root, transform=None, download=False, train=True, datasetname="caltech101"):
        self.root = os.path.join(root, datasetname)
        self.transform = transform
        self.train = train
        self.download = download

        # Use torchvision's Caltech101 dataset class
        self.dataset = datasets.Caltech101(root=root, target_type='category', transform=None, download=download)
        print(f"Number of samples in dataset: {len(self.dataset)}")
        # Set the classes and number of classes
        self.classes = self.dataset.categories
        self.num_classes = len(self.classes)
        
        # Split data into training and validation sets per class
        self.train_paths, self.val_paths, self.train_labels, self.val_labels = self.split_data_by_class()

        # Choose paths and labels based on the `train` flag
        if self.train:
            self.data_paths = self.train_paths
            self.data_labels = self.train_labels
        else:
            self.data_paths = self.val_paths
            self.data_labels = self.val_labels

    def split_data_by_class(self):
        paths = []
        labels = []

        for i in range(len(self.dataset)):
            path = os.path.join(
                self.root,
                "101_ObjectCategories",
                self.dataset.categories[self.dataset.y[i]],
                "image_" + f"{self.dataset.index[i]:04d}" + ".jpg"
            )
            paths.append(path)
            labels.append(self.dataset.y[i])

        train_paths, val_paths = [], []
        train_labels, val_labels = [], []

        # Split each class into train and val
        for cls in range(self.num_classes):
            cls_indices = [i for i, label in enumerate(labels) if label == cls]
            cls_paths = [paths[i] for i in cls_indices]
            cls_labels = [labels[i] for i in cls_indices]

            # Split data for the current class
            cls_train_paths, cls_val_paths, cls_train_labels, cls_val_labels = train_test_split(
                cls_paths, cls_labels, test_size=0.2, random_state=42
            )

            train_paths.extend(cls_train_paths)
            train_labels.extend(cls_train_labels)
            val_paths.extend(cls_val_paths)
            val_labels.extend(cls_val_labels)

        return train_paths, val_paths, train_labels, val_labels

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        image_path = self.data_paths[index]
        label = self.data_labels[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "index": index,
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "caption": str(self.classes[label]),
            "path": image_path
        }

class Caltech101DatasetLMDB(Dataset):
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

        # Convert numpy array to PIL Image
        image = Image.fromarray(np.uint8(image), 'RGB')

        if self.transform:
            image = self.transform(image)

        return {
            "index": index,
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "caption": str(label),
            "path": str(path_real)
        }

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = Caltech101Dataset(root="data", transform=transform, download=True)
    print(f"Caltech101 dataset successfully loaded.")
    print(f"Number of images: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Classes: {dataset.classes}")
    print(f"Sample data: {dataset[0]}")
    print(f"Sample image shape: {dataset[0]['image'].size()}")
