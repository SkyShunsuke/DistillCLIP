import os
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torchvision
from torchvision.datasets.utils import download_url
import json

FOOD101_CLASSES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare", "beet_salad", "beignets",
    "bibimbap", "bread_pudding", "breakfast_burrito", "bruschetta", "caesar_salad", "cannoli", "caprese_salad",
    "carrot_cake", "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder", "club_sandwich",
    "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes", "deviled_eggs", "donuts", "dumplings",
    "edamame", "eggs_benedict", "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice", "frozen_yogurt",
    "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich", "grilled_salmon", "guacamole",
    "gyoza", "hamburger", "hot_and_sour_soup", "hot_dog", "huevos_rancheros", "hummus", "ice_cream",
    "lasagna", "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
    "mussels", "nachos", "omelette", "onion_rings", "oysters", "pad_thai", "paella", "pancakes",
    "panna_cotta", "peking_duck", "pho", "pizza", "pork_chop", "poutine", "prime_rib", "pulled_pork_sandwich",
    "ramen", "ravioli", "red_velvet_cake", "risotto", "samosa", "sashimi", "scallops", "seaweed_salad",
    "shrimp_and_grits", "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
    "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles"
]

class Food101Dataset(Dataset):
    def __init__(self, root, transform=None, train=True, download=False, base_prompt="A photo of ", caption_file=None):
        self.root = root
        self.transform = transform
        self.train = train
        self.base_prompt = base_prompt
        self.caption_file = caption_file

        if download:
            self.download()

        split = "train" if self.train else "test"
        label_file = os.path.join(self.root, "food-101", "meta", f"{split}.json")

        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Label file for {split} split not found at {label_file}")

        with open(label_file, "r") as f:
            label_dict = json.load(f)

        if self.caption_file is not None:
            with open(self.caption_file, "r") as f:
                self.captions = json.load(f)

        self.classes = FOOD101_CLASSES
        self.num_classes = len(self.classes)
        
        self.samples = []
        for cls in self.classes:
            cls_files = label_dict[cls]
            for file in cls_files:
                assert os.path.exists(os.path.join(self.root, "food-101", "images", file + ".jpg")), f"File not found: {file}"
                self.samples.append((file + ".jpg", self.classes.index(cls), cls))

    def download(self):
        # Download the Food101 dataset using torchvision's functionality
        torchvision.datasets.Food101(self.root, download=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, cls = self.samples[index]

        image_path = os.path.join(self.root, "food-101", "images", path)
        # Load and transform the image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        if self.caption_file is not None:
            caption = self.captions[image_path]
        else:
            caption = f"{self.base_prompt}{FOOD101_CLASSES[label]}"

        return {
            "index": index,
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "path": image_path,
            "caption": caption,
        }

class Food101DatasetLMDB(Dataset):
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
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = Food101Dataset(root="data/food101", transform=transform, train=True, download=True)
    print(len(dataset))
    print(dataset[0]["image"].size())
    print(dataset[0]["caption"])
    print(dataset[0]["label"])
