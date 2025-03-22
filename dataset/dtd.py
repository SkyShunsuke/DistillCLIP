import os
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torchvision.datasets.utils import download_url

LABEL_TO_CLASSNAME = {
    0: "banded",
    1: "blotchy",
    2: "braided",
    3: "bubbly",
    4: "bumpy",
    5: "chequered",
    6: "cobwebbed",
    7: "cracked",
    8: "crosshatched",
    9: "crystalline",
    10: "dotted",
    11: "fibrous",
    12: "flecked",
    13: "freckled",
    14: "frilly",
    15: "gauzy",
    16: "grid",
    17: "grooved",
    18: "honeycombed",
    19: "interlaced",
    20: "knitted",
    21: "lacelike",
    22: "lined",
    23: "marbled",
    24: "matted",
    25: "meshed",
    26: "paisley",
    27: "perforated",
    28: "pitted",
    29: "pleated",
    30: "polka-dotted",
    31: "porous",
    32: "potholed",
    33: "scaly",
    34: "smeared",
    35: "spiralled",
    36: "sprinkled",
    37: "stained",
    38: "stratified",
    39: "striped",
    40: "studded",
    41: "swirly",
    42: "veined",
    43: "waffled",
    44: "woven",
    45: "wrinkled",
    46: "zigzagged"
}
DTD_CLASSES = list(LABEL_TO_CLASSNAME.values())

class DTDDataset(Dataset):
    """
    Dataset class for the Describable Textures Dataset (DTD).

    Args:
        root (str): Root directory where the dataset is stored or will be downloaded.
        split (str): Dataset split to use ('train', 'val', or 'test').
        transform (callable, optional): A function/transform to apply to images.
        download (bool, optional): If True, downloads the dataset if not already present.
    """
    def __init__(self, root, split='train', transform=None, download=False, base_prompt="A photo of a "):
        self.root = root
        self.split = split
        self.transform = transform
        self.data_dir = os.path.join(root, "dtd")
        self.base_prompt = base_prompt

        if download:
            self.download()

        # Load the split file
        self.samples = []
        split_files = list(Path(self.data_dir, "labels").glob(f"{split}*.txt"))
        for split_file in split_files:
            if not os.path.exists(split_file):
                raise FileNotFoundError(f"Split file not found: {split_file}")
            
            with open(split_file, "r") as f:
                for line in f:
                    image_path = line.split()[0]
                    clsname = image_path.split('/')[0]
                    label = DTD_CLASSES.index(clsname)
                    self.samples.append((image_path, label, clsname))
                    
        self.classes = DTD_CLASSES
        self.num_classes = len(self.classes)

    def download(self):
        """Download the DTD dataset."""
        if os.path.exists(self.data_dir):
            print("Dataset already downloaded.")
            return

        url = "https://thor.robots.ox.ac.uk/dtd/dtd-r1.0.1.tar.gz"
        archive_path = os.path.join(self.root, "dtd-r1.0.1.tar.gz")

        print("Downloading dataset...")
        download_url(url, self.root, filename="dtd-r1.0.1.tar.gz")

        print("Extracting dataset...")
        import tarfile
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=self.root)

        os.rename(os.path.join(self.root, "dtd"), self.data_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Returns:
            dict: A dictionary containing:
                - "index": Index of the sample.
                - "image": Transformed image tensor.
                - "label": Label tensor.
                - "path": Path to the image.
        """
        image_path, label, clsname = self.samples[index]
        full_image_path = os.path.join(self.data_dir, "images", image_path)
        
        # Load the image
        image = Image.open(full_image_path).convert("RGB")

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return {
            "index": index,
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "path": full_image_path, 
            "caption": f"{self.base_prompt}{clsname} texture",
        }

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((96, 96)),  # DTD images are 96x96
        transforms.ToTensor(),
    ])

    ## For real
    dataset = DTDDataset(root="data", transform=transform, download=False, split="train")
    print(len(dataset))
    print(dataset[-1]["image"].size())
    print(dataset[-1]["label"])
    print(dataset[-1]["path"])
    print(dataset[-1])

    # save image to file
    img = dataset[0]["image"]
    img = transforms.ToPILImage()(img)
    img.save("dtd_sample.jpg")
    