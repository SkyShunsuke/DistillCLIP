import os
import tarfile
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from PIL import Image
import json
import scipy.io

FLOWERS102_CLASSES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea", "english marigold",
    "tiger lily", "moon orchid", "bird of paradise", "monkshood", "globe thistle", "snapdragon", "colt's foot",
    "king protea", "spear thistle", "yellow iris", "globe flower", "purple coneflower", "peruvian lily",
    "balloon flower", "giant white arum lily", "fire lily", "pincushion flower", "fritillary", "red ginger",
    "grape hyacinth", "corn poppy", "prince of wales feathers", "stemless gentian", "artichoke", "sweet william",
    "carnation", "garden phlox", "love in the mist", "mexican aster", "alpine sea holly", "ruby-lipped cattleya",
    "cape flower", "great masterwort", "siam tulip", "lenten rose", "barbeton daisy", "daffodil", "sword lily",
    "poinsettia", "bolero deep blue", "wallflower", "marigold", "buttercup", "oxeye daisy", "common dandelion",
    "petunia", "wild pansy", "primula", "sunflower", "pelargonium", "bishop of llandaff", "gaura", "geranium",
    "orange dahlia", "pink-yellow dahlia", "cautleya spicata", "japanese anemone", "black-eyed susan",
    "silverbush", "californian poppy", "osteospermum", "spring crocus", "bearded iris", "windflower",
    "tree poppy", "gazania", "azalea", "water lily", "rose", "thorn apple", "morning glory", "passion flower",
    "lotus", "toad lily", "anthurium", "frangipani", "clematis", "hibiscus", "columbine", "desert-rose",
    "tree mallow", "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum", "bee balm", "ball moss",
    "foxglove", "bougainvillea", "camellia", "mallow", "mexican petunia", "bromelia", "blanket flower",
    "trumpet creeper", "blackberry lily"
]

class Flowers102Dataset(Dataset):
    base_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
    files = {
        "images": "102flowers.tgz",
        "labels": "imagelabels.mat",
        "splits": "setid.mat"
    }

    def __init__(self, root, transform=None, download=False, train=True, caption_file=None, base_prompt="A photo of a "):
        self.root = os.path.join(root, "flowers")
        self.transform = transform
        self.split = 'train' if train else 'test'
        self.caption_file = caption_file
        self.base_prompt = base_prompt

        if download:
            self.download()

        self.images, self.labels = self.load_data()

        if self.caption_file is not None:
            with open(self.caption_file, "r") as f:
                self.captions = json.load(f)
                
        self.classes = FLOWERS102_CLASSES
        self.num_classes = len(self.classes)

    def download(self):
        os.makedirs(self.root, exist_ok=True)
        for key, filename in self.files.items():
            url = self.base_url + filename
            download_url(url, self.root, filename)
            if filename.endswith(".tgz"):
                self.extract_file(os.path.join(self.root, filename))

    def extract_file(self, file_path):
        if file_path.endswith("tgz"):
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(path=self.root)

    def load_data(self):
        # Load labels
        labels_path = os.path.join(self.root, self.files['labels'])
        labels = scipy.io.loadmat(labels_path)['labels'][0] - 1

        # Load splits
        splits_path = os.path.join(self.root, self.files['splits'])
        splits = scipy.io.loadmat(splits_path)
        if self.split == 'train':
            indices = np.concatenate((splits['trnid'][0], splits['valid'][0])) - 1
        else:
            indices = splits['tstid'][0] - 1

        images = []
        selected_labels = []
        images_dir = os.path.join(self.root, "jpg")
        for idx in indices:
            image_name = f"image_{idx + 1:05d}.jpg"
            images.append(os.path.join(images_dir, image_name))
            selected_labels.append(labels[idx])

        self.classes = sorted(set(selected_labels))
        self.num_classes = len(self.classes)
        
        return images, selected_labels
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label = self.labels[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        
        if self.caption_file is not None:
            caption = self.captions[image_path]
        else:
            caption = f"{self.base_prompt}{FLOWERS102_CLASSES[label]}"
            
        return {
            "index": index,
            "image": image,
            "label": torch.tensor(label),
            "caption": caption,
            "path": image_path
        }
    
class Flowers102DatasetLMDB(Dataset):
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
    dataset = Flowers102Dataset(root="data", download=True, train=True)
    test_dataset = Flowers102Dataset(root="data", download=True, train=False)
    print(f"Flowers102 dataset successfully loaded.")
    print(f"Number of images: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Classes: {dataset.classes}")
    print(f"Sample image shape: {dataset[0]['image'].size()}")
    print(f"Sample label: {dataset[0]['label']}")
    
