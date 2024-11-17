import argparse
from torchvision import transforms as tfms
from dataset import PetsDataset, PetsDatasetLMDB, StanfordCarsDataset, StanfordCarsDatasetLMDB, \
    Flowers102Dataset, Flowers102DatasetLMDB, Caltech101Dataset, Caltech101DatasetLMDB, STL10Dataset

def build_dataset(dataset_config: dict, args: argparse.Namespace):
    if args.dataset == "pets":
        pets_transform = tfms.Compose(
            [
                tfms.Resize((args.image_size, args.image_size)),
                tfms.ToTensor(),
                tfms.RandomHorizontalFlip(p=0.5),
                tfms.RandomRotation(degrees=[0.0, 360.0], interpolation=tfms.InterpolationMode.NEAREST, expand=False, fill=0),
            ]
        )
        dataset_config["transform"] = pets_transform
        train_dataset = PetsDataset(**dataset_config) if not args.syn_dataset else PetsDatasetLMDB(**dataset_config)
        dataset_config["train"] = False
        dataset_config["transform"] = tfms.Compose(
            [
                tfms.Resize((args.image_size, args.image_size)),
                tfms.ToTensor(),
            ]
        )
        val_dataset = PetsDataset(**dataset_config) if not args.syn_dataset else PetsDatasetLMDB(**dataset_config)
        return train_dataset, val_dataset
    elif args.dataset == "cars":
        cars_transform = tfms.Compose(
            [
                tfms.Resize((args.image_size, args.image_size)),
                tfms.ToTensor(),
                tfms.RandomHorizontalFlip(p=0.5),
                tfms.RandomRotation(degrees=[0.0, 360.0], interpolation=tfms.InterpolationMode.NEAREST, expand=False, fill=0),
            ]
        )
        dataset_config["transform"] = cars_transform
        train_dataset = StanfordCarsDataset(**dataset_config) if not args.syn_dataset else StanfordCarsDatasetLMDB(**dataset_config)
        dataset_config["train"] = False
        dataset_config["transform"] = tfms.Compose(
            [
                tfms.Resize((args.image_size, args.image_size)),
                tfms.ToTensor(),
            ]
        )
        val_dataset = StanfordCarsDataset(**dataset_config) if not args.syn_dataset else StanfordCarsDatasetLMDB(**dataset_config)
        return train_dataset, val_dataset
    elif args.dataset == "flowers":
        flowers_transform = tfms.Compose(
            [
                tfms.Resize((args.image_size, args.image_size)),
                tfms.ToTensor(),
                tfms.RandomHorizontalFlip(p=0.5),
                tfms.RandomRotation(degrees=[0.0, 360.0], interpolation=tfms.InterpolationMode.NEAREST, expand=False, fill=0),
            ]
        )
        dataset_config["transform"] = flowers_transform
        train_dataset = Flowers102Dataset(**dataset_config) if not args.syn_dataset else Flowers102DatasetLMDB(**dataset_config)
        dataset_config["train"] = False
        dataset_config["transform"] = tfms.Compose(
            [
                tfms.Resize((args.image_size, args.image_size)),
                tfms.ToTensor(),
            ]
        )
        val_dataset = Flowers102Dataset(**dataset_config) if not args.syn_dataset else Flowers102DatasetLMDB(**dataset_config)
        return train_dataset, val_dataset
    elif args.dataset == "caltech":
        caltech_transform = tfms.Compose(
            [
                tfms.Resize((args.image_size, args.image_size)),
                tfms.ToTensor(),
                tfms.RandomHorizontalFlip(p=0.5),
                tfms.RandomRotation(degrees=[0.0, 360.0], interpolation=tfms.InterpolationMode.NEAREST, expand=False, fill=0),
            ]
        )
        dataset_config["transform"] = caltech_transform
        train_dataset = Caltech101Dataset(**dataset_config) if not args.syn_dataset else Caltech101DatasetLMDB(**dataset_config)
        dataset_config["train"] = False
        dataset_config["transform"] = tfms.Compose(
            [
                tfms.Resize((args.image_size, args.image_size)),
                tfms.ToTensor(),
            ]
        )
        val_dataset = Caltech101Dataset(**dataset_config) if not args.syn_dataset else Caltech101DatasetLMDB(**dataset_config)
        return train_dataset, val_dataset
    elif args.dataset == "stl":
        stl_transform = tfms.Compose(
            [
                tfms.Resize((args.image_size, args.image_size)),
                tfms.ToTensor(),
            ]
        )
        dataset_config["transform"] = stl_transform
        train_dataset = STL10Dataset(**dataset_config)
        dataset_config["train"] = False
        val_dataset = STL10Dataset(**dataset_config)
        return train_dataset, val_dataset
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")