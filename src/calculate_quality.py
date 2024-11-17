"""
Script for caulucate the quality of the generated samples. 
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
from pathlib import Path

import time
import random
import numpy as np
import torch
import cv2
import h5py
from tqdm.auto import tqdm
from accelerate import PartialState
from torch.utils.data import DataLoader
from torchvision import transforms as tfms
from diffusers import StableDiffusionPipeline, DDIMScheduler
import warnings
warnings.filterwarnings("ignore")

from dataset import PetsDataset, PetsDatasetLMDB, Caltech101Dataset, Caltech101DatasetLMDB, StanfordCarsDataset, StanfordCarsDatasetLMDB, \
    Flowers102Dataset, Flowers102DatasetLMDB
from sampling import ddim_sampling

def calculate_fid(real_loader, syn_loader, device, num_features=2048, normalize=True) -> float:
    """
    Calculate the FID score between the real and synthetic samples.
    For reference: https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html
    """
    from torcheval.metrics import FrechetInceptionDistance as FID
    fid = FID(feature_dim=num_features, device=device)
    
    for real_batch, syn_batch in tqdm(zip(real_loader, syn_loader), total=len(real_loader)):
        if len(real_batch['image']) != len(syn_batch['image']):
            print(f"Batch size mismatch: {len(real_batch['image'])} vs {len(syn_batch['image'])}, skipping...")
            continue
        fid.update(real_batch['image'].to(device), is_real=True)
        fid.update(syn_batch['image'].to(device), is_real=False)  
    fid_score = fid.compute().item()
    
    return fid_score

def calculate_ssim(real_loader, syn_loader, device) -> float:
    """
    Calculate the SSIM score between the real and synthetic samples.
    For reference: https://lightning.ai/docs/torchmetrics/stable/image/ssim.html
    """
    from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
    ssim = SSIM()
    
    real_imgs = []
    syn_imgs = []
    for real_batch, syn_batch in tqdm(zip(real_loader, syn_loader), total=len(real_loader)):
        if len(real_batch['image']) != len(syn_batch['image']):
            print(f"Batch size mismatch: {len(real_batch['image'])} vs {len(syn_batch['image'])}, skipping...")
            continue
        real_imgs.append(real_batch['image'])
        syn_imgs.append(syn_batch['image'])
    real_imgs = torch.cat(real_imgs, dim=0)
    syn_imgs = torch.cat(syn_imgs, dim=0)
    
    ssim_score = ssim(syn_imgs, real_imgs).item()
    return ssim_score

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='pets', help='Dataset name')
    parser.add_argument('--real_data_dir', type=str, default='data/pets', help='Path to the dataset')
    parser.add_argument('--syn_data_path', type=str, default='data/pets', help='Path to the dataset')
    parser.add_argument('--metrics', type=str, default='fid', help='Metrics to calculate')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--split', type=str, default='train', help='Dataset split')
    
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    
    args = parser.parse_args()
    config = vars(args)
    print(f"Config: {config}")
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Load the dataset
    real_data_dir = Path(args.real_data_dir)
    syn_data_path = Path(args.syn_data_path)
    
    # construct real dataset
    train_transform = tfms.Compose([
        tfms.Resize((args.img_size, args.img_size)),
        tfms.ToTensor(),
    ])
    is_train = (args.split == 'train')
    if args.dataset_name == 'pets':
        real_dataset = PetsDataset(real_data_dir, train=is_train, transform=train_transform)
        real_loader = DataLoader(real_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)
        
        syn_dataset = PetsDatasetLMDB(syn_data_path, train=is_train, transform=train_transform)
        syn_loader = DataLoader(syn_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)
    elif args.dataset_name == 'caltech':
        real_dataset = Caltech101Dataset(real_data_dir, train=is_train, transform=train_transform)
        real_loader = DataLoader(real_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)
        
        syn_dataset = Caltech101DatasetLMDB(syn_data_path, train=is_train, transform=train_transform)
        syn_loader = DataLoader(syn_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)
    elif args.dataset_name == 'cars':
        real_dataset = StanfordCarsDataset(real_data_dir, train=is_train, transform=train_transform)
        real_loader = DataLoader(real_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)
        
        syn_dataset = StanfordCarsDatasetLMDB(syn_data_path, train=is_train, transform=train_transform)
        syn_loader = DataLoader(syn_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)
    elif args.dataset_name == 'flowers':
        real_dataset = Flowers102Dataset(real_data_dir, train=is_train, transform=train_transform)
        real_loader = DataLoader(real_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)
        
        syn_dataset = Flowers102DatasetLMDB(syn_data_path, train=is_train, transform=train_transform)
        syn_loader = DataLoader(syn_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset_name}")
    
    if args.metrics == 'fid':
        print("Calculating FID score...")
        fid = calculate_fid(real_loader, syn_loader, args.device)
        print(f"FID (on {args.split} set): {fid}")
    elif args.metrics == 'ssim':
        print("Calculating SSIM score...")
        ssim = calculate_ssim(real_loader, syn_loader, args.device)
        print(f"SSIM (on {args.split} set): {ssim}")
    else:
        raise ValueError(f"Invalid metrics: {args.metrics}")

if __name__ == '__main__':
    main()
    