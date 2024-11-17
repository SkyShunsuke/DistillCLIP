"""
Generate captions for images using large language model.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
from pathlib import Path

from typing import List, Dict, Any
import base64
from pprint import pprint
import time
import numpy as np
import torch
import cv2
import json
from tqdm.auto import tqdm
from accelerate import PartialState
from torch.utils.data import DataLoader
from torchvision import transforms as tfms
from diffusers import StableDiffusionPipeline, DDIMScheduler, StableUnCLIPImg2ImgPipeline
import warnings
from PIL import Image
warnings.filterwarnings("ignore")
from dotenv import load_dotenv

from openai import OpenAI

from dataset import PetsDataset, StanfordCarsDataset, Flowers102Dataset, Caltech101Dataset, STL10DatasetFolder

SUPPORTED_MODELS = [
    'gpt-4o-mini',
]

DEFAULT_PROMPTS = [
    "Describe this image in {} words.",
    "Generate a caption for this image in {} words.",
    "Generate a caption for each image with a maximum of {} words. Separate the captions for each image using a colon (':').",
]

def separate_captions(captions: str, num_captions: int, separator=":",):
    results = False
    captions = captions.split(separator)
    if len(captions) == num_captions:
        results = True
    return results, captions

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def create_message(encoded_images: List[str], detail_level: str, prompt: str, role="user"):
    messages = [
        {
            "role": role,
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
            ]
        }
    ]
    for i in range(len(encoded_images)):
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_images[i]}",
                "detail": detail_level
            },
        })
    return messages

def save_captions(captions: List[str], img_files: List[str], output_dir: Path, caption_file: str):
    """Save captions to a json file."""
    caption_data = {}
    for i, img_file in enumerate(img_files):
        caption_data[img_file] = captions[i]
    with open(output_dir / caption_file, 'w') as f:
        json.dump(caption_data, f)
    print(f"Captions saved to {output_dir / caption_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate captions for images using large language model.')
    # dataset
    parser.add_argument('--dataset', type=str, default='stl', help='Dataset to use for generating captions.')
    parser.add_argument('--data_path', type=str, default='data/stl', help='Path to dataset.')
    parser.add_argument('--with_test', action='store_true', help='Use test set instead of validation set.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for generating captions.')
    parser.add_argument('--image_size', type=int, default=224, help='Size of images to generate captions for.')
    parser.add_argument('--with_unlabeled', action='store_true', help='Use unlabelled data for generating captions.')
    parser.set_defaults(with_unlabeled=False)
    
    # llm
    parser.add_argument('--llm_type', type=str, default='gpt-4o-mini', help='Type of large language model to use.')
    parser.add_argument('--detail_level', type=str, default='low', help='Detail level of captions to generate.')
    parser.add_argument('--dotenv_path', type=str, default='.env', help='Path to .env file.')
    parser.add_argument('--num_captions_per_image', type=int, default=1, help='Number of captions to generate per image.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling captions.')
    parser.add_argument('--top_k', type=int, default=50, help='Top k for sampling captions.')
    parser.add_argument('--context_length', type=int, default=77, help='Maximum length of captions to generate.')
    parser.add_argument('--max_retry', type=int, default=3, help='Maximum number of retries for generating captions.')
    
    # output
    parser.add_argument('--output_dir', type=str, default='caption_data', help='Directory to save generated captions.')
    parser.add_argument('--caption_file', type=str, default='captions.h5', help='File to save generated captions.')
    
    args = parser.parse_args()
    
    config = {
        'dataset': args.dataset,
        'data_path': args.data_path,
        'with_test': args.with_test,
        'with_unlabeled': args.with_unlabeled,
        'batch_size': args.batch_size,
        'image_size': args.image_size,
        'llm_type': args.llm_type,
        'detail_level': args.detail_level,
        'num_captions_per_image': args.num_captions_per_image,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'context_length': args.context_length,
        'output_dir': args.output_dir,
        'caption_file': args.caption_file
    }
    pprint(config)
    
    if args.llm_type not in SUPPORTED_MODELS:
        raise ValueError(f"Large language model {args.llm_type} not supported. Supported models: {SUPPORTED_MODELS}")
    
    # Load environment variables
    if os.path.exists(args.dotenv_path):
        load_dotenv(args.dotenv_path)
    else:
        raise ValueError(f"Dotenv file not found at {args.dotenv_path}")
    
    # create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # load dataset
    transform = tfms.Compose([
        tfms.Resize((args.image_size, args.image_size)),
        tfms.ToTensor()
    ])
    if args.dataset == 'pets':
        train_dataset = PetsDataset(args.data_path, transform=transform, split='train')
        test_dataset = PetsDataset(args.data_path, transform=transform, split='test')
    elif args.dataset == 'stanford_cars':
        train_dataset = StanfordCarsDataset(args.data_path, transform=transform, split='train')
        test_dataset = StanfordCarsDataset(args.data_path, transform=transform, split='test')
    elif args.dataset == 'flowers102':
        train_dataset = Flowers102Dataset(args.data_path, transform=transform, split='train')
        test_dataset = Flowers102Dataset(args.data_path, transform=transform, split='test')
    elif args.dataset == 'caltech101':
        train_dataset = Caltech101Dataset(args.data_path, transform=transform, split='train')
        test_dataset = Caltech101Dataset(args.data_path, transform=transform, split='test')
    elif args.dataset == 'stl10':
        train_dataset = STL10DatasetFolder(args.data_path, transform=transform, train=True, with_unlabeled=args.with_unlabeled)
        test_dataset = STL10DatasetFolder(args.data_path, transform=transform, train=False, with_unlabeled=args.with_unlabeled)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    print(f"Train dataset: {len(train_dataset)}")
    print(f"Test dataset: {len(test_dataset)}")
    
    # load llm
    client = OpenAI()
    prompt = DEFAULT_PROMPTS[2].format(args.context_length)
    print(f"Generation Configuration: \n =====================")
    print(f"Prompt: {prompt}")
    print(f"Temperature: {args.temperature}")
    print(f"Top k: {args.top_k}")
    print(f"Context length: {args.context_length}")
    print(f"Detail level: {args.detail_level}")
    print(f" =====================")
    
    
    print("Generating captions for training set...")
    print(" =====================")
    train_captions = []
    train_files = []
    tqdm_loader = tqdm(train_loader, total=len(train_loader))
    for i, batch in enumerate(tqdm_loader):
        img_files: List[str] = batch['path']
        train_files.extend(img_files)
        encoded_images: List[str] = [encode_image(img_file) for img_file in img_files]
        msg = create_message(encoded_images, args.detail_level, prompt)
        
        cnt = 0
        while cnt < args.max_retry:
            response = client.chat.completions.create(
                model=args.llm_type,
                messages=msg,
                max_tokens=args.context_length * args.num_captions_per_image * args.batch_size,
            ).choices[0].message.content
            results, captions = separate_captions(response, args.batch_size * args.num_captions_per_image)
            if results:
                train_captions.extend(captions)
                break
            else:
                cnt += 1
                print(f"Retry {cnt} for batch {i}")
        
        if cnt == args.max_retry:
            print(f"Failed to generate captions for batch {i}")
            break
        
        
        tqdm_loader.set_description(f"Batch {i}")
    
    assert len(train_captions) == len(train_dataset), f"Number of captions generated {len(train_captions)} does not match number of images {len(train_dataset)}"
    
    if args.with_test:
        print("Generating captions for test set...")
        print(" =====================")
        test_captions = []
        test_files = []
        tqdm_loader = tqdm(test_loader, total=len(test_loader))
        for i, batch in enumerate(tqdm_loader):
            img_files: List[str] = batch['path']
            test_files.extend(img_files)
            encoded_images: List[str] = [encode_image(img_file) for img_file in img_files]
            msg = create_message(encoded_images, args.detail_level, prompt)
            
            cnt = 0
            while cnt < args.max_retry:
                response = client.chat.completions.create(
                    model=args.llm_type,
                    messages=msg,
                    max_tokens=args.context_length * args.num_captions_per_image * args.batch_size,
                ).choices[0].message.content
                results, captions = separate_captions(response, args.batch_size * args.num_captions_per_image)
                if results:
                    test_captions.extend(captions)
                    break
                else:
                    cnt += 1
                    print(f"Retry {cnt} for batch {i}")
            
            tqdm_loader.set_description(f"Batch {i}")
        
        assert len(test_captions) == len(test_dataset), f"Number of captions generated {len(test_captions)} does not match number of images {len(test_dataset)}"
    
    # save captions
    save_captions(train_captions, train_files, output_dir, f"train_{args.caption_file}")
    if args.with_test:
        save_captions(test_captions, test_files, output_dir, f"test_{args.caption_file}")
    
    print("Captions generated and saved successfully!")
    
if __name__ == '__main__':
    main()

    
    