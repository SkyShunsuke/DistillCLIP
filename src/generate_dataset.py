"""
Generate synthetic dataset.
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
from diffusers import StableDiffusionPipeline, DDIMScheduler, StableUnCLIPImg2ImgPipeline
import warnings
from PIL import Image
warnings.filterwarnings("ignore")

from dataset import PetsDataset, StanfordCarsDataset, Flowers102Dataset, Caltech101Dataset, STL10Dataset, STL10DatasetFolder
from sampling import ddim_sampling, unclip_sampling, text_sampling, textual_inversion

def get_diffusion_pipe(model_name, device):    
    print(f"Loading model: {model_name}")
    if model_name == "runwayml/stable-diffusion-v1-5":
        pipe = StableDiffusionPipeline.from_pretrained(model_name).to(device)
    elif model_name == "stabilityai/stable-diffusion-2-1-unclip":
        pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    elif model_name == "stabilityai/stable-diffusion-2-1":
        pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    return pipe

def postprocess(images, image_size):
    images = [(img * 255.0).astype(np.uint8) for img in images]
    images = [cv2.resize(img, (image_size, image_size)) for img in images]
    return images
    
def save_to_hdf5(hdf5_path, img_path, caption, cls, img, split, sample_idx, compression="gzip", n_trials=10, n_sleep=1):
    success = False
    for i in range(n_trials):
        try:
            with h5py.File(hdf5_path, "a") as f:
                filename = img_path.split("/")[-1]  
                clsname = filename.split("_")[0]
                save_path = f"{split}/{cls}/{filename}/{sample_idx}"
                f.create_dataset(save_path, data=img, compression=compression)
                
                f[save_path].attrs["label"] = cls
                f[save_path].attrs["filename"] = filename
                f[save_path].attrs["class"] = clsname
                f[save_path].attrs["path_real"] = str(img_path)
                f[save_path].attrs["caption"] = str(caption)
                success = True
                break
        except BlockingIOError:
            time.sleep(n_sleep)
    assert success, f"Failed to save {img_path} to {hdf5_path}"
        
def generate(dist, dataloader, pipe, args, hdf5_path, split, prompt=""):
    num_process = dist.num_processes
    assert args.batch_size % num_process == 0, "Batch size should be divisible by the number of processes"
    
    for i, batch in tqdm(enumerate(dataloader)):
        assert args.batch_size % num_process == 0, "Batch size should be divisible by the number of processes"
        batch_size_on_node = args.batch_size // num_process
        
        images = batch["image"].pixel_values[0]
        paths = np.array(batch["path"])
        labels = np.array(batch["label"])
        captions = np.array(batch["caption"])        
        base_prompt = dataloader.dataset.base_prompt
        with dist.split_between_processes(list(range(len(images)))) as idxs:
            assert len(idxs) == batch_size_on_node, f"Batch size on node: {len(idxs)} should be equal to {batch_size_on_node}"
            current_process_id = dist.process_index
            print(f"Current process => [{current_process_id}/{num_process}], Split: {split}, Batch: {i}/{len(dataloader)}")
            if args.generation_type == "ddim":
                samples = ddim_sampling(
                    pipe=pipe,
                    images=images[idxs],
                    prompt=prompt,
                    num_inference_steps=args.num_steps,
                    device=args.device,
                    num_samples=args.num_samples,
                    ddim_sampling_step=args.ddim_sampling_step,
                    guidance_scale=args.guidance_scale,
                    do_classifier_free_guidance=True,
                    negative_prompt="",
                    prefix=f"[{split}] {i}/{len(dataloader)}",
                )  # (B, H, W, C)
        
                samples = postprocess(samples, args.image_size)
            elif args.generation_type == "unclip":
                samples = unclip_sampling(
                    pipe=pipe,
                    images=images[idxs],
                    prompt=prompt,
                    num_inference_steps=args.num_steps,
                    batch_size=batch_size_on_node,
                    device=args.device,
                    num_samples=args.num_samples,
                    guidance_scale=args.guidance_scale,
                    noise_level=0,
                    output_type="image",
                    prefix=f"[{split}] {i}/{len(dataloader)}",
                )
                samples = postprocess(samples, args.image_size)
            elif args.generation_type == "txt2img":
                samples = text_sampling(
                    pipe=pipe,
                    prompts=captions[idxs],
                    num_inference_steps=args.num_steps,
                    num_samples=args.num_samples,
                    guidance_scale=args.guidance_scale,
                    img_size=args.image_size,
                    prefix=f"[{split}] {i}/{len(dataloader)}",
                )
            elif args.generation_type == "inversion":
                sample = textual_inversion(
                    prompt_embeds=args.embeds_path, 
                    pipe=pipe,
                    captions=captions[idxs],
                    num_inference_steps=args.num_steps,
                    num_samples=args.num_samples,
                    guidance_scale=args.guidance_scale,
                    img_size=args.image_size,
                    prefix=f"[{split}] {i}/{len(dataloader)}",
                    base_prompt=base_prompt,
                )
            else:
                raise NotImplementedError(f"Invalid generation type: {args.generation_type}")
            # sample = samples[0]
            # import matplotlib.pyplot as plt
            # plt.imsave(f"recon_{split}_{i}.png", sample)
            # break
            for j, img in enumerate(samples):
                img_idx = j // args.num_samples
                sample_idx = j % args.num_samples
                save_to_hdf5(hdf5_path, paths[idxs][img_idx], captions[idxs][img_idx], labels[idxs][img_idx], img, split, sample_idx)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_type", type=str, default="ddim", help="Type of generation: ddim, unclip, or txt2img")
    parser.add_argument("--inversion_type", type=str, default="ddim", help="Type of inversion: ddim or diffusion")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per real image or caption")
    
    # perturbation parameters
    parser.add_argument("--ddim_perturbation", type=str, default=None, help="Perturbation type for DDIM inversion: None, x0, xt, xT, enc, dec, bottleneck")
    parser.add_argument("--diffusion_perturbation", type=str, default=None, help="Perturbation type for diffusion inversion: None, x0, xt, xT, enc, dec, bottleneck")
    parser.add_argument("--perturbation_scale", type=float, default=1.0, help="Perturbation scale")
    parser.add_argument("--pertubation_dist", type=str, default="uniform", help="Perturbation distribution: uniform or normal")
    
    # diffusion parameters
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5", help="Model name for diffusion")
    parser.add_argument("--num_steps", type=int, default=100, help="Number of diffusion steps")
    parser.add_argument("--ddim_sampling_step", type=int, default=10, help="Number of reverse steps for DDIM inversion")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale for DDIM inversion")
    parser.add_argument("--empty_prompt", action="store_true", help="Use empty prompt for DDIM inversion")
    
    parser.add_argument("--dataset", type=str, default="pets", help="Dataset name")
    parser.add_argument("--data_path", type=str, default="data", help="Directory to save generated samples")
    parser.add_argument("--caption_path", type=str, default=None, help="Path to the caption file")
    parser.add_argument("--with_test", action="store_true", help="Generate samples for test set")
    parser.add_argument("--image_size", type=int, default=256, help="Image size")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--new_dataset_name", type=str, default="new_dataset", help="Name of the new dataset")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save generated samples")
    
    args = parser.parse_args()
    
    # setup distributed inference
    distributed_state = PartialState()  # this automatically sets up distributed inference
    args.device = distributed_state.device 
    
    # create output directory
    if not os.path.exists(args.output_dir): print(f"Creating output directory: {args.output_dir}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # set random seed
    # np.random.seed(args.seed)
    # random.seed(args.seed)
    # torch.manual_seed(args.seed)
    config = vars(args)
    print(f"Config: {config}")
    
    pipe = get_diffusion_pipe(args.model_name, args.device)
    if args.inversion_type == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
        )
    
    dataset_config = {
        "root": args.data_path,
        "download": False,
        "transform": None,
        "train": True,
    }
    if "pets" in args.dataset:
        if args.generation_type == "ddim":
            pets_transform = tfms.Compose(
                [
                    tfms.transforms.Resize((args.image_size, args.image_size)),
                    tfms.transforms.ToTensor(),
                ]
            )
        else:
            pets_transform = pipe.feature_extractor
        dataset_config["transform"] = pets_transform
        train_dataset = PetsDataset(**dataset_config)
        dataset_config["train"] = False
        val_dataset = PetsDataset(**dataset_config)
    elif "cars" in args.dataset:
        if args.generation_type == "ddim":
            cars_transform = tfms.Compose(
                [
                    tfms.transforms.Resize((args.image_size, args.image_size)),
                    tfms.transforms.ToTensor(),
                ]
            )
        else:
            cars_transform = pipe.feature_extractor
        dataset_config["transform"] = cars_transform
        train_dataset = StanfordCarsDataset(**dataset_config)
        dataset_config["train"] = False
        val_dataset = StanfordCarsDataset(**dataset_config)
    elif "flowers" in args.dataset:
        if args.generation_type == "ddim":
            flowers_transform = tfms.Compose(
                [
                    tfms.transforms.Resize((args.image_size, args.image_size)),
                    tfms.transforms.ToTensor(),
                ]
            )
        else:
            flowers_transform = pipe.feature_extractor
        dataset_config["transform"] = flowers_transform
        train_dataset = Flowers102Dataset(**dataset_config)
        dataset_config["train"] = False
        val_dataset = Flowers102Dataset(**dataset_config)
    elif "caltech" in args.dataset:
        if args.generation_type == "ddim":
            caltech_transform = tfms.Compose(
                [
                    tfms.transforms.Resize((args.image_size, args.image_size)),
                    tfms.transforms.ToTensor(),
                ]
            )
        else:
            caltech_transform = pipe.feature_extractor
        dataset_config["transform"] = caltech_transform
        train_dataset = Caltech101Dataset(**dataset_config)
        dataset_config["train"] = False
        val_dataset = Caltech101Dataset(**dataset_config)
    elif "stl" in args.dataset:
        if args.generation_type == "ddim":
            stl_transform = tfms.Compose(
                [
                    tfms.transforms.Resize((args.image_size, args.image_size)),
                    tfms.transforms.ToTensor(),
                ]
            )
        else:
            stl_transform = pipe.feature_extractor
        dataset_config["transform"] = stl_transform

        if args.caption_path is not None:
            dataset_config["caption_file"] = args.caption_path
            train_dataset = STL10DatasetFolder(**dataset_config)
            dataset_config["train"] = False
            val_dataset = STL10DatasetFolder(**dataset_config)
        else:
            dataset_config["caption_file"] = args.caption_path
            train_dataset = STL10DatasetFolder(**dataset_config)
            dataset_config["train"] = False
            val_dataset = STL10DatasetFolder(**dataset_config)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    
    train_dataloader = DataLoader(
        train_dataset, args.batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=True
    )
    test_dataloader = DataLoader(
        val_dataset, args.batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=True
    )
    print(f"Drop last {len(train_dataset)% args.batch_size} samples from the dataloader")
    
    classes = train_dataset.classes
    print(f"Classes: {classes}")
    hdf5_path = os.path.join(args.output_dir, f"{args.generation_type}_{args.new_dataset_name}.hdf5")
    
    if args.generation_type == "ddim":
        if args.inversion_type == "ddim":
            prompt = "" if args.empty_prompt else NotImplementedError("Prompt is not empty")
            generate(distributed_state, train_dataloader, pipe, args, hdf5_path, split="train", prompt=prompt)

            if args.with_test:
                generate(distributed_state, test_dataloader, pipe, args, hdf5_path, split="test", prompt=prompt)
        else:
            raise NotImplementedError(f"Invalid inversion type: {args.inversion_type}")
    elif args.generation_type == "unclip":
        generate(distributed_state, train_dataloader, pipe, args, hdf5_path, split="train")
        if args.with_test:
            generate(distributed_state, test_dataloader, pipe, args, hdf5_path, split="test")
    elif args.generation_type == "txt2img":
        generate(distributed_state, train_dataloader, pipe, args, hdf5_path, split="train")
        if args.with_test:
            generate(distributed_state, test_dataloader, pipe, args, hdf5_path, split="test")
    else:
        raise NotImplementedError(f"Invalid generation type: {args.generation_type}")
    
    print(f"Generated samples are saved to {hdf5_path}")
                
                
if __name__ == "__main__":
    main()