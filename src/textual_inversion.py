from diffusers import DDIMScheduler, StableDiffusionPipeline
from transformers.models.clip.modeling_clip import _create_4d_causal_attention_mask
import numpy as np
import torch
from torch import nn
from torchvision import transforms
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import argparse

from pprint import pprint
import inspect
from typing import List, Optional, Tuple, Union

import tensorboardX as tbx

from dataset import STL10Dataset
from utils import LogMeter

def main():
    parser = argparse.ArgumentParser(description="Optimize prompt for training data")
    
    # data
    parser.add_argument("--data", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--split", type=str, default="train", help="Split to use")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for data loader")
    parser.add_argument("--image_size", type=int, default=256, help="Image size")
    
    # model
    parser.add_argument("--model_name", type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--num_inference_steps", type=int, default=1000, help="Number of inference steps")
    parser.add_argument("--embedding_type", type=str, default="clip")
    parser.add_argument("--num_embeds_per_class", type=int, default=1, help="Number of embeddings per class")
    parser.add_argument("--initialization", type=str, default="random", help="Initialization of prompt embeddings")
    
    # optimization
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer to use")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument("--scheduler", type=str, default=None, help="Scheduler to use")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--save_freq", type=int, default=10, help="Save frequency")
    
    # save
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--save_name", type=str, default="prompt.pt", help="Name of the saved prompt")
    
    # logging
    parser.add_argument("--log_freq", type=int, default=10, help="Log frequency")
    parser.add_argument("--log_name", type=str, default="log.txt", help="Name of the log file")
    parser.add_argument("--log_images", type=bool, default=False, help="Log images")
    
    # Other
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    pprint(vars(args))
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # Load dataset
    basic_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    if args.dataset == "stl10":
        dataset = STL10Dataset(root=args.data, transform=basic_transform, train=(args.split == "train"))
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    
    # Load model
    pipe = StableDiffusionPipeline.from_pretrained(args.model_name, torch_dtype=torch.float32)
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
    )
    pipe = pipe.to(args.device)

    classes = dataset.classes
    base_prompt = dataset.base_prompt
    num_classes = dataset.num_classes
    num_dims = 1024 # TODO: It is for SD2.1
    num_embeds_per_class = args.num_embeds_per_class
    
    # Initialize prompt embeddings
    if args.initialization == "random":
        class_weights = nn.Parameter(torch.randn(1, num_classes, num_embeds_per_class, num_dims, device=args.device))
    elif args.initialization == "zero":
        class_weights = nn.Parameter(torch.zeros(1, num_classes, num_embeds_per_class, num_dims, device=args.device))
    elif args.initialization == "text_embed":
        class_weights = nn.Parameter(init_from_text_embeds(pipe, classes, args.embedding_type, num_embeds_per_class, num_dims, args.device))
    
    # Retrieve timesteps
    timesteps, _ = retrieve_timesteps(
        pipe.scheduler, args.num_inference_steps, args.device, timesteps=None, sigmas=None
    )
    
    # Initialize optimizer
    param_groups = [{"params": class_weights}]
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(param_groups, lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {args.optimizer} not implemented")
    
    # Initialize scheduler
    if args.scheduler is not None:
        scheduler = torch.optim.lr_scheduler.__dict__[args.scheduler](optimizer)
    else:
        scheduler = None
    
    # Initialize logging
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = tbx.SummaryWriter(log_dir)
    
    # Train Loop
    print("Starting training...")
    for epoch in range(args.num_epochs):
        loss_meter = LogMeter()
        for i, batch in enumerate(dataset_loader):
            images = batch["image"].to(args.device)
            prompts = batch["caption"]
            
            # Encode images
            z = encode_image(images, pipe, args.device, args.batch_size)
            
            # Get class embeddings
            embeds = get_embeddings(prompts, class_weights, num_embeds_per_class, classes, base_prompt, args.device)
            
            if args.embedding_type == "clip":
                # Get latents from prompt embeddings (optimzie in the clip latent space)
                embeds = get_clip_latents(embeds, pipe, -1, base_prompt, args.device)
            elif args.embedding_type == "text":
                # Get latents from prompt embeddings (optimize in the text latent space)
                embeds = forward_clip_latents(embeds, pipe, -1, base_prompt, args.device)
            
            # Scheduler class provides the function to add noise to the image. 
            # i.e., x_t = \sqrt{alpha_t_bar} * x + sqrt{1 - alpha_t_bar} * N(0, I)
            t_idx= torch.randint(0, args.num_inference_steps, (1,)).item()
            eps = torch.randn(*z.shape).to(args.device)
            noised_x = pipe.scheduler.add_noise(z, eps, timesteps[t_idx])
            noised_x = noised_x.to(args.device)
            
            # Denoise the image
            noise_pred = pipe.unet(
                noised_x, timesteps[t_idx], encoder_hidden_states=embeds, timestep_cond=None, return_dict=False,
            )[0]
            
            loss = torch.nn.functional.mse_loss(noise_pred, eps)  
            
            # Update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log loss
            loss_meter.update(loss.item())
            if i % args.log_freq == 0:
                print(f"Epoch: {epoch}, Iter: {i}, Loss: {loss_meter.avg}")
                logger.add_scalar("loss", loss_meter.avg, epoch * len(dataset_loader) + i)
        
        if scheduler is not None:
            scheduler.step()
        
        # Save embeds
        if epoch % args.save_freq == 0:
            save_path = os.path.join(args.output_dir, args.save_name + f"_{epoch}.pt")
            save_prompt(save_path, class_weights, num_classes, args, base_prompt)
    
    # Save final prompt
    save_path = os.path.join(args.output_dir, args.save_name + f"_latest.pt")
    save_prompt(save_path, class_weights, num_classes, args, base_prompt)
    print("Training completed!")

def init_from_text_embeds(pipe, classes: List[str], embed_type: str, num_embeds_per_class: int, num_dims: int, device: str = "cuda", base_prompt: str = "A photo of a "):
    """Initialize prompt embeddings from text embeddings. 
    Args:
        pipe (StableDiffusionPipeline): Diffusion pipeline
        classes (List[str]): List of classes
        embed_type (str): Type of embeddings
        num_embeds_per_class (int): Number of embeddings per class
        num_dims (int): Number of dimensions
        device (str): Device
        base_prompt (str): Base prompt
    Returns:
        torch.Tensor: Initialized prompt embeddings, (1, K, N, D)
    """
    # TODO: We assume class name is just one word.
    class_embeds = torch.zeros(1, len(classes), num_embeds_per_class, num_dims, device=device)
    if embed_type == "clip":
        for i, class_name in enumerate(classes):
            prompt = base_prompt + class_name
            prmpt_embed = pipe.encode_prompt(
                prompt, device, num_embeds_per_class, True, negative_prompt=None
            )[1]  # (N, M, D)
            length = len(pipe.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids[0])
            cls_embed = prmpt_embed[:, length - 2]  # (N, D)
            class_embeds[0, i] = cls_embed
    elif embed_type == "text":
        model = pipe.text_encoder.text_model
        all_class_text = " ".join(classes)
        input_ids = pipe.tokenizer(all_class_text, padding="longest", return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        embeds = model.embeddings(input_ids)  # (1, 1 + K + 1, D)
        # extract first and start tokens
        class_embeds = embeds[:, 1:-1]  # (1, K, D)
        class_embeds = class_embeds.unsqueeze(2).repeat(1, 1, num_embeds_per_class, 1)  # (1, K, N, D)
    return class_embeds
        
def save_prompt(
    save_path: str, class_weights: torch.Tensor, num_classes, args: argparse.Namespace, base_prompt: str = "A photo of a "
):
    state_dict = {
        "num_embeds_per_prompt": args.num_embeds_per_class,
        "num_classes": num_classes,
        "base_prompt": base_prompt,
        "embedding_type": args.embedding_type,
        "prompt_embeds": class_weights,
    }
    torch.save(state_dict, save_path)
    print(f"Saved prompt to {save_path}")
        
        
def encode_image(
    images: torch.Tensor, pipe: StableDiffusionPipeline, device: str = "cuda", batch_size: int = 1
):
    """Encode images into VAE latents
    Args: 
        images (torch.Tensor): Images, (B, C, H, W)
        pipe (StableDiffusionPipeline): Diffusion pipeline
        device (str): Device
        batch_size (int): Batch size
    Returns:
        torch.Tensor: Latents, (B, d, h, w)
    """
    with torch.no_grad():
        z_dist = pipe.vae.encode(images)
    z = z_dist.latent_dist.sample()
    return z

def get_embeddings(
    prompts: List[str], prompt_embeds: torch.Tensor, num_embeds_per_class: int, classes: List[str], \
        base_prompt: str = "A photo of a ", device: str = "cuda"
):
    """Get prompt embeddings for each class
    Args:
        prompts (List[str]): List of prompts, length of B
        prompt_embeds (torch.Tensor): Prompt embeddings, (1, K, N, D)
        num_embeds_per_class (int): Number of embeddings per class
        classes (List[str]): List of classes
    Returns:
        torch.Tensor: embeddings for each class, (B, N, D)
    """
    B = len(prompts)
    prompt_embeds = prompt_embeds.repeat(B, 1, 1, 1) # (B, K, N, D)
    class_indices = [classes.index(p.split(base_prompt)[1]) for p in prompts]
    class_indices = torch.Tensor(class_indices).long().to(device).unsqueeze(1).unsqueeze(2).unsqueeze(3) # (B, 1, 1, 1)
    class_indices = class_indices.expand(B, 1, num_embeds_per_class, prompt_embeds.size(-1)) # (B, 1, N, D)
    class_embeds = torch.gather(prompt_embeds, 1, class_indices).squeeze(1)   # (B, N, D)
    return class_embeds

def forward_clip_latents(
    class_embeds: torch.Tensor, pipe: StableDiffusionPipeline, max_length: int = -1, base_prompt: str = "A photo of a ", device: str = "cuda"
):
    """Compute latents from prompt embeddings
    Args:
        embeds (torch.Tensor): Embeddings for each class, (B, num_embeds_per_class, D)
        pipe (StableDiffusionPipeline): Diffusion pipeline
        max_length (int): Maximum number of tokens
        base_prompt (str): Base prompt
        num_inference_steps (int): Number of inference steps
    Returns:
        torch.Tensor: Latents for each class, (1, M, D), M is the maximum number of tokens.
    """
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder.text_model
    base_ids = tokenizer(base_prompt, padding="longest", return_tensors="pt").input_ids.to(device)
    base_embeds = text_encoder.embeddings(base_ids)  # (1, 1 + B + 1, D)
    # extract first and end tokens
    start_token = base_embeds[:, 0]  # (1, D)
    end_token = base_embeds[:, -1]  # (1, D)
    base_embeds = base_embeds[:, 1:-1]  # (1, b, D)
    pad_token = text_encoder.embeddings(torch.tensor([[0]]).to(device)).squeeze(0)  # (1, D)
    
    # Concat class_embeds with base_embeds
    B, N, D = class_embeds.size()
    input_shape = tokenizer(base_prompt, padding="max_length", return_tensors="pt").input_ids.size()  # (1, M)
    max_length = max_length if max_length > 0 else input_shape[1]
    pad_length = max_length - (N + base_embeds.size(1) + 2)
    pad_token = pad_token.unsqueeze(0).expand(B, pad_length, -1)  # (B, pad_length, D)
    end_token = end_token.unsqueeze(0).expand(B, 1, -1)  # (B, 1, D)
    start_token = start_token.unsqueeze(0).expand(B, 1, -1)  # (B, 1, D)
    base_embeds = base_embeds.expand(B, -1, -1)  # (B, b, D)
    new_embeds = torch.cat([start_token, base_embeds, class_embeds, end_token, pad_token], dim=1)  # (B, M, D)
    
    # Feed new_embeds to the model
    causal_attention_mask = _create_4d_causal_attention_mask(
        input_shape, new_embeds.dtype, device=new_embeds.device
    )

    last_hidden_state = text_encoder.encoder(
        inputs_embeds=new_embeds,
        causal_attention_mask=causal_attention_mask,
        return_dict=True,
    )[0]
    last_hidden_state = text_encoder.final_layer_norm(last_hidden_state)
    return last_hidden_state

def get_clip_latents(
    class_embeds: torch.Tensor, pipe: StableDiffusionPipeline, max_length: int = -1, base_prompt: str = "A photo of a ", device: str = "cuda"
):
    """Compute latents from prompt embeddings
    Args:
        embeds (torch.Tensor): Embeddings for each class, (B, num_embeds_per_class, D)
        pipe (StableDiffusionPipeline): Diffusion pipeline
        max_length (int): Maximum number of tokens
        base_prompt (str): Base prompt
        num_inference_steps (int): Number of inference steps
    Returns:
        torch.Tensor: Latents for each class, (1, M, D), M is the maximum number of tokens.
    """
    # First compute prompt_embeddings for base prompt
    B, N = class_embeds.size(0), class_embeds.size(1)
    with torch.no_grad():
        prompt_embed = pipe.encode_prompt(
            base_prompt, device, B, True, negative_prompt=None
        )[1]  # (B, M, D)
    max_length = prompt_embed.size(1)
    base_prompt_length = len(pipe.tokenizer(base_prompt, padding="longest", return_tensors="pt").input_ids[0])
    
    # Then we extract special token from prompt_embeds
    pad_token = prompt_embed[:, -1] # (B, D)
    end_token = prompt_embed[:, base_prompt_length - 1] # (B, D)
    base_embeds = prompt_embed[:, :base_prompt_length - 1] # (B, base_prompt_length - 1, D)
    
    # We concatenate base_embeds with class_embeds
    pad_length = max_length - (N + base_prompt_length)
    assert pad_length >= 0, f"Number of embeddings per class: {N} is too large"
    end_token = end_token.unsqueeze(1)  # (B, 1, D)
    pad_token = pad_token.unsqueeze(1).expand(B, pad_length, -1) # (B, pad_length, D)
    new_embeds = torch.cat([base_embeds, class_embeds, end_token, pad_token], dim=1) # (B, M, D)
    assert new_embeds.size(1) == max_length, f"Size of new_embeds: {new_embeds.size(1)} is not equal to max_length: {max_length}"
    
    return new_embeds

def generate_images(
    class_embeds: torch.Tensor, pipe: StableDiffusionPipeline, num_inference_steps: int, device: str = "cuda", base_prompt: str = "A photo of a ", \
        num_images_per_prompt: int = 1, logger: tbx.SummaryWriter = None, prefix: str = None
):
    """Generate images from prompt embeddings
    Args:
        class_embeds (torch.Tensor): Embeddings for each class, (B, N, D)
        pipe (StableDiffusionPipeline): Diffusion pipeline
        num_inference_steps (int): Number of inference steps
        base_prompt (str): Base prompt
        device (str): Device
        base_prompt (str): Base prompt
        num_images_per_prompt (int): Number of images per prompt
        logger (tbx.SummaryWriter): Logger
        prefix (str): Prefix for logging
    Returns:
        torch.Tensor: Generated images, (B, C, H, W)
    """
    raise NotImplementedError("Function not implemented")
    
    
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    