import torch
from torch import Tensor
import tqdm
import cv2
import numpy as np

from sampling_functions import encode, sample, invert, \
    encode_image, encode_prompt, prepare_latents
from llm import generate_captions

from typing import List

def ddim_sampling(
    pipe,
    images: Tensor,
    prompt: str,
    num_inference_steps: int,
    device: str,
    num_samples: int=1,
    ddim_sampling_step=None,
    guidance_scale: float=3.5,
    do_classifier_free_guidance: bool=True,
    negative_prompt: str="",
    prefix="",
):
    if ddim_sampling_step is None:
        ddim_sampling_step = 0
    
    # encode image to latent space
    start_latents = encode(pipe, images, device=device)  # (B, c, h, w)  
    
    # invert latent space to image space. 
    inverted_latents = invert(
        pipe=pipe,
        start_latents=start_latents,
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_samples,
        do_classifier_free_guidance=do_classifier_free_guidance,
        negative_prompt=negative_prompt,
        device=device,
        prefix=prefix,
    )  # (n-2, B, c, h, w)
    
    if ddim_sampling_step < num_inference_steps:
        inverted_latents = inverted_latents[-(ddim_sampling_step + 1)]
    else:
        inverted_latents = inverted_latents[-1]
        
    # sample from inverted latent space
    samples = sample(
        pipe=pipe,
        prompt=prompt,
        start_latents=inverted_latents,
        start_step=ddim_sampling_step,
        num_inference_steps=num_inference_steps,
        prefix=prefix,
    )  # (B, c, h, w)
    
    return samples

@torch.no_grad()
def unclip_sampling(
    pipe, 
    images: Tensor,
    prompt: str,
    num_inference_steps: int,
    batch_size: int,
    device: str,
    num_samples: int=1,
    guidance_scale: float=3.5,
    noise_level: int = 0,
    output_type: str="image",
    prefix="",
    height: int=None,
    width: int=None,
) -> np.ndarray:
    do_classifier_free_guidance = guidance_scale > 0.0
    # Get Height and Width
    height = height or pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = width or pipe.unet.config.sample_size * pipe.vae_scale_factor

    # Encode prompt
    prompt_embeds, uncond_prompt_embeds = encode_prompt(pipe, prompt, batch_size, device, num_samples, do_classifier_free_guidance)
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([prompt_embeds, uncond_prompt_embeds])
    
    # Encode image
    image_embeds = encode_image(pipe, images, device, batch_size, num_samples, do_classifier_free_guidance, noise_level, None)
    image_embeds = image_embeds.to(device, dtype=prompt_embeds.dtype)
    
    # scheduler
    pipe.scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    
    num_channels_latents = pipe.unet.config.in_channels
    # Prepare latents
    latents = prepare_latents(
        pipe,
        batch_size*num_samples, 
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        latents=None,
    )
    
    print(f"{prefix}: Sampling Images for {num_inference_steps} steps...")
    for i, t in enumerate(pipe.progress_bar(timesteps)):
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        with torch.no_grad():
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                class_labels=image_embeds,
                return_dict=False
            )[0]
        
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # compute the previous noisy sample x_t -> x_t-1
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    if not output_type == "latent":
        images = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    else:
        images = latents

    images = images.cpu().detach()
    images = pipe.image_processor.postprocess(images, output_type=output_type)
    return images  # (B, C, H, W)

def text_sampling(
    pipe, 
    prompts: List[str],
    num_inference_steps: int,
    num_samples: int=1,
    guidance_scale: float=9,
    img_size: int=224,
    prefix="",
):  
    print(f"{prefix}: Sampling Images for {num_inference_steps} steps...")
    # tensor to list
    prompts = prompts.tolist()
    images = pipe(prompts, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, num_images_per_prompt=num_samples).images
    # images: [K*num_samples, PIL.Image], K is the number of prompts
    images = [np.array(img) for img in images]
    images = [cv2.resize(img, (img_size, img_size)) for img in images]
    return images  # (B, C, H, W)
    
    
    
    
    
    
    
    