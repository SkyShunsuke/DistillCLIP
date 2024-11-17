import torch
from torch import Tensor
import tqdm
from typing import Optional
from diffusers.models.embeddings import get_timestep_embedding
from PIL import Image

@torch.no_grad()
def encode(pipe, images: Tensor, device):
    images_scaled = images.to(device) * 2 - 1  # scale to [-1, 1]
    latents = pipe.vae.encode(images_scaled)
    l = 0.18215 * latents.latent_dist.sample()  # empriical value for ensure the latent has unit variance.
    return l

@torch.no_grad()
def sample(
    pipe,
    prompt,
    start_step=0,
    start_latents=None,
    guidance_scale=3.5,
    num_inference_steps=30,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device="cuda",
    prefix="",
):
    batch_size = num_images_per_prompt if start_latents is None else start_latents.shape[0]
    # encode prompt
    text_emb = pipe._encode_prompt(
        prompt, device, batch_size, do_classifier_free_guidance, negative_prompt
    )
    
    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    
    # create a random starting point if we don't have one
    if start_latents is None:
        start_latents = torch.randn(batch_size, 4, 64, 64, device=device)  # (1, 4, 64, 64) is the latent shape for diffusion models. (4 is the number of dimensions of the encoder)
        start_latents *= pipe.scheduler.init_noise_sigma
    
    latents = start_latents.clone()
    
    if prefix is not None:
        print(f"{prefix}: Sampling Images for {num_inference_steps} steps...")
    
    for i in tqdm.tqdm(range(start_step, num_inference_steps)):
        
        t = pipe.scheduler.timesteps[i]
        
        # expand latents for classifier guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents  # (2*B, C, H, W)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t) 
        
        # Predict the noise
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_emb).sample  # (2*B, C, H, W)
        
        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)  # (B, C, H, W), (B, C, H, W)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        prev_t = pipe.scheduler.timesteps[i+1] if i+1 < num_inference_steps else pipe.scheduler.timesteps[-1]  # t-1
        alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
        alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]
        predicted_x0 = (latents - (1-alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        direction_pointing_to_xt = (1-alpha_t_prev).sqrt() * noise_pred
        latents = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt
    
    # last generative step
    images = pipe.decode_latents(latents)  # pass stable diffusion decoder
    
    return images


@torch.no_grad()
def invert(
    pipe,
    start_latents, 
    prompt,
    guidance_scale=3.5,
    num_inference_steps=80,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device="cuda",
    prefix="",
):
    # import pdb; pdb.set_trace()
    batch_size = start_latents.shape[0]
    
    # encode prompt, #TODO: implement per image prompt encoding, currently it is same for all images.
    text_emb = pipe._encode_prompt(
        prompt, device, batch_size, do_classifier_free_guidance, negative_prompt
    )
    
    latents = start_latents.clone()
    
    # TODO: why keep intermediate latents
    intermediate_latents = []
    
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    
    timesteps = reversed(pipe.scheduler.timesteps)
    
    if prefix is not None:
        print(f"{prefix}: Encoding Images for {num_inference_steps} steps...")
        
    for i in tqdm.tqdm(range(1, num_inference_steps), total=num_inference_steps-1):
        #TODO: why skip final iteration, why start from 1.
        if i >= num_inference_steps - 1: continue
        
        t = timesteps[i]
        
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_emb).sample  # (2*B, C, H, W)
        
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)  # (B, C, H, W), (B, C, H, W)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
        current_t = max(0, t.item() - (1000//num_inference_steps))  # TODO: why?
        next_t = t
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]
        
        latents = (latents - (1-alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt()/alpha_t.sqrt()) + (1-alpha_t_next).sqrt() * noise_pred
        
        intermediate_latents.append(latents)
        
    return  intermediate_latents

def encode_prompt(
    pipe,
    prompt,
    batch_size,
    device,
    num_images_per_prompt,
    do_classifier_free_guidance,
    prompt_embeds: Optional[torch.Tensor] = None,
):
    r"""
    Encodes the prompt into text encoder hidden states.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            prompt to be encoded
        device: (`torch.device`):
            torch device
        num_images_per_prompt (`int`):
            number of images that should be generated per prompt
        do_classifier_free_guidance (`bool`):
            whether to use classifier free guidance or not
        prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
    """

    if prompt_embeds is None:
        text_inputs = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = pipe.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = pipe.tokenizer.batch_decode(
                untruncated_ids[:, pipe.tokenizer.model_max_length - 1 : -1]
            )

        if hasattr(pipe.text_encoder.config, "use_attention_mask") and pipe.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        prompt_embeds = pipe.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
        prompt_embeds = prompt_embeds[0]

    if pipe.text_encoder is not None:
        prompt_embeds_dtype = pipe.text_encoder.dtype
    elif pipe.unet is not None:
        prompt_embeds_dtype = pipe.unet.dtype
    else:
        prompt_embeds_dtype = prompt_embeds.dtype

    prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

    bs_embed, seq_len, _ = prompt_embeds.shape  # (1, S, C)
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(batch_size * num_images_per_prompt, 1, 1)  # (B, S, C)

    # get unconditional embeddings for classifier free guidance
    if do_classifier_free_guidance:
        uncond_tokens = [""] * batch_size

        max_length = prompt_embeds.shape[1]
        uncond_input = pipe.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        if hasattr(pipe.text_encoder.config, "use_attention_mask") and pipe.text_encoder.config.use_attention_mask:
            attention_mask = uncond_input.attention_mask.to(device)
        else:
            attention_mask = None

        uncond_prompt_embeds = pipe.text_encoder(
            uncond_input.input_ids.to(device),
            attention_mask=attention_mask,
        )
        uncond_prompt_embeds = uncond_prompt_embeds[0]

    if do_classifier_free_guidance:
        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = uncond_prompt_embeds.shape[1]

        uncond_prompt_embeds = uncond_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        uncond_prompt_embeds = uncond_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        uncond_prompt_embeds = uncond_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, uncond_prompt_embeds

def noise_image_embeddings(pipe, image_embeds, noise_level, noise=None):
    if noise is None:
        noise = torch.randn_like(image_embeds, device=image_embeds.device, dtype=image_embeds.dtype)
    
    noise_level = torch.tensor([noise_level] * image_embeds.shape[0], device=image_embeds.device)
    pipe.image_normalizer.to(image_embeds.device)
    
    image_embeds = pipe.image_normalizer.scale(image_embeds)
    image_embeds = pipe.image_noising_scheduler.add_noise(image_embeds, timesteps=noise_level, noise=noise)
    image_embeds = pipe.image_normalizer.unscale(image_embeds)
    
    noise_level = get_timestep_embedding(
        timesteps=noise_level, embedding_dim=image_embeds.shape[-1], flip_sin_to_cos=True, downscale_freq_shift=0
    ).to(image_embeds.device)
    
    image_embeds = torch.cat([image_embeds, noise_level], dim=1)  
    return image_embeds

def encode_image(pipe, image, device, batch_size, num_images_per_prompt, do_classifier_guidance, noise_level, image_embeds):
    dtype = next(pipe.image_encoder.parameters()).dtype 
    if isinstance(image, Image.Image):
        repeat_by = batch_size
    else:
        repeat_by = num_images_per_prompt
    
    if image_embeds is None:
        if not isinstance(image, torch.Tensor):
            image = pipe.feature_extractor(images=image, return_tensors="pt").pixel_values
        image = image.to(device, dtype=dtype)   
        image_embeds = pipe.image_encoder(image).image_embeds
    
    # add noise to image embeddings
    image_embeds = noise_image_embeddings(
        pipe, image_embeds, noise_level
    )  # (batch_size, image_embeds_dim)
    
    image_embeds = image_embeds.unsqueeze(1)
    bs_embed, seq_len, _ = image_embeds.shape
    image_embeds = image_embeds.repeat(1, repeat_by, 1)
    image_embeds = image_embeds.view(bs_embed * repeat_by, seq_len, -1)
    image_embeds = image_embeds.squeeze(1)
    # (batch_size * num_images_per_prompt, image_embeds_dim)
    
    if do_classifier_guidance:
        # for unconditional guidance, we use zeroed out embeddings
        uncond_image_embeds = torch.zeros_like(image_embeds)
        image_embeds = torch.cat([uncond_image_embeds, image_embeds])
    return image_embeds

def prepare_latents(pipe, batch_size, num_channels_latents, height, width, dtype, device, latents=None):
    shape = (
        batch_size,
        num_channels_latents,
        int(height) // pipe.vae_scale_factor,
        int(width) // pipe.vae_scale_factor,
    )
    if latents is None:
        latents = torch.randn(shape, dtype=dtype, device=device)
    else:
        latents = latents.to(dtype=dtype, device=device)

    # scale the initial noise by the standard deviation of the unet
    latents = latents * pipe.scheduler.init_noise_sigma
    return latents

