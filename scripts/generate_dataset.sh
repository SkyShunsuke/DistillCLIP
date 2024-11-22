export CUDA_VISIBLE_DEVICES=1,2,3,4,5 # ,2,3,4,5
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# accelerate launch --num_processes 5 src/generate_dataset.py \
#     --generation_type txt2img \
#     --num_samples 1 \
#     --model_name stabilityai/stable-diffusion-2-1 \
#     --num_steps 20 \
#     --guidance_scale 9 \
#     --dataset stl10 \
#     --data_path data/stl10_images \
#     --caption_path caption_data/train_stl10_train_labeled_gpt4omini_low.json \
#     --image_size 96 \
#     --device cuda \
#     --output_dir syn_data \
#     --batch_size 50 \
#     --new_dataset_name stl10_captiontxi_s9_n20_x1 \

accelerate launch --num_processes 5 src/generate_dataset.py \
    --generation_type unclip \
    --num_samples 2 \
    --model_name stabilityai/stable-diffusion-2-1-unclip \
    --num_steps 20 \
    --guidance_scale 9 \
    --dataset stl10 \
    --data_path data/stl10_images \
    --image_size 96 \
    --device cuda \
    --output_dir syn_data \
    --batch_size 25 \
    --new_dataset_name stl10_imagetxi_s9_n20_x2 \

accelerate launch --num_processes 5 src/generate_dataset.py \
    --generation_type unclip \
    --num_samples 4 \
    --model_name stabilityai/stable-diffusion-2-1-unclip \
    --num_steps 20 \
    --guidance_scale 9 \
    --dataset stl10 \
    --data_path data/stl10_images \
    --image_size 96 \
    --device cuda \
    --output_dir syn_data \
    --batch_size 15 \
    --new_dataset_name stl10_imagetxi_s9_n20_x4 \

accelerate launch --num_processes 5 src/generate_dataset.py \
    --generation_type unclip \
    --num_samples 8 \
    --model_name stabilityai/stable-diffusion-2-1-unclip \
    --num_steps 20 \
    --guidance_scale 9 \
    --dataset stl10 \
    --data_path data/stl10_images \
    --image_size 96 \
    --device cuda \
    --output_dir syn_data \
    --batch_size 5 \
    --new_dataset_name stl10_imagetxi_s9_n20_x8 \

accelerate launch --num_processes 5 src/generate_dataset.py \
    --generation_type unclip \
    --num_samples 16 \
    --model_name stabilityai/stable-diffusion-2-1-unclip \
    --num_steps 20 \
    --guidance_scale 9 \
    --dataset stl10 \
    --data_path data/stl10_images \
    --image_size 96 \
    --device cuda \
    --output_dir syn_data \
    --batch_size 5 \
    --new_dataset_name stl10_imagetxi_s9_n20_x16 \



# accelerate launch --num_processes 5 src/generate_dataset.py \
#     --generation_type txt2img \
#     --num_samples 2 \
#     --model_name stabilityai/stable-diffusion-2-1 \
#     --num_steps 20 \
#     --guidance_scale 9 \
#     --dataset stl10 \
#     --data_path data/stl10_images \
#     --caption_path caption_data/train_stl10_train_labeled_gpt4omini_low.json \
#     --image_size 96 \
#     --device cuda \
#     --output_dir syn_data \
#     --batch_size 25 \
#     --new_dataset_name stl10_captiontxi_s9_n20_x2 \

# accelerate launch --num_processes 5 src/generate_dataset.py \
#     --generation_type txt2img \
#     --num_samples 4 \
#     --model_name stabilityai/stable-diffusion-2-1 \
#     --num_steps 20 \
#     --guidance_scale 9 \
#     --dataset stl10 \
#     --data_path data/stl10_images \
#     --caption_path caption_data/train_stl10_train_labeled_gpt4omini_low.json \
#     --image_size 96 \
#     --device cuda \
#     --output_dir syn_data \
#     --batch_size 15 \
#     --new_dataset_name stl10_captiontxi_s9_n20_x4 \

# accelerate launch --num_processes 5 src/generate_dataset.py \
#     --generation_type txt2img \
#     --num_samples 8 \
#     --model_name stabilityai/stable-diffusion-2-1 \
#     --num_steps 20 \
#     --guidance_scale 9 \
#     --dataset stl10 \
#     --data_path data/stl10_images \
#     --caption_path caption_data/train_stl10_train_labeled_gpt4omini_low.json \
#     --image_size 96 \
#     --device cuda \
#     --output_dir syn_data \
#     --batch_size 10 \
#     --new_dataset_name stl10_captiontxi_s9_n20_x8 \

# accelerate launch --num_processes 5 src/generate_dataset.py \
#     --generation_type txt2img \
#     --num_samples 16 \
#     --model_name stabilityai/stable-diffusion-2-1 \
#     --num_steps 20 \
#     --guidance_scale 9 \
#     --dataset stl10 \
#     --data_path data/stl10_images \
#     --caption_path caption_data/train_stl10_train_labeled_gpt4omini_low.json \
#     --image_size 96 \
#     --device cuda \
#     --output_dir syn_data \
#     --batch_size 5 \
#     --new_dataset_name stl10_captiontxi_s9_n20_x16 \

# accelerate launch --num_processes 5 src/generate_dataset.py \
#     --generation_type txt2img \
#     --num_samples 16 \
#     --model_name stabilityai/stable-diffusion-2-1 \
#     --num_steps 20 \
#     --guidance_scale 9 \
#     --dataset stl10 \
#     --data_path data \
#     --image_size 96 \
#     --device cuda \
#     --output_dir syn_data \
#     --batch_size 5 \
#     --new_dataset_name stl10_classtxi_s9_n20_x16_rev \

# accelerate launch --num_processes 5 src/generate_dataset.py \
#     --generation_type txt2img \
#     --num_samples 8 \
#     --model_name stabilityai/stable-diffusion-2-1 \
#     --num_steps 20 \
#     --guidance_scale 9 \
#     --dataset stl10 \
#     --data_path data \
#     --image_size 96 \
#     --device cuda \
#     --output_dir syn_data \
#     --batch_size 5 \
#     --new_dataset_name stl10_classtxi_s9_n20_x8 \

# accelerate launch --num_processes 5 src/generate_dataset.py \
#     --generation_type txt2img \
#     --num_samples 16 \
#     --model_name stabilityai/stable-diffusion-2-1 \
#     --num_steps 20 \
#     --guidance_scale 9 \
#     --dataset stl10 \
#     --data_path data \
#     --image_size 96 \
#     --device cuda \
#     --output_dir syn_data \
#     --batch_size 5 \
#     --new_dataset_name stl10_classtxi_s9_n20_x16 \

# accelerate launch src/generate_dataset.py \
#     --generation_type unclip \
#     --inversion_type ddim \
#     --num_samples 2 \
#     --model_name stabilityai/stable-diffusion-2-1-unclip \
#     --num_steps 20 \
#     --ddim_sampling_step 20 \
#     --guidance_scale 10 \
#     --empty_prompt \
#     --dataset caltech \
#     --data_path data \
#     --with_test \
#     --image_size 256 \
#     --device cuda \
#     --output_dir syn_data \
#     --batch_size 30 \
#     --new_dataset_name caltech_unclip_s10_n20_x2 \

# accelerate launch src/generate_dataset.py \
#     --generation_type unclip \
#     --inversion_type ddim \
#     --num_samples 4 \
#     --model_name stabilityai/stable-diffusion-2-1-unclip \
#     --num_steps 20 \
#     --ddim_sampling_step 20 \
#     --guidance_scale 10 \
#     --empty_prompt \
#     --dataset caltech \
#     --data_path data \
#     --with_test \
#     --image_size 256 \
#     --device cuda \
#     --output_dir syn_data \
#     --batch_size 15 \
#     --new_dataset_name caltech_unclip_s10_n20_x4 \

# accelerate launch src/generate_dataset.py \
#     --generation_type unclip \
#     --inversion_type ddim \
#     --num_samples 8 \
#     --model_name stabilityai/stable-diffusion-2-1-unclip \
#     --num_steps 20 \
#     --ddim_sampling_step 20 \
#     --guidance_scale 10 \
#     --empty_prompt \
#     --dataset caltech \
#     --data_path data \
#     --with_test \
#     --image_size 256 \
#     --device cuda \
#     --output_dir syn_data \
#     --batch_size 10 \
#     --new_dataset_name caltech_unclip_s10_n20_x8 \

    # parser.add_argument("--generation_type", type=str, default="img2img", help="Type of generation: img2img or img2txt")
    # parser.add_argument("--inversion_type", type=str, default="ddim", help="Type of inversion: ddim or diffusion")
    # parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per real image or caption")
    
    # # perturbation parameters
    # parser.add_argument("--ddim_perturbation", type=str, default=None, help="Perturbation type for DDIM inversion: None, x0, xt, xT, enc, dec, bottleneck")
    # parser.add_argument("--diffusion_perturbation", type=str, default=None, help="Perturbation type for diffusion inversion: None, x0, xt, xT, enc, dec, bottleneck")
    # parser.add_argument("--perturbation_scale", type=float, default=1.0, help="Perturbation scale")
    # parser.add_argument("--pertubation_dist", type=str, default="uniform", help="Perturbation distribution: uniform or normal")
    
    # # diffusion parameters
    # parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5", help="Model name for diffusion")
    # parser.add_argument("--num_steps", type=int, default=100, help="Number of diffusion steps")
    # parser.add_argument("--ddim_sampling_step", type=int, default=10, help="Number of reverse steps for DDIM inversion")
    # parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale for DDIM inversion")
    # parser.add_argument("--empty_prompt", action="store_true", help="Use empty prompt for DDIM inversion")
    
    # parser.add_argument("--data_path", type=str, default="data", help="Directory to save generated samples")
    # parser.add_argument("--with_test", action="store_true", help="Generate samples for test set")
    # parser.add_argument("--image_size", type=int, default=256, help="Image size")
    # parser.add_argument("--device", type=str, default="cuda", help="Device to run the model")
    # parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # parser.add_argument("--output_dir", type=str, default="output", help="Directory to save generated samples")