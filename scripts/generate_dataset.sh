export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 # ,2,3,4,5
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

accelerate launch --num_processes 6 src/generate_dataset.py \
    --generation_type unclip \
    --num_samples 4 \
    --model_name stabilityai/stable-diffusion-2-1-unclip \
    --num_steps 20 \
    --guidance_scale 9 \
    --dataset food101 \
    --data_path data/food101 \
    --image_size 224 \
    --device cuda \
    --output_dir syn_data \
    --batch_size 60 \
    --new_dataset_name food101_img2img_s9_n20_x4 \