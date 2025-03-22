export CUDA_VISIBLE_DEVICES=1,2,3,4,5
python src/calculate_quality.py \
    --dataset_name  caltech \
    --real_data_dir data/ \
    --syn_data_path ./syn_data/unclip_caltech_unclip_s10_n20_x1.hdf5 \
    --metrics fid \
    --img_size 256 \
    --split train \
    --batch_size 64 \
    --num_workers 4 \
    --device cuda \
    --seed 42