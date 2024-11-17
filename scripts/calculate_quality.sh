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

    # parser.add_argument('--real_data_dir', type=str, default='data/pets', help='Path to the dataset')
    # parser.add_argument('--syn_data_path', type=str, default='data/pets', help='Path to the dataset')
    # parser.add_argument('--metrics', type=str, default='fid', help='Metrics to calculate')
    # parser.add_argument('--img_size', type=int, default=256, help='Image size')
    # parser.add_argument('--split', type=str, default='train', help='Dataset split')
    
    # parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    # parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    # parser.add_argument('--device', type=str, default='cuda', help='Device')
    # parser.add_argument('--seed', type=int, default=42, help='Seed')