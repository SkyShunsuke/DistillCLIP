    # parser.add_argument("--model_name", type=str, default="resnet50")
    # parser.add_argument("--pretrained", action="store_true", default=False)
    # parser.add_argument("--dataset", type=str, default="pets")
    # parser.add_argument("--data_path", type=str, default="./data")
    # parser.add_argument("--batch_size", type=int, default=256)
    # parser.add_argument("--image_size", type=int, default=256)
    # parser.add_argument("--image_channel", type=int, default=3)
    # parser.add_argument("--epochs", type=int, default=100)
    # parser.add_argument("--optimzier", type=str, default="adam")
    # parser.add_argument("--lr", type=float, default=1e-3)
    # parser.add_argument("--weight_decay", type=float, default=1e-4)
    # parser.add_argument("--scheduler", type=str, default="cosine")
    # parser.add_argument("--scheduler_epochs", type=int, default=100)
    # parser.add_argument("--wandb_project", type=str, default="apap")
    # parser.add_argument("--wandb_run_name", type=str, default="apap-pets-real-resnet50")
    # parser.add_argument("--save_model", action="store_true", default=False)
    # parser.add_argument("--save_freq", type=int, default=None)
    # parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--eval_only", action="store_true", default=False)
    # parser.add_argument("--model_path", type=str, default=None)
    # parser.add_argument("--save_best_model", action="store_true", default=False)

export CUDA_VISIBLE_DEVICES=1
python src/train_classifier.py \
    --model_name resnet18 \
    --model_path /home/haselab/projects/Adversarial-Perception-aware-Perturbation/syn_models/classifier_rn18_flowers_synx1_ep500.pth \
    --dataset flowers \
    --data_path ./data \
    --batch_size 64 \
    --image_size 256 \
    --image_channel 3 \
    --epochs 250 \
    --optimzier adam \
    --lr 5e-4 \
    --weight_decay 0 \
    --save_model \
    --save_freq 500 \
    --save_best_model \
    --scheduler cosine \
    --scheduler_epochs 200 \
    --wandb_project unclip \
    --wandb_run_name flowers-sym2real-synx1-resnet18

python src/train_classifier.py \
    --model_name resnet18 \
    --model_path /home/haselab/projects/Adversarial-Perception-aware-Perturbation/syn_models/classifier_rn18_flowers_synx2_ep500.pth \
    --dataset flowers \
    --data_path ./data \
    --batch_size 64 \
    --image_size 256 \
    --image_channel 3 \
    --epochs 250 \
    --optimzier adam \
    --lr 5e-4 \
    --weight_decay 0 \
    --save_model \
    --save_freq 500 \
    --save_best_model \
    --scheduler cosine \
    --scheduler_epochs 200 \
    --wandb_project unclip \
    --wandb_run_name flowers-sym2real-synx2-resnet18

python src/train_classifier.py \
    --model_name resnet18 \
    --model_path /home/haselab/projects/Adversarial-Perception-aware-Perturbation/syn_models/classifier_rn18_flowers_synx4_ep500.pth \
    --dataset flowers \
    --data_path ./data \
    --batch_size 64 \
    --image_size 256 \
    --image_channel 3 \
    --epochs 250 \
    --optimzier adam \
    --lr 5e-4 \
    --weight_decay 0 \
    --save_model \
    --save_freq 500 \
    --save_best_model \
    --scheduler cosine \
    --scheduler_epochs 200 \
    --wandb_project unclip \
    --wandb_run_name flowers-sym2real-synx4-resnet18

python src/train_classifier.py \
    --model_name resnet18 \
    --model_path /home/haselab/projects/Adversarial-Perception-aware-Perturbation/syn_models/classifier_rn18_flowers_synx8_ep500.pth \
    --dataset flowers \
    --data_path ./data \
    --batch_size 64 \
    --image_size 256 \
    --image_channel 3 \
    --epochs 250 \
    --optimzier adam \
    --lr 5e-4 \
    --weight_decay 0 \
    --save_model \
    --save_freq 500 \
    --save_best_model \
    --scheduler cosine \
    --scheduler_epochs 200 \
    --wandb_project unclip \
    --wandb_run_name flowers-sym2real-synx8-resnet18
