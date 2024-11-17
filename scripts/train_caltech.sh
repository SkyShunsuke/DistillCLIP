export CUDA_VISIBLE_DEVICES=1
python src/train_classifier.py \
    --model_name resnet18 \
    --model_path /home/haselab/projects/Adversarial-Perception-aware-Perturbation/syn_models/classifier_rn18_caltech_synx1_ep500.pth \
    --dataset caltech \
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
    --wandb_run_name caltech-sym2real-synx1-resnet18

python src/train_classifier.py \
    --model_name resnet18 \
    --model_path /home/haselab/projects/Adversarial-Perception-aware-Perturbation/syn_models/classifier_rn18_caltech_synx2_ep500.pth \
    --dataset caltech \
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
    --wandb_run_name caltech-sym2real-synx2-resnet18

python src/train_classifier.py \
    --model_name resnet18 \
    --model_path /home/haselab/projects/Adversarial-Perception-aware-Perturbation/syn_models/classifier_rn18_caltech_synx4_ep500.pth \
    --dataset caltech \
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
    --wandb_run_name caltech-sym2real-synx4-resnet18

python src/train_classifier.py \
    --model_name resnet18 \
    --model_path /home/haselab/projects/Adversarial-Perception-aware-Perturbation/syn_models/classifier_rn18_caltech_synx8_ep500.pth \
    --dataset caltech \
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
    --wandb_run_name caltech-sym2real-synx8-resnet18