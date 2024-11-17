import torch
from torch import nn
import wandb

from utils import LogMeter


import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
from pathlib import Path

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
from torchvision import transforms as tfms

from dataset import PetsDataset, PetsDatasetLMDB, StanfordCarsDataset, StanfordCarsDatasetLMDB, \
    Flowers102Dataset, Flowers102DatasetLMDB, Caltech101Dataset, Caltech101DatasetLMDB
from dataset import build_dataset
from models import get_model
from validate import validate

import logging
import wandb
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--pretrained", action="store_true", default=False)
    parser.add_argument("--dataset", type=str, default="pets")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--image_channel", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--optimzier", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine")
    parser.add_argument("--scheduler_epochs", type=int, default=100)
    parser.add_argument("--wandb_project", type=str, default="apap")
    parser.add_argument("--wandb_run_name", type=str, default="apap-pets-real-resnet50")
    parser.add_argument("--save_model", action="store_true", default=False)
    parser.add_argument("--save_freq", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--save_best_model", action="store_true", default=False)
    
    # synthetic dataset
    parser.add_argument("--syn_dataset", action="store_true", default=False)
    parser.add_argument("--syn_data_path", type=str, default="./data")
    parser.add_argument("--eval_on_real", action="store_true", default=False)
    parser.add_argument("--train_mode", type=str, default="syn2real")  # syn2real, mix
    
    args = parser.parse_args()
    
    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    args = parser.parse_args()
    config = vars(args)
    logging.info(f"Config: {config}")
    
    # dataset 
    dataset_config = {
        "root": args.data_path if not args.syn_dataset else args.syn_data_path,
        "download": False,
        "transform": None,
        "train": True,
    }
    train_dataset, val_dataset = build_dataset(dataset_config, args)
    
    train_dataloader = DataLoader(
        train_dataset, args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False
    )
    
    # model
    model = get_model(args, args.model_name, in_channels=args.image_channel, num_classes=train_dataset.num_classes)

    model = model.cuda()
    logging.info(f"Model: {model}")
    if args.model_path is not None:
        logging.info(f"Load model from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path))
    
    if args.eval_only:
        logging.info("Start evaluation")
        val_stats = validate(model, val_dataloader, 0, args)
        logging.info(f"val_loss: {val_stats['loss']}, val_acc: {val_stats['acc']}")
        return    
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # scheduler
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.scheduler_epochs)
    else:
        raise ValueError(f"Invalid scheduler: {args.scheduler}")
    
    # logging
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=config)
    
    logging.info(f"Number of training samples: {len(train_dataset)}")
    logging.info(f"Start training for {args.epochs} epochs")
    current_acc_max_idx = 0
    current_acc_max = 0
    for i in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_dataloader, optimizer, scheduler, i, args)
        logging.info(f"[Epoch {i}] train_loss: {train_stats['loss']}")
        
        if i % 10 == 0:
            val_stats = validate(model, val_dataloader, i, args)
            current_acc_max_idx = i if current_acc_max < val_stats["acc"] else current_acc_max
            logging.info(f"[Epoch {i}] val_loss: {val_stats['loss']}, val_acc: {val_stats['acc']}")
        
        if args.save_freq is not None and i % args.save_freq == 0:
            save_dir = os.path.join(wandb.run.dir, "models")
            Path(save_dir).mkdir(parents=True, exist_ok=True)   
            logging.info(f"Save model to {save_dir}")
            torch.save(model.state_dict(), os.path.join(save_dir, f"classifier_{i}.pth"))
    
    logging.info(f"Overall Training finished")
    
    if args.save_model:
        save_dir = os.path.join(wandb.run.dir, "models")
        Path(save_dir).mkdir(parents=True, exist_ok=True)   
        logging.info(f"Save model to {save_dir}")
        torch.save(model.state_dict(), os.path.join(save_dir, "classifier.pth"))
        
        if args.save_best_model:
            logging.info(f"Save best model to {save_dir}")
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_classifier_{current_acc_max_idx}.pth"))
    
    if args.eval_on_real:
        logging.info(f"Start evaluation on real dataset")
        real_dataset = PetsDataset(args.data_path, train=False, transform=val_dataset.transform)
        real_dataloader = DataLoader(real_dataset, args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        val_stats = validate(model, real_dataloader, args.epochs + 1, args)
        logging.info(f"[Real] val_loss: {val_stats['loss']}, val_acc: {val_stats['acc']}")
        logging.info(f"Finish evaluation on real dataset")
    
    wandb.finish()
        
    
def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    args
):
    model.train()
    loss_meter = LogMeter()
    acc_meter = LogMeter()
    acc_meter.reset()
    loss_meter.reset()
    
    for i, batch in enumerate(dataloader):
        imgs, labels = batch["image"], batch["label"]
        imgs = imgs.cuda()
        labels = labels.cuda()
        
        preds = model(imgs)
        
        loss = nn.CrossEntropyLoss()(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        
        acc_meter.update((preds.argmax(1) == labels).float().mean().item())
        loss_meter.update(loss.item())
        
        if i % 100 == 0:
            wandb.log(
                {"train/loss": loss.item()}
            )
            print(f"[Epoch {epoch}] Step {i}: loss: {loss.item()}")

    print(f"[Epoch {epoch}] loss: {loss_meter.avg}, acc: {acc_meter.avg}")
        
    return {
        "loss": loss_meter.avg,
        "acc": acc_meter.avg
    }
        
if __name__ == "__main__":
    main()