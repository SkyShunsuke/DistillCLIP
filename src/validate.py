import torch
from torch import nn
import wandb

from utils import LogMeter

def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    epoch: int,
    args,
):
    model.eval()
    loss_meter = LogMeter()
    loss_meter.reset()
    
    results = []
    for i, batch in enumerate(dataloader):
        imgs, labels = batch["image"], batch["label"]
        imgs = imgs.cuda()
        labels = labels.cuda()
        with torch.inference_mode():
            preds = model(imgs)
            loss = nn.CrossEntropyLoss()(preds, labels)
        pred_labels = preds.argmax(1)        
        results.append(pred_labels == labels)
        loss_meter.update(loss.item())
    acc = torch.cat(results).float().mean().item()
    
    if not args.eval_only:
        wandb.log(
            {"val/loss": loss_meter.avg, 
                "val/acc": acc
            }
        ) 
    print(f"(Validation) [Epoch {epoch}] loss: {loss_meter.avg} acc: {acc}")
    
    return {
        "loss": loss_meter.avg,
        "acc": acc
    }

        