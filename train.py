"""Train the mushroom classifier.

Default: fine-tune an ImageNet-pretrained EfficientNet-B0.
  1. A few epochs with the backbone frozen (only the new head learns).
  2. Unfreeze everything and fine-tune at a lower learning rate with a
     cosine schedule.
The best epoch (by validation accuracy) is saved, and training stops early
when validation stops improving.

Examples:
    python train.py                                   # full run, pretrained EfficientNet-B0
    python train.py --arch convnet --epochs 40 --lr 1e-3 --freeze-epochs 0
    python train.py --subset 10 --epochs 2 --no-pretrained   # quick smoke test
"""
import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from data import class_weights, load_datasets, make_loaders
from evaluate import predict_all, print_summary, summarize, write_report
from model import ARCHS, build_model, load_checkpoint, save_checkpoint, set_backbone_trainable


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data_split", help="folder with train/ val/ test/")
    ap.add_argument("--arch", choices=ARCHS, default="efficientnet_b0")
    ap.add_argument("--no-pretrained", action="store_true", help="start from random weights")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--freeze-epochs", type=int, default=3, help="epochs training only the head")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3, help="head learning rate")
    ap.add_argument("--finetune-lr", type=float, default=2e-4, help="learning rate after unfreezing")
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--no-class-weights", action="store_true", help="disable imbalance-aware loss")
    ap.add_argument("--patience", type=int, default=5, help="early-stopping patience (epochs)")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--subset", type=int, default=None, help="use at most N images per class")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    ap.add_argument("--run-name", default=None)
    return ap.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_optimizer(model, lr, weight_decay):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def cosine_with_warmup(optimizer, total_steps, warmup_steps):
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, use_amp):
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        seen += inputs.size(0)
    return total_loss / seen, 100.0 * correct / seen


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    torch.backends.cudnn.benchmark = True  # fixed input size, so let cuDNN pick the fastest kernels
    print(f"Device: {device}  (mixed precision: {use_amp})")

    train_ds, val_ds, test_ds = load_datasets(args.data, args.img_size, args.subset)
    classes = train_ds.classes
    print(f"Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)} | {len(classes)} classes")
    train_loader, val_loader, test_loader = make_loaders(
        train_ds, val_ds, test_ds, args.batch_size, args.num_workers)

    pretrained = not args.no_pretrained and args.arch != "convnet"
    model = build_model(args.arch, len(classes), pretrained=pretrained).to(device)

    weights = None if args.no_class_weights else class_weights(train_ds.targets, len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    freeze_epochs = min(args.freeze_epochs, args.epochs) if pretrained else 0
    steps_per_epoch = len(train_loader)

    def start_phase(lr, n_epochs):
        opt = make_optimizer(model, lr, args.weight_decay)
        total = max(1, n_epochs * steps_per_epoch)
        return opt, cosine_with_warmup(opt, total, warmup_steps=min(steps_per_epoch, total // 10 + 1))

    if freeze_epochs:
        set_backbone_trainable(model, args.arch, False)
        optimizer, scheduler = start_phase(args.lr, freeze_epochs)
        print(f"Phase 1: training head only for {freeze_epochs} epochs")
    else:
        lr = args.lr if not pretrained else args.finetune_lr
        optimizer, scheduler = start_phase(lr, args.epochs)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="MushroomSpeciesPredictor", name=args.run_name, config=vars(args))

    os.makedirs(args.out, exist_ok=True)
    best_path = os.path.join(args.out, "best.pt")
    best_acc, best_epoch, bad_epochs = -1.0, 0, 0

    for epoch in range(1, args.epochs + 1):
        if pretrained and epoch == freeze_epochs + 1 and freeze_epochs:
            set_backbone_trainable(model, args.arch, True)
            optimizer, scheduler = start_phase(args.finetune_lr, args.epochs - freeze_epochs)
            print(f"Phase 2: fine-tuning all layers at lr={args.finetune_lr}")

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, use_amp)
        logits, targets = predict_all(model, val_loader, device)
        val_loss = nn.functional.cross_entropy(logits, targets).item()
        val = summarize(logits, targets, classes)

        print(f"Epoch {epoch:3d}/{args.epochs} | train loss {train_loss:.4f} acc {train_acc:6.2f}% "
              f"| val loss {val_loss:.4f} acc {val['top1']:6.2f}% top-5 {val['top5']:6.2f}% "
              f"balanced {val['balanced_acc']:6.2f}% | {time.time() - t0:.0f}s")
        if run:
            run.log({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
                     "val_loss": val_loss, "val_accuracy": val["top1"], "val_top5": val["top5"],
                     "val_balanced_accuracy": val["balanced_acc"],
                     "lr": optimizer.param_groups[0]["lr"]})

        if val["top1"] > best_acc:
            best_acc, best_epoch, bad_epochs = val["top1"], epoch, 0
            save_checkpoint(best_path, model, args.arch, classes,
                            extra={"img_size": args.img_size, "epoch": epoch, "val_acc": best_acc})
            print(f"  saved new best -> {best_path}")
        else:
            bad_epochs += 1
            # don't stop while the head-only phase is still running
            if bad_epochs >= args.patience and epoch > freeze_epochs:
                print(f"Early stopping: no improvement for {args.patience} epochs")
                break

    print(f"\nBest val accuracy {best_acc:.2f}% at epoch {best_epoch}. Evaluating it on the test set...")
    model, _, _ = load_checkpoint(best_path, device)
    logits, targets = predict_all(model, test_loader, device)
    test = summarize(logits, targets, classes)
    print_summary(test, classes, "test")
    report = write_report(test, classes, os.path.join(args.out, "report"))
    print(f"Per-class report and confusion matrix: {os.path.dirname(report)}")

    if run:
        run.log({"test_accuracy": test["top1"], "test_top5": test["top5"],
                 "test_balanced_accuracy": test["balanced_acc"]})
        run.finish()


if __name__ == "__main__":
    main()
