"""Evaluate a checkpoint: top-1/top-5, per-class accuracy, confusion matrix.

    python evaluate.py --checkpoint checkpoints/best.pt --data data_split --split test
"""
import argparse
import csv
import os

import torch

from data import load_datasets, make_loaders
from model import load_checkpoint


@torch.no_grad()
def predict_all(model, loader, device):
    """Return (logits, targets) for a whole loader."""
    model.eval()
    all_logits, all_targets = [], []
    for inputs, targets in loader:
        all_logits.append(model(inputs.to(device, non_blocking=True)).float().cpu())
        all_targets.append(targets)
    return torch.cat(all_logits), torch.cat(all_targets)


def summarize(logits, targets, classes):
    """Overall and per-class metrics from logits."""
    n = len(classes)
    pred = logits.argmax(1)
    k = min(5, n)
    top5 = logits.topk(k, dim=1).indices.eq(targets[:, None]).any(1)

    confusion = torch.zeros(n, n, dtype=torch.long)
    confusion.index_put_((targets, pred), torch.ones_like(targets), accumulate=True)

    support = confusion.sum(1)
    correct = confusion.diag()
    present = support > 0
    per_class_acc = torch.where(present, correct.float() / support.clamp(min=1), torch.nan)

    return {
        "top1": pred.eq(targets).float().mean().item() * 100,
        "top5": top5.float().mean().item() * 100,
        # mean of per-class accuracy: not inflated by the large classes
        "balanced_acc": per_class_acc[present].mean().item() * 100,
        "per_class_acc": per_class_acc,
        "support": support,
        "confusion": confusion,
    }


def write_report(metrics, classes, out_dir):
    """Save per-class CSV and a confusion-matrix image; return the CSV path."""
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "per_class.csv")
    confusion = metrics["confusion"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "support", "accuracy", "most_confused_with"])
        for i, name in enumerate(classes):
            row = confusion[i].clone()
            row[i] = 0
            other = classes[row.argmax()] if row.sum() > 0 else ""
            acc = metrics["per_class_acc"][i].item()
            w.writerow([name, metrics["support"][i].item(), "" if acc != acc else f"{acc:.4f}", other])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = confusion.float() / confusion.sum(1, keepdim=True).clamp(min=1)
    fig, ax = plt.subplots(figsize=(18, 18))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=6)
    ax.set_yticks(range(len(classes)), classes, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalized)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=120)
    plt.close(fig)
    return csv_path


def print_summary(metrics, classes, name="test", worst=10):
    print(f"\n=== {name}: top-1 {metrics['top1']:.2f}% | top-5 {metrics['top5']:.2f}% "
          f"| balanced {metrics['balanced_acc']:.2f}% ===")
    acc = metrics["per_class_acc"]
    order = [i for i in acc.argsort().tolist() if acc[i] == acc[i]][:worst]
    print(f"Weakest {len(order)} classes:")
    for i in order:
        print(f"  {acc[i].item() * 100:6.2f}%  (n={metrics['support'][i].item():4d})  {classes[i]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--data", default="data_split")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, ckpt = load_checkpoint(args.checkpoint, device)
    _, val, test = load_datasets(args.data, img_size=ckpt.get("img_size", 224))
    if val.classes != classes:
        raise SystemExit("Class list in the data split does not match the checkpoint.")
    _, val_loader, test_loader = make_loaders(val, val, test, args.batch_size, args.num_workers)
    loader = test_loader if args.split == "test" else val_loader

    logits, targets = predict_all(model, loader, device)
    metrics = summarize(logits, targets, classes)
    print_summary(metrics, classes, args.split)
    path = write_report(metrics, classes, args.out)
    print(f"Per-class report: {path}")


if __name__ == "__main__":
    main()
