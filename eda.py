"""Quick look at the dataset: image counts per class and a grid of samples.

    python eda.py --data data_split
"""
import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp")


def list_images(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_split")
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)

    # class counts per split
    splits = [s for s in ("train", "val", "test") if os.path.isdir(os.path.join(args.data, s))]
    train_dir = os.path.join(args.data, "train")
    classes = sorted(os.listdir(train_dir))
    counts = {s: {c: len(list_images(os.path.join(args.data, s, c)))
                  if os.path.isdir(os.path.join(args.data, s, c)) else 0
                  for c in classes} for s in splits}
    for s in splits:
        n = counts[s]
        empty = [c for c, v in n.items() if v == 0]
        print(f"{s:5s}: {sum(n.values()):6d} images | min {min(n.values())} max {max(n.values())}"
              + (f" | empty: {', '.join(empty)}" if empty else ""))

    order = sorted(classes, key=lambda c: counts["train"][c])
    fig, ax = plt.subplots(figsize=(10, 16))
    ax.barh(order, [counts["train"][c] for c in order], color="#4a7c59")
    ax.tick_params(axis="y", labelsize=6)
    ax.set_xlabel("Training images")
    ax.set_title("Training images per class")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "class_counts.png"), dpi=120)
    plt.close(fig)

    # random sample grid from the training split
    pool = [(c, f) for c in classes for f in list_images(os.path.join(train_dir, c))]
    picks = rng.sample(pool, min(args.samples, len(pool)))
    side = int(len(picks) ** 0.5 + 0.999)
    fig = plt.figure(figsize=(2 * side, 2 * side))
    sizes = set()
    for i, (c, f) in enumerate(picks, 1):
        with Image.open(os.path.join(train_dir, c, f)) as img:
            sizes.add(img.size)
            ax = fig.add_subplot(side, side, i)
            ax.imshow(img.convert("RGB"))
        ax.set_title(c[:18], fontsize=6)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "sample_grid.png"), dpi=100)
    plt.close(fig)
    print(f"{len(sizes)} distinct image sizes among the samples")
    print(f"Saved {args.out}/class_counts.png and {args.out}/sample_grid.png")


if __name__ == "__main__":
    main()
