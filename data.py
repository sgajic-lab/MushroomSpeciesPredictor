"""Datasets, transforms and data loaders."""
import os
import random
from collections import Counter

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224


def build_transforms(train, img_size=IMG_SIZE):
    """Train: light augmentation that keeps colour intact (colour is a key
    feature for telling species apart). Eval: resize + centre crop, which
    keeps the aspect ratio instead of squashing the image."""
    if train:
        return v2.Compose([
            v2.ToImage(),
            v2.RandomResizedCrop(img_size, scale=(0.5, 1.0), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(20),
            # small hue shift only: hue=0.5 (the old value) turned red caps green
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            v2.RandomErasing(p=0.2),
        ])
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(int(img_size * 256 / 224), antialias=True),
        v2.CenterCrop(img_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class FixedClassImageFolder(ImageFolder):
    """ImageFolder that uses a given class list instead of the folders it finds.

    Without this, a class folder missing from val/test silently shifts every
    later label index.
    """

    def __init__(self, root, classes, **kwargs):
        self._fixed_classes = list(classes)
        try:
            super().__init__(root, allow_empty=True, **kwargs)
        except TypeError:  # torchvision older than 0.18 has no allow_empty
            super().__init__(root, **kwargs)

    def find_classes(self, directory):
        return self._fixed_classes, {c: i for i, c in enumerate(self._fixed_classes)}


def _subset_per_class(dataset, n_per_class, seed=0):
    """Keep at most ``n_per_class`` random images of each class (for quick runs)."""
    rng = random.Random(seed)
    by_class = {}
    for idx, target in enumerate(dataset.targets):
        by_class.setdefault(target, []).append(idx)
    keep = []
    for idxs in by_class.values():
        rng.shuffle(idxs)
        keep.extend(idxs[:n_per_class])
    sub = Subset(dataset, sorted(keep))
    sub.targets = [dataset.targets[i] for i in sub.indices]
    sub.classes = dataset.classes
    return sub


def load_datasets(split_dir, img_size=IMG_SIZE, subset=None):
    """Return (train, val, test) datasets that share the train class indices."""
    train = ImageFolder(os.path.join(split_dir, "train"), transform=build_transforms(True, img_size))
    classes = train.classes
    eval_tf = build_transforms(False, img_size)
    val = FixedClassImageFolder(os.path.join(split_dir, "val"), classes, transform=eval_tf)
    test = FixedClassImageFolder(os.path.join(split_dir, "test"), classes, transform=eval_tf)

    missing = {
        name: [c for c in classes if c not in os.listdir(os.path.join(split_dir, name))]
        for name in ("val", "test")
    }
    for name, miss in missing.items():
        if miss:
            print(f"Warning: {name} split has no images for: {', '.join(miss)}")

    if subset:
        train, val, test = (_subset_per_class(d, subset) for d in (train, val, test))
    return train, val, test


def class_weights(targets, num_classes, power=0.5):
    """Inverse-frequency loss weights, softened by ``power`` (0.5 = inverse sqrt).

    The dataset is heavily imbalanced (a handful to thousands of images per class), so
    without weights the model learns to favour the big classes.
    """
    counts = Counter(targets)
    freq = torch.tensor([max(counts.get(i, 0), 1) for i in range(num_classes)], dtype=torch.float)
    w = (freq.max() / freq) ** power
    return w / w.mean()


def make_loaders(train, val, test, batch_size=32, num_workers=None):
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 1)
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return (
        # drop the last partial batch (BatchNorm dislikes tiny batches) unless that leaves nothing
        DataLoader(train, shuffle=True, drop_last=len(train) > batch_size, **common),
        DataLoader(val, shuffle=False, **common),
        DataLoader(test, shuffle=False, **common),
    )
