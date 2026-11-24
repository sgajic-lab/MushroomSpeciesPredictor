"""Predict the species of one or more mushroom photos.

    python predict.py "Amanita Amerirubescens.jpg"
    python predict.py photo1.jpg photo2.jpg --top-k 5

DISCLAIMER: this is a student machine-learning project. Never use its output
to decide whether a mushroom is safe to eat.
"""
import argparse

import torch
from PIL import Image

from data import build_transforms
from model import load_checkpoint

LOW_CONFIDENCE = 0.5


class MushroomPredictor:
    """Loads the checkpoint once and classifies any number of images."""

    def __init__(self, checkpoint="checkpoints/best.pt", device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.classes, ckpt = load_checkpoint(checkpoint, self.device)
        self.transform = build_transforms(train=False, img_size=ckpt.get("img_size", 224))

    @torch.no_grad()
    def predict(self, image, top_k=3):
        """Return [(species, probability), ...] for a path or PIL image."""
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        batch = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = self.model(batch).softmax(1)[0]
        top = probs.topk(min(top_k, len(self.classes)))
        return [(self.classes[i], p) for p, i in zip(top.values.tolist(), top.indices.tolist())]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    predictor = MushroomPredictor(args.checkpoint)
    for path in args.images:
        preds = predictor.predict(path, args.top_k)
        print(f"\n{path}")
        for rank, (name, p) in enumerate(preds, 1):
            print(f"  {rank}. {name:<32} {p * 100:5.1f}%")
        if preds[0][1] < LOW_CONFIDENCE:
            print("  (low confidence: the model is unsure about this photo)")
    print("\nNot for deciding whether a mushroom is edible.")


if __name__ == "__main__":
    main()
