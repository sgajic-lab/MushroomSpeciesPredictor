# MushroomSpeciesPredictor

An image classifier that predicts a mushroom's species from a photo across 169 species.
By default it fine-tunes an ImageNet-pretrained EfficientNet-B0. A small CNN trained
from scratch is also available for comparison.

> **Disclaimer:** this is a machine-learning project, not a field guide. Never use its
> predictions to decide whether a mushroom is safe to eat.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

For GPU training, install the CUDA build of PyTorch from <https://pytorch.org/get-started/locally/>.

## Data

The data is the Kaggle dataset
[`dariobaumberger/combined-kaggle-mushrooms-dataset`](https://www.kaggle.com/datasets/dariobaumberger/combined-kaggle-mushrooms-dataset).
It has 169 species and about 104,000 images, in an `images/` folder with one subfolder
per species. Split it 70/15/15 with a fixed seed:

```bash
python split_data.py path/to/raw_images --out data_split
python eda.py --data data_split          # class counts + sample grid in reports/
```

## Train

```bash
python train.py                          # pretrained EfficientNet-B0 (recommended)
python train.py --arch resnet50          # bigger backbone
python train.py --arch convnet --freeze-epochs 0 --epochs 40   # from-scratch CNN
python train.py --subset 10 --epochs 2 --no-pretrained          # quick smoke test
python train.py --wandb --run-name effnet_b0_v1                 # log to Weights & Biases
```

The training recipe:

- **Transfer learning.** The head trains for 3 epochs with the backbone frozen, then
  all layers are fine-tuned at a lower learning rate.
- **AdamW** with linear warmup and a cosine learning-rate schedule.
- **Imbalance handling.** The loss uses class weights (inverse square root of class
  frequency) plus label smoothing. The dataset ranges from 211 to 5,823 images per
  class.
- **Augmentation that keeps color.** Random crops, flips and small rotations are used,
  with only a slight hue shift.
- **Best checkpoint and early stopping.** The best epoch by validation accuracy is saved
  to `checkpoints/best.pt`, and training stops early when validation stops improving.
- **Mixed precision** on CUDA GPUs.

At the end of training, the best checkpoint is evaluated on the test set. Results go to
`checkpoints/report/`: top-1, top-5, balanced accuracy, a per-class CSV, and a confusion
matrix.

## Train on a free Kaggle GPU

`kaggle_train.ipynb` runs the whole pipeline on Kaggle. Setup and instructions are in the
notebook's first cell. In short:

1. Create a new notebook on Kaggle and import this `.ipynb` file.
2. Add the dataset above as an input.
3. In the notebook settings, choose the **GPU T4 x2** accelerator and turn **Internet on**.
4. Use **Save Version → Save & Run All**. The notebook clones this repository automatically.

The checkpoint and report appear in the version's **Output** tab.

## Evaluate and predict

```bash
python evaluate.py --checkpoint checkpoints/best.pt --split test
python predict.py "Amanita Amerirubescens.jpg" --top-k 3
```

`predict.py` shows the top-k species with their probabilities and flags low-confidence
predictions.

## Files

| File | Purpose |
|---|---|
| `data.py` | Transforms, datasets (val/test share the train class indices), loaders, class weights |
| `model.py` | `ConvNet`, pretrained backbones, freezing, checkpoint save/load |
| `train.py` | Training loop |
| `evaluate.py` | Metrics, per-class report, confusion matrix |
| `predict.py` | Command-line and Python prediction (`MushroomPredictor`) |
| `eda.py` | Dataset statistics and sample grid |
| `split_data.py` | Reproducible train/val/test split |
