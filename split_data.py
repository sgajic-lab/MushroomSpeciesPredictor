"""Split a folder of class subfolders into train/val/test (70/15/15), reproducibly.

    python split_data.py path/to/images --out data_split
    python split_data.py /kaggle/input/.../images --out data_split --link   # Linux: no copying

The source folder must contain one subfolder per species. Each class is
shuffled with a fixed seed and split by ratio, so every class gets images in
every split and the same input always gives the same split.
"""
import argparse
import os
import random
import shutil

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp")
SPLITS = ("train", "val", "test")


def split_class(files, ratio, rng):
    files = sorted(files)
    rng.shuffle(files)
    n = len(files)
    n_val = max(1, round(n * ratio[1])) if n >= 3 else 0
    n_test = max(1, round(n * ratio[2])) if n >= 3 else 0
    n_train = n - n_val - n_test
    return {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("--out", default="data_split")
    ap.add_argument("--ratio", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--link", action="store_true", help="symlink instead of copying (fast; Linux/macOS)")
    args = ap.parse_args()

    if abs(sum(args.ratio) - 1) > 1e-6:
        raise SystemExit("--ratio values must add up to 1")
    if os.path.exists(args.out) and os.listdir(args.out):
        raise SystemExit(f"{args.out} already exists and is not empty; choose another --out")

    classes = sorted(d for d in os.listdir(args.source) if os.path.isdir(os.path.join(args.source, d)))
    totals = dict.fromkeys(SPLITS, 0)
    skipped = 0
    for cls in classes:
        src_dir = os.path.join(args.source, cls)
        files = [f for f in os.listdir(src_dir) if f.lower().endswith(IMAGE_EXTS)]
        if not files:
            print(f"Skipping {cls!r}: no images")
            skipped += 1
            continue
        # seed per class so adding/removing a class doesn't reshuffle the others
        rng = random.Random(f"{args.seed}-{cls}")
        for split, names in split_class(files, args.ratio, rng).items():
            dst_dir = os.path.join(args.out, split, cls)
            os.makedirs(dst_dir, exist_ok=True)
            for name in names:
                src = os.path.abspath(os.path.join(src_dir, name))
                dst = os.path.join(dst_dir, name)
                if args.link:
                    try:
                        os.symlink(src, dst)
                    except OSError as e:
                        raise SystemExit(f"Could not create a symlink ({e}). On Windows, run without --link.")
                else:
                    shutil.copy2(src, dst)
            totals[split] += len(names)

    print(f"{len(classes) - skipped} classes -> " + ", ".join(f"{s}: {totals[s]}" for s in SPLITS))
    print(f"Split written to {args.out}")


if __name__ == "__main__":
    main()
