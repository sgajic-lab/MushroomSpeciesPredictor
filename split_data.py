"""Split a folder of class subfolders into train/val/test (70/15/15), reproducibly.

    python split_data.py path/to/images --out data_split                 # copy files
    python split_data.py path/to/images --out data_split --link          # symlink (Linux/macOS)
    python split_data.py path/to/images --out data_split --resize 256    # shrink once, save as JPEG

The source folder must contain one subfolder per species. Each class is
shuffled with a fixed seed and split by ratio, so every class gets images in
every split and the same input always gives the same split.

--resize makes training faster when the CPU is the bottleneck: each image is
decoded and shrunk once here instead of on every epoch.
"""
import argparse
import os
import random
import shutil
import statistics
from concurrent.futures import ProcessPoolExecutor

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp")
SPLITS = ("train", "val", "test")

# The Kaggle dataset combines several sources that use different names for the same
# species (old name vs. current name after reclassification). Left as separate classes,
# the model gets marked wrong for "confusing" a species with itself. Each folder on the
# left is merged into the class on the right.
SYNONYMS = {
    "Lycoperdon pyriforme": "Apioperdon pyriforme",
    "Armillaria tabescens": "Desarmillaria tabescens",
    "Merulius tremellosus": "Phlebia tremellosa",
    "Psathyrella candolleana": "Candolleomyces candolleanus",
    "Clitocybe nuda": "Collybia nuda",
    "Lepista nuda": "Collybia nuda",
    "Gyromitra infula": "Paragyromitra infula",
    "Boletus pulcherrimus": "Rubroboletus pulcherrimus",
    "Boletus badius": "Imleria badia",
    "Lactarius volemus": "Lactifluus volemus",
    "Polyporus squamosus": "Cerioporus squamosus",
}

# Folders labelled with a genus or family instead of a species. They overlap with
# real species classes, so they are left out.
DROP = {
    "Pleurotus": "genus, not a species",
    "Chroogomphus": "genus, not a species",
    "Clavariaceae": "family, not a species",
    "Clavulinaceae": "family, not a species",
}


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


def resize_one(job):
    """Shrink so the short side is ``size`` (never enlarge) and save as JPEG."""
    src, dst, size = job
    from PIL import Image
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = size / min(w, h)
            if scale < 1:
                im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
            im.save(dst, "JPEG", quality=90)
        return None
    except Exception as e:  # unreadable image: skip it, report it
        return f"{src}: {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("--out", default="data_split")
    ap.add_argument("--ratio", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    ap.add_argument("--seed", type=int, default=42)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--link", action="store_true", help="symlink instead of copying (fast; Linux/macOS)")
    mode.add_argument("--resize", type=int, metavar="SIZE", help="shrink images to SIZE px (short side) as JPEG")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="processes for --resize")
    ap.add_argument("--keep-names", action="store_true",
                    help="don't merge synonym folders or drop non-species folders")
    args = ap.parse_args()

    if abs(sum(args.ratio) - 1) > 1e-6:
        raise SystemExit("--ratio values must add up to 1")
    if os.path.exists(args.out) and os.listdir(args.out):
        raise SystemExit(f"{args.out} already exists and is not empty; choose another --out")

    folders = sorted(d for d in os.listdir(args.source) if os.path.isdir(os.path.join(args.source, d)))

    # group source folders into classes, merging synonyms and dropping non-species folders
    groups = {}
    for folder in folders:
        if not args.keep_names and folder in DROP:
            print(f"Dropping {folder!r}: {DROP[folder]}")
            continue
        cls = folder if args.keep_names else SYNONYMS.get(folder, folder)
        groups.setdefault(cls, []).append(folder)
    for cls, members in groups.items():
        if len(members) > 1:
            print(f"Merging {' + '.join(repr(m) for m in members)} -> {cls!r}")

    totals = dict.fromkeys(SPLITS, 0)
    per_class = {}
    jobs = []
    used = set()
    for cls in sorted(groups):
        # (folder, file) pairs, so files from merged folders can't clash
        files = [(folder, f) for folder in groups[cls]
                 for f in os.listdir(os.path.join(args.source, folder)) if f.lower().endswith(IMAGE_EXTS)]
        if not files:
            print(f"Skipping {cls!r}: no images")
            continue
        per_class[cls] = len(files)
        # seed per class so adding/removing a class doesn't reshuffle the others
        rng = random.Random(f"{args.seed}-{cls}")
        for split, pairs in split_class(files, args.ratio, rng).items():
            dst_dir = os.path.join(args.out, split, cls)
            os.makedirs(dst_dir, exist_ok=True)
            for folder, name in pairs:
                src = os.path.abspath(os.path.join(args.source, folder, name))
                stem, ext = os.path.splitext(name)
                ext = ".jpg" if args.resize else ext
                dst = os.path.join(dst_dir, stem + ext)
                n = 1
                while dst in used:  # same file name in two merged folders
                    dst = os.path.join(dst_dir, f"{stem}_{n}{ext}")
                    n += 1
                used.add(dst)
                if args.resize:
                    jobs.append((src, dst, args.resize))
                elif args.link:
                    try:
                        os.symlink(src, dst)
                    except OSError as e:
                        raise SystemExit(f"Could not create a symlink ({e}). On Windows, run without --link.")
                else:
                    shutil.copy2(src, dst)
            totals[split] += len(pairs)

    if jobs:
        print(f"Resizing {len(jobs)} images to {args.resize}px with {args.workers} processes...", flush=True)
        failed = []
        with ProcessPoolExecutor(args.workers) as pool:
            for i, err in enumerate(pool.map(resize_one, jobs, chunksize=64), 1):
                if err:
                    failed.append(err)
                if i % 20000 == 0:
                    print(f"  {i}/{len(jobs)}", flush=True)
        if failed:
            print(f"Skipped {len(failed)} unreadable images, e.g. {failed[0]}")

    counts = sorted(per_class.values())
    print(f"{len(per_class)} classes, {sum(counts)} images "
          f"(per class: min {counts[0]}, median {statistics.median(counts):.0f}, max {counts[-1]})")
    for limit in (20, 50, 100):
        print(f"  classes with fewer than {limit} images: {sum(c < limit for c in counts)}")
    print("Split -> " + ", ".join(f"{s}: {totals[s]}" for s in SPLITS))
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
