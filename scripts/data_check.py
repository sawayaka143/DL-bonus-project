"""Dataset statistics, label sanity checks, near-duplicate check, and ground-truth previews.

Outputs (in results/data/):
  stats.json              images/objects per split and per class
  label_issues.txt        anything suspicious found in label files
  duplicates.txt          near-duplicate image pairs (perceptual hash), within and across splits
  gt_examples.jpg         grid of training images with ground-truth boxes
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import imagehash
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["buffalo", "elephant", "rhino", "zebra"]
COLORS = ["tab:orange", "tab:blue", "tab:green", "tab:red"]
SPLITS = ["train", "val", "test"]
SEED = 0
HASH_THRESHOLD = 6  # Hamming distance on 64-bit pHash; <=6 is treated as a near-duplicate


def read_labels(path):
    rows = []
    for line in path.read_text().strip().splitlines():
        parts = line.split()
        if parts:
            rows.append((int(float(parts[0])), *map(float, parts[1:5]), len(parts)))
    return rows


def main():
    global DATA, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="african-wildlife-clean", help="folder name under datasets/")
    ap.add_argument("--out", default="results/data/clean")
    args = ap.parse_args()
    DATA, OUT = ROOT / "datasets" / args.data, ROOT / args.out
    OUT.mkdir(parents=True, exist_ok=True)
    stats, issues = {}, []
    hashes = {}  # (split, name) -> phash

    for split in SPLITS:
        img_dir, lbl_dir = DATA / "images" / split, DATA / "labels" / split
        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        objs, imgs_with = Counter(), Counter()
        empty = 0
        for img in images:
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                issues.append(f"{split}/{img.name}: missing label file")
                continue
            rows = read_labels(lbl)
            if not rows:
                empty += 1
                issues.append(f"{split}/{img.name}: empty label file (no objects)")
            seen = set()
            for cls, x, y, w, h, ncol in rows:
                if ncol != 5:
                    issues.append(f"{split}/{img.name}: line with {ncol} columns")
                if cls not in range(len(NAMES)):
                    issues.append(f"{split}/{img.name}: invalid class id {cls}")
                if not all(0 <= v <= 1 for v in (x, y, w, h)):
                    issues.append(f"{split}/{img.name}: coordinates outside [0,1] {x:.3f} {y:.3f} {w:.3f} {h:.3f}")
                if w * h < 1e-4:
                    issues.append(f"{split}/{img.name}: tiny box (area {w*h:.6f} of image)")
                key = (cls, round(x, 3), round(y, 3), round(w, 3), round(h, 3))
                if key in seen:
                    issues.append(f"{split}/{img.name}: duplicated box {key}")
                seen.add(key)
                objs[NAMES[cls] if cls < len(NAMES) else cls] += 1
            for c in {r[0] for r in rows}:
                imgs_with[NAMES[c] if c < len(NAMES) else c] += 1
            with Image.open(img) as im:
                hashes[(split, img.name)] = imagehash.phash(im.convert("RGB"))
        stats[split] = {
            "images": len(images),
            "objects": sum(objs.values()),
            "images_without_objects": empty,
            "objects_per_class": {n: objs.get(n, 0) for n in NAMES},
            "images_containing_class": {n: imgs_with.get(n, 0) for n in NAMES},
        }

    # near-duplicate search over all pairs (~1.1M comparisons of 64-bit hashes)
    keys = list(hashes)
    pairs = defaultdict(list)
    for a, b in combinations(keys, 2):
        d = hashes[a] - hashes[b]
        if d <= HASH_THRESHOLD:
            kind = "within " + a[0] if a[0] == b[0] else f"ACROSS {a[0]}/{b[0]}"
            pairs[kind].append((d, f"{a[0]}/{a[1]}", f"{b[0]}/{b[1]}"))

    with open(OUT / "duplicates.txt", "w") as f:
        f.write(f"pHash Hamming distance <= {HASH_THRESHOLD}\n")
        for kind in sorted(pairs):
            f.write(f"\n## {kind}: {len(pairs[kind])} pairs\n")
            for d, a, b in sorted(pairs[kind]):
                f.write(f"{d}\t{a}\t{b}\n")
    (OUT / "label_issues.txt").write_text("\n".join(issues) + "\n" if issues else "No issues found.\n")
    stats["near_duplicate_pairs"] = {k: len(v) for k, v in pairs.items()}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"{len(issues)} label issues")

    # ground-truth preview grid
    random.seed(SEED)
    train_imgs = sorted((DATA / "images" / "train").iterdir())
    sample = random.sample(train_imgs, 8)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for ax, img in zip(axes.ravel(), sample):
        im = Image.open(img).convert("RGB")
        W, H = im.size
        ax.imshow(im)
        for cls, x, y, w, h, _ in read_labels(DATA / "labels" / "train" / (img.stem + ".txt")):
            ax.add_patch(patches.Rectangle(((x - w / 2) * W, (y - h / 2) * H), w * W, h * H,
                                           fill=False, lw=2, ec=COLORS[cls]))
            ax.text((x - w / 2) * W, (y - h / 2) * H, NAMES[cls], color="white", fontsize=10,
                    bbox=dict(fc=COLORS[cls], ec="none", pad=1))
        ax.set_title(img.name)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "gt_examples.jpg", dpi=80)


if __name__ == "__main__":
    main()
