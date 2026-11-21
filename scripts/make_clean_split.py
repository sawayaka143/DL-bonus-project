"""Build a leakage-free version of the official African Wildlife split.

The official split contains identical / near-identical photos in different splits
(e.g. train/1 (151).jpg == test/1 (103).jpg). Method:
  1. Compute a 64-bit perceptual hash (pHash) for every image.
  2. Link two images if their Hamming distance is <= 6, and take connected components
     (union-find) as duplicate groups.
  3. A group that spans several splits is moved entirely to the highest-priority split it touches,
     with priority train > val > test. This only removes leaked copies from val/test; no image is
     moved *into* an evaluation split, and no image is dropped.
The procedure is deterministic (no randomness, so no seed is needed).

Output: datasets/african-wildlife-clean/{images,labels}/{train,val,test}
        results/data/split_changes.txt  (every moved file)
"""
import shutil
from itertools import combinations
from pathlib import Path

import imagehash
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "datasets" / "african-wildlife"
DST = ROOT / "datasets" / "african-wildlife-clean"
SPLITS = ["train", "val", "test"]  # also the priority order
THRESHOLD = 6


def main():
    items = []  # (split, filename)
    hashes = []
    for split in SPLITS:
        for img in sorted((SRC / "images" / split).iterdir()):
            items.append((split, img.name))
            with Image.open(img) as im:
                hashes.append(imagehash.phash(im.convert("RGB")))

    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in combinations(range(len(items)), 2):
        if hashes[i] - hashes[j] <= THRESHOLD:
            parent[find(i)] = find(j)

    groups = {}
    for i in range(len(items)):
        groups.setdefault(find(i), []).append(i)

    if DST.exists():
        shutil.rmtree(DST)
    moves = []
    for members in groups.values():
        target = min((items[i][0] for i in members), key=SPLITS.index)
        for i in members:
            split, name = items[i]
            if split != target:
                moves.append(f"{split}/{name} -> {target}/ (duplicate of a {target} image)")
            stem = Path(name).stem
            for kind, fname in (("images", name), ("labels", stem + ".txt")):
                out = DST / kind / target
                out.mkdir(parents=True, exist_ok=True)
                # prefix with the original split to avoid filename collisions
                new_name = fname if split == target else f"from-{split}_{fname}"
                shutil.copy2(SRC / kind / split / fname, out / new_name)

    report = ROOT / "results" / "data" / "split_changes.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(f"{len(moves)} images moved (pHash distance <= {THRESHOLD}):\n" + "\n".join(sorted(moves)) + "\n")
    for split in SPLITS:
        print(split, len(list((DST / "images" / split).iterdir())))
    print(f"{len(moves)} images moved; see {report}")


if __name__ == "__main__":
    main()
