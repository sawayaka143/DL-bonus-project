"""Compare the two runs on the validation set, pick the final run, and evaluate it on the test set.

Steps:
  1. Validate runs/<run>/weights/best.pt for every run on the VAL split -> results/comparison.{json,md}
  2. Choose the final run by validation mAP@0.50 (the test set is never used for this choice).
  3. Evaluate the chosen checkpoint once on the TEST split -> results/test_metrics.json
  4. Save test images with predicted boxes -> results/predictions/
  5. Match predictions to ground truth on every test image and save the worst cases
     (missed objects, false detections, wrong class, poor localisation) -> results/failures/

Usage: python scripts/evaluate.py [--runs baseline_640 exp_320]
"""
import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from ultralytics import YOLO

from common import ROOT, data_yaml

RES = ROOT / "results"
NAMES = ["buffalo", "elephant", "rhino", "zebra"]
COLORS = ["tab:orange", "tab:blue", "tab:green", "tab:red"]

# Evaluation settings (Ultralytics defaults for mAP computation)
EVAL_CONF = 0.001  # NMS confidence threshold used when computing P/R/mAP (keeps the full PR curve)
EVAL_IOU = 0.7     # NMS IoU threshold
MATCH_IOU = 0.5    # a prediction is a true positive if IoU >= 0.5 with a same-class GT box (mAP@0.50)
VIS_CONF = 0.25    # confidence threshold for drawn predictions and the failure analysis


def val_metrics(weights, split, imgsz, name):
    model = YOLO(weights)
    m = model.val(data=data_yaml(), split=split, imgsz=imgsz, batch=16, conf=EVAL_CONF, iou=EVAL_IOU,
                  project=str(ROOT / "runs" / "eval"), name=name, exist_ok=True, plots=True, verbose=False)
    per_class = {
        NAMES[c]: {"precision": float(m.box.p[i]), "recall": float(m.box.r[i]), "mAP50": float(m.box.ap50[i])}
        for i, c in enumerate(m.box.ap_class_index)
    }
    return {
        "precision": float(m.box.mp), "recall": float(m.box.mr),
        "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map), "per_class": per_class,
    }


def read_gt(img_path):
    lbl = img_path.parent.parent.parent / "labels" / img_path.parent.name / (img_path.stem + ".txt")
    W, H = Image.open(img_path).size
    boxes = []
    for line in lbl.read_text().split("\n"):
        if line.strip():
            c, x, y, w, h = map(float, line.split()[:5])
            boxes.append((int(c), (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H))
    return boxes


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def analyse_image(gt, preds):
    """Greedy matching by confidence. Returns lists of error descriptions per type."""
    errors = {"missed": [], "false_positive": [], "wrong_class": [], "poor_box": []}
    used = set()
    for pc, conf, pbox in sorted(preds, key=lambda p: -p[1]):
        best_j, best_iou = None, 0
        for j, (gc, *gbox) in enumerate(gt):
            if j in used:
                continue
            v = iou(pbox, gbox)
            if v > best_iou:
                best_j, best_iou = j, v
        if best_j is not None and best_iou >= MATCH_IOU:
            used.add(best_j)
            if gt[best_j][0] != pc:
                errors["wrong_class"].append(f"GT {NAMES[gt[best_j][0]]} predicted as {NAMES[pc]} ({conf:.2f})")
        elif best_j is not None and best_iou >= 0.1 and gt[best_j][0] == pc:
            used.add(best_j)
            errors["poor_box"].append(f"{NAMES[pc]} box IoU {best_iou:.2f} ({conf:.2f})")
        else:
            errors["false_positive"].append(f"{NAMES[pc]} ({conf:.2f}), best IoU {best_iou:.2f}")
    for j, (gc, *gbox) in enumerate(gt):
        if j not in used:
            errors["missed"].append(f"{NAMES[gc]} ({(gbox[2]-gbox[0]):.0f}x{(gbox[3]-gbox[1]):.0f}px)")
    return errors


def draw(ax, img_path, boxes, title, with_conf):
    im = Image.open(img_path).convert("RGB")
    ax.imshow(im)
    for b in boxes:
        c, conf, (x1, y1, x2, y2) = (b[0], b[1], b[2]) if with_conf else (b[0], None, b[1:])
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=2, ec=COLORS[c]))
        label = NAMES[c] + (f" {conf:.2f}" if conf is not None else "")
        ax.text(x1, y1, label, color="white", fontsize=9, bbox=dict(fc=COLORS[c], ec="none", pad=1))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


SIZE_BINS = [("small (<10% of image side)", 0, 0.10), ("medium (10-30%)", 0.10, 0.30), ("large (>30%)", 0.30, 9)]


def recall_by_size(weights, split_dir, imgsz):
    """Recall at conf 0.25 / IoU 0.5 (class-aware), split by GT box size relative to the image.
    Size = sqrt(box area / image area), i.e. the box side as a fraction of the image side."""
    model = YOLO(weights)
    hit, tot = [0] * len(SIZE_BINS), [0] * len(SIZE_BINS)
    for res in model.predict([str(p) for p in sorted(split_dir.iterdir())], imgsz=imgsz, conf=VIS_CONF,
                             verbose=False, stream=True):
        p = Path(res.path)
        W, H = res.orig_shape[1], res.orig_shape[0]
        preds = list(zip(res.boxes.cls.int().tolist(), res.boxes.xyxy.tolist()))
        used = set()
        for c, *g in read_gt(p):
            rel = (((g[2] - g[0]) * (g[3] - g[1])) / (W * H)) ** 0.5
            b = next(i for i, (_, lo, hi) in enumerate(SIZE_BINS) if lo <= rel < hi)
            tot[b] += 1
            best = max(((iou(pb, g), k) for k, (pc, pb) in enumerate(preds) if pc == c and k not in used),
                       default=(0, None))
            if best[0] >= MATCH_IOU:
                used.add(best[1])
                hit[b] += 1
    return {name: {"objects": t, "recall": round(h / t, 3) if t else None}
            for (name, _, _), h, t in zip(SIZE_BINS, hit, tot)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["baseline_640", "exp_320"])
    args = ap.parse_args()
    RES.mkdir(exist_ok=True)

    # 1-2. validation comparison and model selection
    rows = []
    for run in args.runs:
        summ = json.loads((ROOT / "runs" / run / "summary.json").read_text())
        m = val_metrics(ROOT / "runs" / run / "weights" / "best.pt", "val", summ["imgsz"], f"{run}_val")
        rows.append({"run": run, "imgsz": summ["imgsz"], "train_time_min": summ["train_time_min"], **m})
    final = max(rows, key=lambda r: r["mAP50"])
    (RES / "comparison.json").write_text(json.dumps({"runs": rows, "selected": final["run"]}, indent=2))
    md = ["| Run | Changed setting (imgsz) | Val precision | Val recall | Val mAP@0.50 | Val mAP@0.50:0.95 | Train time (min) |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['run']} | {r['imgsz']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['mAP50']:.3f} | "
                  f"{r['mAP50-95']:.3f} | {r['train_time_min']:.1f} |")
    # recall by object size on the validation set (tests the hypothesis of the experiment)
    val_dir = Path(yaml.safe_load(Path(data_yaml()).read_text())["path"]) / "images" / "val"
    md += ["", "Validation recall by object size (conf 0.25, IoU 0.5):", "",
           "| Run | " + " | ".join(n for n, _, _ in SIZE_BINS) + " |", "|---|" + "---|" * len(SIZE_BINS)]
    for r in rows:
        r["recall_by_size"] = recall_by_size(ROOT / "runs" / r["run"] / "weights" / "best.pt", val_dir, r["imgsz"])
        md.append(f"| {r['run']} | " + " | ".join(f"{v['recall']:.3f} (n={v['objects']})"
                                                  for v in r["recall_by_size"].values()) + " |")
    (RES / "comparison.json").write_text(json.dumps({"runs": rows, "selected": final["run"]}, indent=2))
    (RES / "comparison.md").write_text("\n".join(md) + f"\n\nSelected by val mAP@0.50: **{final['run']}**\n")
    print("\n".join(md), "\nselected:", final["run"])

    # 3. single test evaluation of the selected checkpoint
    weights = ROOT / "runs" / final["run"] / "weights" / "best.pt"
    test = val_metrics(weights, "test", final["imgsz"], f"{final['run']}_test")
    test.update(run=final["run"], imgsz=final["imgsz"], nms_conf=EVAL_CONF, nms_iou=EVAL_IOU, match_iou=MATCH_IOU,
                note="P and R are Ultralytics' values at the confidence that maximises mean F1 on the PR curve")
    (RES / "test_metrics.json").write_text(json.dumps(test, indent=2))
    print(json.dumps(test, indent=2))
    for f in ("confusion_matrix.png", "BoxPR_curve.png"):
        src = ROOT / "runs" / "eval" / f"{final['run']}_test" / f
        if src.exists():
            shutil.copy(src, RES / f"test_{f}")

    # 4-5. predictions on the test set and failure mining
    model = YOLO(weights)
    test_dir = Path(yaml.safe_load(Path(data_yaml()).read_text())["path"]) / "images" / "test"
    images = sorted(test_dir.iterdir())
    per_image = []
    for res in model.predict([str(p) for p in images], imgsz=final["imgsz"], conf=VIS_CONF, verbose=False, stream=True):
        p = Path(res.path)
        preds = [(int(c), float(s), tuple(map(float, b)))
                 for c, s, b in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(), res.boxes.xyxy.tolist())]
        gt = read_gt(p)
        errs = analyse_image(gt, preds)
        per_image.append({"image": p.name, "path": str(p), "gt": gt, "preds": preds, "errors": errs,
                          "n_errors": sum(len(v) for v in errs.values())})

    out = RES / "predictions"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    sample = [per_image[i] for i in sorted(rng.choice(len(per_image), 8, replace=False))]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for ax, r in zip(axes.ravel(), sample):
        draw(ax, r["path"], r["preds"], r["image"], with_conf=True)
    plt.tight_layout()
    plt.savefig(out / "test_predictions_grid.jpg", dpi=80)
    plt.close()

    fail_dir = RES / "failures"
    fail_dir.mkdir(exist_ok=True)
    failures = sorted([r for r in per_image if r["n_errors"]], key=lambda r: -r["n_errors"])
    summary = {k: sum(len(r["errors"][k]) for r in per_image) for k in ("missed", "false_positive", "wrong_class", "poor_box")}
    summary["images_with_errors"] = len(failures)
    summary["test_images"] = len(per_image)
    # one strongest example of each error type, then the images with most errors
    picked, seen = [], set()
    for k in ("missed", "false_positive", "wrong_class", "poor_box"):
        for r in sorted(failures, key=lambda r: -len(r["errors"][k])):
            if r["errors"][k] and r["image"] not in seen:
                picked.append(r); seen.add(r["image"]); break
    for r in failures:
        if len(picked) >= 6:
            break
        if r["image"] not in seen:
            picked.append(r); seen.add(r["image"])
    for i, r in enumerate(picked, 1):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))
        draw(a1, r["path"], r["gt"], f"{r['image']} - ground truth", with_conf=False)
        draw(a2, r["path"], r["preds"], "prediction (conf >= 0.25)", with_conf=True)
        fig.suptitle("; ".join(f"{k}: {', '.join(v)}" for k, v in r["errors"].items() if v), fontsize=9)
        plt.tight_layout()
        plt.savefig(fail_dir / f"failure_{i}.jpg", dpi=80)
        plt.close()
    (fail_dir / "failures.json").write_text(json.dumps(
        {"summary": summary, "visualised": [{"file": f"failure_{i}.jpg", "image": r["image"], "errors": r["errors"]}
                                             for i, r in enumerate(picked, 1)],
         "all_images_with_errors": [{"image": r["image"], "errors": r["errors"]} for r in failures]}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
