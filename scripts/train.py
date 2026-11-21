"""Fine-tune YOLO11n on the cleaned African Wildlife split.

Usage:
  python scripts/train.py --name baseline_640 --imgsz 640
  python scripts/train.py --name exp_320 --imgsz 320

Everything except --imgsz (and --name) is fixed so the two runs form a controlled experiment.
Outputs go to runs/<name>/ (Ultralytics' args.yaml, results.csv, curves, weights/best.pt);
a small summary with the training time is written to runs/<name>/summary.json.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO

from common import data_yaml

ROOT = Path(__file__).resolve().parents[1]

# Settings shared by both runs (do not change between runs).
FIXED = dict(
    epochs=50,
    batch=16,
    optimizer="SGD",  # set explicitly; "auto" would choose the optimizer/LR by itself
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=5e-4,
    warmup_epochs=3,
    cos_lr=False,
    patience=100,  # no early stopping within 50 epochs
    seed=0,
    deterministic=True,
    workers=4,
    pretrained=True,
    plots=True,
)
WEIGHTS = "yolo11n.pt"  # COCO-pretrained YOLO11 nano checkpoint, downloaded by Ultralytics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=FIXED["epochs"], help="override only for quick smoke tests")
    ap.add_argument("--device", default=None, help="e.g. 0 or cpu; default: auto")
    args = ap.parse_args()

    cfg = {**FIXED, "epochs": args.epochs, "data": data_yaml()}
    model = YOLO(WEIGHTS)
    t0 = time.time()
    model.train(**cfg, imgsz=args.imgsz, project=str(ROOT / "runs"), name=args.name, exist_ok=True,
                device=args.device)
    minutes = (time.time() - t0) / 60

    run_dir = ROOT / "runs" / args.name
    summary = {
        "run": args.name,
        "framework": f"ultralytics {ultralytics.__version__}, torch {torch.__version__}",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "initial_weights": WEIGHTS,
        "imgsz": args.imgsz,
        **{k: cfg[k] for k in ("epochs", "batch", "optimizer", "lr0", "seed")},
        "train_time_min": round(minutes, 2),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
