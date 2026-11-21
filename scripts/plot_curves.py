"""Plot validation learning curves of both runs on one figure -> results/learning_curves_comparison.png"""
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ROOT

RUNS = {"baseline_640": "imgsz 640 (baseline)", "exp_320": "imgsz 320 (experiment)"}
PANELS = [("metrics/mAP50(B)", "val mAP@0.50"), ("metrics/precision(B)", "val precision"),
          ("metrics/recall(B)", "val recall"), ("train/box_loss", "train box loss"), ("val/box_loss", "val box loss")]

fig, axes = plt.subplots(1, len(PANELS), figsize=(24, 4.5))
for run, label in RUNS.items():
    rows = list(csv.DictReader(open(ROOT / "runs" / run / "results.csv")))
    ep = [int(r["epoch"]) for r in rows]
    for ax, (key, title) in zip(axes, PANELS):
        ax.plot(ep, [float(r[key]) for r in rows], label=label)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
axes[0].legend()
plt.tight_layout()
plt.savefig(ROOT / "results" / "learning_curves_comparison.png", dpi=80)
print("saved results/learning_curves_comparison.png")
