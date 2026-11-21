| Run | Changed setting (imgsz) | Val precision | Val recall | Val mAP@0.50 | Val mAP@0.50:0.95 | Train time (min) |
|---|---|---|---|---|---|---|
| baseline_640 | 640 | 0.972 | 0.879 | 0.959 | 0.801 | 9.0 |
| exp_320 | 320 | 0.951 | 0.872 | 0.932 | 0.765 | 5.6 |

Validation recall by object size (conf 0.25, IoU 0.5):

| Run | small (<10% of image side) | medium (10-30%) | large (>30%) |
|---|---|---|---|
| baseline_640 | 0.500 (n=8) | 0.842 (n=95) | 0.947 (n=263) |
| exp_320 | 0.250 (n=8) | 0.821 (n=95) | 0.935 (n=263) |

Selected by val mAP@0.50: **baseline_640**
