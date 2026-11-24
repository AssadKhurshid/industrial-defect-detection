# industrial-defect-detection

Find surface defects (scratches, cracks, contamination, missing parts) in product images for automated visual inspection. A classical OpenCV baseline is compared with a deep anomaly model trained only on good parts, and the model runs as a CPU service with measured latency.

> Work in progress. Deep model results land with milestone 3, latency with milestone 4.

## Results
Image-level, MVTec AD test split, per category. Each model is fit on that category's `train/good` images only. Baseline: one Kaggle CPU run, about 50 min for all 15 categories.

| Category | Test images (good / defect) | Baseline AUROC | Baseline F1-max | F1 if every image is called defect | Deep model AUROC | Deep model F1-max |
| --- | --- | --- | --- | --- | --- | --- |
| bottle | 20 / 63 | 0.977 | 0.976 | 0.863 | | |
| cable | 58 / 92 | 0.596 | 0.763 | 0.760 | | |
| capsule | 23 / 109 | 0.729 | 0.911 | 0.905 | | |
| carpet | 28 / 89 | 0.459 | 0.864 | 0.864 | | |
| grid | 21 / 57 | 0.681 | 0.844 | 0.844 | | |
| hazelnut | 40 / 70 | 0.765 | 0.845 | 0.778 | | |
| leather | 32 / 92 | 0.786 | 0.852 | 0.852 | | |
| metal_nut | 22 / 93 | 0.587 | 0.894 | 0.894 | | |
| pill | 26 / 141 | 0.606 | 0.916 | 0.916 | | |
| screw | 41 / 119 | 0.454 | 0.853 | 0.853 | | |
| tile | 33 / 84 | 0.936 | 0.922 | 0.836 | | |
| toothbrush | 12 / 30 | 0.975 | 0.966 | 0.833 | | |
| transistor | 60 / 40 | 0.835 | 0.753 | 0.571 | | |
| wood | 19 / 60 | 0.946 | 0.950 | 0.863 | | |
| zipper | 32 / 119 | 0.775 | 0.918 | 0.881 | | |
| **mean** | | **0.740** | **0.882** | 0.834 | | |

Read F1 with care: most MVTec test sets have more defect than good images, so calling *every* image a defect already scores the F1 in the "every image is called defect" column. In 9 of 15 categories the baseline's best F1 is no better than that, which is why AUROC is the main metric here.

The baseline (per-pixel mean and spread of good images) works where every part sits in the same place and the surface is even: bottle, toothbrush, wood, tile reach AUROC 0.94-0.98. It fails where good parts vary in position or rotation (screw 0.454, metal_nut 0.587) and on carpet (0.459), worse than a coin flip, because a pixel there has no fixed "normal" value.

Latency (one image, CPU): _tbd_

## Run it
```bash
pip install -e ".[dev]"
pytest
```

## How it works
_tbd_

## Limitations
_tbd_

## Data
[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) (Bergmann et al., 2019/2021), licence [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/): non-commercial use only. Not stored in this repo.
