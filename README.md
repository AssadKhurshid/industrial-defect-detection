# industrial-defect-detection

Find surface defects (scratches, cracks, contamination, missing parts) in product images for automated visual inspection. A classical OpenCV baseline is compared with a deep anomaly model trained only on good parts, and the model runs as a CPU service with measured latency.

> Work in progress. Results land with milestones 2-4.

## Results
Image-level, MVTec AD test split, per category.
| Category | OpenCV baseline AUROC | Deep model AUROC | Deep model F1 |
| --- | --- | --- | --- |
| _tbd_ | | | |

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
