"""Kaggle job: OpenCV pixel-statistics baseline on all 15 MVTec AD categories (CPU only).

Push from the repo root:  kaggle kernels push -p notebooks/kaggle_baseline
Outputs in /kaggle/working: baseline.json (per-category image AUROC and best F1).
"""

import glob
import json
import os
import subprocess
import sys
import time

# Installs the latest main from GitHub, so push code before starting a run.
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q",
        "industrial-defect-detection @ git+https://github.com/AssadKhurshid/industrial-defect-detection.git",
    ],
    check=True,
)

import numpy as np  # installed with the package above

from idd.baseline import PixelStatsBaseline, read_gray
from idd.data import load_split
from idd.metrics import auroc, f1_max

# A category is any folder that has train/good/ inside.
category_dirs = sorted(os.path.dirname(os.path.dirname(p)) for p in glob.glob("/kaggle/input/**/train/good", recursive=True))

results = []
for category_dir in category_dirs:
    start = time.time()
    train, test = load_split(category_dir, "train"), load_split(category_dir, "test")
    model = PixelStatsBaseline().fit(read_gray(s.image) for s in train)
    labels = [s.label for s in test]
    scores = [model.score(read_gray(s.image)) for s in test]
    f1, _ = f1_max(labels, scores)
    results.append({
        "category": os.path.basename(category_dir),
        "train_good": len(train),
        "test_good": labels.count(0),
        "test_defect": labels.count(1),
        "auroc": round(auroc(labels, scores), 3),
        "f1_max": round(f1, 3),
        "seconds": round(time.time() - start, 1),
    })
    print(results[-1], flush=True)

summary = {
    "model": "PixelStatsBaseline(blur=5, eps=5.0), 256x256 grayscale",
    "categories": len(results),
    "mean_auroc": round(float(np.mean([r["auroc"] for r in results])), 3),
    "mean_f1_max": round(float(np.mean([r["f1_max"] for r in results])), 3),
    "results": results,
}
with open("baseline.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
