"""Kaggle job: PatchCore-lite on all 15 MVTec AD categories (GPU).

Push from the repo root:  kaggle kernels push -p notebooks/kaggle_patchcore
Outputs in /kaggle/working: patchcore.json (per-category image AUROC and best F1).
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
import torch

from idd.data import load_split
from idd.metrics import auroc, f1_max
from idd.patchcore import PatchCore, load_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 32


def batches(samples):
    for i in range(0, len(samples), BATCH):
        yield torch.stack([load_image(s.image) for s in samples[i : i + BATCH]])


category_dirs = sorted(os.path.dirname(os.path.dirname(p)) for p in glob.glob("/kaggle/input/**/train/good", recursive=True))
print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "", flush=True)

results = []
model = PatchCore(coreset_ratio=0.1, device=DEVICE)  # backbone loaded once, memory bank refit per category
for category_dir in category_dirs:
    start = time.time()
    train, test = load_split(category_dir, "train"), load_split(category_dir, "test")
    model.fit(batches(train))
    fit_seconds = time.time() - start
    labels = [s.label for s in test]
    scores = torch.cat([model.score(b) for b in batches(test)]).cpu().tolist()
    f1, _ = f1_max(labels, scores)
    results.append({
        "category": os.path.basename(category_dir),
        "train_good": len(train),
        "memory_bank": len(model.memory),
        "test_good": labels.count(0),
        "test_defect": labels.count(1),
        "auroc": round(auroc(labels, scores), 3),
        "f1_max": round(f1, 3),
        "fit_seconds": round(fit_seconds, 1),
        "seconds": round(time.time() - start, 1),
    })
    print(results[-1], flush=True)

summary = {
    "model": "PatchCore-lite: Wide ResNet-50 layer2+3, 224 px, 10% greedy coreset",
    "device": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
    "categories": len(results),
    "mean_auroc": round(float(np.mean([r["auroc"] for r in results])), 3),
    "mean_f1_max": round(float(np.mean([r["f1_max"] for r in results])), 3),
    "results": results,
}
with open("patchcore.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
