"""Kaggle job: fit PatchCore-lite on one MVTec category (GPU), export it to ONNX, and check on
the real test images that ONNX Runtime gives the same scores as PyTorch.

Push from the repo root:  kaggle kernels push -p notebooks/kaggle_export
Outputs in /kaggle/working: model.onnx + model.onnx.data (weights), export.json.
"""

import glob
import json
import os
import subprocess
import sys

# Installs the latest main from GitHub, so push code before starting a run.
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q",
        "industrial-defect-detection[deep] @ git+https://github.com/AssadKhurshid/industrial-defect-detection.git",
    ],
    check=True,
)

import numpy as np  # installed with the package above
import onnxruntime as ort
import torch

from idd.data import load_split
from idd.export import export_onnx
from idd.metrics import auroc
from idd.patchcore import PatchCore, load_image

CATEGORY = "bottle"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

(category_dir,) = [os.path.dirname(os.path.dirname(p)) for p in glob.glob(f"/kaggle/input/**/{CATEGORY}/train/good", recursive=True)]
train, test = load_split(category_dir, "train"), load_split(category_dir, "test")

model = PatchCore(coreset_ratio=0.1, device=DEVICE)
model.fit(torch.stack([load_image(s.image) for s in train[i : i + 32]]) for i in range(0, len(train), 32))

export_onnx(model, "model.onnx")  # also moves the model to the CPU
session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

labels, torch_scores, onnx_scores = [], [], []
for s in test:
    x = load_image(s.image)[None]
    labels.append(s.label)
    torch_scores.append(float(model.score(x)[0]))
    onnx_scores.append(float(session.run(None, {"image": x.numpy()})[0].max()))

summary = {
    "category": CATEGORY,
    "memory_bank": len(model.memory),
    "test_images": len(test),
    "auroc_pytorch": round(auroc(labels, torch_scores), 3),
    "auroc_onnx": round(auroc(labels, onnx_scores), 3),
    "max_abs_score_diff": float(np.max(np.abs(np.array(torch_scores) - np.array(onnx_scores)))),
    "max_score": max(torch_scores),
    "onnx_mb": round(sum(os.path.getsize(f) for f in glob.glob("model.onnx*")) / 1e6),
}
with open("export.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
