"""Kaggle job: heatmap figure for the README. For a few categories, the first defect image of
the test split (sorted order, not hand-picked): image, true defect mask, baseline map, PatchCore map.

Push from the repo root:  kaggle kernels push -p notebooks/kaggle_heatmaps
Output in /kaggle/working: heatmaps.png
"""

import glob
import os
import subprocess
import sys

# Installs the latest main from GitHub, so push code before starting a run.
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q",
        "industrial-defect-detection @ git+https://github.com/AssadKhurshid/industrial-defect-detection.git",
    ],
    check=True,
)

import cv2  # installed with the package above
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from idd.baseline import PixelStatsBaseline, read_gray
from idd.data import load_split
from idd.patchcore import PatchCore, load_image

CATEGORIES = ["bottle", "carpet", "metal_nut", "screw"]  # one easy case + the baseline's 3 failures
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CROP = slice(16, 240)  # PatchCore sees the centre 224 px of the image resized to 256

root = os.path.dirname(os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/bottle/train/good", recursive=True)[0])))
model = PatchCore(coreset_ratio=0.1, device=DEVICE)
fig, axes = plt.subplots(len(CATEGORIES), 4, figsize=(10, 2.6 * len(CATEGORIES)), squeeze=False)

for row, category in zip(axes, CATEGORIES, strict=True):
    category_dir = os.path.join(root, category)
    train, test = load_split(category_dir, "train"), load_split(category_dir, "test")
    sample = next(s for s in test if s.label == 1)

    baseline = PixelStatsBaseline().fit(read_gray(s.image) for s in train)
    base_map = baseline.anomaly_map(read_gray(sample.image))[CROP, CROP]

    model.fit(torch.stack([load_image(s.image) for s in train[i : i + 32]]) for i in range(0, len(train), 32))
    pc_map = model.anomaly_map(load_image(sample.image)[None])  # (1, 28, 28)
    pc_score = float(pc_map.max())
    pc_map = F.interpolate(pc_map[None], size=224, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    pc_map = cv2.GaussianBlur(pc_map, (0, 0), 4)  # smooth the 8 px patch grid, as in the PatchCore paper

    image = cv2.cvtColor(cv2.resize(cv2.imread(str(sample.image)), (256, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)[CROP, CROP]
    mask = cv2.resize(cv2.imread(str(sample.mask), cv2.IMREAD_GRAYSCALE), (256, 256), interpolation=cv2.INTER_NEAREST)[CROP, CROP]

    panels = [
        (image, None, f"{category} / {sample.defect}"),
        (mask, "gray", "true defect"),
        (base_map, "inferno", f"baseline, score {base_map.max():.1f}"),
        (pc_map, "inferno", f"PatchCore, score {pc_score:.2f}"),
    ]
    for ax, (img, cmap, title) in zip(row, panels, strict=True):
        ax.imshow(img, cmap=cmap)  # maps scaled to their own min..max: compare where, not how much
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    print(category, sample.image.name, sample.defect, flush=True)

fig.tight_layout()
fig.savefig("heatmaps.png", dpi=80)
print("saved heatmaps.png", os.path.getsize("heatmaps.png") // 1000, "KB")
