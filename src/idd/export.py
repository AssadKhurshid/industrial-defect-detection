"""Export a fitted PatchCore to one ONNX file: image in, anomaly map out.

The graph holds the CNN, the patch pooling and the memory bank, so ONNX Runtime needs nothing
else from this repo (no torch) to score an image.
"""

import torch
from torch import nn

from idd.patchcore import PatchCore


class ScoringNet(nn.Module):
    """PatchCore.anomaly_map as one torch module, written with ops that export cleanly."""

    def __init__(self, model: PatchCore):
        super().__init__()
        self.model = model
        self.net = getattr(model.backbone, "net", None)  # registered so its weights go into the file
        memory = model.memory.float().cpu()
        self.register_buffer("memory_t", memory.T.contiguous())
        self.register_buffer("memory_sq", (memory**2).sum(1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats, (h, w) = self.model.patch_features(images)
        # Squared distance to every bank entry, |f|^2 - 2 f.m + |m|^2: one matrix product
        # instead of cdist, so the graph is a plain MatMul that every ONNX runtime supports.
        d2 = (feats**2).sum(-1, keepdim=True) - 2 * feats @ self.memory_t + self.memory_sq
        return d2.amin(-1).clamp_min(0).sqrt().reshape(-1, h, w)


def export_onnx(model: PatchCore, path, size: int = 224) -> None:
    """Write the fitted model to path, for (1, 3, size, size) ImageNet-normalised images."""
    model.device = "cpu"
    if model.memory.device.type != "cpu":
        model.memory = model.memory.cpu()
    if hasattr(model.backbone, "net"):
        model.backbone.net.cpu()
    torch.onnx.export(
        ScoringNet(model).eval(),
        (torch.zeros(1, 3, size, size),),
        str(path),
        input_names=["image"],
        output_names=["anomaly_map"],
        verbose=False,  # its progress lines contain emoji that crash a Windows cp1252 console
    )
