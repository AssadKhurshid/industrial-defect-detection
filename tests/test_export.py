import numpy as np
import onnxruntime as ort
import torch
from test_patchcore import tiny_backbone

from idd.export import export_onnx
from idd.patchcore import PatchCore


def test_onnx_matches_pytorch(tmp_path):
    torch.manual_seed(0)
    model = PatchCore(backbone=tiny_backbone, coreset_ratio=0.5).fit([torch.rand(4, 3, 64, 64)])
    image = torch.rand(1, 3, 64, 64)
    expected = model.anomaly_map(image).numpy()

    export_onnx(model, tmp_path / "model.onnx", size=64)
    session = ort.InferenceSession(str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"])
    (got,) = session.run(None, {"image": image.numpy()})

    assert got.shape == expected.shape == (1, 8, 8)
    np.testing.assert_allclose(got, expected, atol=1e-4)
