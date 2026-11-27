import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from test_patchcore import tiny_backbone

from idd import api
from idd.export import export_onnx
from idd.patchcore import PatchCore
from idd.preprocess import to_input


@pytest.fixture
def client(tmp_path, monkeypatch):
    """API with a tiny exported model fit on dark, slightly noisy images."""
    rng = np.random.default_rng(0)
    good = [rng.integers(0, 25, (80, 80, 3), dtype=np.uint8) for _ in range(4)]
    batch = torch.stack([torch.from_numpy(to_input(img, 64)) for img in good])  # same path as the API
    model = PatchCore(backbone=tiny_backbone, coreset_ratio=0.5).fit([batch])
    export_onnx(model, tmp_path / "model.onnx", size=64)
    monkeypatch.setenv("IDD_MODEL", str(tmp_path / "model.onnx"))
    api.session.cache_clear()
    yield TestClient(api.app)
    api.session.cache_clear()


def png(image: np.ndarray) -> bytes:
    return cv2.imencode(".png", image)[1].tobytes()


def test_inspect_flags_the_bright_image(client, monkeypatch):
    monkeypatch.setenv("IDD_THRESHOLD", "1.0")
    dark = np.full((80, 80, 3), 12, np.uint8)
    bright = dark.copy()
    bright[20:50, 20:50] = 255

    good = client.post("/inspect", files={"image": ("good.png", png(dark))}).json()
    bad = client.post("/inspect", files={"image": ("bad.png", png(bright))}).json()

    assert bad["score"] > good["score"]
    assert (good["is_defect"], bad["is_defect"]) == (False, True)


def test_inspect_rejects_a_non_image(client):
    response = client.post("/inspect", files={"image": ("x.png", b"not an image")})
    assert response.status_code == 422
