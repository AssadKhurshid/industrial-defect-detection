"""HTTP API: upload a product image, get its anomaly score back.

Run:  pip install -e ".[api]"  then  IDD_MODEL=out/onnx/model.onnx uvicorn idd.api:app
Needs only ONNX Runtime, numpy and OpenCV: no PyTorch.

Settings (environment variables):
- IDD_MODEL: path to the ONNX file from idd.export (default /models/model.onnx, the Docker mount)
- IDD_THRESHOLD: score above which an image counts as a defect. Set it per product from good
  parts on your line; without it the API returns the score and is_defect = null.
"""

import os
import time
from functools import lru_cache

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, UploadFile

from idd.preprocess import to_input

app = FastAPI(title="industrial-defect-detection")


@lru_cache(maxsize=1)
def session() -> ort.InferenceSession:
    """Loaded on the first request, so /health answers even before a model is mounted."""
    return ort.InferenceSession(os.environ.get("IDD_MODEL", "/models/model.onnx"), providers=["CPUExecutionProvider"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/inspect")
def inspect(image: UploadFile) -> dict:
    """Score one image. A plain `def`: FastAPI runs it in a worker thread, so a slow model
    call doesn't block other requests."""
    decoded = cv2.imdecode(np.frombuffer(image.file.read(), np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=422, detail="not a readable image")
    model = session()
    size = model.get_inputs()[0].shape[-1]  # the size the model was exported for
    start = time.perf_counter()
    (anomaly_map,) = model.run(None, {"image": to_input(decoded, size)[None]})
    model_ms = (time.perf_counter() - start) * 1000
    score = float(anomaly_map.max())
    threshold = float(os.environ["IDD_THRESHOLD"]) if os.environ.get("IDD_THRESHOLD") else None
    return {
        "score": round(score, 4),
        "threshold": threshold,
        "is_defect": None if threshold is None else score > threshold,
        "model_ms": round(model_ms, 1),
    }
