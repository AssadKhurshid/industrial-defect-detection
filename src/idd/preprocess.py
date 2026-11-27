"""Image -> model input, with numpy and OpenCV only, so the API image needs no PyTorch."""

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_input(image_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    """OpenCV BGR image -> (3, size, size) float32: resize, center crop, ImageNet normalise
    (the backbone was trained on images prepared this way)."""
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resize = round(size * 256 / 224)  # resize 256 then crop 224, scaled to any size
    image = cv2.resize(image, (resize, resize), interpolation=cv2.INTER_AREA)
    top = (resize - size) // 2
    image = image[top : top + size, top : top + size].astype(np.float32) / 255
    return ((image - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).copy()
