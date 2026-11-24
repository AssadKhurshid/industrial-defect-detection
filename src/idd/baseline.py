"""Classical baseline: per-pixel statistics of good images, no learning beyond mean and spread.

Works when parts are photographed in the same position on an even surface. On MVTec AD it
does well on bottle, toothbrush, wood and tile, and fails where good parts vary in position or
rotation (screw, metal_nut) and on carpet: there a pixel has no fixed "normal" value.
"""

import cv2
import numpy as np


def read_gray(path, size: int = 256) -> np.ndarray:
    """Grayscale image resized to size x size, as float32."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"cannot read image {path}")
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)


class PixelStatsBaseline:
    """Learn mean and standard deviation of every pixel over good images. A pixel is suspicious
    when it is many standard deviations away from its mean (a z-score)."""

    def __init__(self, blur: int = 5, eps: float = 5.0):
        self.blur = blur  # odd Gaussian kernel size: smooths noise before and after comparing
        self.eps = eps  # added to the std, so pixels that never vary don't explode the z-score

    def fit(self, images) -> "PixelStatsBaseline":
        stack = np.stack([self._smooth(img) for img in images])
        self.mean, self.std = stack.mean(axis=0), stack.std(axis=0)
        return self

    def anomaly_map(self, image: np.ndarray) -> np.ndarray:
        z = np.abs(self._smooth(image) - self.mean) / (self.std + self.eps)
        return self._smooth(z)  # a defect is a blob, not a single noisy pixel

    def score(self, image: np.ndarray) -> float:
        return float(self.anomaly_map(image).max())

    def _smooth(self, image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image, (self.blur, self.blur), 0)
