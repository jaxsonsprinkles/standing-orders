"""T0 change gate: per-region frame difference with lighting normalization. CPU only.

Each region (a zone polygon, or the whole frame) is compared with the last frame that PASSED the
gate, so slow changes accumulate until they cross the threshold instead of slipping through.
Lighting: pixels are z-scored inside the region (zero mean, unit variance, with a std floor) before
differencing, so a brightness or contrast shift alone, e.g. a lamp turning on, scores near zero."""

from __future__ import annotations
import cv2
import numpy as np

from .program import FRAME_ZONE, Point

ANALYSIS_WIDTH = 96      # work on a small image: cheap, and averages out sensor noise
STD_FLOOR = 6.0          # gray levels; stops a flat region's noise being amplified to full scale
DEFAULT_THRESHOLD = 0.25  # mean |z-score difference|; tune on recorded clips (scripts/measure_t0.py)


class ChangeGate:
    def __init__(self, regions: dict[str, list[Point]], threshold: float = DEFAULT_THRESHOLD):
        """regions: name -> polygon in normalized coordinates. Use FRAME_ZONE for the whole frame."""
        self.regions, self.threshold = regions, threshold
        self._masks: dict[str, np.ndarray] = {}
        self._ref: dict[str, np.ndarray] = {}

    def _mask(self, name: str, shape: tuple[int, int]) -> np.ndarray:
        if name not in self._masks or self._masks[name].shape != shape:
            h, w = shape
            m = np.zeros(shape, np.uint8)
            if name == FRAME_ZONE:
                m[:] = 1
            else:
                pts = np.array([[round(x * (w - 1)), round(y * (h - 1))] for x, y in self.regions[name]], np.int32)
                cv2.fillPoly(m, [pts], 1)
                cv2.polylines(m, [pts], True, 1)  # a sliver still covers its outline pixels
            self._masks[name] = m.astype(bool)
        return self._masks[name]

    @staticmethod
    def _normalize(pixels: np.ndarray) -> np.ndarray:
        p = pixels.astype(np.float32)
        return (p - p.mean()) / max(float(p.std()), STD_FLOOR)

    def score(self, name: str, image: np.ndarray) -> float | None:
        """Difference from the reference for one region. None if there is no reference yet."""
        gray = self._gray(image)
        ref = self._ref.get(name)
        if ref is None:
            return None
        return float(np.abs(self._normalize(gray[self._mask(name, gray.shape)]) - ref).mean())

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        small = cv2.resize(image, (ANALYSIS_WIDTH, max(1, round(h * ANALYSIS_WIDTH / w))), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small

    def update(self, image: np.ndarray) -> dict[str, bool]:
        """True per region if it changed enough to need a closer look. The first frame is a change."""
        gray = self._gray(image)
        out = {}
        for name in self.regions:
            norm = self._normalize(gray[self._mask(name, gray.shape)])
            ref = self._ref.get(name)
            if ref is None or float(np.abs(norm - ref).mean()) >= self.threshold:
                self._ref[name] = norm
                out[name] = True
            else:
                out[name] = False
        return out
