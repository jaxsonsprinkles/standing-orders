"""T1 image/text embeddings. Implementations: ClipOnnxEmbedder (real), FakeEmbedder (tests)."""
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed_image(self, image: np.ndarray) -> np.ndarray:
        """BGR array -> L2-normalized float32 vector."""
        ...

    def embed_text(self, text: str) -> np.ndarray:
        """L2-normalized float32 vector in the same space as embed_image."""
        ...


class ClipOnnxEmbedder:
    """Xenova/clip-vit-base-patch32 quantized ONNX (revision and sha256 pinned in docs/day1-findings.md).
    Weights are downloaded at setup (download_models), checksum-verified, never committed."""

    def __init__(self, model_dir: str):
        raise NotImplementedError

    def embed_image(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def embed_text(self, text: str) -> np.ndarray:
        raise NotImplementedError


def download_models(model_dir: str) -> None:
    """Fetch the pinned files, verify sha256, raise on mismatch. Weights only, never code."""
    raise NotImplementedError
