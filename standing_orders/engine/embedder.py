"""T1 image/text embeddings. Implementations: ClipOnnxEmbedder (real), FakeEmbedder (tests)."""
import hashlib
import os
import pathlib
import urllib.request
from typing import Protocol

import cv2
import numpy as np

MODEL_REPO = "Xenova/clip-vit-base-patch32"
MODEL_REVISION = "6b7c523703ae26e39be4b03774966f8571c7f523"
# file -> sha256 (None: fetched from the same pinned revision but not checksummed; small config files)
MODEL_FILES: dict[str, str | None] = {
    "onnx/vision_model_quantized.onnx": "583fd1110a514667812fee7d684952aaf82a99b959760c8d7dca7e0ab9839299",
    "onnx/text_model_quantized.onnx": "73baab855d406190da9faa498cfedf65f15cf309f4cc7385b7b032e6d08e5c3a",
    "tokenizer.json": None,
    "preprocessor_config.json": None,
}
DEFAULT_MODEL_DIR = os.path.expanduser("~/.cache/standing-orders-models")
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


class Embedder(Protocol):
    def embed_image(self, image: np.ndarray) -> np.ndarray:
        """BGR array -> L2-normalized float32 vector."""
        ...

    def embed_text(self, text: str) -> np.ndarray:
        """L2-normalized float32 vector in the same space as embed_image."""
        ...


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def models_present(model_dir: str) -> bool:
    return all((pathlib.Path(model_dir) / f).is_file() for f in MODEL_FILES)


class ClipOnnxEmbedder:
    """Xenova/clip-vit-base-patch32 quantized ONNX (revision and sha256 pinned in docs/day1-findings.md).
    Weights are downloaded at setup (download_models), checksum-verified, never committed."""

    def __init__(self, model_dir: str):
        import onnxruntime as ort          # heavy imports stay out of the engine's import path
        from tokenizers import Tokenizer

        d = pathlib.Path(model_dir)
        missing = [f for f in MODEL_FILES if not (d / f).is_file()]
        if missing:
            raise FileNotFoundError(f"model files missing in {d}: {', '.join(missing)} (run download_models)")
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self._vision = ort.InferenceSession(str(d / "onnx/vision_model_quantized.onnx"), opts,
                                            providers=["CPUExecutionProvider"])
        self._text = ort.InferenceSession(str(d / "onnx/text_model_quantized.onnx"), opts,
                                          providers=["CPUExecutionProvider"])
        self._tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self._tok.no_padding()
        self._tok.enable_truncation(max_length=77)
        self._text_cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _preprocess(image: np.ndarray) -> np.ndarray:
        """CLIP preprocessing: bicubic resize of the short side to 224, center crop, normalize."""
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h, w = image.shape[:2]
        scale = 224 / min(h, w)
        nh, nw = max(224, round(h * scale)), max(224, round(w * scale))
        image = cv2.resize(image[..., :3], (nw, nh), interpolation=cv2.INTER_CUBIC)
        y0, x0 = (nh - 224) // 2, (nw - 224) // 2
        rgb = cv2.cvtColor(image[y0:y0 + 224, x0:x0 + 224], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return ((rgb - _MEAN) / _STD).transpose(2, 0, 1)[None].astype(np.float32)

    def embed_image(self, image: np.ndarray) -> np.ndarray:
        out = self._vision.run(["image_embeds"], {"pixel_values": self._preprocess(image)})[0]
        return _unit(out[0])

    def embed_text(self, text: str) -> np.ndarray:
        if text not in self._text_cache:
            ids = np.array([self._tok.encode(text).ids], np.int64)
            self._text_cache[text] = _unit(self._text.run(["text_embeds"], {"input_ids": ids})[0][0])
        return self._text_cache[text]


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_models(model_dir: str = DEFAULT_MODEL_DIR) -> None:
    """Fetch the pinned files, verify sha256, raise on mismatch. Weights only, never code.
    Files already present with a matching checksum are kept. A mismatch deletes the download."""
    root = pathlib.Path(model_dir)
    for rel, want in MODEL_FILES.items():
        dest = root / rel
        if dest.is_file() and (want is None or _sha256(dest) == want):
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        url = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{rel}"
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        if want is not None and _sha256(tmp) != want:
            tmp.unlink()
            raise ValueError(f"sha256 mismatch for {rel}: refusing to use the download")
        tmp.replace(dest)


class FakeEmbedder:
    """Deterministic test embedder with no model. Images are described by a coarse colour histogram
    with the most common colour (the background) removed, so a box of any colour on a plain desk is
    its own direction and an empty desk is another. Text maps to the image given for it in `concepts`;
    unknown text gets a fixed direction of its own, unrelated to every image."""

    BINS, DIM = 4, 4 ** 3 + 2

    def __init__(self, concepts: dict[str, np.ndarray] | None = None):
        self.concepts = {k: self.embed_image(v) for k, v in (concepts or {}).items()}

    def embed_image(self, image: np.ndarray) -> np.ndarray:
        px = np.asarray(image).reshape(-1, image.shape[-1] if np.ndim(image) == 3 else 1)[:, :3]
        if px.shape[1] == 1:
            px = np.repeat(px, 3, axis=1)
        q = (px.astype(np.int32) * self.BINS // 256)
        idx = q[:, 0] * self.BINS ** 2 + q[:, 1] * self.BINS + q[:, 2]
        hist = np.bincount(idx, minlength=self.DIM).astype(np.float32)
        hist[int(hist.argmax())] = 0.0               # drop the background
        if hist.sum() < 0.002 * len(idx):             # nothing but background (and noise)
            hist = np.zeros(self.DIM, np.float32)
            hist[-2] = 1.0
        return _unit(hist)

    def embed_text(self, text: str) -> np.ndarray:
        if text in self.concepts:
            return self.concepts[text]
        v = np.zeros(self.DIM, np.float32)
        v[-1] = 1.0
        return v


def maybe_embedder(model_dir: str = DEFAULT_MODEL_DIR):
    """The real embedder if its weights were downloaded at setup, else None (cascade then skips T1)."""
    return ClipOnnxEmbedder(model_dir) if models_present(model_dir) else None
