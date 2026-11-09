import hashlib
import io

import cv2
import numpy as np
import pytest

from standing_orders.engine import embedder as emb
from standing_orders.engine.embedder import ClipOnnxEmbedder, FakeEmbedder, download_models, models_present
from tests.helpers import scene

needs_models = pytest.mark.skipif(not models_present(emb.DEFAULT_MODEL_DIR), reason="CLIP models not downloaded")


def test_fake_embedder_separates_box_colours_and_empty():
    e = FakeEmbedder()
    white, red = scene(True), scene(True).copy()
    red[150:190, 140:180] = (0, 0, 255)
    empty = scene(False)
    for v in (e.embed_image(white), e.embed_image(red), e.embed_image(empty)):
        assert v.dtype == np.float32 and abs(np.linalg.norm(v) - 1) < 1e-5
    assert e.embed_image(white) @ e.embed_image(white) > 0.999
    assert e.embed_image(white) @ e.embed_image(red) < 0.1
    assert e.embed_image(white) @ e.embed_image(empty) < 0.1
    assert e.embed_image(empty) @ e.embed_image(scene(False, brightness=60)) > 0.999   # lighting-proof empties


def test_fake_embedder_text_follows_concepts():
    e = FakeEmbedder({"a mug": scene(True)})
    assert e.embed_text("a mug") @ e.embed_image(scene(True)) > 0.999
    assert e.embed_text("anything else") @ e.embed_image(scene(True)) == 0


def test_download_verifies_checksum_and_never_keeps_bad_file(tmp_path, monkeypatch):
    bodies = {}

    def fake_open(url, timeout=0):
        rel = url.split("/resolve/")[1].split("/", 1)[1]
        assert emb.MODEL_REVISION in url and url.startswith("https://huggingface.co/Xenova/")
        return io.BytesIO(bodies.get(rel, b"not the real weights"))

    monkeypatch.setattr(emb.urllib.request, "urlopen", fake_open)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        download_models(str(tmp_path))
    assert not list(tmp_path.rglob("*.onnx")) and not list(tmp_path.rglob("*.part"))
    good = b"weights"
    monkeypatch.setitem(emb.MODEL_FILES, "onnx/vision_model_quantized.onnx", hashlib.sha256(good).hexdigest())
    monkeypatch.setitem(emb.MODEL_FILES, "onnx/text_model_quantized.onnx", hashlib.sha256(good).hexdigest())
    bodies.update({"onnx/vision_model_quantized.onnx": good, "onnx/text_model_quantized.onnx": good})
    download_models(str(tmp_path))
    assert models_present(str(tmp_path))
    download_models(str(tmp_path))   # already present and verified: no refetch


def test_missing_models_raise_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_models"):
        ClipOnnxEmbedder(str(tmp_path))


@needs_models
def test_real_clip_sanity():
    e = ClipOnnxEmbedder(emb.DEFAULT_MODEL_DIR)
    circle = np.full((240, 320, 3), 255, np.uint8)
    cv2.circle(circle, (160, 120), 80, (0, 0, 255), -1)
    square = np.full((240, 320, 3), 255, np.uint8)
    cv2.rectangle(square, (80, 40), (240, 200), (255, 0, 0), -1)
    c, s = e.embed_image(circle), e.embed_image(square)
    assert c.shape == (512,) and abs(np.linalg.norm(c) - 1) < 1e-4
    red_circle, blue_square = e.embed_text("a red circle"), e.embed_text("a blue square")
    assert c @ red_circle > c @ blue_square
    assert s @ blue_square > s @ red_circle
