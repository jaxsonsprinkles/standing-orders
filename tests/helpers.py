"""Synthetic videos for tests: a grey desk with an optional 'mug' (white square) in the desk zone."""
import cv2
import numpy as np

W, H = 320, 240


def scene(mug: bool, person: bool = False, brightness: int = 100) -> np.ndarray:
    img = np.full((H, W, 3), brightness, np.uint8)
    if mug:
        img[150:190, 140:180] = 255
    if person:
        img[20:200, 20:90] = (40, 40, 200)
    return img


def write_video(path: str, segments: list[tuple[dict, float]], fps: int = 10) -> str:
    """segments: [(scene kwargs, seconds), ...]"""
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    assert w.isOpened()
    for kwargs, seconds in segments:
        for _ in range(round(seconds * fps)):
            w.write(scene(**kwargs))
    w.release()
    return path
