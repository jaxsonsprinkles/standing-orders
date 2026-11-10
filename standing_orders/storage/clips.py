"""Event clips: a window of recorded frames written as a small MP4.

Clips stay small because delivery can be slow (docs/day1-findings.md): frames are already downscaled
by the RingBuffer, and playback is a time-lapse at `fps` frames per second."""

from __future__ import annotations
import pathlib

import cv2


def export_clip(frames: list[tuple[float, pathlib.Path]], out_path: str, fps: float = 4.0) -> str:
    """Write frames (as returned by RingBuffer.window) to out_path. Returns out_path."""
    if not frames:
        raise ValueError("no frames to export")
    first = cv2.imread(str(frames[0][1]))
    if first is None:
        raise ValueError(f"unreadable frame: {frames[0][1]}")
    h, w = first.shape[:2]
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    for codec in ("avc1", "mp4v"):  # avc1 (H.264) plays inline in chats; mp4v is the fallback
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*codec), fps, (w, h))
        if writer.isOpened():
            break
        writer.release()
    else:
        raise OSError("no usable video codec")
    try:
        for _, path in frames:
            image = cv2.imread(str(path))
            if image is None or image.shape[:2] != (h, w):
                continue  # a half-written or differently sized frame must not abort the clip
            writer.write(image)
    finally:
        writer.release()
    return out_path
