"""Recorded video as a FrameSource. Timestamps are video time, so replay is deterministic."""
import cv2

from ..engine.types import Frame


class FileSource:
    def __init__(self, path: str, sample_fps: float = 1.0, source_id: str = "file"):
        self.path, self.sample_fps, self.source_id = path, sample_fps, source_id
        self._cap = None
        self._next_t = 0.0

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {self.path}")
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._index, self._next_t = 0, 0.0

    def read(self) -> Frame | None:
        """Next sampled frame, or None at the end of the video."""
        cap = self._cap
        while True:
            t = self._index / self._fps
            self._index += 1
            if t + 1e-9 < self._next_t:
                if not cap.grab():  # skip without decoding
                    return None
                continue
            ok, image = cap.read()
            if not ok:
                return None
            self._next_t = t + 1.0 / self.sample_fps
            return Frame(image, t, self.source_id)

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
