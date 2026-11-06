"""macOS webcam via OpenCV + AVFoundation, sampled at sample_fps with wall-clock timestamps.

macOS ties camera permission to the app that launched the process (docs/day1-findings.md), so a
gateway under launchd gets no frames. start() reports that cleanly instead of hanging."""
import time

import cv2

from ..engine.types import Frame


class CameraError(RuntimeError):
    pass


class AvfSource:
    def __init__(self, index: int = 0, sample_fps: float = 1.0, source_id: str = "cam0",
                 warmup_seconds: float = 5.0, time_fn=time.time, sleep_fn=time.sleep):
        self.index, self.sample_fps, self.source_id = index, sample_fps, source_id
        self.warmup_seconds, self._time, self._sleep = warmup_seconds, time_fn, sleep_fn
        self._cap = None
        self._next_at = 0.0

    def _grab(self, deadline: float, min_mean: float = -1.0):
        while self._time() < deadline:
            ok, image = self._cap.read()
            if ok and image is not None and image.mean() > min_mean:
                return image
            self._sleep(0.1)
        return None

    def start(self) -> None:
        # warmup: the first frames are often black while exposure settles, so wait for a lit one
        self._cap = cv2.VideoCapture(self.index, cv2.CAP_AVFOUNDATION)
        if not self._cap.isOpened() or self._grab(self._time() + self.warmup_seconds, min_mean=1.0) is None:
            self.stop()
            raise CameraError(
                f"no frame from camera {self.index}: grant camera access to the app that launched "
                "this process (System Settings > Privacy & Security > Camera), or run the gateway "
                "from an authorized Terminal")

    def read(self) -> Frame | None:
        """Wait until the next sample is due, then return it. None if the camera stopped delivering."""
        wait = self._next_at - self._time()
        if wait > 0:
            self._sleep(wait)
        image = self._grab(self._time() + 2.0)
        if image is None:
            return None
        now = self._time()
        self._next_at = now + 1.0 / self.sample_fps
        return Frame(image, now, self.source_id)

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
