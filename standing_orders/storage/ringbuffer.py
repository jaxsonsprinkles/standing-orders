"""Rolling local recording: one downscaled JPEG per sampled frame, named by timestamp.

Pruned by age (retention) and by total size (disk cap), oldest first. Age uses frame timestamps,
not the wall clock, so it behaves the same in replay. Plain files: the pane and tools read it
from other processes without coordination."""
import pathlib

import cv2

from ..engine.types import Frame


class RingBuffer:
    def __init__(self, directory: str, retention_seconds: float = 3600, max_bytes: int = 2 << 30,
                 max_width: int = 960, jpeg_quality: int = 80):
        self.dir = pathlib.Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.retention_seconds, self.max_bytes = retention_seconds, max_bytes
        self.max_width, self.quality = max_width, jpeg_quality

    @staticmethod
    def _ts(path: pathlib.Path) -> float:
        return float(path.stem)

    def _files(self) -> list[pathlib.Path]:
        return sorted(self.dir.glob("*.jpg"))

    def add(self, frame: Frame) -> pathlib.Path:
        image = frame.image
        h, w = image.shape[:2]
        if w > self.max_width:
            image = cv2.resize(image, (self.max_width, round(h * self.max_width / w)), interpolation=cv2.INTER_AREA)
        path = self.dir / f"{frame.timestamp:016.3f}.jpg"
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            raise OSError(f"could not encode frame at {frame.timestamp}")
        path.write_bytes(buf.tobytes())
        self.prune(frame.timestamp)
        return path

    def prune(self, now: float) -> None:
        files = self._files()
        keep = [p for p in files if self._ts(p) >= now - self.retention_seconds]
        for p in files[: len(files) - len(keep)]:
            p.unlink(missing_ok=True)
        sizes = [p.stat().st_size for p in keep]
        total = sum(sizes)
        for p, size in zip(keep, sizes):
            if total <= self.max_bytes:
                break
            p.unlink(missing_ok=True)
            total -= size

    def window(self, start: float, end: float) -> list[tuple[float, pathlib.Path]]:
        """Recorded frames with start <= timestamp <= end, oldest first."""
        return [(self._ts(p), p) for p in self._files() if start <= self._ts(p) <= end]

    def purge(self) -> None:
        for p in self._files():
            p.unlink(missing_ok=True)

    def size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self._files())


class RingSource:
    """FrameSource over a recorded window of a RingBuffer, for backtests."""

    def __init__(self, ring: RingBuffer, start: float, end: float, source_id: str = "ring"):
        self.ring, self.start_t, self.end_t, self.source_id = ring, start, end, source_id
        self._frames: list[tuple[float, pathlib.Path]] = []

    def start(self) -> None:
        self._frames = self.ring.window(self.start_t, self.end_t)

    def read(self) -> Frame | None:
        while self._frames:
            t, path = self._frames.pop(0)
            image = cv2.imread(str(path))
            if image is not None:  # pruned or corrupt files are skipped
                return Frame(image, t, self.source_id)
        return None

    def stop(self) -> None:
        self._frames = []
