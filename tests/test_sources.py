import cv2

from standing_orders.engine.types import Frame
from standing_orders.sources.avf_source import AvfSource, CameraError
from standing_orders.sources.file_source import FileSource
from standing_orders.storage.clips import export_clip
from standing_orders.storage.ringbuffer import RingBuffer
from tests.helpers import W, H, scene, write_video


def test_file_source_samples_at_requested_rate(tmp_path):
    path = write_video(str(tmp_path / "v.mp4"), [({"mug": False}, 3), ({"mug": True}, 3)])
    src = FileSource(path, sample_fps=1.0)
    src.start()
    frames = []
    while (f := src.read()) is not None:
        frames.append(f)
    src.stop()
    assert [round(f.timestamp) for f in frames] == [0, 1, 2, 3, 4, 5]
    assert frames[0].image.shape == (H, W, 3)
    assert frames[0].image.mean() < frames[4].image.mean()  # mug appears in the second half


def test_file_source_missing_file_raises(tmp_path):
    src = FileSource(str(tmp_path / "nope.mp4"))
    try:
        src.start()
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


def test_avf_source_reports_missing_permission_instead_of_hanging():
    t = [0.0]
    src = AvfSource(index=99, warmup_seconds=0.2, time_fn=lambda: t[0],
                    sleep_fn=lambda s: t.__setitem__(0, t[0] + s))
    try:
        src.start()
    except CameraError as e:
        assert "Privacy & Security" in str(e)
        return
    raise AssertionError("expected CameraError")


def test_ringbuffer_prunes_by_age(tmp_path):
    rb = RingBuffer(str(tmp_path / "rb"), retention_seconds=10)
    for t in range(0, 30, 2):
        rb.add(Frame(scene(False), float(t), "cam0"))
    stamps = [ts for ts, _ in rb.window(0, 1e9)]
    assert min(stamps) >= 28 - 10 and max(stamps) == 28


def test_ringbuffer_prunes_by_size_and_downscales(tmp_path):
    big = Frame(scene(True), 0.0, "cam0")
    rb = RingBuffer(str(tmp_path / "rb"), max_width=100)
    path = rb.add(big)
    assert cv2.imread(str(path)).shape[1] == 100
    one = path.stat().st_size
    rb2 = RingBuffer(str(tmp_path / "rb2"), max_bytes=one * 3 + 1, max_width=100)
    for t in range(10):
        rb2.add(Frame(scene(True), float(t), "cam0"))
    assert len(rb2.window(0, 99)) == 3 and rb2.window(0, 99)[-1][0] == 9
    rb2.purge()
    assert rb2.size_bytes() == 0


def test_export_clip_roundtrip(tmp_path):
    rb = RingBuffer(str(tmp_path / "rb"))
    for t in range(6):
        rb.add(Frame(scene(t >= 3), float(t), "cam0"))
    out = export_clip(rb.window(1, 5), str(tmp_path / "clips" / "e.mp4"))
    cap = cv2.VideoCapture(out)
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 5


def test_avf_source_read_accepts_dark_frames_after_warmup():
    import numpy as np
    from standing_orders.sources.avf_source import AvfSource

    class Cap:
        def __init__(self): self.n = 0
        def read(self):
            self.n += 1
            return True, np.full((4, 4, 3), 50 if self.n == 1 else 0, np.uint8)  # lit warmup, then lights off
        def release(self): pass

    src = AvfSource(time_fn=lambda: 0.0, sleep_fn=lambda s: None)
    src._cap = Cap()
    assert src._grab(1.0, min_mean=1.0) is not None   # warmup consumed the lit frame
    assert src.read() is not None                      # dark room: still a frame, loop continues
