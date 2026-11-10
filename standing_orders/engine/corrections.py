"""Corrections: save the crop as an example, refit thresholds, replay labeled events, accept only if
the target event flips and no previously correct event breaks.

How a correction takes effect: the crop becomes a saved example for the predicate (polarity 1 =
"this is a real one", 0 = "this is not"). The Cascade's T1 treats a crop nearly identical to a saved
example (cosine >= example_sim) as decided by that example, ahead of the prompt margins, so one
correction fixes that exact look without retuning the prompt thresholds for everything else."""

from __future__ import annotations
import inspect
import json
import pathlib
import uuid
from dataclasses import dataclass, field

import cv2
import numpy as np

from .cascade import Cascade
from .clock import FakeClock
from .program import WatchProgram, parse_program
from .runner import ListSink, Runner, StaticStore
from .types import Frame

REPLAY_MAX_FRAMES = 120


@dataclass
class CorrectionResult:
    accepted: bool
    flipped: bool                                          # target event no longer fires
    broken: list[str] = field(default_factory=list)        # ids of previously-correct events that broke
    message: str = ""


def load_examples(conn, order_id: str) -> dict[str, list[tuple[int, np.ndarray]]]:
    """Saved examples of an order as {predicate: [(polarity, unit vector)]}, for Cascade(examples=...)."""
    out: dict[str, list[tuple[int, np.ndarray]]] = {}
    for predicate, polarity, blob in conn.execute(
            "SELECT predicate, polarity, embedding FROM examples WHERE order_id=?", (order_id,)):
        out.setdefault(predicate, []).append((int(polarity), np.frombuffer(blob, np.float32).copy()))
    return out


class _ClipSource:
    """Frames of a clip with timestamps spread evenly over `seconds`. Event clips are time-lapses
    of the recording window, so the true spacing is not in the file; the window length is."""

    def __init__(self, images: list[np.ndarray], seconds: float):
        n = len(images)
        step = max(1, -(-n // REPLAY_MAX_FRAMES))
        dt = seconds / max(n, 1)
        self._frames = [Frame(images[i], i * dt, "replay") for i in range(0, n, step)]
        self._i = 0

    def read(self) -> Frame | None:
        if self._i >= len(self._frames):
            return None
        self._i += 1
        return self._frames[self._i - 1]


def _read_clip(path: str | None) -> list[np.ndarray]:
    if not path or not pathlib.Path(path).is_file():
        raise FileNotFoundError(f"clip not available: {path}")
    cap = cv2.VideoCapture(path)
    images = []
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            images.append(image)
    finally:
        cap.release()
    if not images:
        raise ValueError(f"clip has no readable frames: {path}")
    return images


def _make_verifier(verifier_factory, clock):
    try:
        takes_clock = len(inspect.signature(verifier_factory).parameters) >= 1
    except (TypeError, ValueError):
        takes_clock = False
    return verifier_factory(clock) if takes_clock else verifier_factory()


def _fires(program: WatchProgram, images: list[np.ndarray], examples, embedder, verifier_factory,
           clip_seconds: float) -> bool:
    """Would this program fire on this clip? Same Runner, FakeClock, no budget: determinism comes from
    the verifier (a cached or scripted one), not from limits."""
    clock = FakeClock()
    verifier = _make_verifier(verifier_factory, clock)
    sink = ListSink()
    runner = Runner(_ClipSource(images, clip_seconds), StaticStore([program]),
                    lambda p: Cascade(p, verifier, clock, embedder=embedder, examples=examples),
                    clock, sink, kind="backtest")
    try:
        runner.run()
    finally:
        runner.close()
    return bool(sink.events)


def _best_candidate(program: WatchProgram, frame: Frame, embedder) -> str:
    """The predicate whose prompt the last frame matches best against its negatives."""
    probe = Cascade(program, None, FakeClock(), embedder=embedder)
    best, best_margin = next(iter(program.predicates)), -2.0
    for name, spec in program.predicates.items():
        vec = embedder.embed_image(probe.crop_image(frame, spec.zone))
        margin = float(vec @ embedder.embed_text(spec.prompt)) - max(
            (float(vec @ embedder.embed_text(t)) for t in spec.negatives), default=0.0)
        if margin > best_margin:
            best, best_margin = name, margin
    return best


def apply_correction(conn, order_id: str, event_id: str, label: str, predicate: str | None,
                     note: str, *, embedder, verifier_factory, clip_dir: str,
                     clip_seconds: float = 20.0) -> CorrectionResult:
    """label: "correct" | "wrong". For "wrong", `predicate` is the predicate that was falsely TRUE
    (None: pick the best candidate). Writes examples/label only if accepted; else leaves state unchanged
    and returns the conflict.

    verifier_factory: clock -> Verifier (or no-arg). It must be deterministic for the same pixels
    (a CachedVerifier over a stored run, or a scripted one): replays are only evidence if they repeat.
    clip_seconds: length of the recording window an event clip covers (the Runner's clip_seconds);
    replay timestamps are spread over it. Example crops are written under clip_dir/examples/.
    "correct" without a predicate only records the label."""
    if label not in ("correct", "wrong"):
        return CorrectionResult(False, False, [], f"label must be 'correct' or 'wrong', got {label!r}")
    row = conn.execute("SELECT program_json FROM orders WHERE id=?", (order_id,)).fetchone()
    event = conn.execute("SELECT id, clip_path FROM events WHERE id=? AND order_id=?", (event_id, order_id)).fetchone()
    if row is None or event is None:
        return CorrectionResult(False, False, [], "unknown order or event")
    program = parse_program(json.loads(row[0]))
    if predicate is not None and predicate not in program.predicates:
        return CorrectionResult(False, False, [], f"unknown predicate {predicate!r}")

    def finish(accepted: bool, flipped: bool, broken: list[str], message: str, new_examples=()) -> CorrectionResult:
        if accepted:
            with conn:
                for ex in new_examples:
                    conn.execute("INSERT INTO examples (id, order_id, predicate, polarity, embedding, crop_path, "
                                 "source_event) VALUES (?,?,?,?,?,?,?)", ex)
                conn.execute("UPDATE events SET label=?, label_note=? WHERE id=?", (label, note, event_id))
        return CorrectionResult(accepted, flipped, broken, message)

    if label == "correct" and predicate is None:
        return finish(True, False, [], "recorded; no example saved (name the predicate to teach one)")

    try:
        target_images = _read_clip(event[1])
    except (OSError, ValueError) as exc:
        return CorrectionResult(False, False, [], f"cannot verify a correction without the clip: {exc}")
    last = Frame(target_images[-1], 0.0, "replay")
    predicate = predicate or _best_candidate(program, last, embedder)
    spec = program.predicates[predicate]
    crop = Cascade(program, None, FakeClock(), embedder=embedder).crop_image(last, spec.zone)
    ex_id = f"ex_{uuid.uuid4().hex[:12]}"
    polarity = 1 if label == "correct" else 0
    vec = embedder.embed_image(crop).astype(np.float32)
    crop_path = pathlib.Path(clip_dir) / "examples" / f"{ex_id}.jpg"
    candidate = {k: list(v) for k, v in load_examples(conn, order_id).items()}
    candidate.setdefault(predicate, []).append((polarity, vec))

    target_fires = _fires(program, target_images, candidate, embedder, verifier_factory, clip_seconds)
    flipped = (not target_fires) if label == "wrong" else target_fires
    if not flipped:
        return CorrectionResult(False, False, [], "conflict: the correction would not change this event "
                                f"({'still fires' if label == 'wrong' else 'would stop firing'}); nothing saved")
    broken, unchecked = [], []
    for other_id, clip in conn.execute(
            "SELECT id, clip_path FROM events WHERE order_id=? AND label='correct' AND id<>?", (order_id, event_id)):
        try:
            images = _read_clip(clip)
        except (OSError, ValueError):
            unchecked.append(other_id)
            continue
        if not _fires(program, images, candidate, embedder, verifier_factory, clip_seconds):
            broken.append(other_id)
    if broken:
        return CorrectionResult(False, True, broken,
                                f"conflict: this would stop {len(broken)} event(s) you marked correct; nothing saved")
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(crop_path), crop)
    message = f"accepted; saved {'positive' if polarity else 'negative'} example for {predicate!r}"
    if unchecked:
        message += f" ({len(unchecked)} labeled event(s) had no clip and could not be rechecked)"
    return finish(True, True, [], message,
                  [(ex_id, order_id, predicate, polarity, vec.tobytes(), str(crop_path), event_id)])
