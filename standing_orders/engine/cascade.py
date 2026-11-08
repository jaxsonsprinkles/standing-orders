"""Per-order detection cascade.

Today: T0 change gate -> verifier (T2) on the regions that changed. Unchanged regions keep their last
value. Day 3 adds T1 local embeddings in front of the verifier, the heartbeat, the hourly vision
budget and the Confirm step; the Cascade/Verifier interfaces stay as they are."""
from dataclasses import dataclass

import cv2

from .change_gate import ChangeGate
from .clock import Clock
from .program import FRAME_ZONE, WatchProgram
from .types import Frame, Tri, Verdict


@dataclass
class CascadeStats:
    frames: int = 0
    t0_passed: int = 0      # predicate checks the change gate let through
    t2_calls: int = 0       # verifier calls (one call may check several predicates)


class Cascade:
    def __init__(self, program: WatchProgram, verifier, clock: Clock, t0_threshold: float | None = None):
        self.program, self.verifier, self.clock = program, verifier, clock
        regions = {spec.zone: program.zones.get(spec.zone, []) for spec in program.predicates.values()}
        self.gate = ChangeGate(regions, **({} if t0_threshold is None else {"threshold": t0_threshold}))
        self.verdicts: dict[str, Verdict] = {n: Verdict(Tri.UNKNOWN, 0.0, "not checked yet") for n in program.predicates}
        self.stats = CascadeStats()

    def _crop(self, frame: Frame, zone: str) -> bytes:
        image = frame.image
        if zone != FRAME_ZONE:
            h, w = image.shape[:2]
            xs, ys = zip(*self.program.zones[zone])
            x0, x1 = int(min(xs) * w), max(int(max(xs) * w), int(min(xs) * w) + 1)
            y0, y1 = int(min(ys) * h), max(int(max(ys) * h), int(min(ys) * h) + 1)
            image = image[y0:y1, x0:x1]
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise ValueError(f"could not encode crop of zone {zone!r}")
        return buf.tobytes()

    async def update(self, frame: Frame) -> dict[str, Tri]:
        """Return the current value of every predicate after this frame."""
        self.stats.frames += 1
        changed = self.gate.update(frame.image)
        todo = {n: s for n, s in self.program.predicates.items() if changed[s.zone]}
        if todo:
            self.stats.t0_passed += len(todo)
            self.stats.t2_calls += 1
            crops = [self._crop(frame, s.zone) for s in todo.values()]
            self.verdicts.update(await self.verifier.check(crops, todo))
        return {n: v.value for n, v in self.verdicts.items()}
