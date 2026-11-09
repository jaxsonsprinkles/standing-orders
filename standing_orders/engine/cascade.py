"""Per-order detection cascade.

T0 change gate -> (T1 local embedding check) -> T2 verifier, on the regions that changed. Unchanged
regions keep their last value. Failure policy: a raising verifier or an exhausted vision budget makes
the affected predicates UNKNOWN. UNKNOWN never fires, and nothing here ever raises into the Runner."""
from dataclasses import dataclass

import cv2
import numpy as np

from .change_gate import ChangeGate
from .clock import Clock
from .program import FRAME_ZONE, PredicateSpec, WatchProgram
from .types import Frame, Tri, Verdict

RETRY_SECONDS = 5.0   # a predicate left UNKNOWN by a failure is looked at again after this long
# Image-text cosine margins in CLIP are small (a clear match beats a clear mismatch by ~0.05-0.1),
# so the defaults are tight; tune per camera. Image-image similarity to a saved example is on a
# different scale (same scene ~0.9+), hence its own threshold.
T1_TRUE_MARGIN = 0.04
T1_FALSE_MARGIN = 0.04
EXAMPLE_SIM = 0.92

Examples = dict  # predicate name -> [(polarity 1|0, unit embedding)]


@dataclass
class CascadeStats:
    frames: int = 0
    t0_passed: int = 0          # predicate checks the change gate let through
    t1_decided: int = 0         # predicate checks answered locally (no verifier call)
    t2_calls: int = 0           # verifier calls, Confirm included (one call may check several predicates)
    fires: int = 0              # events emitted (the Runner increments)
    budget_exhausted: bool = False   # the latest vision call was refused by the budget
    verifier_errors: int = 0
    unresolved_frames: int = 0  # frames that ended with a predicate left UNKNOWN by failure/budget


def _unknown(names, reason: str) -> dict[str, Verdict]:
    return {n: Verdict(Tri.UNKNOWN, 0.0, reason) for n in names}


class Cascade:
    def __init__(self, program: WatchProgram, verifier, clock: Clock, t0_threshold: float | None = None,
                 *, embedder=None, budget=None, heartbeat_seconds: float = 60.0,
                 t1_true_margin: float = T1_TRUE_MARGIN, t1_false_margin: float = T1_FALSE_MARGIN,
                 example_sim: float = EXAMPLE_SIM, examples: Examples | None = None):
        """embedder: enables T1 (see _t1). examples: saved corrections, {predicate: [(polarity, vec)]}
        (see corrections.load_examples). Without an embedder every changed zone goes to the verifier."""
        self.program, self.verifier, self.clock = program, verifier, clock
        self.embedder, self.budget, self.heartbeat_seconds = embedder, budget, heartbeat_seconds
        self.t1_true_margin, self.t1_false_margin = t1_true_margin, t1_false_margin
        self.example_sim, self.examples = example_sim, examples or {}
        self._text: dict[str, tuple[np.ndarray, list[np.ndarray]]] = {}
        regions = {spec.zone: program.zones.get(spec.zone, []) for spec in program.predicates.values()}
        self.gate = ChangeGate(regions, **({} if t0_threshold is None else {"threshold": t0_threshold}))
        self.verdicts: dict[str, Verdict] = {n: Verdict(Tri.UNKNOWN, 0.0, "not checked yet") for n in program.predicates}
        self.stats = CascadeStats()
        self._pending: set[str] = set()          # UNKNOWN because a call failed or was refused
        self._last_attempt: dict[str, float] = {}

    def crop_image(self, frame: Frame, zone: str) -> np.ndarray:
        image = frame.image
        if zone != FRAME_ZONE:
            h, w = image.shape[:2]
            xs, ys = zip(*self.program.zones[zone])
            x0, x1 = int(min(xs) * w), max(int(max(xs) * w), int(min(xs) * w) + 1)
            y0, y1 = int(min(ys) * h), max(int(max(ys) * h), int(min(ys) * h) + 1)
            image = image[y0:y1, x0:x1]
        return image

    def _crop(self, frame: Frame, zone: str) -> bytes:
        image = self.crop_image(frame, zone)
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise ValueError(f"could not encode crop of zone {zone!r}")
        return buf.tobytes()

    def _t1(self, name: str, spec: PredicateSpec, vec: np.ndarray) -> Verdict | None:
        """Local decision from the crop embedding, or None when uncertain (the verifier decides).
        1. a saved example that is nearly identical to the crop decides (corrections win);
        2. else prompt similarity versus the best negative prompt, with a margin on both sides."""
        pos = max((float(vec @ e) for pol, e in self.examples.get(name, []) if pol), default=-1.0)
        neg = max((float(vec @ e) for pol, e in self.examples.get(name, []) if not pol), default=-1.0)
        if max(pos, neg) >= self.example_sim:
            return Verdict(Tri.TRUE if pos > neg else Tri.FALSE, max(pos, neg),
                           f"t1: matches a saved {'positive' if pos > neg else 'negative'} example")
        if name not in self._text:
            self._text[name] = (self.embedder.embed_text(spec.prompt),
                                [self.embedder.embed_text(t) for t in spec.negatives])
        prompt, negatives = self._text[name]
        if not negatives:
            return None                      # nothing to compare against
        margin = float(vec @ prompt) - max(float(vec @ n) for n in negatives)
        conf = min(1.0, 0.5 + abs(margin) * 5)
        if margin >= self.t1_true_margin:
            return Verdict(Tri.TRUE, conf, f"t1: matches '{spec.prompt}' (margin {margin:+.3f})")
        if margin <= -self.t1_false_margin:
            return Verdict(Tri.FALSE, conf, f"t1: closer to a negative than '{spec.prompt}' (margin {margin:+.3f})")
        return None

    def _spend(self, n: int = 1) -> bool:
        if self.budget is None:
            return True
        ok = self.budget.try_spend(n)
        self.stats.budget_exhausted = not ok
        return ok

    async def _t2(self, frame: Frame, todo: dict[str, PredicateSpec]) -> None:
        """Ask the verifier about `todo`. Any failure leaves those predicates UNKNOWN and pending."""
        now = self.clock.now()
        for n in todo:
            self._last_attempt[n] = now
        if not self._spend():
            self.verdicts.update(_unknown(todo, "vision budget exhausted"))
            self._pending.update(todo)
            return
        self.stats.t2_calls += 1
        try:
            crops = [self._crop(frame, s.zone) for s in todo.values()]
            got = await self.verifier.check(crops, todo)
            out = {n: got.get(n) for n in todo}
            if any(not isinstance(v, Verdict) for v in out.values()):
                raise TypeError("verifier returned no verdict for some predicate")
        except Exception as exc:  # a vision failure must never take the engine down
            self.stats.verifier_errors += 1
            self.verdicts.update(_unknown(todo, f"verifier error: {type(exc).__name__}"))
            self._pending.update(todo)
            return
        self.verdicts.update(out)
        self._pending.difference_update(todo)

    def _run_t1(self, frame: Frame, todo: dict[str, PredicateSpec], now: float) -> dict[str, PredicateSpec]:
        """Decide what T1 can; return the predicates that still need the verifier (uncertain, T1 error,
        or due a heartbeat look: T1 never gets to say FALSE for longer than heartbeat_seconds)."""
        vecs: dict[str, np.ndarray] = {}
        rest = {}
        for n, spec in todo.items():
            self._last_attempt.setdefault(n, now)
            try:
                if spec.zone not in vecs:
                    vecs[spec.zone] = self.embedder.embed_image(self.crop_image(frame, spec.zone))
                verdict = self._t1(n, spec, vecs[spec.zone])
            except Exception:  # a broken embedder degrades to T2, never to a firing
                verdict = None
            if verdict is None or now - self._last_attempt[n] >= self.heartbeat_seconds:
                rest[n] = spec
                continue
            self.verdicts[n] = verdict
            self._pending.discard(n)
            self.stats.t1_decided += 1
        return rest

    async def update(self, frame: Frame) -> dict[str, Tri]:
        """Return the current value of every predicate after this frame."""
        self.stats.frames += 1
        now = self.clock.now()
        changed = self.gate.update(frame.image)
        todo = {n: s for n, s in self.program.predicates.items() if changed[s.zone]}
        self.stats.t0_passed += len(todo)
        for n in self._pending:  # retry what a failure left UNKNOWN, but not on every frame
            if n not in todo and now - self._last_attempt.get(n, float("-inf")) >= RETRY_SECONDS:
                todo[n] = self.program.predicates[n]
        if todo and self.embedder is not None:
            todo = self._run_t1(frame, todo, now)
        if todo:
            await self._t2(frame, todo)
        if self._pending:
            self.stats.unresolved_frames += 1
        return {n: v.value for n, v in self.verdicts.items()}
