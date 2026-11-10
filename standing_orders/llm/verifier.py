import hashlib
import json
import time
from typing import Callable, Protocol

from ..engine.clock import Clock
from ..engine.program import PredicateSpec
from ..engine.types import Tri, Verdict


class Verifier(Protocol):
    """T2 and Confirm. Implementations: LlmVerifier (ctx.llm), CachedVerifier (replay),
    ScriptedVerifier (tests).

    crops[i] is the JPEG crop for the i-th predicate of `predicates` (dict order). Returns one
    Verdict per predicate name. A verifier that cannot decide returns Tri.UNKNOWN."""

    async def check(self, crops: list[bytes], predicates: dict[str, PredicateSpec]) -> dict[str, Verdict]: ...


Script = Callable[[str, float], Tri]   # (predicate name, clock time) -> value


def timeline(entries: dict[str, list[tuple[float, Tri]]]) -> Script:
    """A script from {predicate: [(from_time, value), ...]}. Before the first entry: UNKNOWN."""
    def script(name: str, now: float) -> Tri:
        value = Tri.UNKNOWN
        for start, v in sorted(entries.get(name, []), key=lambda e: e[0]):
            if start <= now:
                value = v
        return value
    return script


class ScriptedVerifier:
    """Answers from a script keyed by predicate and clock time, so tests and replays are exact.
    `calls` records (time, predicate names, crop sizes) for assertions."""

    def __init__(self, script: Script, clock: Clock):
        self.script, self.clock = script, clock
        self.calls: list[tuple[float, list[str], list[int]]] = []

    async def check(self, crops: list[bytes], predicates: dict[str, PredicateSpec]) -> dict[str, Verdict]:
        now = self.clock.now()
        self.calls.append((now, list(predicates), [len(c) for c in crops]))
        return {n: Verdict(self.script(n, now), 1.0, f"scripted at t={now:g}") for n in predicates}


def predicate_hash(name: str, spec: PredicateSpec) -> str:
    return hashlib.sha256(json.dumps([name, spec.prompt, list(spec.negatives)]).encode()).hexdigest()


class CachedVerifier:
    """Memoizes an inner verifier in the `verifications` table, keyed by (sha256 of the crop bytes,
    hash of predicate name + prompt + negatives). A hit costs no inner call; a miss calls the inner
    verifier once for all missed predicates and stores what it decided. UNKNOWN is never stored (it is
    a failure or a shrug, not knowledge), and inner exceptions propagate without caching anything.
    Replays and corrections run through this so the same pixels always get the same answer."""

    def __init__(self, inner, conn, now: Callable[[], float] = time.time):
        self.inner, self.conn, self._now = inner, conn, now
        self.hits = self.misses = 0

    def _get(self, frame_hash: str, pred_hash: str) -> Verdict | None:
        row = self.conn.execute("SELECT verdict_json FROM verifications WHERE frame_hash=? AND predicate_hash=?",
                                (frame_hash, pred_hash)).fetchone()
        if row is None:
            return None
        d = json.loads(row[0])
        return Verdict(Tri(d["value"]), d["confidence"], d.get("reason", ""))

    async def check(self, crops: list[bytes], predicates: dict[str, PredicateSpec]) -> dict[str, Verdict]:
        names = list(predicates)
        keys = {n: (hashlib.sha256(c).hexdigest(), predicate_hash(n, predicates[n])) for n, c in zip(names, crops)}
        out: dict[str, Verdict] = {}
        for n in names:
            hit = self._get(*keys[n])
            if hit is not None:
                out[n] = hit
        self.hits += len(out)
        missed = [n for n in names if n not in out]
        if missed:
            self.misses += len(missed)
            got = await self.inner.check([crops[names.index(n)] for n in missed], {n: predicates[n] for n in missed})
            for n in missed:
                v = got[n]
                out[n] = v
                if v.value is not Tri.UNKNOWN:
                    self.conn.execute("INSERT OR REPLACE INTO verifications VALUES (?,?,?,?)",
                                      (*keys[n], json.dumps({"value": v.value.value, "confidence": v.confidence,
                                                             "reason": v.reason}), self._now()))
            self.conn.commit()
        return {n: out[n] for n in names}
