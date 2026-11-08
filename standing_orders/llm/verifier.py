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
