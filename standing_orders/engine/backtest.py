"""Backtest: replay recorded footage through the same Runner with a FakeClock."""

from __future__ import annotations
import tempfile
from dataclasses import dataclass, field

from .budget import CallCap
from .cascade import Cascade
from .clock import FakeClock
from .program import WatchProgram
from .runner import ListSink, Runner, StaticStore
from .types import Event


@dataclass
class BacktestResult:
    events: list[Event] = field(default_factory=list)   # would-have-fired, kind="backtest", with clips
    frames_checked: int = 0
    t2_calls: int = 0
    unresolved_frames: int = 0   # frames whose predicates stayed UNKNOWN because the T2 cap ran out


def backtest(program: WatchProgram, source, verifier, *, embedder=None, max_t2_calls: int = 10,
             clip_dir: str | None = None, clock: FakeClock | None = None,
             clip_seconds: float = 20.0) -> BacktestResult:
    """source: a FrameSource over recent footage (e.g. a ringbuffer-backed source). The caller starts
    and stops it. Never silently treats unresolved frames as FALSE: they count in unresolved_frames.

    verifier: a Verifier, or a callable clock -> Verifier (for ScriptedVerifier, which reads the clock
    the backtest runs on; or pass `clock` yourself to share your own FakeClock).
    max_t2_calls caps every vision call, Confirm included; nothing is spent beyond it, so a capped run
    reports what it could not look at instead of guessing."""
    clock = clock or FakeClock()
    if not hasattr(verifier, "check") and callable(verifier):
        verifier = verifier(clock)
    sink, cascades = ListSink(), []

    def factory(p: WatchProgram) -> Cascade:
        cascades.append(Cascade(p, verifier, clock, embedder=embedder, budget=CallCap(max_t2_calls)))
        return cascades[-1]

    with tempfile.TemporaryDirectory(prefix="so-backtest-") as tmp:
        recorder = None
        if clip_dir is not None:
            from ..storage.ringbuffer import RingBuffer
            recorder = RingBuffer(tmp, retention_seconds=clip_seconds)
        runner = Runner(source, StaticStore([program]), factory, clock, sink, kind="backtest",
                        recorder=recorder, clip_dir=clip_dir, clip_seconds=clip_seconds)
        try:
            runner.run()
        finally:
            runner.close()
    stats = cascades[0].stats if cascades else None
    return BacktestResult(events=sink.events, frames_checked=stats.frames if stats else 0,
                          t2_calls=stats.t2_calls if stats else 0,
                          unresolved_frames=stats.unresolved_frames if stats else 0)
