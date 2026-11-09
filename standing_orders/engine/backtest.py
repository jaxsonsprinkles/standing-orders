"""Backtest: replay recorded footage through the same Runner with a FakeClock."""
from dataclasses import dataclass, field

from .program import WatchProgram
from .types import Event


@dataclass
class BacktestResult:
    events: list[Event] = field(default_factory=list)   # would-have-fired, kind="backtest", with clips
    frames_checked: int = 0
    t2_calls: int = 0
    unresolved_frames: int = 0   # frames whose predicates stayed UNKNOWN because the T2 cap ran out


def backtest(program: WatchProgram, source, verifier, *, embedder=None, max_t2_calls: int = 10,
             clip_dir: str | None = None) -> BacktestResult:
    """source: a FrameSource over recent footage (e.g. a ringbuffer-backed source).
    Never silently treats unresolved frames as FALSE: they count in unresolved_frames."""
    raise NotImplementedError
