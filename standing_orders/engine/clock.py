
from __future__ import annotations
import time
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...
    def observe(self, source_time: float) -> None:
        """Called by the Runner with each frame's timestamp. Replay clocks follow it; live clocks ignore it."""


class RealClock:
    def now(self) -> float:
        return time.time()

    def observe(self, source_time: float) -> None:
        pass


class FakeClock:
    """Injected clock: live and recorded video run through identical rule code."""

    def __init__(self, start: float = 0.0):
        self._t = start

    def now(self) -> float:
        return self._t

    def observe(self, source_time: float) -> None:
        self._t = max(self._t, source_time)

    def set(self, t: float) -> None:
        self._t = t

    def advance(self, seconds: float) -> None:
        self._t += seconds
