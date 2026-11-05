import time
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...


class RealClock:
    def now(self) -> float:
        return time.time()


class FakeClock:
    """Injected clock: live and recorded video run through identical rule code."""

    def __init__(self, start: float = 0.0):
        self._t = start

    def now(self) -> float:
        return self._t

    def set(self, t: float) -> None:
        self._t = t

    def advance(self, seconds: float) -> None:
        self._t += seconds
