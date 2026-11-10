"""Hourly vision-call budget (T2 and Confirm). Sliding 3600 s window on the injected clock."""

from __future__ import annotations
from collections import deque

from .clock import Clock

WINDOW_SECONDS = 3600.0


class VisionBudget:
    def __init__(self, per_hour: int, clock: Clock):
        self.per_hour, self.clock = max(0, int(per_hour)), clock
        self._spent: deque[float] = deque()   # one entry per call, oldest first

    def _prune(self) -> None:
        cutoff = self.clock.now() - WINDOW_SECONDS
        while self._spent and self._spent[0] <= cutoff:
            self._spent.popleft()

    def try_spend(self, n: int = 1) -> bool:
        """Spend n calls if the window has room. False (and spend nothing) when exhausted."""
        self._prune()
        if n < 0 or len(self._spent) + n > self.per_hour:
            return False
        now = self.clock.now()
        self._spent.extend([now] * n)
        return True

    def remaining(self) -> int:
        self._prune()
        return max(0, self.per_hour - len(self._spent))

    @property
    def exhausted(self) -> bool:
        return self.remaining() == 0


class CallCap:
    """Fixed total number of calls, no window. Same interface as VisionBudget (used by backtest)."""

    def __init__(self, total: int):
        self.total, self._used = max(0, int(total)), 0

    def try_spend(self, n: int = 1) -> bool:
        if n < 0 or self._used + n > self.total:
            return False
        self._used += n
        return True

    def remaining(self) -> int:
        return self.total - self._used

    @property
    def exhausted(self) -> bool:
        return self.remaining() == 0
