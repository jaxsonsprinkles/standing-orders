"""Hourly vision-call budget (T2 and Confirm). Sliding 3600 s window on the injected clock."""
from .clock import Clock


class VisionBudget:
    def __init__(self, per_hour: int, clock: Clock):
        raise NotImplementedError

    def try_spend(self, n: int = 1) -> bool:
        """Spend n calls if the window has room. False (and spend nothing) when exhausted."""
        raise NotImplementedError

    def remaining(self) -> int:
        raise NotImplementedError

    @property
    def exhausted(self) -> bool:
        raise NotImplementedError
