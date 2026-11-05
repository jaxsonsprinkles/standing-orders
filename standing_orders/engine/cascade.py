"""T0 change gate, T1 local check, T2 vision check, per predicate. Day 3."""
from .program import WatchProgram
from .types import Frame, Tri


class Cascade:
    def __init__(self, program: WatchProgram, verifier, clock):
        self.program, self.verifier, self.clock = program, verifier, clock

    async def update(self, frame: Frame) -> dict[str, Tri]:
        """Return the current value of every predicate after this frame. Day 3."""
        raise NotImplementedError
