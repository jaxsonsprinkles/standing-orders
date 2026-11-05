"""Frame loop. step() is deterministic given its inputs, so replay equals live."""
from .types import Event


class Runner:
    def __init__(self, source, program_store, cascade, clock, sink):
        self.source, self.program_store, self.cascade = source, program_store, cascade
        self.clock, self.sink = clock, sink

    def step(self) -> list[Event]:
        """Read one frame, update predicates, evaluate rules, confirm candidates, emit events. Day 2."""
        raise NotImplementedError
