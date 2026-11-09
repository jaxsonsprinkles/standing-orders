"""Orders and events in SQLite (the bus between engine, tools and pane)."""
from ..engine.program import WatchProgram
from ..engine.types import Event


class OrderRepo:
    def __init__(self, conn):
        self.conn = conn

    def save_draft(self, program: WatchProgram) -> None: raise NotImplementedError
    def get(self, order_id: str) -> tuple[WatchProgram, str] | None: raise NotImplementedError  # (program, status)
    def list(self) -> list[tuple[WatchProgram, str]]: raise NotImplementedError
    def set_status(self, order_id: str, status: str) -> None: raise NotImplementedError  # draft|armed|paused
    def armed(self) -> list[WatchProgram]: raise NotImplementedError  # satisfies runner.ProgramStore
    def add_event(self, event: Event) -> None: raise NotImplementedError  # also usable as a Runner sink: emit
    def emit(self, event: Event) -> None: raise NotImplementedError
    def events(self, order_id: str) -> list[Event]: raise NotImplementedError
    def add_counters(self, bucket_start: float, order_id: str, **deltas: int) -> None: raise NotImplementedError
