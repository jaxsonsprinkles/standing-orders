
from __future__ import annotations
from typing import Protocol

from ..engine.types import Frame


class FrameSource(Protocol):
    def start(self) -> None: ...
    def read(self) -> Frame | None: ...   # None at end of a recording
    def stop(self) -> None: ...
