"""Shared value types. No Hermes imports anywhere under engine/."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Tri(Enum):
    """Predicate value. UNKNOWN never causes a firing."""
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Frame:
    image: Any          # numpy BGR array
    timestamp: float    # seconds, from the source (wall clock live, video time in replay)
    source_id: str


@dataclass(frozen=True)
class Verdict:
    value: Tri
    confidence: float
    reason: str = ""


@dataclass
class Event:
    order_id: str
    fired_at: float
    kind: str                     # "live" | "backtest"
    reason: str = ""
    confidence: float = 0.0
    clip_path: str | None = None
    id: str = ""
    predicates: dict[str, Tri] = field(default_factory=dict)
