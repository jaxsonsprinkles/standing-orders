"""WatchProgram: the JSON the compiler produces. Data only, never executable code."""
from dataclasses import dataclass, field
from typing import Any

Point = tuple[float, float]  # normalized 0..1

# Rule = predicate name (str) or a one-key dict, e.g.
# {"held_for": {"seconds": 10, "of": {"all": ["mug", {"not": "person"}]}}}
Rule = str | dict[str, Any]

OPERATORS = ("all", "any", "not", "held_for", "rose", "fell", "within")


@dataclass
class PredicateSpec:
    zone: str                   # a zone name, or "frame" for the whole frame
    prompt: str
    negatives: list[str] = field(default_factory=list)


@dataclass
class Action:
    mode: str                   # "agent_turn"
    instruction: str


@dataclass
class WatchProgram:
    id: str
    text: str
    source: str
    zones: dict[str, list[Point]]
    predicates: dict[str, PredicateSpec]
    rule: Rule
    action: Action
    cooldown_seconds: int
    vision_budget_per_hour: int


class ProgramError(ValueError):
    """Validation failure; message goes back to the compiler for one repair attempt."""


def parse_program(data: dict) -> WatchProgram:
    """Build a WatchProgram from compiler JSON. Day 2."""
    raise NotImplementedError


def validate_program(program: WatchProgram) -> None:
    """Reject unknown operators, undefined predicates/zones, bad polygons, missing cooldown. Day 2."""
    raise NotImplementedError
