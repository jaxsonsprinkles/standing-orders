"""WatchProgram: the JSON the compiler produces. Data only, never executable code."""

from __future__ import annotations
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
    mode: str                   # "agent_turn" | "add_list_item"
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


ACTION_MODES = ("agent_turn", "add_list_item")
FRAME_ZONE = "frame"


def _require(data: Any, keys: tuple[str, ...], where: str) -> None:
    if not isinstance(data, dict):
        raise ProgramError(f"{where}: expected an object, got {type(data).__name__}")
    missing = [k for k in keys if k not in data]
    if missing:
        raise ProgramError(f"{where}: missing {', '.join(missing)}")


def parse_program(data: dict) -> WatchProgram:
    """Build a WatchProgram from compiler JSON. Shape only; validate_program checks meaning."""
    _require(data, ("id", "text", "source", "zones", "predicates", "rule", "action",
                    "cooldown_seconds", "vision_budget_per_hour"), "program")
    if not isinstance(data["zones"], dict) or not isinstance(data["predicates"], dict):
        raise ProgramError("program: 'zones' and 'predicates' must be objects")
    predicates = {}
    for name, spec in data["predicates"].items():
        _require(spec, ("zone", "prompt"), f"predicate {name!r}")
        predicates[name] = PredicateSpec(spec["zone"], spec["prompt"], list(spec.get("negatives", [])))
    _require(data["action"], ("mode", "instruction"), "action")
    try:
        zones = {n: [(float(x), float(y)) for x, y in pts] for n, pts in data["zones"].items()}
    except (TypeError, ValueError):
        raise ProgramError("zones: each zone must be a list of [x, y] number pairs")
    return WatchProgram(
        id=data["id"], text=data["text"], source=data["source"], zones=zones,
        predicates=predicates, rule=data["rule"],
        action=Action(data["action"]["mode"], data["action"]["instruction"]),
        cooldown_seconds=data["cooldown_seconds"],
        vision_budget_per_hour=data["vision_budget_per_hour"],
    )


def _polygon_area(pts: list[Point]) -> float:
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]))) / 2


def validate_program(program: WatchProgram) -> None:
    """Reject unknown operators, undefined predicates/zones, bad polygons, bad cooldown/budget.
    Raises ProgramError; the message is fed back to the compiler for its one repair attempt."""
    from .rules import compile_rule, referenced_predicates  # rules imports this module

    for name, pts in program.zones.items():
        if name == FRAME_ZONE:
            raise ProgramError(f"zone name {FRAME_ZONE!r} is reserved for the whole frame")
        if len(pts) < 3 or any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in pts):
            raise ProgramError(f"zone {name!r}: needs 3+ points with x,y in 0..1")
        if _polygon_area(pts) < 1e-4:
            raise ProgramError(f"zone {name!r}: polygon has no area")
    if not program.predicates:
        raise ProgramError("program needs at least one predicate")
    for name, spec in program.predicates.items():
        if not isinstance(spec.zone, str) or not isinstance(spec.prompt, str):
            raise ProgramError(f"predicate {name!r}: 'zone' and 'prompt' must be strings")
        if spec.zone != FRAME_ZONE and spec.zone not in program.zones:
            raise ProgramError(f"predicate {name!r} uses undefined zone {spec.zone!r}")
        if not spec.prompt.strip():
            raise ProgramError(f"predicate {name!r}: prompt is empty")
    compile_rule(program.rule)
    undefined = referenced_predicates(program.rule) - program.predicates.keys()
    if undefined:
        raise ProgramError(f"rule uses undefined predicates: {', '.join(sorted(undefined))}")
    if program.action.mode not in ACTION_MODES:  # unhashable modes fail the membership test too
        raise ProgramError(f"action.mode must be one of {ACTION_MODES}, got {program.action.mode!r}")
    if not isinstance(program.action.instruction, str) or not program.action.instruction.strip():
        raise ProgramError("action.instruction is empty")
    for field_, value in (("cooldown_seconds", program.cooldown_seconds),
                          ("vision_budget_per_hour", program.vision_budget_per_hour)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ProgramError(f"{field_} must be a non-negative integer, got {value!r}")
