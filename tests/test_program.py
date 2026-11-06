import copy

import pytest

from standing_orders.engine.program import ProgramError, parse_program, validate_program

MUG = {
    "id": "ord_mug_left",
    "text": "If I leave my mug on the desk for 10 seconds after I walk away, tell me",
    "source": "cam0",
    "zones": {"desk": [[0.1, 0.55], [0.9, 0.55], [0.9, 1.0], [0.1, 1.0]]},
    "predicates": {
        "mug": {"zone": "desk", "prompt": "a coffee mug on a desk",
                "negatives": ["an empty desk", "a bowl on a desk"]},
        "person": {"zone": "frame", "prompt": "a person in the room", "negatives": ["an empty room"]},
    },
    "rule": {"held_for": {"seconds": 10, "of": {"all": ["mug", {"not": "person"}]}}},
    "action": {"mode": "agent_turn", "instruction": "Tell the user the mug was left behind. Attach the clip."},
    "cooldown_seconds": 300,
    "vision_budget_per_hour": 20,
}


def mutated(**changes):
    d = copy.deepcopy(MUG)
    d.update(changes)
    return d


def test_spec_example_parses_and_validates():
    p = parse_program(MUG)
    validate_program(p)
    assert p.predicates["mug"].negatives == ["an empty desk", "a bowl on a desk"]
    assert p.zones["desk"][0] == (0.1, 0.55)


@pytest.mark.parametrize("key", ["id", "zones", "rule", "action", "cooldown_seconds"])
def test_missing_key_rejected_at_parse(key):
    d = copy.deepcopy(MUG)
    del d[key]
    with pytest.raises(ProgramError, match=key):
        parse_program(d)


@pytest.mark.parametrize("changes, match", [
    ({"rule": {"bogus": "mug"}}, "unknown operator"),
    ({"rule": "ghost"}, "undefined predicates: ghost"),
    ({"rule": {"all": ["mug", "ghost"]}}, "ghost"),
    ({"zones": {"desk": [[0, 0], [1, 1]]}}, "3\\+ points"),
    ({"zones": {"desk": [[0, 0], [1, 0], [2, 1]]}}, "0..1"),
    ({"zones": {"desk": [[0, 0], [0.5, 0.5], [1, 1]]}}, "no area"),
    ({"zones": {"frame": [[0, 0], [1, 0], [1, 1]]}}, "reserved"),
    ({"cooldown_seconds": -1}, "cooldown_seconds"),
    ({"cooldown_seconds": 1.5}, "cooldown_seconds"),
    ({"vision_budget_per_hour": "20"}, "vision_budget_per_hour"),
    ({"action": {"mode": "shell", "instruction": "rm"}}, "action.mode"),
])
def test_semantic_errors_are_actionable(changes, match):
    with pytest.raises(ProgramError, match=match):
        validate_program(parse_program(mutated(**changes)))


def test_predicate_with_undefined_zone_rejected():
    d = copy.deepcopy(MUG)
    d["predicates"]["mug"]["zone"] = "shelf"
    with pytest.raises(ProgramError, match="undefined zone 'shelf'"):
        validate_program(parse_program(d))


def test_malformed_zone_points_rejected_at_parse():
    with pytest.raises(ProgramError, match="zones"):
        parse_program(mutated(zones={"desk": [[0, 0], [1], [1, 1]]}))
