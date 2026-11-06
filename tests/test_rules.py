import pytest

from standing_orders.engine.clock import FakeClock
from standing_orders.engine.program import ProgramError
from standing_orders.engine.rules import RuleEvaluator, compile_rule, referenced_predicates
from standing_orders.engine.types import Tri

T, F, U = Tri.TRUE, Tri.FALSE, Tri.UNKNOWN
MUG_RULE = {"held_for": {"seconds": 10, "of": {"all": ["mug", {"not": "person"}]}}}


def run(rule, steps, hyst=0.0):
    """steps: (seconds_since_previous_step, values). Returns the list of results."""
    clock = FakeClock(100)
    ev = RuleEvaluator(rule, clock, hyst)
    out = []
    for dt, values in steps:
        clock.advance(dt)
        out.append(ev.update(values))
    return out


def test_all_any_not_kleene():
    assert run({"all": ["a", "b"]}, [(0, {"a": T, "b": T}), (1, {"a": T, "b": U}), (1, {"a": F, "b": U})]) == [T, U, F]
    assert run({"any": ["a", "b"]}, [(0, {"a": F, "b": F}), (1, {"a": F, "b": U}), (1, {"a": T, "b": U})]) == [F, U, T]
    assert run({"not": "a"}, [(0, {"a": T}), (1, {"a": F}), (1, {"a": U})]) == [F, T, U]


def test_missing_predicate_is_unknown():
    assert run("a", [(0, {})]) == [U]


def test_held_for_fires_after_continuous_seconds():
    steps = [(0, {"mug": T, "person": F})] + [(1, {"mug": T, "person": F})] * 10
    res = run(MUG_RULE, steps)
    assert res[:10] == [F] * 10 and res[10] is T


def test_held_for_resets_when_person_returns():
    steps = [(0, {"mug": T, "person": F})] + [(1, {"mug": T, "person": F})] * 5
    steps += [(1, {"mug": T, "person": T})] + [(1, {"mug": T, "person": F})] * 9
    assert T not in run(MUG_RULE, steps)  # 9 s since the reset: not enough


def test_unknown_gap_breaks_hold_and_never_fires():
    steps = [(0, {"mug": T, "person": F})] + [(1, {"mug": T, "person": F})] * 6
    steps += [(1, {"mug": U, "person": F})] + [(1, {"mug": T, "person": F})] * 9
    res = run(MUG_RULE, steps)
    assert T not in res and U in res


def test_unknown_only_stream_never_fires():
    assert T not in run(MUG_RULE, [(1, {"mug": U, "person": U})] * 30)


def test_rose_and_fell_are_one_update_edges():
    assert run({"rose": "a"}, [(0, {"a": F}), (1, {"a": T}), (1, {"a": T}), (1, {"a": F})]) == [F, T, F, F]
    assert run({"fell": "a"}, [(0, {"a": T}), (1, {"a": F}), (1, {"a": F})]) == [F, T, F]


def test_rose_from_unknown_is_not_an_edge():
    assert run({"rose": "a"}, [(0, {"a": U}), (1, {"a": T})]) == [U, F]


def test_within_window():
    rule = {"within": {"seconds": 5, "after": "door", "then": "entered"}}
    inside = run(rule, [(0, {"door": F, "entered": F}), (1, {"door": T, "entered": F}), (2, {"door": T, "entered": T})])
    assert inside == [F, F, T]
    late = run(rule, [(0, {"door": F, "entered": F}), (1, {"door": T, "entered": F}), (6, {"door": T, "entered": T})])
    assert late[-1] is F
    never = run(rule, [(0, {"door": F, "entered": T})])
    assert never == [F]


def test_hysteresis_swallows_one_frame_flicker():
    steps = [(0, {"a": T})] + [(1, {"a": T})] * 3 + [(0.5, {"a": F}), (0.5, {"a": T})] + [(1, {"a": T})] * 2
    assert F not in run("a", steps, hyst=1.0)[1:]
    # a real change still gets through after the hold time
    res = run("a", [(0, {"a": T}), (1, {"a": F}), (1, {"a": F}), (1, {"a": F})], hyst=1.5)
    assert res == [T, T, T, F]


def test_hysteresis_passes_unknown_immediately():
    assert run("a", [(0, {"a": T}), (1, {"a": U})], hyst=5.0) == [T, U]


@pytest.mark.parametrize("bad", [
    {"bogus": "a"}, {"all": []}, {"all": "a"}, {"held_for": {"seconds": 0, "of": "a"}},
    {"held_for": {"seconds": 5}}, {"within": {"seconds": 5, "after": "a"}}, {"a": 1, "b": 2}, 5, ["a"],
])
def test_bad_rules_rejected(bad):
    with pytest.raises(ProgramError):
        compile_rule(bad)


def test_referenced_predicates():
    assert referenced_predicates(MUG_RULE) == {"mug", "person"}
    assert referenced_predicates({"within": {"seconds": 5, "after": "x", "then": {"not": "y"}}}) == {"x", "y"}


@pytest.mark.parametrize("secs", [float("nan"), float("inf"), -1, "5", True])
def test_non_finite_or_bad_seconds_rejected(secs):
    with pytest.raises(ProgramError):
        compile_rule({"held_for": {"seconds": secs, "of": "a"}})
