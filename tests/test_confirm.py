"""Unit 4: Confirm before a fire. Fail closed on disagreement, errors and an exhausted budget."""
from standing_orders.engine.budget import VisionBudget
from standing_orders.engine.types import Frame, Tri
from standing_orders.llm.verifier import ScriptedVerifier
from tests.helpers_d3 import build
from tests.test_runner_gate import ALONE, HAND_ON_MUG, script_for

LEAVE = [(HAND_ON_MUG, 3), (ALONE, 30)]


def scripted(segs):
    return lambda clock: ScriptedVerifier(script_for(segs), clock)


def go(tmp_path, segs, **kw):
    runner, cascades, verifier, sink, source = build(tmp_path, segs, **kw)
    try:
        return runner.run(), cascades[0], verifier, runner
    finally:
        source.stop()
        runner.close()


def test_confirm_passes_and_costs_three_calls(tmp_path):
    events, cascade, verifier, _ = go(tmp_path, LEAVE, verifier_for=scripted(LEAVE))
    assert [e.fired_at for e in events] == [13.0]
    assert cascade.stats.fires == 1 and cascade.stats.confirm_suppressed == 0
    assert len(verifier.calls) == cascade.stats.t2_calls
    assert [c[0] for c in verifier.calls][-3:] == [13.0, 13.0, 13.0]   # three confirm calls at fire time


class Flipper:
    """Truthful until `after` calls have been made, then claims a person is present."""

    def __init__(self, inner, after):
        self.inner, self.after, self.n = inner, after, 0

    async def check(self, crops, predicates):
        self.n += 1
        got = await self.inner.check(crops, predicates)
        if self.n > self.after and "person" in got:
            from standing_orders.engine.types import Verdict
            got["person"] = Verdict(Tri.TRUE, 1.0, "flipped")
        return got


def test_disagreement_suppresses_event_and_goes_quiet(tmp_path):
    events, cascade, verifier, _ = go(tmp_path, LEAVE, verifier_for=lambda c: Flipper(scripted(LEAVE)(c), after=2))
    assert events == [] and cascade.stats.fires == 0
    assert cascade.stats.confirm_suppressed >= 1
    # one failed confirm stops at the first disagreeing frame, then 15 s of quiet before the next try
    assert cascade.stats.confirm_suppressed <= 2


def test_confirm_error_suppresses(tmp_path):
    class DiesOnConfirm:
        def __init__(self, inner):
            self.inner, self.n = inner, 0

        async def check(self, crops, predicates):
            self.n += 1
            if self.n > 2:
                raise ConnectionError("llm down")
            return await self.inner.check(crops, predicates)

    events, cascade, _, _ = go(tmp_path, LEAVE, verifier_for=lambda c: DiesOnConfirm(scripted(LEAVE)(c)))
    assert events == [] and cascade.stats.verifier_errors >= 1


def test_budget_too_small_for_confirm_never_fires(tmp_path):
    # 2 calls for the cascade (first frame, person leaves) + confirm needs 3 more: allow only 3 in total
    events, cascade, verifier, _ = go(tmp_path, LEAVE, verifier_for=scripted(LEAVE),
                                      budget_for=lambda c: VisionBudget(3, c))
    assert events == [] and cascade.stats.budget_exhausted
    assert cascade.stats.confirm_suppressed >= 1


def test_confirm_off_fires_without_extra_calls(tmp_path):
    events, cascade, verifier, _ = go(tmp_path, LEAVE, verifier_for=scripted(LEAVE), confirm=False)
    assert len(events) == 1 and cascade.stats.t2_calls == 2


def test_confirm_frames_are_evenly_spread(tmp_path):
    runner, *_rest = build(tmp_path, LEAVE)
    try:
        for t in range(0, 30):
            runner._recent.append(Frame(None, float(t), "x"))
        picked = runner._confirm_frames(29.0)
        assert [f.timestamp for f in picked] == [23.0, 26.0, 29.0]
    finally:
        _rest[3].stop()
        runner.close()
