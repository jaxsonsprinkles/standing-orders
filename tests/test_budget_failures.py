"""Unit 1: VisionBudget, verifier failures, exhausted budget. None of these may fire or raise."""
from standing_orders.engine.budget import VisionBudget
from standing_orders.engine.cascade import Cascade
from standing_orders.engine.clock import FakeClock
from standing_orders.engine.program import parse_program
from standing_orders.engine.runner import ListSink, Runner, StaticStore
from standing_orders.llm.verifier import ScriptedVerifier
from standing_orders.sources.file_source import FileSource
from tests.helpers import write_video
from tests.test_program import MUG
from tests.test_runner_gate import ALONE, HAND_ON_MUG, QUIET, script_for

LEAVE = [(HAND_ON_MUG, 3), (ALONE, 20)]


def test_budget_is_sliding_window():
    clock = FakeClock()
    b = VisionBudget(3, clock)
    assert b.try_spend(2) and b.remaining() == 1 and not b.exhausted
    assert not b.try_spend(2) and b.remaining() == 1   # refused: nothing spent
    assert b.try_spend()
    assert b.exhausted and not b.try_spend()
    clock.set(3599.0)
    assert b.exhausted
    clock.set(3600.0)                                   # the first two calls (t=0) age out
    assert b.remaining() == 3 and b.try_spend()


def test_zero_budget_is_always_exhausted():
    assert VisionBudget(0, FakeClock()).exhausted


class Raising:
    async def check(self, crops, predicates):
        raise RuntimeError("llm down")


def run(tmp_path, segments, verifier_for, budget_for=None, **kw):
    program = parse_program(MUG)
    clock = FakeClock()
    verifier = verifier_for(clock)
    source = FileSource(write_video(str(tmp_path / "c.mp4"), segments), sample_fps=1.0)
    source.start()
    cascades = []

    def factory(p):
        c = Cascade(p, verifier, clock, budget=budget_for(clock) if budget_for else None, **kw)
        cascades.append(c)
        return c

    runner = Runner(source, StaticStore([program]), factory, clock, ListSink())
    try:
        return runner.run(), cascades[0], verifier
    finally:
        source.stop()
        runner.close()


def test_raising_verifier_gives_unknown_and_never_fires(tmp_path):
    events, cascade, _ = run(tmp_path, LEAVE, lambda clock: Raising())
    assert events == []
    assert all(v.value.value == "unknown" for v in cascade.verdicts.values())
    assert cascade.stats.verifier_errors >= 1 and cascade.stats.fires == 0
    assert cascade.stats.unresolved_frames > 0


def test_failed_predicates_are_retried_and_recover(tmp_path):
    class Flaky:
        def __init__(self, inner):
            self.inner, self.n = inner, 0

        async def check(self, crops, predicates):
            self.n += 1
            if self.n <= 2:
                raise TimeoutError
            return await self.inner.check(crops, predicates)

    events, cascade, _ = run(tmp_path, [(HAND_ON_MUG, 3), (ALONE, 40)],
                             lambda clock: Flaky(ScriptedVerifier(script_for([(HAND_ON_MUG, 3), (ALONE, 40)]), clock)))
    assert cascade.stats.verifier_errors == 2
    assert cascade.verdicts["mug"].value.value == "true"   # retried without a new change
    assert len(events) == 1


def test_exhausted_budget_never_fires(tmp_path):
    segs = [(HAND_ON_MUG, 3), (ALONE, 40)]
    events, cascade, verifier = run(tmp_path, segs, lambda c: ScriptedVerifier(script_for(segs), c),
                                    budget_for=lambda c: VisionBudget(1, c))
    assert events == []                          # the one allowed call saw "person present"
    assert len(verifier.calls) == 1
    assert cascade.stats.budget_exhausted and cascade.stats.unresolved_frames > 0
    assert cascade.verdicts["person"].value.value == "unknown"


def test_budget_zero_makes_no_calls(tmp_path):
    events, cascade, verifier = run(tmp_path, LEAVE, lambda c: ScriptedVerifier(script_for(LEAVE), c),
                                    budget_for=lambda c: VisionBudget(0, c))
    assert events == [] and verifier.calls == [] and cascade.stats.t2_calls == 0


def test_stats_count_frames_and_t0(tmp_path):
    events, cascade, verifier = run(tmp_path, [(QUIET, 20)], lambda c: ScriptedVerifier(script_for([(QUIET, 20)]), c))
    s = cascade.stats
    assert s.frames == 20 and s.t0_passed == 2 and s.t2_calls == 1 and events == []
