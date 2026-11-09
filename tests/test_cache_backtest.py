"""Unit 5: CachedVerifier and backtest()."""
import asyncio
import os

from standing_orders.engine.backtest import backtest
from standing_orders.engine.program import PredicateSpec, parse_program
from standing_orders.engine.types import Tri, Verdict
from standing_orders.llm.verifier import CachedVerifier, ScriptedVerifier
from standing_orders.sources.file_source import FileSource
from standing_orders.storage.db import connect
from tests.helpers_d3 import write_clip
from tests.test_program import MUG
from tests.test_runner_gate import ALONE, HAND_ON_MUG, QUIET, script_for

SPEC = PredicateSpec("desk", "a mug", ["an empty desk"])


class Counting:
    def __init__(self):
        self.calls = []

    async def check(self, crops, predicates):
        self.calls.append(list(predicates))
        return {n: Verdict(Tri.TRUE, 0.9, "inner") for n in predicates}


def test_cached_verifier_hit_skips_inner_and_miss_stores(tmp_path):
    conn = connect(str(tmp_path / "so.db"))
    inner = Counting()
    v = CachedVerifier(inner, conn)
    first = asyncio.run(v.check([b"crop-a"], {"mug": SPEC}))
    again = asyncio.run(v.check([b"crop-a"], {"mug": SPEC}))
    assert inner.calls == [["mug"]] and first == again and again["mug"].reason == "inner"
    assert (v.hits, v.misses) == (1, 1)
    # different pixels, or a different prompt/negatives, are different keys
    asyncio.run(v.check([b"crop-b"], {"mug": SPEC}))
    asyncio.run(v.check([b"crop-a"], {"mug": PredicateSpec("desk", "a mug", ["a bowl"])}))
    assert len(inner.calls) == 3
    # persisted: a fresh verifier over the same db answers without the inner one
    inner2 = Counting()
    assert asyncio.run(CachedVerifier(inner2, conn).check([b"crop-a"], {"mug": SPEC}))["mug"].value is Tri.TRUE
    assert inner2.calls == []


def test_cached_verifier_mixed_batch_only_asks_for_misses(tmp_path):
    conn = connect(str(tmp_path / "so.db"))
    inner = Counting()
    v = CachedVerifier(inner, conn)
    asyncio.run(v.check([b"x"], {"mug": SPEC}))
    out = asyncio.run(v.check([b"x", b"y"], {"mug": SPEC, "person": PredicateSpec("frame", "a person")}))
    assert inner.calls == [["mug"], ["person"]] and list(out) == ["mug", "person"]


def test_unknown_and_errors_are_not_cached(tmp_path):
    conn = connect(str(tmp_path / "so.db"))

    class Shrug:
        async def check(self, crops, predicates):
            return {n: Verdict(Tri.UNKNOWN, 0.0) for n in predicates}

    class Boom:
        async def check(self, crops, predicates):
            raise RuntimeError

    asyncio.run(CachedVerifier(Shrug(), conn).check([b"x"], {"mug": SPEC}))
    try:
        asyncio.run(CachedVerifier(Boom(), conn).check([b"x"], {"mug": SPEC}))
    except RuntimeError:
        pass
    assert conn.execute("SELECT COUNT(*) FROM verifications").fetchone()[0] == 0


def mug_left_clip(tmp_path):
    segs = [(HAND_ON_MUG, 3), (ALONE, 30)]
    return segs, write_clip(str(tmp_path / "left.mp4"), segs)


def test_backtest_finds_the_would_have_fired_event(tmp_path):
    segs, path = mug_left_clip(tmp_path)
    src = FileSource(path, sample_fps=1.0)
    src.start()
    try:
        result = backtest(parse_program(MUG), src, lambda clock: ScriptedVerifier(script_for(segs), clock),
                          clip_dir=str(tmp_path / "clips"))
    finally:
        src.stop()
    assert [e.fired_at for e in result.events] == [13.0] and result.events[0].kind == "backtest"
    assert os.path.getsize(result.events[0].clip_path) > 0
    assert result.frames_checked == 33 and result.unresolved_frames == 0
    assert result.t2_calls == 5   # first frame, person leaves, 3 confirm frames


def test_backtest_cap_reports_unresolved_frames_and_does_not_guess(tmp_path):
    segs, path = mug_left_clip(tmp_path)
    src = FileSource(path, sample_fps=1.0)
    src.start()
    try:
        result = backtest(parse_program(MUG), src, lambda clock: ScriptedVerifier(script_for(segs), clock),
                          max_t2_calls=1)
    finally:
        src.stop()
    assert result.events == [] and result.t2_calls == 1
    assert result.unresolved_frames > 0


def test_backtest_quiet_footage_is_cheap(tmp_path):
    path = write_clip(str(tmp_path / "q.mp4"), [(QUIET, 60)])
    src = FileSource(path, sample_fps=1.0)
    src.start()
    try:
        result = backtest(parse_program(MUG), src, lambda c: ScriptedVerifier(script_for([(QUIET, 60)]), c))
    finally:
        src.stop()
    assert result.events == [] and result.frames_checked == 60 and result.t2_calls == 1
