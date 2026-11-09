"""Unit 3: T1 local embedding check, heartbeat, and the cost of a quiet scene."""
import asyncio

import numpy as np

from standing_orders.engine.cascade import Cascade
from standing_orders.engine.clock import FakeClock
from standing_orders.engine.program import parse_program
from standing_orders.engine.types import Frame, Tri
from standing_orders.llm.verifier import ScriptedVerifier
from tests.helpers_d3 import always_true, build, mug_embedder
from tests.test_program import MUG
from tests.test_runner_gate import ALONE, HAND_ON_MUG, QUIET, script_for


def run_all(tmp_path, segments, **kw):
    runner, cascades, verifier, sink, source = build(tmp_path, segments, **kw)
    try:
        events = runner.run()
    finally:
        source.stop()
        runner.close()
    return events, cascades[0], verifier


def test_t1_decided_frames_cost_no_verifier_calls(tmp_path):
    events, cascade, verifier = run_all(tmp_path, [(QUIET, 25)], embedder=mug_embedder())
    assert events == [] and verifier.calls == []
    assert cascade.stats.t1_decided == 2 and cascade.stats.t2_calls == 0   # both predicates, first frame


def test_leave_mug_still_fires_with_t1_and_calls_the_verifier_less(tmp_path):
    segs = [(HAND_ON_MUG, 3), (ALONE, 20)]
    scripted = lambda clock: ScriptedVerifier(script_for(segs), clock)
    base, _, v_base = run_all(tmp_path, segs, verifier_for=scripted)
    with_t1, cascade, v_t1 = run_all(tmp_path, segs, embedder=mug_embedder(), verifier_for=scripted)
    assert [e.fired_at for e in with_t1] == [e.fired_at for e in base] == [13.0]
    assert len(v_t1.calls) < len(v_base.calls)
    assert cascade.stats.t1_decided >= 3 and cascade.stats.t2_calls == len(v_t1.calls)


def test_uncertain_goes_to_the_verifier(tmp_path):
    # person-less frame holding only a mug: no prompt matches the whole frame, so T2 must decide
    _, cascade, verifier = run_all(tmp_path, [(ALONE, 5)], embedder=mug_embedder())
    assert any("person" in names for _, names, _ in verifier.calls)
    assert cascade.verdicts["mug"].reason.startswith("t1")


def test_no_negatives_means_uncertain(tmp_path):
    prog = parse_program({**MUG, "predicates": {"mug": {"zone": "desk", "prompt": "a coffee mug on a desk"}},
                          "rule": "mug"})
    clock = FakeClock()
    verifier = always_true(clock)
    c = Cascade(prog, verifier, clock, embedder=mug_embedder())
    from tests.helpers_d3 import scene3
    asyncio.run(c.update(Frame(scene3(mug=True), 0.0, "x")))
    assert c.stats.t1_decided == 0 and c.stats.t2_calls == 1


class FalseEverywhere:
    """Every crop looks exactly like the negative prompt."""

    def embed_image(self, image):
        return np.array([1.0, 0.0], np.float32)

    def embed_text(self, text):
        return np.array([1.0, 0.0], np.float32) if "empty" in text else np.array([0.0, 1.0], np.float32)


def test_heartbeat_makes_t2_look_even_when_t1_keeps_saying_false():
    prog = parse_program({**MUG, "predicates": {"mug": MUG["predicates"]["mug"]}, "rule": "mug"})
    clock = FakeClock()
    verifier = ScriptedVerifier(lambda n, t: Tri.TRUE, clock)
    c = Cascade(prog, verifier, clock, embedder=FalseEverywhere(), heartbeat_seconds=60.0)
    rng = np.random.default_rng(0)
    for t in range(0, 200):
        clock.set(float(t))
        img = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)   # every frame changes: T0 always passes
        asyncio.run(c.update(Frame(img, float(t), "x")))
    assert c.stats.t0_passed == 200
    assert [round(call[0]) for call in verifier.calls] == [60, 120, 180]   # one look per heartbeat
    assert c.stats.t1_decided == 197


def test_broken_embedder_falls_back_to_t2():
    class Broken(FalseEverywhere):
        def embed_image(self, image):
            raise RuntimeError("onnx exploded")

    prog = parse_program(MUG)
    clock = FakeClock()
    verifier = always_true(clock)
    c = Cascade(prog, verifier, clock, embedder=Broken())
    from tests.helpers_d3 import scene3
    asyncio.run(c.update(Frame(scene3(mug=True), 0.0, "x")))
    assert c.stats.t1_decided == 0 and c.stats.t2_calls == 1
    assert c.verdicts["mug"].value is Tri.TRUE
