"""Unit 6 and the Day 3/4 engine gate: a bowl false-fires, a correction flips it, the true mug survives.
All offline: synthetic clips, FakeEmbedder, a scripted T2 that is wrong about bowls."""
import json

import pytest

from standing_orders.engine.cascade import Cascade
from standing_orders.engine.clock import FakeClock
from standing_orders.engine.corrections import apply_correction, load_examples
from standing_orders.engine.program import parse_program
from standing_orders.engine.runner import ListSink, Runner, StaticStore
from standing_orders.engine.budget import VisionBudget
from standing_orders.sources.file_source import FileSource
from standing_orders.storage.db import connect
from tests.helpers_d3 import always_true, build, mug_embedder, write_clip
from tests.test_program import MUG
from standing_orders.llm.verifier import ScriptedVerifier
from tests.test_runner_gate import ALONE, HAND_ON_MUG, QUIET, script_for

MUG_SEGS = [(HAND_ON_MUG, 3), (ALONE, 17)]
BOWL_SEGS = [({"bowl": True, "person": True}, 3), ({"bowl": True}, 17)]


def live_run(path, examples=None, embedder=None, budget_per_hour=None):
    """What the engine does live: T0 -> T1 (embedder + examples) -> T2 (wrong about bowls) -> Confirm."""
    clock, sink = FakeClock(), ListSink()
    src = FileSource(path, sample_fps=4.0)
    src.start()
    cascades = []

    def factory(p):
        budget = VisionBudget(budget_per_hour, clock) if budget_per_hour is not None else None
        cascades.append(Cascade(p, always_true(clock), clock, embedder=embedder or mug_embedder(),
                                examples=examples, budget=budget))
        return cascades[-1]

    runner = Runner(src, StaticStore([parse_program(MUG)]), factory, clock, sink)
    try:
        return runner.run(), cascades[0]
    finally:
        src.stop()
        runner.close()


@pytest.fixture
def world(tmp_path):
    conn = connect(str(tmp_path / "so.db"))
    conn.execute("INSERT INTO orders VALUES (?,?,?,?,?,?)", (MUG["id"], MUG["text"], json.dumps(MUG), "armed", 0, 0))
    paths = {"mug": write_clip(str(tmp_path / "mug.mp4"), MUG_SEGS), "bowl": write_clip(str(tmp_path / "bowl.mp4"), BOWL_SEGS)}
    ids = {}
    for name, path in paths.items():
        events, _ = live_run(path)
        assert len(events) == 1, f"{name} must fire before any correction"   # the bowl FALSE-fires
        ids[name] = f"ev_{name}"
        conn.execute("INSERT INTO events (id, order_id, fired_at, kind, clip_path) VALUES (?,?,?,?,?)",
                     (ids[name], MUG["id"], events[0].fired_at, "live", path))
    conn.execute("UPDATE events SET label='correct' WHERE id=?", (ids["mug"],))
    conn.commit()
    return conn, ids, paths, tmp_path


def correct(conn, event_id, label, predicate, tmp_path, **kw):
    return apply_correction(conn, MUG["id"], event_id, label, predicate, "note", embedder=mug_embedder(),
                            verifier_factory=always_true, clip_dir=str(tmp_path / "clips"), **kw)


def test_gate_bowl_false_fires_correction_flips_it_and_the_mug_survives(world):
    conn, ids, paths, tmp_path = world
    res = correct(conn, ids["bowl"], "wrong", "mug", tmp_path)
    assert res.accepted and res.flipped and res.broken == []
    rows = conn.execute("SELECT predicate, polarity, source_event, crop_path FROM examples").fetchall()
    assert [r[:3] for r in rows] == [("mug", 0, ids["bowl"])] and rows[0][3].endswith(".jpg")
    assert conn.execute("SELECT label, label_note FROM events WHERE id=?", (ids["bowl"],)).fetchone() == ("wrong", "note")
    # the engine, given the saved examples, now stays quiet on the bowl and still fires on the mug
    examples = load_examples(conn, MUG["id"])
    assert live_run(paths["bowl"], examples)[0] == []
    assert len(live_run(paths["mug"], examples)[0]) == 1


def test_picks_the_best_candidate_predicate_when_none_given(world):
    conn, ids, _, tmp_path = world
    res = correct(conn, ids["bowl"], "wrong", None, tmp_path)
    assert res.accepted
    assert conn.execute("SELECT predicate FROM examples").fetchone() == ("mug",)


def test_conflict_when_a_correct_event_would_break_changes_nothing(world):
    conn, ids, paths, tmp_path = world
    # a second event with the same look as the labeled-correct mug, marked wrong: contradicts the first
    conn.execute("INSERT INTO events (id, order_id, fired_at, kind, clip_path) VALUES ('ev_dup', ?, 13, 'live', ?)",
                 (MUG["id"], paths["mug"]))
    conn.commit()
    res = correct(conn, "ev_dup", "wrong", "mug", tmp_path)
    assert not res.accepted and res.broken == [ids["mug"]] and "nothing saved" in res.message
    assert conn.execute("SELECT COUNT(*) FROM examples").fetchone()[0] == 0
    assert conn.execute("SELECT label FROM events WHERE id='ev_dup'").fetchone() == (None,)
    assert not list((tmp_path / "clips").rglob("*.jpg"))


def test_conflict_when_the_correction_does_not_change_the_target(world):
    conn, ids, _, tmp_path = world
    res = correct(conn, ids["bowl"], "wrong", "person", tmp_path)   # wrong predicate: bowl still fires
    assert not res.accepted and not res.flipped
    assert conn.execute("SELECT COUNT(*) FROM examples").fetchone()[0] == 0


def test_correct_label_with_predicate_saves_a_positive_example(world):
    conn, ids, _, tmp_path = world
    res = correct(conn, ids["mug"], "correct", "mug", tmp_path)
    assert res.accepted
    assert conn.execute("SELECT predicate, polarity FROM examples").fetchall() == [("mug", 1)]


def test_correct_without_predicate_only_labels(world):
    conn, ids, _, tmp_path = world
    assert correct(conn, ids["bowl"], "correct", None, tmp_path).accepted
    assert conn.execute("SELECT COUNT(*) FROM examples").fetchone()[0] == 0


def test_missing_clip_or_unknown_ids_are_refused(world):
    conn, ids, _, tmp_path = world
    conn.execute("UPDATE events SET clip_path='/nope.mp4' WHERE id=?", (ids["bowl"],))
    assert not correct(conn, ids["bowl"], "wrong", "mug", tmp_path).accepted
    assert not correct(conn, "ev_missing", "wrong", "mug", tmp_path).accepted
    assert not correct(conn, ids["bowl"], "meh", "mug", tmp_path).accepted
    assert not correct(conn, ids["bowl"], "wrong", "ghost", tmp_path).accepted
    assert conn.execute("SELECT COUNT(*) FROM examples").fetchone()[0] == 0


# --- the rest of the Day 3/4 engine gate ----------------------------------------------------------

def test_gate_leave_mug_fires_with_the_full_cascade(world):
    _, _, paths, _ = world
    events, cascade = live_run(paths["mug"])
    assert len(events) == 1 and events[0].fired_at >= 13.0
    assert cascade.stats.t1_decided > 0 and cascade.stats.fires == 1


def test_gate_quiet_scene_makes_a_handful_of_verifier_calls(tmp_path):
    """300 quiet frames. Cost is asserted, not hoped for."""
    quiet = lambda c: ScriptedVerifier(script_for([(QUIET, 300)]), c)
    runner, cascades, verifier, sink, source = build(tmp_path, [(QUIET, 300)], sample_fps=1.0, verifier_for=quiet)
    try:
        runner.run()
    finally:
        source.stop()
        runner.close()
    assert len(verifier.calls) <= 2 and cascades[0].stats.frames == 300 and sink.events == []
    # and with T1 in front of it: zero
    runner, cascades, verifier, sink, source = build(tmp_path, [(QUIET, 300)], embedder=mug_embedder(), clip_name="q2.mp4",
                                                              verifier_for=quiet)
    try:
        runner.run()
    finally:
        source.stop()
        runner.close()
    assert len(verifier.calls) == 0 and cascades[0].stats.t1_decided == 2


@pytest.mark.parametrize("per_hour", [0, 1, 3])
def test_gate_exhausted_budget_never_fires(world, per_hour):
    _, _, paths, _ = world
    for clip in paths.values():
        events, cascade = live_run(clip, embedder=_NoEmbedder(), budget_per_hour=per_hour)
        assert events == [], f"budget {per_hour} fired"
        assert cascade.stats.budget_exhausted


class _NoEmbedder:
    """Forces every decision to T2, where the budget bites."""

    def embed_image(self, image):
        raise RuntimeError

    def embed_text(self, text):
        raise RuntimeError
