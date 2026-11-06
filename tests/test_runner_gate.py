"""Day 2 exit gate: a hand-written order fires correctly on a recorded clip with a scripted verifier."""
import pytest

from standing_orders.engine.cascade import Cascade
from standing_orders.engine.clock import FakeClock
from standing_orders.engine.program import parse_program, validate_program
from standing_orders.engine.runner import ListSink, Runner, StaticStore
from standing_orders.engine.types import Tri
from standing_orders.llm.verifier import ScriptedVerifier, timeline
from standing_orders.sources.file_source import FileSource
from standing_orders.storage.ringbuffer import RingBuffer
from tests.helpers import write_video
from tests.test_program import MUG

QUIET = {"mug": False, "person": False}
HAND_ON_MUG = {"mug": True, "person": True}
ALONE = {"mug": True, "person": False}

SCENARIOS = {  # name -> (segments, expected fire times in video seconds)
    "leave_mug": ([(HAND_ON_MUG, 3), (ALONE, 20)], [13.0]),
    "take_mug": ([(HAND_ON_MUG, 3), (QUIET, 20)], []),
    "return_within_5s": ([(HAND_ON_MUG, 3), (ALONE, 5), (HAND_ON_MUG, 3), (ALONE, 9)], []),
    "quiet": ([(QUIET, 25)], []),
}


def script_for(segments):
    entries, t = {"mug": [], "person": []}, 0.0
    for kwargs, seconds in segments:
        for name in entries:
            entries[name].append((t, Tri.TRUE if kwargs[name] else Tri.FALSE))
        t += seconds
    return timeline(entries)


def run_scenario(tmp_path, segments, **runner_kwargs):
    program = parse_program(MUG)
    validate_program(program)
    clock = FakeClock()
    verifier = ScriptedVerifier(script_for(segments), clock)
    sink = ListSink()
    source = FileSource(write_video(str(tmp_path / "clip.mp4"), segments), sample_fps=1.0)
    source.start()
    runner = Runner(source, StaticStore([program]), lambda p: Cascade(p, verifier, clock), clock, sink,
                    **runner_kwargs)
    try:
        events = runner.run()
        return events, verifier, runner
    finally:
        source.stop()
        runner.close()


@pytest.mark.parametrize("name", SCENARIOS)
def test_scenario(tmp_path, name):
    segments, expected = SCENARIOS[name]
    events, _, _ = run_scenario(tmp_path, segments)
    assert [e.fired_at for e in events] == expected
    for e in events:
        assert e.order_id == "ord_mug_left" and e.kind == "live" and e.confidence == 1.0
        assert e.predicates == {"mug": Tri.TRUE, "person": Tri.FALSE}


def test_cooldown_suppresses_repeat_firing(tmp_path):
    # mug stays alone for 60 s; cooldown is 300 s, so exactly one event
    events, _, _ = run_scenario(tmp_path, [(HAND_ON_MUG, 3), (ALONE, 60)])
    assert len(events) == 1


def test_t0_keeps_verifier_calls_far_below_frame_count(tmp_path):
    segments = [(HAND_ON_MUG, 3), (ALONE, 20)]
    events, verifier, runner = run_scenario(tmp_path, segments)
    stats = next(iter(runner._states.values())).cascade.stats
    assert stats.frames == 23
    assert stats.t2_calls <= 3 and len(verifier.calls) == stats.t2_calls  # first frame + the person leaving
    assert all(sizes and all(s > 0 for s in sizes) for _, _, sizes in verifier.calls)


def test_event_gets_a_clip_from_the_recording(tmp_path):
    rb = RingBuffer(str(tmp_path / "rb"))
    events, _, _ = run_scenario(tmp_path, SCENARIOS["leave_mug"][0], recorder=rb, clip_dir=str(tmp_path / "clips"))
    import cv2, os
    assert events[0].clip_path and os.path.getsize(events[0].clip_path) > 0
    assert cv2.VideoCapture(events[0].clip_path).isOpened()


def test_end_of_source_sets_ended(tmp_path):
    _, _, runner = run_scenario(tmp_path, [(QUIET, 2)])
    assert runner.ended


def test_clip_failure_still_emits_event_and_starts_cooldown(tmp_path, monkeypatch):
    import standing_orders.storage.clips as clips
    monkeypatch.setattr(clips, "export_clip", lambda *a, **k: (_ for _ in ()).throw(OSError("no codec")))
    rb = RingBuffer(str(tmp_path / "rb"))
    events, _, _ = run_scenario(tmp_path, [(HAND_ON_MUG, 3), (ALONE, 40)], recorder=rb, clip_dir=str(tmp_path / "c"))
    assert len(events) == 1 and events[0].clip_path is None


def test_disarmed_and_edited_orders_are_dropped_or_rebuilt(tmp_path):
    program = parse_program(MUG)
    store = StaticStore([program])
    clock = FakeClock()
    segs = [(ALONE, 30)]
    source = FileSource(write_video(str(tmp_path / "c.mp4"), segs), sample_fps=1.0)
    source.start()
    runner = Runner(source, store, lambda p: Cascade(p, ScriptedVerifier(script_for(segs), clock), clock),
                    clock, ListSink())
    runner.step()
    assert set(runner._states) == {"ord_mug_left"}
    store._programs = []
    runner.step()
    assert runner._states == {}
    runner.close()
    source.stop()
