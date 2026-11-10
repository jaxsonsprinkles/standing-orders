"""Day 3/4 test helpers: bowl scenes, a FakeEmbedder wired to the MUG program, runner builders."""
import cv2
import numpy as np

from standing_orders.engine.cascade import Cascade
from standing_orders.engine.clock import FakeClock
from standing_orders.engine.embedder import FakeEmbedder
from standing_orders.engine.program import parse_program
from standing_orders.engine.runner import ListSink, Runner, StaticStore
from standing_orders.llm.verifier import ScriptedVerifier, Script, timeline
from standing_orders.engine.types import Tri
from standing_orders.sources.file_source import FileSource
from tests.helpers import H, W, scene
from tests.test_program import MUG

BOWL_BGR = (40, 60, 220)   # a red box where the mug's white one goes


def scene3(mug: bool = False, person: bool = False, bowl: bool = False) -> np.ndarray:
    img = scene(mug, person)
    if bowl:
        img[150:190, 140:180] = BOWL_BGR
    return img


def write_clip(path: str, segments: list[tuple[dict, float]], fps: int = 4) -> str:
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    assert w.isOpened()
    for kwargs, seconds in segments:
        for _ in range(round(seconds * fps)):
            w.write(scene3(**kwargs))
    w.release()
    return path


def mug_embedder() -> FakeEmbedder:
    """Knows the mug, the empty desk, a person and an empty room, but NOT bowls."""
    desk = lambda img: img[int(.55 * H):, int(.1 * W):int(.9 * W)]
    return FakeEmbedder({
        "a coffee mug on a desk": desk(scene3(mug=True)),
        "an empty desk": desk(scene3()),
        "a person in the room": scene3(person=True),
        "an empty room": scene3(),
    })


def always_true(clock) -> ScriptedVerifier:
    """A T2 that says the mug is there and nobody is around: wrong for bowls, right for mugs."""
    return ScriptedVerifier(lambda name, now: Tri.TRUE if name == "mug" else Tri.FALSE, clock)


def build(tmp_path, segments, *, embedder=None, verifier_for=None, budget_for=None, fps=4, clip_name="c.mp4",
          sample_fps=1.0, **runner_kw):
    """Runner over a synthetic video. Returns (runner, cascade_list, verifier, sink, source)."""
    program = parse_program(MUG)
    clock = FakeClock()
    verifier = (verifier_for or always_true)(clock)
    sink = ListSink()
    source = FileSource(write_clip(str(tmp_path / clip_name), segments, fps), sample_fps=sample_fps)
    source.start()
    cascades = []

    def factory(p):
        c = Cascade(p, verifier, clock, embedder=embedder, budget=budget_for(clock) if budget_for else None)
        cascades.append(c)
        return c

    runner = Runner(source, StaticStore([program]), factory, clock, sink, **runner_kw)
    return runner, cascades, verifier, sink, source
