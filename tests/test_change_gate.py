import numpy as np

from standing_orders.engine.change_gate import ChangeGate
from standing_orders.engine.program import FRAME_ZONE
from tests.helpers import scene

DESK = [(0.1, 0.55), (0.9, 0.55), (0.9, 1.0), (0.1, 1.0)]


def noisy(img, seed, sigma=2.0):
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def textured(mug: bool, brightness=100, seed=0):
    """A desk with texture, so lighting normalization has structure to work with."""
    img = scene(mug, brightness=brightness).astype(np.float32)
    img += np.random.default_rng(7).normal(0, 12, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_first_frame_passes_then_static_scene_is_quiet():
    g = ChangeGate({"desk": DESK, FRAME_ZONE: []})
    assert g.update(textured(False)) == {"desk": True, FRAME_ZONE: True}
    for i in range(20):
        assert g.update(noisy(textured(False), i)) == {"desk": False, FRAME_ZONE: False}


def test_mug_appearing_in_zone_passes():
    g = ChangeGate({"desk": DESK})
    g.update(textured(False))
    assert g.update(textured(True)) == {"desk": True}


def test_change_outside_zone_is_ignored_by_that_zone():
    g = ChangeGate({"desk": DESK, FRAME_ZONE: []})
    g.update(scene(False))
    img = scene(False)
    img[10:100, 20:90] = (20, 20, 20)  # person in the upper-left, above the desk zone (y > 0.55)
    r = g.update(img)
    assert r["desk"] is False and r[FRAME_ZONE] is True


def test_lamp_on_off_does_not_pass():
    g = ChangeGate({"desk": DESK, FRAME_ZONE: []})
    g.update(textured(False, brightness=90))
    for b in (150, 60, 200, 90):  # brightness swings, same scene
        assert not any(g.update(textured(False, brightness=b)).values()), b


def test_lamp_with_a_mug_still_passes():
    g = ChangeGate({"desk": DESK})
    g.update(textured(False, brightness=90))
    assert g.update(textured(True, brightness=170)) == {"desk": True}


def test_slow_change_accumulates_against_last_passed_reference():
    g = ChangeGate({"desk": DESK}, threshold=0.25)
    g.update(scene(False))
    base = scene(False)
    fired = False
    for k in range(1, 41):  # mug fades in over 40 frames
        img = base.copy()
        img[150:190, 140:180] = (100 + 155 * k / 40)
        fired |= g.update(img)["desk"]
    assert fired


def test_sliver_polygon_still_has_pixels():
    g = ChangeGate({"sliver": [(0.5, 0.5), (0.5005, 0.5), (0.5, 0.5005)]})
    assert g.update(scene(False)) == {"sliver": True}   # no NaN: first frame passes
    g.update(scene(True))  # must not raise
    assert g._mask("sliver", (54, 96)).any()
