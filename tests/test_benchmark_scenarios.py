"""benchmark/scenarios.yaml must stay consistent with the fixtures and the program validator."""
import pathlib

import cv2
import pytest
import yaml

from standing_orders.engine.program import parse_program, validate_program

BENCH = pathlib.Path(__file__).parent.parent / "benchmark"
SPEC = yaml.safe_load((BENCH / "scenarios.yaml").read_text())


def test_ten_scenarios_with_unique_ids():
    assert [s["id"] for s in SPEC["scenarios"]] == list(range(1, 11))


@pytest.mark.parametrize("name", SPEC["orders"])
def test_orders_validate(name):
    validate_program(parse_program(SPEC["orders"][name]))


@pytest.mark.parametrize("s", SPEC["scenarios"], ids=lambda s: f"s{s['id']}")
def test_fixture_is_small_silent_video(s):
    assert s["order"] in SPEC["orders"] and isinstance(s["fire"], bool)
    path = BENCH / s["clip"]
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
    assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 640
    assert path.stat().st_size < 400_000
    assert b"mdta" not in path.read_bytes()[:4096]   # no Apple/location metadata box
