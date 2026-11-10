"""The committed benchmark must replay offline: right answers, no cache misses, same on any machine."""
import pytest

from standing_orders.benchmark import run_all, summarize
from standing_orders.engine.embedder import DEFAULT_MODEL_DIR, models_present

pytestmark = pytest.mark.skipif(not models_present(DEFAULT_MODEL_DIR), reason="run `hermes orders setup` first")


def test_committed_cache_replays_every_scenario_correctly():
    results = run_all()
    assert [r.id for r in results if not r.correct] == []
    s = summarize(results)
    assert s["cache_misses"] == 0
    assert s["vision_calls"] < s["polling_calls"] / 5
    assert all(r.injection_ok for r in results if r.injection_ok is not None)
