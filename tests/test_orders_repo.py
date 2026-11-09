import pytest

from standing_orders.engine.program import parse_program, validate_program
from standing_orders.engine.types import Event
from standing_orders.storage.db import connect
from standing_orders.storage.orders import OrderRepo, program_to_dict


def make_program(oid="o1"):
    p = parse_program({
        "id": oid, "text": "tell me if mug left", "source": "cam0",
        "zones": {"desk": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]},
        "predicates": {"mug": {"zone": "desk", "prompt": "a mug", "negatives": ["a cup holder"]}},
        "rule": {"held_for": {"seconds": 5, "of": {"not": "mug"}}},
        "action": {"mode": "agent_turn", "instruction": "say so"},
        "cooldown_seconds": 60, "vision_budget_per_hour": 10})
    validate_program(p)
    return p


@pytest.fixture
def repo(tmp_path):
    t = [1000.0]
    r = OrderRepo(connect(str(tmp_path / "so.db")), time_fn=lambda: t[0])
    r.t = t
    return r


def test_round_trip(repo):
    p = make_program()
    repo.create(p)
    assert repo.get("o1") == p
    assert repo.get("nope") is None
    assert program_to_dict(p)["zones"]["desk"][0] == [0.1, 0.1]


def test_draft_not_armed_until_status_changes(repo):
    repo.create(make_program())
    assert repo.status("o1") == "draft" and repo.armed() == []
    repo.set_status("o1", "armed")
    assert [p.id for p in repo.armed()] == ["o1"]
    repo.set_status("o1", "paused")
    assert repo.armed() == []


def test_status_validated(repo):
    with pytest.raises(ValueError):
        repo.create(make_program(), status="bogus")
    repo.create(make_program())
    with pytest.raises(ValueError):
        repo.set_status("o1", "firing")
    with pytest.raises(KeyError):
        repo.set_status("missing", "armed")


def test_armed_reflects_program_edit(repo):
    p = make_program()
    repo.create(p, "armed")
    assert repo.armed()[0].cooldown_seconds == 60
    p.cooldown_seconds = 5
    repo.t[0] += 1
    repo.update_program(p)
    assert repo.armed()[0].cooldown_seconds == 5


def test_list(repo):
    repo.create(make_program("a"))
    repo.t[0] += 1
    repo.create(make_program("b"), "armed")
    assert [(o["id"], o["status"]) for o in repo.list()] == [("a", "draft"), ("b", "armed")]


def test_events_and_labels(repo):
    repo.create(make_program())
    repo.emit(Event("o1", 10.0, "live", "mug gone", 0.9, "/tmp/c.mp4", "o1@10"))
    repo.add_event(Event("o1", 5.0, "backtest", "bt", 0.8, None, "o1@5"))
    assert [e["id"] for e in repo.events("o1")] == ["o1@10", "o1@5"]
    assert [e["id"] for e in repo.events("o1", kind="backtest")] == ["o1@5"]
    repo.label_event("o1@10", "wrong", "it was a plate")
    assert repo.get_event("o1@10")["label"] == "wrong"
    with pytest.raises(ValueError):
        repo.label_event("o1@10", "maybe")
    with pytest.raises(KeyError):
        repo.label_event("zzz", "correct")
    with pytest.raises(ValueError):
        repo.add_event(Event("o1", 1.0, "weird", id="x"))
    repo.delete_backtest_events("o1")
    assert [e["id"] for e in repo.events("o1")] == ["o1@10"]


def test_counters_accumulate_per_hour_bucket(repo):
    repo.add_counters("o1", frames=3, t2_calls=1, now=3600.0)
    repo.add_counters("o1", frames=2, t2_calls=2, now=3700.0)
    repo.add_counters("o1", frames=7, now=7300.0)
    assert repo.counters("o1", since=0)["frames"] == 12
    assert repo.counters("o1", since=3600)["t2_calls"] == 3
    assert repo.counters("o1", since=7200)["frames"] == 7
    assert repo.counters(since=0)["frames"] == 12
    repo.create(make_program())
    repo.emit(Event("o1", 1.0, "live", id="o1@1"))
    assert repo.counters("o1", since=0)["fires"] == 1
