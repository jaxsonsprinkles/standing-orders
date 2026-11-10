"""Benchmark: replay the labeled fixtures through the real engine (T0, T1, T2 stub-or-live, Confirm, rules)
and report precision/recall and vision calls against fixed-interval polling.

No Hermes imports. Time comes from a FakeClock that follows the video, so a 10-minute scenario runs in seconds.
T2 answers come from a recorded cache keyed by (scenario, video time, call index, predicates), never by
pixels, so the same engine run asks the same questions on any machine. A question missing from the cache is
reported as a cache miss and answered UNKNOWN; with a live LLM it is asked and recorded instead."""
import hashlib
import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any

from .engine.budget import VisionBudget
from .engine.cascade import Cascade
from .engine.clock import FakeClock
from .engine.embedder import maybe_embedder
from .engine.program import PredicateSpec, parse_program, validate_program
from .engine.runner import ListSink, Runner, StaticStore
from .engine.types import Frame, Tri, Verdict
from .hermes.engine_thread import build_fire_message
from .llm.llm_verifier import LlmVerifier
from .sources.file_source import FileSource

ROOT = pathlib.Path(__file__).parent.parent / "benchmark"
POLL_INTERVAL = 2.0   # seconds; the naive baseline the spec compares against (43,200 calls per day)


def load_spec(path: pathlib.Path | None = None) -> dict:
    path = path or ROOT / "scenarios.yaml"
    try:
        import yaml
        return yaml.safe_load(path.read_text())
    except ImportError:
        from ruamel.yaml import YAML   # the Hermes runtime ships ruamel
        return YAML(typ="safe").load(path.read_text())


class LoopedSource:
    """Plays a recording forward then backward until `total_seconds` of video time have passed."""

    def __init__(self, inner, total_seconds: float, sample_fps: float):
        self.inner, self.total, self.dt = inner, total_seconds, 1.0 / sample_fps
        self._frames: list[Frame] = []
        self._n = 0

    def start(self) -> None:
        self.inner.start()
        while (f := self.inner.read()) is not None:
            self._frames.append(f)
        self.inner.stop()
        self._n = 0

    def read(self) -> Frame | None:
        t = self._n * self.dt
        if t >= self.total or not self._frames:
            return None
        count = len(self._frames)
        period = max(1, 2 * (count - 1))
        i = self._n % period
        i = i if i < count else period - i
        self._n += 1
        return Frame(self._frames[i].image, t, self._frames[i].source_id)

    def stop(self) -> None:
        self._frames = []


class T2Cache:
    """Recorded T2 answers. JSON on disk so it is committed with the fixtures and diffable."""

    def __init__(self, path: pathlib.Path | None = None):
        self.path = path or ROOT / "cache" / "t2_answers.json"
        data = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.meta: dict[str, Any] = data.get("meta", {})
        self.answers: dict[str, dict] = data.get("answers", {})
        self.dirty = False

    def save(self, meta: dict | None = None) -> None:
        self.meta.update(meta or {})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"meta": self.meta, "answers": dict(sorted(self.answers.items()))},
                                        indent=1, sort_keys=True) + "\n")


def _phash(predicates: dict[str, PredicateSpec]) -> str:
    blob = json.dumps([[n, s.prompt, list(s.negatives)] for n, s in predicates.items()])
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


class BenchVerifier:
    """Verifier for one scenario. Cache hit -> stored answer. Miss -> `inner` if given (and record), else UNKNOWN."""

    def __init__(self, cache: T2Cache, scenario: int, clock, inner=None):
        self.cache, self.scenario, self.clock, self.inner = cache, scenario, clock, inner
        self.misses = 0
        self._seen: dict[str, int] = {}

    async def check(self, crops, predicates):
        now = f"{self.clock.now():.3f}"
        k = self._seen[now] = self._seen.get(now, -1) + 1
        key = f"{self.scenario}|{now}|{k}|{_phash(predicates)}"
        hit = self.cache.answers.get(key)
        if hit is not None:
            return {n: Verdict(Tri(hit[n]["value"]), hit[n]["confidence"], hit[n]["reason"]) for n in predicates}
        if self.inner is None:
            self.misses += 1
            return {n: Verdict(Tri.UNKNOWN, 0.0, "benchmark cache miss") for n in predicates}
        got = await self.inner.check(crops, predicates)
        if all(v.value is not Tri.UNKNOWN for v in got.values()):   # never record a failure as knowledge
            self.cache.answers[key] = {n: {"value": v.value.value, "confidence": v.confidence, "reason": v.reason,
                                           "crop_sha256": hashlib.sha256(c).hexdigest()[:16]}
                                       for (n, v), c in zip(got.items(), crops)}
            self.cache.dirty = True
        return got


@dataclass
class ScenarioResult:
    id: int
    title: str
    expected: bool
    fired: bool
    fire_times: list[float]
    duration_s: float
    frames: int = 0
    t0_passed: int = 0
    t1_decided: int = 0
    t2_calls: int = 0
    confirm_suppressed: int = 0
    verifier_errors: int = 0
    cache_misses: int = 0
    wall_s: float = 0.0
    cpu_s: float = 0.0
    injection_ok: bool | None = None
    could: bool = False

    @property
    def correct(self) -> bool:
        return self.fired == self.expected

    @property
    def polling_calls(self) -> int:
        return int(self.duration_s / POLL_INTERVAL)


def run_scenario(scenario: dict, spec: dict, cache: T2Cache, embedder, llm=None) -> ScenarioResult:
    order = spec["orders"][scenario["order"]]
    program = parse_program(order)
    validate_program(program)
    fps = float(spec.get("sample_fps", 2.0))
    clock = FakeClock()
    source = FileSource(str(ROOT / scenario["clip"]), sample_fps=fps)
    if scenario.get("loop_seconds"):
        source = LoopedSource(source, float(scenario["loop_seconds"]), fps)
    inner = LlmVerifier(llm, clock=clock, max_calls_per_hour=program.vision_budget_per_hour) if llm else None
    verifier = BenchVerifier(cache, scenario["id"], clock, inner)
    cascades: list[Cascade] = []

    def factory(p):
        cascades.append(Cascade(p, verifier, clock, embedder=embedder,
                                budget=VisionBudget(p.vision_budget_per_hour, clock)))
        return cascades[-1]

    sink = ListSink()
    runner = Runner(source, StaticStore([program]), factory, clock, sink, kind="backtest")
    source.start()
    wall0, cpu0 = time.perf_counter(), time.process_time()
    try:
        runner.run()
    finally:
        runner.close()
        source.stop()
    wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
    st = cascades[0].stats
    result = ScenarioResult(
        id=scenario["id"], title=scenario["title"], expected=scenario["fire"], fired=bool(sink.events),
        fire_times=[e.fired_at for e in sink.events], duration_s=clock.now(), frames=st.frames,
        t0_passed=st.t0_passed, t1_decided=st.t1_decided, t2_calls=st.t2_calls,
        confirm_suppressed=st.confirm_suppressed, verifier_errors=st.verifier_errors,
        cache_misses=verifier.misses, wall_s=wall, cpu_s=cpu, could=bool(scenario.get("could")))
    if scenario.get("injection") and sink.events:
        msg = build_fire_message(program, sink.events[0])
        result.injection_ok = f"Your instruction for this order, fixed when it was created: {json.dumps(order['action']['instruction'])}" in msg
    return result


def run_all(spec: dict | None = None, *, scenarios: list[int] | None = None, cache: T2Cache | None = None,
            embedder="auto", llm=None, include_could: bool = True) -> list[ScenarioResult]:
    spec = spec or load_spec()
    cache = cache or T2Cache()
    if embedder == "auto":
        embedder = maybe_embedder()
    out = []
    for s in spec["scenarios"]:
        if scenarios and s["id"] not in scenarios:
            continue
        if not include_could and s.get("could"):
            continue
        out.append(run_scenario(s, spec, cache, embedder, llm))
    if cache.dirty:
        cache.save()
        cache.dirty = False
    return out


def summarize(results: list[ScenarioResult]) -> dict:
    tp = sum(r.fired and r.expected for r in results)
    fp = sum(r.fired and not r.expected for r in results)
    fn = sum(not r.fired and r.expected for r in results)
    calls, polling, video = (sum(getattr(r, k) for r in results) for k in ("t2_calls", "polling_calls", "duration_s"))
    return {"precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
            "scenarios": len(results), "correct": sum(r.correct for r in results),
            "vision_calls": calls, "polling_calls": polling, "video_seconds": video,
            "cache_misses": sum(r.cache_misses for r in results),
            "frames_per_cpu_second": (sum(r.frames for r in results) / max(1e-9, sum(r.cpu_s for r in results)))}


def report(results: list[ScenarioResult]) -> str:
    lines = ["| # | Scenario | Expect | Got | First fire | Frames | T0 | T1 | Vision calls | Polling calls | |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        mark = "ok" if r.correct else "WRONG"
        if r.cache_misses:
            mark += f" ({r.cache_misses} cache misses)"
        ff = f"{r.fire_times[0]:.1f}s" if r.fire_times else "-"
        lines.append(f"| {r.id} | {r.title} | {'fire' if r.expected else 'quiet'} | {'fired' if r.fired else 'quiet'} | "
                     f"{ff} | {r.frames} | {r.t0_passed} | {r.t1_decided} | {r.t2_calls} | {r.polling_calls} | {mark} |")
    s = summarize(results)
    per_day = lambda a, b: f"{a / max(1.0, b) * 86400:,.0f}"
    lines += ["", f"Correct: {s['correct']}/{s['scenarios']}.  Precision: {s['precision']}  Recall: {s['recall']}",
              f"Vision calls: {s['vision_calls']} over {s['video_seconds']:.0f} s of video "
              f"({per_day(s['vision_calls'], s['video_seconds'])} per day at this rate) versus {s['polling_calls']} "
              f"for polling every {POLL_INTERVAL:g} s ({per_day(s['polling_calls'], s['video_seconds'])} per day).",
              f"Engine speed: {s['frames_per_cpu_second']:.0f} frames per CPU second (T1 CLIP on this machine)."]
    for r in results:
        if r.injection_ok is not None:
            lines.append(f"Injection scenario {r.id}: fired action unchanged = {r.injection_ok}")
    return "\n".join(lines)
