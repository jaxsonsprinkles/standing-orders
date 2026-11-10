"""Frame loop. step() is deterministic given its inputs, so replay equals live.

The Runner drives every armed order from one frame source. Per order it keeps a Cascade (predicate
values) and a RuleEvaluator (time logic). Time comes only from the injected clock, which follows the
frame timestamps (Clock.observe), so recorded video and live video run identical code."""
import asyncio
from collections import deque
import pathlib
from typing import Callable, Protocol

import cv2

from .cascade import Cascade
from .clock import Clock
from .program import WatchProgram
from .rules import RuleEvaluator
from .types import Event, Tri


class ProgramStore(Protocol):
    def armed(self) -> list[WatchProgram]: ...


class StaticStore:
    def __init__(self, programs: list[WatchProgram]):
        self._programs = programs

    def armed(self) -> list[WatchProgram]:
        return list(self._programs)


class ListSink:
    def __init__(self):
        self.events: list[Event] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)


class _OrderState:
    def __init__(self, program: WatchProgram, cascade: Cascade, rules: RuleEvaluator):
        self.program, self.cascade, self.rules = program, cascade, rules
        self.quiet_until = float("-inf")  # cooldown


class Runner:
    def __init__(self, source, program_store: ProgramStore, cascade_factory: Callable[[WatchProgram], Cascade],
                 clock: Clock, sink, *, kind: str = "live", hysteresis_seconds: float = 0.0,
                 recorder=None, clip_dir: str | None = None, clip_seconds: float = 20.0,
                 confirm: bool = True, confirm_frames: int = 3, confirm_window_seconds: float = 6.0,
                 confirm_quiet_seconds: float = 15.0):
        """recorder: optional RingBuffer fed every frame; with clip_dir, a fired event gets a clip
        of the last clip_seconds. sink needs emit(event).
        confirm: before a fire is emitted, cascade.confirm() re-checks `confirm_frames` frames spread over
        the last `confirm_window_seconds`; anything but TRUE suppresses the event and silences the order
        for `confirm_quiet_seconds`."""
        self.source, self.program_store, self.cascade_factory = source, program_store, cascade_factory
        self.clock, self.sink, self.kind, self.hysteresis = clock, sink, kind, hysteresis_seconds
        self.recorder, self.clip_dir, self.clip_seconds = recorder, clip_dir, clip_seconds
        self.confirm, self.confirm_frames = confirm, max(1, confirm_frames)
        self.confirm_window, self.confirm_quiet = confirm_window_seconds, confirm_quiet_seconds
        self._recent: deque = deque(maxlen=64)
        self.ended = False
        self._states: dict[str, _OrderState] = {}
        self._loop = asyncio.new_event_loop()

    def close(self) -> None:
        self._loop.close()

    def _sync_orders(self) -> None:
        armed = {p.id: p for p in self.program_store.armed()}
        for oid in list(self._states):
            if oid not in armed or self._states[oid].program != armed[oid]:
                del self._states[oid]  # disarmed or edited: start clean
        for oid, program in armed.items():
            if oid not in self._states:
                self._states[oid] = _OrderState(
                    program, self.cascade_factory(program),
                    RuleEvaluator(program.rule, self.clock, self.hysteresis))

    def _confirm_frames(self, now: float) -> list:
        """`confirm_frames` frames evenly spread over the recent window (fewer if there are fewer)."""
        recent = [f for f in self._recent if f.timestamp >= now - self.confirm_window]
        if len(recent) <= self.confirm_frames:
            return recent
        last = len(recent) - 1
        return [recent[round(i * last / (self.confirm_frames - 1))] for i in range(self.confirm_frames)] \
            if self.confirm_frames > 1 else [recent[-1]]

    def _confirmed(self, st: "_OrderState", now: float) -> bool:
        try:
            verdict = self._loop.run_until_complete(st.cascade.confirm(self._confirm_frames(now)))
        except Exception:  # fail closed
            return False
        return verdict.value is Tri.TRUE

    def _clip(self, event: Event, now: float) -> str | None:
        if self.recorder is None or self.clip_dir is None:
            return None
        from ..storage.clips import export_clip  # storage is plain files + cv2, no Hermes either
        frames = self.recorder.window(now - self.clip_seconds, now)
        return export_clip(frames, str(pathlib.Path(self.clip_dir) / f"{event.id}.mp4")) if frames else None

    def step(self) -> list[Event]:
        """Read one frame, update predicates, evaluate rules, emit events. [] at end of the source."""
        frame = self.source.read()
        if frame is None:
            self.ended = True
            return []
        self.clock.observe(frame.timestamp)
        now = self.clock.now()
        if self.recorder is not None:
            self.recorder.add(frame)
        self._recent.append(frame)
        self._sync_orders()
        events = []
        for oid, st in self._states.items():
            values = self._loop.run_until_complete(st.cascade.update(frame))
            if st.rules.update(values) is not Tri.TRUE or now < st.quiet_until:
                continue
            if self.confirm and not self._confirmed(st, now):
                st.quiet_until = now + self.confirm_quiet
                st.cascade.stats.confirm_suppressed += 1
                continue
            true_verdicts = [v for v in st.cascade.verdicts.values() if v.value is Tri.TRUE]
            event = Event(
                order_id=oid, fired_at=now, kind=self.kind, id=f"{oid}@{now:.3f}",
                reason="; ".join(v.reason for v in true_verdicts if v.reason) or st.program.text,
                confidence=min((v.confidence for v in true_verdicts), default=0.0),
                predicates=dict(values))
            try:
                event.clip_path = self._clip(event, now)
            except (OSError, ValueError, cv2.error):
                event.clip_path = None  # a clip problem must not swallow the firing
            st.quiet_until = now + st.program.cooldown_seconds
            self.sink.emit(event)
            st.cascade.stats.fires += 1
            events.append(event)
        return events

    def run(self) -> list[Event]:
        """Step until the source ends. For recorded video; live runs call step() from the engine thread."""
        out = []
        while not self.ended:
            out += self.step()
        return out
