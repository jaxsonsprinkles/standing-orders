# Day 2 notes (Nov 6, 2026)

Built: rule operators, program parser/validator, FileSource, AvfSource, rolling recording, clip export,
T0 change gate, Runner, minimal Cascade (T0 -> verifier), ScriptedVerifier. 74 tests, `hermes plugins doctor .` passes.

## Exit gate
"Hand-written order fires on a recorded clip with a scripted verifier": MET on synthetic clips
(tests/test_runner_gate.py: leave-mug fires once at t=13.0, take-mug / return-within-5s / quiet never fire).
NOT yet run on real recorded clips (none recorded). Next: record clips, then run scripts/measure_t0.py on them.

## Decisions to know about
- Runner takes a `cascade_factory(program)` (was `cascade`): one Runner drives every armed order from one camera.
  `program_store.armed()` is polled every step. Editing an order rebuilds its state (resets cooldown).
- Clock gained `observe(t)`: the Runner feeds frame timestamps, so replay clocks follow the video; RealClock ignores it.
- `within` shape: `{"within": {"seconds": N, "after": A, "then": B}}`, TRUE while B is TRUE within N s of A rising.
  It does NOT fire on B's absence (use `held_for` + `not`).
- Rule hysteresis exists (`Runner(hysteresis_seconds=...)`) but defaults to 0; tune on clips.
- Verifier contract: `crops[i]` is the crop for the i-th predicate in dict order.
- T0 is grayscale: colour-only changes between equal-brightness objects are invisible to it.
- Synthetic T0 scores: sensor noise <= 0.11, lamp swings 0.002, mug appearing 0.52; threshold 0.25. Real numbers pending.

## Measured / not measured
- Webcam through AvfSource worked from the agent shell earlier (6 frames at 2 fps in 2.7 s, 1080p);
  a later retry returned no frames (camera permission/in-use state), so the live Runner run is unverified.
- Clips encode as H.264; 6 frames at 960 px wide was 135 KB. Delivery is slow on the hotspot; shrink if needed.

## Known gaps (Day 3)
T1 embeddings, heartbeat, vision budget, Confirm, verifier errors (a raising verifier currently propagates),
DB-backed program store, adapter wiring, counters flush.
