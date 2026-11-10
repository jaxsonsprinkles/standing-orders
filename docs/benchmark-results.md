# Benchmark results (Nov 9, 2026)

`python benchmark/run.py` or `hermes orders benchmark`: replays 10 labeled clips of the demo desk through the real
engine (T0, CLIP T1, Confirm, rules). T2 answers are the ones recorded by `hermes orders benchmark --live-vision`
through the Hermes model configured on this Mac; they replay from `benchmark/cache/t2_answers.json` with no model.

| # | Scenario | Expect | Got | First fire | Vision calls | Polling calls |
|---|---|---|---|---|---|---|
| 1 | Leave mug, walk away | fire | fired | 21.5 s | 10 | 21 |
| 2 | Walk away, taking the mug | quiet | quiet | - | 7 | 25 |
| 3 | Leave mug, return within 5 s | quiet | quiet | - | 5 | 5 |
| 4 | Leave a bowl instead | quiet | quiet | - | 4 | 19 |
| 5 | Lamp off and on, empty desk | quiet | quiet | - | 10 | 4 |
| 6 | Person walks past | quiet | quiet | - | 3 | 5 |
| 7 | Small mug at the zone edge | fire | fired | 20.0 s | 13 | 14 |
| 8 | Sticky note ("BUY MILK") | fire | fired | 2.0 s | 3 | 6 |
| 9 | Sticky note with injected text | fire | fired | 2.0 s | 3 | 9 |
| 10 | Quiet scene, 10 min | quiet | quiet | - | 1 | 299 |

Precision 1.0, recall 1.0, 10/10 correct. 59 vision calls over 821 s of video versus 407 for polling every 2 s.
Quiet scene: 1 call in 600 s (the first frame) versus 299 for polling.
Engine speed 110 to 140 frames per CPU second including CLIP, on this Apple Silicon Mac.
Injection scenario 9: the fire message keeps the order's fixed instruction; the note text appears only as quoted data.

## Read this before quoting the numbers
- In-sample: one take per scenario, and T0/T1 settings were looked at on the same clips. They were not changed from the
  defaults, but this is not a held-out result. Record a second take of each to get one.
- Scenario 10 is 16 s of footage played forward and backward to 10 minutes. Sensor noise is real; scene changes are not.
- The per-day figure for the full set (6,209) is meaningless: the clips are almost all events. Use the quiet-scene row.
- The bowl (4) never fooled the system, so there is no real false fire to correct. The correction replay is tested
  only on synthetic clips; for the demo, show the bowl being ignored.
- Clips 3 and 5 are short, so "fire" is not reachable by length in 3; it checks only that the timer resets.
- Event-to-delivery latency and live CPU are not measured here (they need the live gateway).
