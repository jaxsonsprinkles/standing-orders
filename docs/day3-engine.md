# Day 3/4 engine notes

Offline only: synthetic video (tests/helpers.py, tests/helpers_d3.py), FakeEmbedder, ScriptedVerifier.
No camera, no real clips. Run: `/tmp/so-venv/bin/python -m pytest -q` (116 pass, ~9 s).

## Built (one commit each)

1. `engine/budget.py` VisionBudget (sliding 3600 s on the injected clock; `CallCap` = fixed total, used by
   backtest). Cascade: a raising verifier, a malformed answer or a refused budget spend makes the
   affected predicates UNKNOWN (never raises, never fires). Predicates left UNKNOWN by a failure are
   retried every 5 s while the scene is otherwise still. `CascadeStats`: frames, t0_passed, t1_decided,
   t2_calls, fires, budget_exhausted, verifier_errors, unresolved_frames, confirm_suppressed.
2. `engine/embedder.py` ClipOnnxEmbedder (quantized ONNX, CLIP preprocessing, `tokenizers` for text),
   `download_models(model_dir)` (pinned revision, sha256 on both onnx files, bad download deleted),
   `FakeEmbedder`. Models live in `~/.cache/standing-orders-models` (not in git). Needs `onnxruntime`
   and `tokenizers` installed. The real-model test skips when the files are absent.
3. T1 in `Cascade` (`embedder=`, `t1_true_margin`, `t1_false_margin`, `example_sim`, `examples=`).
   Per changed zone: crop embedding vs prompt and negatives; a saved example almost identical to the crop
   (cosine >= `example_sim`) decides first. Margin >= +true -> TRUE, <= -false -> FALSE, else (or no
   negatives, or any embedder error) the verifier decides. T1-decided predicates cost no verifier call.
   Heartbeat: when T0 passes and T1 would decide but the predicate has not had a verifier look for
   `heartbeat_seconds`, T2 looks instead.
4. Confirm. `Cascade.confirm(frames)` re-asks the verifier (never T1) on every TRUE/FALSE predicate, once
   per frame; TRUE only if every frame agrees. `Runner(confirm=True, confirm_frames=3,
   confirm_window_seconds=6, confirm_quiet_seconds=15)` keeps a deque of recent frames and calls it
   before emitting. Not TRUE (disagree, UNKNOWN, budget, error) -> no event, 15 s quiet, cooldown not
   started. Each confirm frame is one call in `t2_calls` and one budget unit.
5. `CachedVerifier` (llm/verifier.py): key (sha256 of crop bytes, sha256 of name+prompt+negatives);
   hit = no inner call; UNKNOWN and errors are never stored. `engine/backtest.py`: Runner + FakeClock,
   kind="backtest", every vision call (confirm too) against `CallCap(max_t2_calls)`;
   `unresolved_frames` counts frames that ended with a predicate left UNKNOWN by the cap.
6. `engine/corrections.py` `apply_correction` + `load_examples(conn, order_id)`. Saves the predicate-zone
   crop of the clip's last frame as a pos/neg example (float32 BLOB), replays the target clip and every
   other order event labeled `correct` with the candidate examples, accepts only if the target flips
   and none break. Conflict leaves DB and files untouched.

## For the adapter (Worker B)

- `Cascade(p, verifier, clock, embedder=emb, budget=VisionBudget(p.vision_budget_per_hour, clock),
  examples=load_examples(conn, p.id))`. Reload (rebuild the Cascade) after an accepted correction.
- Budget is spent by the Cascade. Do not also give the same budget to `LlmVerifier`, or calls count twice.
- `verifier_factory` for `apply_correction` is `clock -> Verifier` (or no-arg); it must be deterministic
  for the same pixels (CachedVerifier over a stored run, or scripted).
- `backtest(program, source, verifier_or_clock_factory, ...)`: the caller starts/stops the source.

## Measured (synthetic)

| scene | verifier calls |
| --- | --- |
| leave-mug, 23 frames, no T1 | 5 (2 cascade + 3 confirm) |
| same, with T1 | 4 (T1 decided 3 predicate checks) |
| quiet, 300 frames, no T1 | 1 |
| quiet, 300 frames, with T1 | 0 |
| backtest leave-mug, 33 frames | 5; cap=1 -> 0 events, unresolved_frames > 0 |

Real CLIP on a laptop CPU: ~22 ms per image embed. Red-circle vs blue-square synthetic image: matching
prompt beats the other by 0.06 to 0.10 cosine; unrelated prompts sit around 0.22. That is why the
default margins are 0.04.

## Gaps

- Margins and `example_sim` (0.04 / 0.04 / 0.92) are guesses from two synthetic images. Not tuned on a
  real camera. FakeEmbedder tests prove the plumbing, not the accuracy of CLIP on mugs.
- Heartbeat only runs when T0 passes. A scene that never changes but is mis-read once by T1 stays
  mis-read. Running it on a timer would cost ~60 calls/hour per order, which exceeds the default budget.
- Correction replay assumes an event clip covers `clip_seconds` evenly. Clips are time-lapses and the
  true spacing is not stored; a clip shorter than the window (fire early after start) is replayed too
  fast. Fix: store frame times with the event, or the window start.
- "Refit thresholds" is implemented as nearest-example override, not a re-estimate of the margins.
- A saved example matches by image similarity to one crop; a different bowl placement may not match.
  Several corrections are needed to generalize.
- The Confirm check needs all frames to agree; one noisy frame suppresses a real fire (15 s later it
  tries again). Cost: 3 vision calls per fire, so a 20/hour budget allows about 6 fires.
- `correct` with no predicate records the label only.
- Backtest does not spend or refund per predicate; one cap unit = one verifier call.
