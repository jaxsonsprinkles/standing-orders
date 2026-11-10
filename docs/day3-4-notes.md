# Day 3 + 4 notes (Nov 9, 2026)

Built (two workers merged, then glue and review fixes in the main thread): VisionBudget, CLIP T1 (pinned
ONNX, checksum-verified download), heartbeat, Confirm, CachedVerifier, backtest, corrections with replay proof,
OrderRepo, compiler with one repair, LlmVerifier, `so_*` tools, engine thread, fire message, skill.
151 tests pass, `hermes plugins doctor .` passes (7 tools). Details: docs/day3-engine.md, docs/day3-hermes.md.

## Exit gates
- Day 4 gate (offline, synthetic): leave-mug fires, bowl false-fires, correction flips the bowl and the mug
  still fires; quiet 300 frames cost 1 verifier call (0 with T1); exhausted budget never fires. MET in tests.
- Day 3 gate (typed sentence -> armed order -> fires on live webcam -> clip in chat): NOT run. Needs your Mac,
  a gateway started from an authorized Terminal, `download_models`, and a real model behind `ctx.llm`.

## Review fixes applied
UNKNOWN from a swallowing verifier is retried; saved examples beat the heartbeat; counters and budget survive
cascade rebuilds; backtest event ids are `bt:`-prefixed; the action instruction is built from the user's
sentence, never from model output.

## Known gaps
- corrections: no baseline replay, and `_ClipSource` spreads a clip evenly over clip_seconds, so rules with
  `held_for` longer than the clip never fire in replay (a "wrong" label then looks accepted). Verify on real clips.
- T1 margins (cascade.py defaults) and T0 threshold are untuned; no real clips exist.
- LlmVerifier keeps its own hourly cap next to VisionBudget (redundant, harmless).
- Platform toolset restriction needs `platform_toolsets.standing_orders: [standing_orders]` in config.yaml.
- Unverified live: config_schema reaching adapter config, camera in the gateway, delivery of clips on a slow network.
