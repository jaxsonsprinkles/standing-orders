# Day 3: Hermes surface and LLM (Worker B)

## Built (unit tested with fakes; NOT verified live)
- `storage/orders.py` OrderRepo: orders, events, hourly counters; `armed()` is the Runner's program store and
  `emit()` its sink. Status changes validated against `ORDER_STATUSES`.
- `llm/compiler.py`: `compile_order(llm, sentence, frame_jpeg)` -> WatchProgram via `acomplete_structured` with the
  reference frame. Identity fields (id, text, source) are set by us, not the model. One repair call with the
  validator's error text; second failure raises ProgramError. `plan_text()` for the user-facing plan.
- `llm/llm_verifier.py`: one structured call per `check()`, zone crops in predicate order. Camera text is declared
  untrusted data in the system prompt. Budget exhausted, any exception or unparseable answer -> `Tri.UNKNOWN`, never
  raises. Has its own rolling-hour call cap as a backstop to A's VisionBudget.
- `hermes/tools.py`: `so_create_order`, `so_arm`, `so_disarm`, `so_list_orders`, `so_status`, `so_backtest`,
  `so_correct`. `plugin.yaml` `provides_tools` equals the registered set (a test enforces it; `hermes plugins doctor .`
  reports 7 tools). Handlers share state only through SQLite + files in `~/.hermes/plugin-data/standing-orders/`
  (`so.db`, `ring/`, `clips/`, `status.json`, `engine.lock`).
- `hermes/engine_thread.py`: one daemon thread per machine, guarded by an `fcntl` lock file. Polls
  `OrderRepo.armed()` every step, records the ring buffer, exports clips, writes events, flushes counters, writes
  `status.json`. Camera failure -> status `unavailable` with the AvfSource message, retried every 30 s; the gateway
  platform stays connected. Fired events call `on_fire(message)`; the adapter schedules `fire()` on the gateway loop
  with `asyncio.run_coroutine_threadsafe`.
- Fire message (`build_fire_message`): order, confidence, camera reason as a JSON-quoted string (whitespace collapsed,
  300 chars), the fixed action instruction, clip path, and an instruction to reply with a `MEDIA:<clip>` line.
- Adapter: PROBE_CAMERA spike and fire.txt watcher removed. Day 1 `hermes send` delivery path kept.
- `skills/standing-orders/SKILL.md` registered with `ctx.register_skill`.

## Assumptions about Worker A's code (check at merge)
- `so_backtest` finds `backtest` in `standing_orders.engine.{backtest,replay,runner}` and `so_correct` finds
  `apply_correction` in `engine.{corrections,correction,learning,cascade}`. They are called through
  `_call_flex` (tools.py), which passes only keyword arguments the function declares from: program, ring/ringbuffer/
  recorder, verifier, repo, minutes, seconds, clip_dir, event, event_id, label, note. If the real signatures differ
  (positional-only, other names) fix the two call sites. A missing function or `NotImplementedError` returns a clear
  `error`, and `so_correct` still saves the label.
- `Cascade(..., budget=)` and `VisionBudget(per_hour, clock)` are wired in `EngineThread._cascade_factory`; if
  VisionBudget's constructor differs the exception is caught and only the LlmVerifier hourly cap applies.
- Counter flush reads `cascade.stats.frames / t0_passed / t2_calls`.

## Gaps
- Restricted toolset: plugins cannot set a platform's toolset at registration. Set it in `config.yaml`, otherwise the
  camera session gets the default `hermes-standing_orders` resolution:
  ```yaml
  platform_toolsets:
    standing_orders: [standing_orders]   # only so_* tools; no terminal, web or file tools
  ```
  Until that is set, the fire message tells the agent to only reply, which is a prompt-level guard, not a hard one.
- Config values (`camera_index`, `sample_fps`, ...) are read from `config.extra`; whether `config_schema` values land
  there was not confirmed. Defaults apply otherwise.
- `so_create_order` compiles against the newest ring-buffer frame, so the gateway must have been running with camera
  access (or pass `reference_frame_path`).
- Tool handlers run `ctx.llm` in a throwaway event loop (`run_sync`); the trust gate was not exercised against a real
  provider.
- Not verified: a live camera session under the gateway (permission follows the launching app, see day1-findings),
  a real LLM compile, `hermes send` delivery of a real clip, fire->agent turn->reply round trip, the adapter importing
  under a running gateway (doctor only checks import and registration).
- Reply routing: the agent's reply goes through the adapter's `send()` -> `hermes send --to <deliver_to>`.
  `deliver_to` default is `telegram`.
