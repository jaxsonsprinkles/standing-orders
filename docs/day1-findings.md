# Day 1 findings (Nov 5, 2026)

## Environment
- Hermes v0.22.7, Python 3.14.9, Apple Silicon. Gateway already running, no platforms configured.
- Telegram is not set up (`TELEGRAM_*` lines in `~/.hermes/.env` are commented template).

## Resolved
- Layout: `plugin.yaml` + root `__init__.py` with `register(ctx)`; `kind: platform` with `provides_tools`.
  `hermes plugins doctor .` passes. `hermes plugins validate . --install-deps` passes (deps resolve on 3.14).
- Gateway platform: `trusted_inbound=True` (camera has no human sender; skips allowlists and pairing).
  Allowed for non-bundled plugins unless core ships a platform of the same name.
- Delivery design: adapter `send()` forwards the turn reply with `ctx.dispatch_tool("send_message",
  {"action": "send", "target": "telegram", "message": ...})`. `send_message` resolves the home channel and
  extracts `MEDIA:` lines. Called via `asyncio.to_thread` so it cannot block the gateway loop. UNTESTED live.
- Embedding model: `Xenova/clip-vit-base-patch32` (card license `mit`; upstream OpenAI CLIP is MIT).
  immich-app/ViT-B-32__openai has no license tag, so it is the fallback only.
  Revision `6b7c523703ae26e39be4b03774966f8571c7f523`. Files (sha256, LFS oid):
  - `onnx/vision_model_quantized.onnx` 89.1 MB `583fd1110a514667812fee7d684952aaf82a99b959760c8d7dca7e0ab9839299`
  - `onnx/text_model_quantized.onnx` 64.5 MB `73baab855d406190da9faa498cfedf65f15cf309f4cc7385b7b032e6d08e5c3a`
  Plus tokenizer/preprocessor json from the same revision. Download at setup, verify checksum, never commit (rule 17).

## Blocked on user
- Camera: option A fails in the agent shell (`not authorized to capture video`). Run
  `python scripts/probe_camera.py` from Terminal (approve the prompt) to test A and B.
- Telegram: create bot with BotFather, set `TELEGRAM_BOT_TOKEN`, `TELEGRAM_ALLOWED_USERS`,
  `TELEGRAM_HOME_CHANNEL`, restart the gateway.
- Live gate: enable plugin and fire the fake event (needs your OK; it touches the running gateway).
