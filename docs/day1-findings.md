# Day 1 findings (Nov 5, 2026)

## Environment
- Hermes v0.22.7, Python 3.14.9, Apple Silicon. Gateway already running, no platforms configured.
- Telegram is not set up (`TELEGRAM_*` lines in `~/.hermes/.env` are commented template).

## Resolved
- Layout: `plugin.yaml` + root `__init__.py` with `register(ctx)`; `kind: platform` with `provides_tools`.
  `hermes plugins doctor .` passes. `hermes plugins validate . --install-deps` passes (deps resolve on 3.14).
- Gateway platform: `trusted_inbound=True` (camera has no human sender; skips allowlists and pairing).
  Allowed for non-bundled plugins unless core ships a platform of the same name.
- Delivery (TESTED): `ctx.dispatch_tool("send_message")` does NOT work (`Unknown tool`; no agent-callable
  send_message in v0.22.7). Working path: adapter shells out to the documented `hermes send --to telegram --json`
  with the text on stdin. The gateway strips `MEDIA:` from the reply and calls `send_image_file` /
  `send_document` / `send_video` on our adapter, so those are overridden to forward `MEDIA:<path>` too.
  Live result: fake order -> agent turn (10-26 s) -> Telegram text delivered (rc 0).
- Media upload is flaky on this network (measured ~13 KB/s upload): files of 35-56 KB sometimes succeed,
  39-100 KB and 678 KB sometimes time out ("Timed out"), independent of size. Confirmed by user: personal hotspot, so a network limit, not a plugin bug. Mitigations to
  build: small clips, one retry, and a text-first fallback with the clip path. Test on demo-day network.
- Camera option A works from the agent shell once camera access is granted: 1920x1080 frame in 0.14 s.
- Plugin is symlinked at `~/.hermes/plugins/standing-orders` and enabled; the platform auto-enables via
  `env_enablement_fn`. Spike trigger: write text to `~/.hermes/plugin-data/standing-orders/fire.txt`.
- Embedding model: `Xenova/clip-vit-base-patch32` (card license `mit`; upstream OpenAI CLIP is MIT).
  immich-app/ViT-B-32__openai has no license tag, so it is the fallback only.
  Revision `6b7c523703ae26e39be4b03774966f8571c7f523`. Files (sha256, LFS oid):
  - `onnx/vision_model_quantized.onnx` 89.1 MB `583fd1110a514667812fee7d684952aaf82a99b959760c8d7dca7e0ab9839299`
  - `onnx/text_model_quantized.onnx` 64.5 MB `73baab855d406190da9faa498cfedf65f15cf309f4cc7385b7b032e6d08e5c3a`
  Plus tokenizer/preprocessor json from the same revision. Download at setup, verify checksum, never commit (rule 17).

## Open
- Confirm on your phone: text from fake events arrived; one clip-less image (q50 JPEG) arrived.
- Camera option C (Desktop) not needed unless the gateway launched under launchd lacks permission:
  TEST by capturing a frame from inside the gateway process (next step).

## Camera inside the gateway (TESTED)
- Option A under the launchd-supervised gateway: `PROBE_CAMERA` returns `no frame` (macOS camera permission
  is tied to the launching app; launchd has none). Same code works from an authorized shell (0.14 s).
- Next: option B (stop the launchd service, run `hermes gateway run` from an authorized Terminal), then
  re-run `echo PROBE_CAMERA > ~/.hermes/plugin-data/standing-orders/fire.txt` and read `camera.txt`.
- Fallback: option C (Desktop renderer posts frames to `plugin_api.py`).
