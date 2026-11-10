"""`hermes orders ...`: setup, doctor, benchmark, purge. Registered with ctx.register_cli_command."""

from __future__ import annotations
import shutil
import sys

from ..benchmark import POLL_INTERVAL, report, run_all, summarize
from ..engine.embedder import DEFAULT_MODEL_DIR, download_models, models_present
from .paths import Layout, data_dir


def _setup(parser) -> None:
    sub = parser.add_subparsers(dest="orders_cmd")
    sub.add_parser("setup", help="download the CLIP weights used by the local check (checksum verified)")
    doc = sub.add_parser("doctor", help="check models, data directory and camera permission")
    doc.add_argument("--camera", action="store_true", help="also try to read one frame from the camera")
    bench = sub.add_parser("benchmark", help="replay the labeled clips and report precision, recall and vision calls")
    bench.add_argument("--scenario", type=int, nargs="*", help="only these scenario numbers")
    bench.add_argument("--skip-could", action="store_true", help="skip the sticky-note scenarios")
    bench.add_argument("--live-vision", action="store_true",
                       help="ask your model for any answer missing from the cache and record it")
    purge = sub.add_parser("purge", help="delete the recording, clips, orders and events")
    purge.add_argument("--yes", action="store_true", help="do not ask")


def _doctor(args) -> int:
    ok = True
    print(f"models: {'present' if models_present(DEFAULT_MODEL_DIR) else 'MISSING (run: hermes orders setup)'} "
          f"({DEFAULT_MODEL_DIR})")
    ok &= models_present(DEFAULT_MODEL_DIR)
    print(f"data dir: {data_dir()}")
    print(f"engine: {Layout(data_dir()).read_status()}")
    if args.camera:
        from ..sources.avf_source import AvfSource, CameraError
        src = AvfSource(warmup_seconds=5.0)
        try:
            src.start()
            frame = src.read()
            print(f"camera: ok, frame {None if frame is None else frame.image.shape}")
        except CameraError as e:
            print(f"camera: FAILED: {e}")
            ok = False
        finally:
            src.stop()
    return 0 if ok else 1


def _benchmark(ctx, args) -> int:
    llm = ctx.llm if args.live_vision else None
    results = run_all(scenarios=args.scenario, include_could=not args.skip_could, llm=llm)
    print(report(results))
    misses = summarize(results)["cache_misses"]
    if misses:
        print(f"\n{misses} T2 questions were not in the cache and were answered UNKNOWN. "
              "Re-run with --live-vision to record them.", file=sys.stderr)
    return 0 if all(r.correct for r in results) and not misses else 1


def _purge(args) -> int:
    layout = Layout(data_dir())
    if not args.yes and input(f"Delete everything under {layout.root}? Type 'yes': ").strip() != "yes":
        print("cancelled")
        return 1
    for path in (layout.ring, layout.clips):
        shutil.rmtree(path, ignore_errors=True)
    for name in ("so.db", "so.db-wal", "so.db-shm", "status.json"):
        (layout.root / name).unlink(missing_ok=True)
    print("purged")
    return 0


def register_cli(ctx) -> None:
    def handler(args) -> int:
        cmd = getattr(args, "orders_cmd", None)
        if cmd == "setup":
            download_models(DEFAULT_MODEL_DIR)
            print(f"models ready in {DEFAULT_MODEL_DIR}")
            return 0
        return {"doctor": lambda: _doctor(args), "benchmark": lambda: _benchmark(ctx, args),
                "purge": lambda: _purge(args)}.get(cmd, lambda: (print("usage: hermes orders {setup,doctor,benchmark,purge}"), 2)[1])()

    ctx.register_cli_command(name="orders", help="Standing Orders: setup, doctor, benchmark, purge",
                             setup_fn=_setup, handler_fn=handler,
                             description="Camera Standing Orders: setup, doctor, benchmark and purge.")
