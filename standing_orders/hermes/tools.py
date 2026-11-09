"""Agent tools. Handlers run in whichever process the agent runs in (CLI, gateway, cron), so SQLite plus
files in the plugin data dir are the only shared state; nothing here talks to the engine thread directly."""
import asyncio
import concurrent.futures
import importlib
import inspect
import json
import time
from typing import Any

from ..engine.program import ProgramError
from ..llm.compiler import compile_order, plan_text
from ..llm.llm_verifier import LlmVerifier
from ..storage.db import connect
from ..storage.orders import OrderRepo
from ..storage.ringbuffer import RingBuffer
from .paths import Layout, data_dir

TOOL_NAMES = ("so_create_order", "so_arm", "so_disarm", "so_list_orders", "so_status", "so_backtest", "so_correct")
TOOLSET = "standing_orders"


def run_sync(coro) -> Any:
    """Run a coroutine to completion from sync code, even if this thread already has a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _engine_fn(name: str, modules: tuple[str, ...]):
    """Find an engine function by name. Engine modules are owned by another stream; absent -> None."""
    for mod in modules:
        try:
            fn = getattr(importlib.import_module(f"standing_orders.engine.{mod}"), name, None)
        except ImportError:
            continue
        if fn is not None:
            return fn
    return None


def _call_flex(fn, **candidates):
    """Call fn passing only the keyword arguments it declares (engine signatures may evolve)."""
    params = inspect.signature(fn).parameters
    kwargs = {k: v for k, v in candidates.items() if k in params}
    missing = [n for n, p in params.items() if p.default is p.empty and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
               and n not in kwargs]
    if missing:
        raise NotImplementedError(f"{fn.__name__} needs arguments this tool cannot supply: {', '.join(missing)}")
    result = fn(**kwargs)
    return run_sync(result) if inspect.isawaitable(result) else result


def _err(message: str, **extra) -> str:
    return json.dumps({"error": message, **extra})


class StandingOrdersTools:
    def __init__(self, llm, layout: Layout | None = None, *, layout_fn=None, time_fn=time.time):
        self.llm, self._layout, self._layout_fn, self._time = llm, layout, layout_fn, time_fn

    @property
    def layout(self) -> Layout:
        if self._layout is None:
            self._layout = Layout(self._layout_fn() if self._layout_fn else data_dir())
        return self._layout

    def repo(self) -> OrderRepo:
        return OrderRepo(connect(self.layout.db), time_fn=self._time)

    # so_create_order
    def create_order(self, args: dict, **kw) -> str:
        text = (args.get("text") or "").strip()
        if not text:
            return _err("text is required: describe in one sentence what to watch for")
        frame = args.get("reference_frame_path")
        path = self.layout.newest_frame() if not frame else frame
        if path is None:
            return _err("no reference frame: the camera has produced no frames yet. Make sure the gateway is "
                        "running with camera access (so_status shows why), or pass reference_frame_path.")
        try:
            jpeg = open(path, "rb").read()
        except OSError as e:
            return _err(f"cannot read reference frame: {e}")
        try:
            program = run_sync(compile_order(self.llm, text, jpeg))
        except ProgramError as e:
            return _err(f"could not compile that order: {e}")
        except Exception as e:
            return _err(f"compiler unavailable: {type(e).__name__}: {e}")
        self.repo().create(program, "draft")
        return json.dumps({
            "order_id": program.id, "status": "draft", "plan": plan_text(program),
            "next": "Show the user this plan. Run so_backtest to test it on recent footage, and arm it with so_arm only "
                    "after the user agrees."})

    # so_arm / so_disarm
    def _set(self, args: dict, status: str) -> str:
        oid = args.get("order_id")
        repo = self.repo()
        current = repo.status(oid) if oid else None
        if current is None:
            return _err(f"no such order: {oid!r}", known=[o["id"] for o in repo.list()])
        repo.set_status(oid, status)
        return json.dumps({"order_id": oid, "status": status, "was": current})

    def arm(self, args: dict, **kw) -> str:
        return self._set(args, "armed")

    def disarm(self, args: dict, **kw) -> str:
        return self._set(args, "paused")

    # so_list_orders
    def list_orders(self, args: dict, **kw) -> str:
        repo = self.repo()
        orders = []
        for o in repo.list():
            events = repo.events(o["id"], kind="live", limit=1)
            orders.append({**o, "last_fired_at": events[0]["fired_at"] if events else None})
        return json.dumps({"orders": orders})

    # so_status
    def status(self, args: dict, **kw) -> str:
        repo, now = self.repo(), self._time()
        since = now - 3600
        orders = []
        for o in repo.list():
            program = repo.get(o["id"])
            c = repo.counters(o["id"], since=since)
            orders.append({"id": o["id"], "status": o["status"], "text": o["text"],
                           "vision_calls_last_hour": c["t2_calls"],
                           "vision_budget_per_hour": program.vision_budget_per_hour if program else None,
                           "fires_last_hour": c["fires"]})
        return json.dumps({"camera": self.layout.read_status(), "counters_last_hour": repo.counters(since=since),
                           "orders": orders})

    # so_backtest
    def backtest(self, args: dict, **kw) -> str:
        oid = args.get("order_id")
        repo = self.repo()
        program = repo.get(oid) if oid else None
        if program is None:
            return _err(f"no such order: {oid!r}")
        fn = _engine_fn("backtest", ("backtest", "replay", "runner"))
        if fn is None:
            return _err("backtest is not available in this build of the engine yet")
        minutes = float(args.get("minutes") or 10)
        ring = RingBuffer(str(self.layout.ring))
        try:
            result = _call_flex(fn, program=program, ring=ring, ringbuffer=ring, recorder=ring,
                                verifier=LlmVerifier(self.llm), repo=repo, minutes=minutes,
                                seconds=minutes * 60, clip_dir=str(self.layout.clips), clips_dir=str(self.layout.clips))
        except NotImplementedError as e:
            return _err(f"backtest is not available yet: {e}")
        except Exception as e:
            return _err(f"backtest failed: {type(e).__name__}: {e}")
        get = (lambda k, d=None: result.get(k, d)) if isinstance(result, dict) else (lambda k, d=None: getattr(result, k, d))
        events = get("events", []) or []
        return json.dumps({
            "order_id": oid, "window_minutes": minutes,
            "would_have_fired": [{"id": e.id, "at": e.fired_at, "reason": e.reason, "confidence": e.confidence,
                                  "clip": e.clip_path} if not isinstance(e, dict) else e for e in events],
            "frames_checked": get("frames_checked", get("frames")),
            "unresolved": get("unresolved", get("unknown")),
            "note": "Backtests never send messages. Share clips with MEDIA:<path> only if the user asks."})

    # so_correct
    def correct(self, args: dict, **kw) -> str:
        eid, label, note = args.get("event_id"), args.get("label"), args.get("note") or ""
        repo = self.repo()
        try:
            repo.label_event(eid, label, note)
        except (ValueError, KeyError) as e:
            return _err(str(e).strip("'"))
        applied = "label saved"
        fn = _engine_fn("apply_correction", ("corrections", "correction", "learning", "cascade"))
        if fn is not None:
            try:
                _call_flex(fn, event=repo.get_event(eid), event_id=eid, label=label, note=note, repo=repo,
                           ring=RingBuffer(str(self.layout.ring)))
                applied = "label saved and applied to this order's examples"
            except Exception as e:
                applied = f"label saved; learning step failed: {type(e).__name__}: {e}"
        else:
            applied = "label saved; the learning step is not available in this build yet"
        return json.dumps({"event_id": eid, "label": label, "result": applied})


def _schema(name: str, description: str, properties: dict | None = None, required: list | None = None) -> dict:
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties or {}, "required": required or []}}


_ORDER_ID = {"order_id": {"type": "string", "description": "Order id from so_create_order or so_list_orders"}}

SCHEMAS = {
    "so_create_order": _schema(
        "so_create_order",
        "Compile a plain-English camera instruction into a draft Standing Order and return a plan (what it watches, "
        "what fires it, max vision calls per hour). Does not arm it. Describe one observable situation, not an action; "
        "the agent action is always 'tell the user' and cannot be chosen by the camera.",
        {"text": {"type": "string", "description": "One sentence, e.g. 'tell me if I leave my mug on the desk for 10 minutes'"},
         "reference_frame_path": {"type": "string", "description": "Optional JPEG to compile against; default is the latest camera frame"}},
        ["text"]),
    "so_arm": _schema("so_arm", "Start watching for an order. Only after the user has seen and approved its plan.", _ORDER_ID, ["order_id"]),
    "so_disarm": _schema("so_disarm", "Pause an order so it stops watching and spending vision calls.", _ORDER_ID, ["order_id"]),
    "so_list_orders": _schema("so_list_orders", "List all Standing Orders with status and last fire time."),
    "so_status": _schema("so_status", "Report Standing Orders camera state (running or unavailable and why), counters and vision budget use per order."),
    "so_backtest": _schema(
        "so_backtest", "Replay an order over recent recorded camera footage and report when it would have fired, "
        "frames checked and how many were unresolved. Sends nothing.",
        {**_ORDER_ID, "minutes": {"type": "number", "description": "How far back to replay (default 10)"}}, ["order_id"]),
    "so_correct": _schema(
        "so_correct", "Mark a fired event as correct or wrong so the order improves.",
        {"event_id": {"type": "string"}, "label": {"type": "string", "enum": ["correct", "wrong"]},
         "note": {"type": "string", "description": "What was actually in view (optional)"}}, ["event_id", "label"]),
}


def register_tools(ctx) -> None:
    tools = StandingOrdersTools(ctx.llm)
    handlers = {"so_create_order": tools.create_order, "so_arm": tools.arm, "so_disarm": tools.disarm,
                "so_list_orders": tools.list_orders, "so_status": tools.status,
                "so_backtest": tools.backtest, "so_correct": tools.correct}
    for name in TOOL_NAMES:
        ctx.register_tool(name=name, toolset=TOOLSET, schema=SCHEMAS[name], handler=handlers[name],
                          description=SCHEMAS[name]["description"])
