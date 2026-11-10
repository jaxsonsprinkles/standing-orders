"""Print the T0 score series for a clip, to tune the threshold and check lighting behavior.
Usage: python scripts/measure_t0.py clip.mp4 [x1,y1;x2,y2;...]   (default: whole frame)"""

from __future__ import annotations
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from standing_orders.engine.change_gate import ChangeGate
from standing_orders.engine.program import FRAME_ZONE
from standing_orders.sources.file_source import FileSource

poly = [tuple(map(float, p.split(","))) for p in sys.argv[2].split(";")] if len(sys.argv) > 2 else None
gate = ChangeGate({"region": poly} if poly else {FRAME_ZONE: []})
src = FileSource(sys.argv[1], sample_fps=2.0)
src.start()
passed = total = 0
while (f := src.read()) is not None:
    name = "region" if poly else FRAME_ZONE
    s = gate.score(name, f.image)
    changed = gate.update(f.image)[name]
    total, passed = total + 1, passed + changed
    print(f"t={f.timestamp:7.1f}s  score={'--' if s is None else f'{s:.3f}'}  {'PASS' if changed else ''}")
print(f"{passed}/{total} frames passed T0")
