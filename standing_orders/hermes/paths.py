"""Where Standing Orders keeps its files. One directory, shared by the gateway, tool handlers and the pane."""
import json
import pathlib
import time


def data_dir() -> pathlib.Path:
    from hermes_constants import get_hermes_home
    d = get_hermes_home() / "plugin-data" / "standing-orders"
    d.mkdir(parents=True, exist_ok=True)
    return d


class Layout:
    def __init__(self, root: pathlib.Path):
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = str(self.root / "so.db")
        self.ring = self.root / "ring"
        self.clips = self.root / "clips"
        self.status_file = self.root / "status.json"
        self.lock_file = self.root / "engine.lock"

    def write_status(self, state: str, detail: str = "", **extra) -> None:
        """Engine thread -> tools. state: starting | running | unavailable | stopped."""
        tmp = self.status_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"state": state, "detail": detail, "updated_at": time.time(), **extra}))
        tmp.replace(self.status_file)

    def read_status(self) -> dict:
        try:
            return json.loads(self.status_file.read_text())
        except (OSError, ValueError):
            return {"state": "not started", "detail": "the gateway has not started the camera engine", "updated_at": None}

    def newest_frame(self) -> pathlib.Path | None:
        frames = sorted(self.ring.glob("*.jpg")) if self.ring.exists() else []
        return frames[-1] if frames else None
