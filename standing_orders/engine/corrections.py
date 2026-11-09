"""Corrections: save the crop as an example, refit thresholds, replay labeled events, accept only if
the target event flips and no previously correct event breaks."""
from dataclasses import dataclass, field


@dataclass
class CorrectionResult:
    accepted: bool
    flipped: bool                                          # target event no longer fires
    broken: list[str] = field(default_factory=list)        # ids of previously-correct events that broke
    message: str = ""


def apply_correction(conn, order_id: str, event_id: str, label: str, predicate: str | None,
                     note: str, *, embedder, verifier_factory, clip_dir: str) -> CorrectionResult:
    """label: "correct" | "wrong". For "wrong", `predicate` is the predicate that was falsely TRUE
    (None: pick the best candidate). Writes examples/label only if accepted; else leaves state unchanged
    and returns the conflict."""
    raise NotImplementedError
