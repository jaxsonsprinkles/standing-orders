"""T2 and Confirm via ctx.llm. Camera text is quoted data, never instructions."""
from ..engine.budget import VisionBudget
from ..engine.program import PredicateSpec
from ..engine.types import Verdict
from .compiler import StructuredLlm


class LlmVerifier:
    """Satisfies llm.verifier.Verifier. Spends one budget unit per check(); budget exhausted or any
    LLM error/unparseable answer -> Tri.UNKNOWN verdicts (never raises, never TRUE)."""

    def __init__(self, llm: StructuredLlm, budget: VisionBudget | None = None):
        raise NotImplementedError

    async def check(self, crops: list[bytes], predicates: dict[str, PredicateSpec]) -> dict[str, Verdict]:
        raise NotImplementedError
