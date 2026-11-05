from typing import Protocol

from ..engine.program import PredicateSpec
from ..engine.types import Verdict


class Verifier(Protocol):
    """T2 and Confirm. Implementations: LlmVerifier (ctx.llm), CachedVerifier (replay),
    ScriptedVerifier (tests)."""

    async def check(self, crops: list[bytes], predicates: dict[str, PredicateSpec]) -> dict[str, Verdict]: ...
