"""Order compiler: sentence + reference frame -> validated WatchProgram. Never generates code."""
from typing import Protocol

from ..engine.program import WatchProgram


class StructuredLlm(Protocol):
    """Thin async seam over ctx.llm.acomplete_structured so tests inject a fake."""
    async def structured(self, instructions: str, text: str, images: list[bytes], schema: dict,
                         purpose: str) -> dict | None: ...


async def compile_order(llm: StructuredLlm, text: str, reference_jpeg: bytes | None, *,
                        order_id: str, source: str = "cam0") -> WatchProgram:
    """One LLM call, parse_program + validate_program; on ProgramError one repair call with the error.
    Second failure raises ProgramError. Returns an unarmed, validated program."""
    raise NotImplementedError


def plan_text(program: WatchProgram) -> str:
    """Plain-language plan shown before arming: what it watches, what fires it, what it can cost."""
    raise NotImplementedError
