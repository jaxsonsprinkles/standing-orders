"""Standing Orders: Hermes plugin entry point. Keep imports light; adapter loads lazily."""
from pathlib import Path


def register(ctx):
    from .standing_orders.hermes.adapter import register_platform
    from .standing_orders.hermes.tools import register_tools

    register_platform(ctx)
    register_tools(ctx)
    skill = Path(__file__).parent / "skills" / "standing-orders" / "SKILL.md"
    if skill.exists():
        ctx.register_skill("standing-orders", skill, "Create, test, arm and correct camera Standing Orders")
