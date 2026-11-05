"""Standing Orders: Hermes plugin entry point. Keep imports light; adapter loads lazily."""


def register(ctx):
    from .standing_orders.hermes.adapter import register_platform
    from .standing_orders.hermes.tools import register_tools

    register_platform(ctx)
    register_tools(ctx)
