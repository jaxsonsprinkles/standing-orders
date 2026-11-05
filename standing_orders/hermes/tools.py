"""Agent tools. Day 1 stub: so_status only."""
import json


def _so_status(args: dict, **kwargs) -> str:
    return json.dumps({"camera": "not started", "orders": 0})


def register_tools(ctx) -> None:
    ctx.register_tool(
        name="so_status",
        toolset="standing_orders",
        schema={
            "name": "so_status",
            "description": "Report Standing Orders camera state, counters and vision budget use.",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=_so_status,
    )
