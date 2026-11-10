"""`/orders` in any chat: status of the camera and every order. Read-only; arming goes through the so_* tools."""

from __future__ import annotations
from .pane_data import summary_text
from .paths import Layout, data_dir


def handle_orders(raw_args: str = "") -> str:
    return summary_text(Layout(data_dir()))


def register_commands(ctx) -> None:
    ctx.register_command("orders", handler=handle_orders, description="Show Standing Orders camera state and orders")
