"""终端交互层中的`command_dispatch`模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from opennova.cli.commands import SlashCommandRegistry


class SlashCommandDispatcher:
    """封装`SlashCommandDispatcher`相关的状态和操作，使调用方通过稳定接口使用该能力。"""

    def __init__(self, registry: SlashCommandRegistry) -> None:
        self.registry = registry

    async def dispatch(
        self,
        target: Any,
        text: str,
        *,
        on_unknown: Callable[[str], None] | None = None,
    ) -> bool:
        command_text, _, args = text.partition(" ")
        command_name = command_text.lower().replace("_", "-")
        command = self.registry.get(command_name)
        if command is None or not command.handler:
            if on_unknown:
                on_unknown(command_name)
            return False
        handler = getattr(target, command.handler, None)
        if not callable(handler):
            if on_unknown:
                on_unknown(command_name)
            return False
        await handler(args)
        return True
