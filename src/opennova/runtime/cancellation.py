"""Agent 核心运行时中的取消控制模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any


class CancellationToken:
    """封装取消控制Token相关的状态和操作，使调用方通过稳定接口使用该能力。"""

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._reason = "Run cancelled"
        self._callbacks: list[Callable[[str], None]] = []

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    def cancel(self, reason: str = "Run cancelled") -> bool:
        """处理取消，并按照当前组件的约定返回结果。

        参数：
            reason: 触发当前状态变化或操作的原因。

        返回：
            表示条件是否成立。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        if self.cancelled:
            return False
        self._reason = reason
        self._event.set()
        callbacks = tuple(self._callbacks)
        self._callbacks.clear()
        for callback in callbacks:
            with suppress(Exception):
                callback(reason)
        return True

    def add_callback(self, callback: Callable[[str], None]) -> Callable[[], None]:
        """添加`add_callback`，必要时执行去重或容量检查。

        参数：
            callback: 在对应事件发生时调用的回调函数。

        返回：
            `Callable[[], None]` 类型的处理结果。
        """
        if self.cancelled:
            callback(self.reason)
            return lambda: None
        self._callbacks.append(callback)

        def unsubscribe() -> None:
            with suppress(ValueError):
                self._callbacks.remove(callback)

        return unsubscribe

    async def wait(self) -> str:
        await self._event.wait()
        return self.reason

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise asyncio.CancelledError(self.reason)


@dataclass
class RunHandle:
    """保存运行处理所需的结构化数据，主要包含 `run_id`、`token`、`task` 字段，便于在组件之间传递或持久化。"""

    run_id: str
    token: CancellationToken = field(default_factory=CancellationToken)
    task: asyncio.Task[Any] | None = None

    @property
    def done(self) -> bool:
        return self.task is None or self.task.done()

    def cancel(self, reason: str = "Run cancelled") -> bool:
        changed = self.token.cancel(reason)
        if self.task is not None and not self.task.done():
            self.task.cancel(reason)
            return True
        return changed

    async def wait(self) -> Any:
        if self.task is None:
            return None
        return await self.task
