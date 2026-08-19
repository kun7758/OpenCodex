"""Agent 核心运行时中的取消控制模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any


class CancellationToken:
    """取消控制令牌，用于在运行时链路中传播取消信号。

    CancellationToken实现了协作式取消机制：
    - 通过cancel()方法触发取消，设置内部事件标志
    - 通过cancelled属性检查是否已取消
    - 通过wait()异步等待取消事件
    - 通过raise_if_cancelled()在取消时抛出CancelledError
    - 支持注册回调函数，在取消时被调用

    典型用法：
        token = CancellationToken()
        token.add_callback(lambda reason: print(f"Cancelled: {reason}"))
        # 在异步操作中检查token.cancelled或调用token.raise_if_cancelled()
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()  # 内部事件，用于信号通知
        self._reason = "Run cancelled"  # 默认取消原因
        self._callbacks: list[Callable[[str], None]] = []  # 取消时触发的回调函数列表

    @property
    def cancelled(self) -> bool:
        """检查是否已触发取消。"""
        return self._event.is_set()

    @property
    def reason(self) -> str:
        """获取取消原因。"""
        return self._reason

    def cancel(self, reason: str = "Run cancelled") -> bool:
        """触发取消操作。

        参数：
            reason: 触发取消的原因描述。

        返回：
            是否成功触发取消（首次取消返回True，重复取消返回False）。

        说明：
            触发取消后会依次调用所有已注册的回调函数，并清空回调列表。
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
        """注册取消回调函数。如果已取消，回调会立即执行。

        参数：
            callback: 取消时调用的回调函数，接收取消原因字符串参数。

        返回：
            取消注册的函数，调用后移除该回调。
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
        """异步等待取消事件触发。阻塞直到cancel()被调用。

        返回：
            取消原因字符串。
        """
        await self._event.wait()
        return self.reason

    def raise_if_cancelled(self) -> None:
        """检查是否已取消，如果是则抛出CancelledError。常用于在长操作中插入取消检查点。"""
        if self.cancelled:
            raise asyncio.CancelledError(self.reason)


@dataclass
class RunHandle:
    """运行句柄，用于管理和跟踪一次Agent运行任务。

    RunHandle封装了一次运行的完整生命周期控制能力，包括：
    - 唯一标识（run_id）
    - 取消控制（token）
    - 异步任务引用（task）

    典型用法：
        run_handle = RunHandle(run_id="xxx")
        # 启动任务后赋值给run_handle.task
        # 通过run_handle.cancel()取消运行
        # 通过run_handle.wait()等待完成
    """

    run_id: str  # 运行的唯一标识符，用于区分不同的运行实例
    token: CancellationToken = field(default_factory=CancellationToken)  # 取消令牌，传播取消信号到整个运行链路
    task: asyncio.Task[Any] | None = None  # 异步任务对象，None表示任务尚未启动或已结束

    @property
    def done(self) -> bool:
        """检查任务是否已完成。任务为None（未启动）或已完成时返回True。"""
        return self.task is None or self.task.done()

    def cancel(self, reason: str = "Run cancelled") -> bool:
        """取消当前运行。同时触发取消令牌和异步任务的取消。

        参数：
            reason: 取消原因，会传播到CancelledError。

        返回：
            是否成功触发取消（首次取消返回True，重复取消返回False）。
        """
        changed = self.token.cancel(reason)
        if self.task is not None and not self.task.done():
            self.task.cancel(reason)
            return True
        return changed

    async def wait(self) -> Any:
        """等待任务完成并返回结果。任务为None时直接返回None。"""
        if self.task is None:
            return None
        return await self.task
