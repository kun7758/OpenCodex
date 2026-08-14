"""Agent 核心运行时中的`event_bus`模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from contextlib import suppress
from threading import RLock
from typing import Any


class RuntimeEventBus:
    """封装`RuntimeEventBus`相关的状态和操作，使调用方通过稳定接口使用该能力。"""

    def __init__(self) -> None:
        self._lock = RLock()
        self._listeners: dict[str, list[Callable[..., Any]]] = defaultdict(list)

    def subscribe(self, event_type: str, listener: Callable[..., Any]) -> Callable[[], None]:
        with self._lock:
            if listener not in self._listeners[event_type]:
                self._listeners[event_type].append(listener)

        def unsubscribe() -> None:
            with self._lock:
                listeners = self._listeners.get(event_type, [])
                if listener in listeners:
                    listeners.remove(listener)
                if not listeners:
                    self._listeners.pop(event_type, None)

        return unsubscribe

    def publish(self, event_type: str, *args: Any, **kwargs: Any) -> None:
        with self._lock:
            listeners = tuple(self._listeners.get(event_type, ()))
        for listener in listeners:
            with suppress(Exception):
                listener(*args, **kwargs)

    def clear(self) -> None:
        with self._lock:
            self._listeners.clear()

    def listener_count(self, event_type: str) -> int:
        with self._lock:
            return len(self._listeners.get(event_type, ()))

    def latest(self, event_type: str) -> Callable[..., Any] | None:
        """处理最近一项，并按照当前组件的约定返回结果。

        参数：
            event_type: 本次操作使用的事件类型。

        返回：
            `Callable[..., Any] | None` 类型的处理结果。
        """
        with self._lock:
            listeners = self._listeners.get(event_type, ())
            return listeners[-1] if listeners else None
