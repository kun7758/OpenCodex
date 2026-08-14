"""内置工具系统中的待办项工具模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from __future__ import annotations

from threading import local
from typing import Any, Literal

from opennova.tools.base import BaseTool, ToolResult

TodoStatus = Literal["pending", "in_progress", "done", "cancelled"]
_compat_state = local()


def _get_compat_todos() -> tuple[dict[str, Any], ...]:
    return getattr(_compat_state, "todos", ())


def _set_compat_todos(todos: list[dict[str, Any]]) -> None:
    _compat_state.todos = tuple(dict(item) for item in todos)


class TodoWriteTool(BaseTool):
    """实现待办项写入工具。模型通过统一工具 Schema 调用它，执行结果使用 ToolResult 返回，并服从运行时安全策略。"""

    name = "todo_write"
    description = (
        "Create or replace a structured todo list for the current task. "
        "Each todo should include id, content, and status."
    )
    max_result_chars = 20_000

    def execute(self, todos: list[dict[str, Any]]) -> ToolResult:
        normalized_or_error = self._normalize_todos(todos)
        if isinstance(normalized_or_error, str):
            return ToolResult(success=False, output="", error=normalized_or_error)

        normalized = normalized_or_error
        state_store = self.config.get("state_store")
        if state_store is not None:
            normalized = state_store.replace_agent_todos(normalized)
        else:
            _set_compat_todos(normalized)
        lines = [f"- [{item['status']}] {item['id']}: {item['content']}" for item in normalized]
        return ToolResult(
            success=True,
            output=f"Updated {len(normalized)} todo(s).\n" + "\n".join(lines),
            metadata={"todos": normalized},
        )

    def is_read_only(self, **kwargs: Any) -> bool:
        return False

    def requires_permission(self, **kwargs: Any) -> bool:
        return False

    @classmethod
    def current_todos(cls, state_store: Any | None = None) -> list[dict[str, Any]]:
        if state_store is not None:
            return state_store.current_todos()
        return [dict(item) for item in _get_compat_todos()]

    @classmethod
    def replace_todos(
        cls,
        todos: list[dict[str, Any]],
        state_store: Any | None = None,
    ) -> list[dict[str, Any]]:
        """替换待办项，并按照当前组件的约定返回结果。

        参数：
            todos: 本次操作使用的待办项。
            state_store: 可选的状态存储。

        返回：
            按调用约定排序的结果列表。
        """
        normalized_or_error = cls._normalize_todos(todos)
        if isinstance(normalized_or_error, str):
            raise ValueError(normalized_or_error)
        normalized = normalized_or_error
        if state_store is not None:
            return state_store.replace_agent_todos(normalized)
        _set_compat_todos(normalized)
        return [dict(item) for item in normalized]

    @staticmethod
    def _normalize_todos(todos: list[dict[str, Any]]) -> list[dict[str, Any]] | str:
        normalized: list[dict[str, Any]] = []
        valid_statuses = {"pending", "in_progress", "done", "cancelled"}
        for index, todo in enumerate(todos, start=1):
            content = str(todo.get("content", "")).strip()
            if not content:
                return f"Todo {index} is missing content"
            status = str(todo.get("status", "pending"))
            if status not in valid_statuses:
                return f"Invalid todo status: {status}"
            normalized.append(
                {
                    "id": str(todo.get("id") or index),
                    "content": content,
                    "status": status,
                }
            )
        return normalized
