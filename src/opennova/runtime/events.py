"""Agent 核心运行时中的事件模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Literal

from opennova.runtime.cancellation import CancellationToken

ToolEventType = Literal[
    "tool_start",
    "permission_request",
    "tool_result",
    "tool_error",
    "tool_cancelled",
]


@dataclass
class ToolUseContext:
    """数据对象 `ToolUseContext` 主要保存
    `tool_id`、`tool_name`、`arguments`、`session_id`、`permission_context`、`read_file_cache`、`abort_signal`、`risk_level`
    等字段，用于在组件之间传递或持久化这组状态。
    """

    tool_id: str
    tool_name: str
    arguments: dict[str, Any]
    session_id: str | None = None
    permission_context: dict[str, Any] = field(default_factory=dict)
    read_file_cache: Any = field(default_factory=dict)
    abort_signal: CancellationToken | None = None
    risk_level: str = "safe"
    diff: str | None = None
    max_result_chars: int | None = None
    non_interactive: bool = False
    started_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolEvent:
    """保存工具事件所需的结构化数据，主要包含
    `type`、`tool_id`、`tool_name`、`arguments`、`started_at`、`duration_ms`、`risk_level`、`success`
    等字段，便于在组件之间传递或持久化。
    """

    type: ToolEventType
    tool_id: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    started_at: float | None = None
    duration_ms: int | None = None
    risk_level: str = "safe"
    success: bool | None = None
    output: str = ""
    error: str | None = None
    diff: str | None = None
    collapsible: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """把工具事件转换为可序列化字典，供事件、会话或 API 边界使用。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        return {
            "type": self.type,
            "tool_id": self.tool_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "risk_level": self.risk_level,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "diff": self.diff,
            "collapsible": self.collapsible,
            "metadata": self.metadata,
        }


_CURRENT_TOOL_CONTEXT: ContextVar[ToolUseContext | None] = ContextVar(
    "opennova_tool_use_context",
    default=None,
)


def set_current_tool_context(context: ToolUseContext | None) -> Token[ToolUseContext | None]:
    """设置当前工具上下文并保持相关派生状态同步。

    参数：
        context: 本次工具调用或运行所使用的上下文。

    返回：
        `Token[ToolUseContext | None]` 类型的处理结果。
    """
    return _CURRENT_TOOL_CONTEXT.set(context)


def reset_current_tool_context(token: Token[ToolUseContext | None]) -> None:
    """恢复 `reset_current_tool_context` 所表示的数据或流程，并遵守当前模块定义的边界与状态约束。

    参数：
        token: 可选的Token。
    """
    _CURRENT_TOOL_CONTEXT.reset(token)


def current_tool_context() -> ToolUseContext | None:
    """读取并返回 `current_tool_context` 所表示的数据或流程，并遵守当前模块定义的边界与状态约束。

    返回：
        `ToolUseContext | None` 类型的处理结果。
    """
    return _CURRENT_TOOL_CONTEXT.get()
