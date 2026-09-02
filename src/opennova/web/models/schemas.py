"""Web API 数据模型定义"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionInfo:
    """会话信息"""

    session_id: str
    created_at: float
    last_active: float
    message_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "message_count": self.message_count,
        }


@dataclass
class WebSocketMessage:
    """WebSocket 消息"""

    type: str
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WebSocketMessage:
        """从字典创建消息"""
        return cls(
            type=data.get("type", "unknown"),
            data=data.get("data", {}),
        )


@dataclass
class ChatRequest:
    """聊天请求"""

    message: str
    mode: str = "act"
    stream: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatRequest:
        """从字典创建请求"""
        return cls(
            message=data.get("message", ""),
            mode=data.get("mode", "act"),
            stream=data.get("stream", True),
        )


@dataclass
class ErrorResponse:
    """错误响应"""

    type: str = "error"
    error: str = ""
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        result = {
            "type": self.type,
            "error": self.error,
        }
        if self.session_id:
            result["session_id"] = self.session_id
        return result


@dataclass
class SuccessResponse:
    """成功响应"""

    type: str = "success"
    data: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        result = {
            "type": self.type,
            **self.data,
        }
        if self.session_id:
            result["session_id"] = self.session_id
        return result
