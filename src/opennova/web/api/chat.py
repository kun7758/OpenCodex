"""WebSocket 聊天端点"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from opennova.web.manager import WebAgentManager

_LOGGER = logging.getLogger(__name__)
router = APIRouter()


class ChatWebSocketHandler:
    """处理单个 WebSocket 连接"""

    def __init__(self, websocket: WebSocket, manager: WebAgentManager):
        self.websocket = websocket
        self.manager = manager
        self.session_id: str | None = None

    async def handle(self) -> None:
        """处理 WebSocket 连接生命周期"""
        await self.websocket.accept()

        try:
            while True:
                data = await self.websocket.receive_json()
                await self._process_message(data)
        except WebSocketDisconnect:
            _LOGGER.info("WebSocket disconnected: %s", self.session_id)
        except Exception as e:
            _LOGGER.error("WebSocket error: %s", e)
            try:
                await self.websocket.close(code=1011, reason=str(e))
            except Exception:
                pass
        finally:
            if self.session_id:
                await self.manager.close_session(self.session_id)

    async def _process_message(self, data: dict[str, Any]) -> None:
        """处理客户端消息"""
        msg_type = data.get("type")

        if msg_type == "create_session":
            await self._handle_create_session()
        elif msg_type == "chat":
            await self._handle_chat(data)
        elif msg_type == "cancel":
            await self._handle_cancel()
        else:
            await self._send_error(f"Unknown message type: {msg_type}")

    async def _handle_create_session(self) -> None:
        """创建会话"""
        try:
            self.session_id = self.manager.create_session()
            await self.websocket.send_json({
                "type": "session_created",
                "session_id": self.session_id,
            })
        except Exception as e:
            _LOGGER.error("Failed to create session: %s", e)
            await self._send_error(f"Failed to create session: {e}")

    async def _handle_chat(self, data: dict[str, Any]) -> None:
        """处理聊天消息"""
        if not self.session_id:
            await self._send_error("No session created")
            return

        session = self.manager.get_session(self.session_id)
        if not session:
            await self._send_error("Session not found")
            return

        message = data.get("message", "")
        mode = data.get("mode", "act")

        if not message.strip():
            await self._send_error("Empty message")
            return

        try:
            async for event in session.client.stream_message(
                self.session_id, message, mode=mode, stream=True
            ):
                await self.websocket.send_json(event.to_dict())
        except Exception as e:
            _LOGGER.error("Error in chat: %s", e)
            await self._send_error(str(e))

    async def _handle_cancel(self) -> None:
        """取消当前运行"""
        if not self.session_id:
            return

        session = self.manager.get_session(self.session_id)
        if session:
            try:
                await session.client.cancel_run(self.session_id)
                await self.websocket.send_json({
                    "type": "run_cancelled",
                    "session_id": self.session_id,
                })
            except Exception as e:
                _LOGGER.error("Error cancelling run: %s", e)
                await self._send_error(f"Failed to cancel: {e}")

    async def _send_error(self, error: str) -> None:
        """发送错误消息"""
        try:
            await self.websocket.send_json({
                "type": "error",
                "error": error,
                "session_id": self.session_id,
            })
        except Exception:
            _LOGGER.error("Failed to send error message: %s", error)


@router.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket) -> None:
    """WebSocket 聊天端点

    每个 WebSocket 连接对应一个独立的 AgentRuntime 实例，
    支持流式消息、工具调用、计划审批等功能。
    """
    from opennova.web.server import get_manager

    manager = get_manager()
    handler = ChatWebSocketHandler(websocket, manager)
    await handler.handle()
