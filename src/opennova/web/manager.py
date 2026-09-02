"""Web 会话管理器，管理多个独立的 OpenNovaClient 实例"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from opennova.config import Config
from opennova.sdk import OpenNovaClient

_LOGGER = logging.getLogger(__name__)


class WebSession:
    """单个 Web 会话的状态"""

    def __init__(self, session_id: str, client: OpenNovaClient):
        self.session_id = session_id
        self.client = client
        self.created_at = time.time()
        self.last_active = self.created_at

    def touch(self) -> None:
        """更新最后活跃时间"""
        self.last_active = time.time()


class WebAgentManager:
    """管理多个独立的 Web 会话

    每个 WebSocket 连接对应一个独立的 OpenNovaClient session，
    每个 session 拥有独立的 AgentRuntime 实例。
    """

    def __init__(self, config: Config):
        self.config = config
        self.sessions: dict[str, WebSession] = {}
        self._cleanup_task: asyncio.Task[None] | None = None
        self._started = False

    async def start(self) -> None:
        """启动管理器"""
        if self._started:
            return
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._started = True
        _LOGGER.info("WebAgentManager started")

    async def stop(self) -> None:
        """停止管理器"""
        if not self._started:
            return

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # 关闭所有会话
        for session_id in list(self.sessions.keys()):
            await self.close_session(session_id)

        self._started = False
        _LOGGER.info("WebAgentManager stopped")

    def create_session(self) -> str:
        """创建新的 Web 会话

        Returns:
            新创建的会话 ID
        """
        client = OpenNovaClient(self.config)
        session_id = client.create_session()
        session = WebSession(session_id, client)
        self.sessions[session_id] = session
        _LOGGER.info("Created web session: %s", session_id)
        return session_id

    def get_session(self, session_id: str) -> WebSession | None:
        """获取会话

        Args:
            session_id: 会话 ID

        Returns:
            会话对象，不存在返回 None
        """
        session = self.sessions.get(session_id)
        if session:
            session.touch()
        return session

    async def close_session(self, session_id: str) -> bool:
        """关闭会话

        Args:
            session_id: 会话 ID

        Returns:
            是否成功关闭
        """
        session = self.sessions.pop(session_id, None)
        if not session:
            return False

        try:
            await session.client.aclose()
        except Exception as e:
            _LOGGER.warning("Error closing session %s: %s", session_id, e)

        _LOGGER.info("Closed web session: %s", session_id)
        return True

    def list_sessions(self) -> list[dict[str, Any]]:
        """列出所有会话

        Returns:
            会话信息列表
        """
        return [
            {
                "session_id": sid,
                "created_at": session.created_at,
                "last_active": session.last_active,
            }
            for sid, session in self.sessions.items()
        ]

    async def _cleanup_loop(self) -> None:
        """定期清理过期会话"""
        while True:
            try:
                await asyncio.sleep(300)  # 每5分钟检查一次
                now = time.time()
                expired = [
                    sid
                    for sid, session in self.sessions.items()
                    if now - session.last_active > 3600  # 1小时过期
                ]
                for sid in expired:
                    await self.close_session(sid)
                    _LOGGER.info("Cleaned up expired session: %s", sid)
            except asyncio.CancelledError:
                break
            except Exception as e:
                _LOGGER.error("Error in cleanup loop: %s", e)
