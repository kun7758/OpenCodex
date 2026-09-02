"""健康检查端点"""

from __future__ import annotations

import time

from fastapi import APIRouter

router = APIRouter()

_start_time = time.time()


def _get_manager():
    """获取 manager 实例"""
    from opennova.web.server import get_manager
    return get_manager()


@router.get("/api/health")
async def health_check() -> dict:
    """健康检查

    Returns:
        服务状态信息
    """
    manager = _get_manager()
    uptime = time.time() - _start_time
    sessions = manager.list_sessions()

    return {
        "status": "healthy",
        "uptime_seconds": round(uptime, 2),
        "active_sessions": len(sessions),
        "version": "0.4.3",
    }


@router.get("/api/info")
async def service_info() -> dict:
    """服务信息

    Returns:
        服务详细信息
    """
    manager = _get_manager()
    sessions = manager.list_sessions()

    return {
        "name": "OpenNova Web UI",
        "version": "0.4.3",
        "description": "Terminal AI coding agent with Web UI",
        "active_sessions": len(sessions),
        "max_sessions": 10,
        "session_timeout_seconds": 3600,
    }
