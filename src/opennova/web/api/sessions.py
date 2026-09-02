"""REST 会话管理端点"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter()


def _get_manager():
    """获取 manager 实例"""
    from opennova.web.server import get_manager
    return get_manager()


@router.get("/api/sessions")
async def list_sessions() -> dict:
    """列出所有会话

    Returns:
        包含会话列表的响应
    """
    manager = _get_manager()
    sessions = manager.list_sessions()
    return {
        "sessions": sessions,
        "total": len(sessions),
    }


@router.post("/api/sessions")
async def create_session() -> dict:
    """创建新会话

    Returns:
        包含新会话 ID 的响应
    """
    manager = _get_manager()
    try:
        session_id = manager.create_session()
        return {
            "session_id": session_id,
            "message": "Session created successfully",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create session: {e}",
        )


@router.get("/api/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    """获取会话信息

    Args:
        session_id: 会话 ID

    Returns:
        会话信息
    """
    manager = _get_manager()
    session = manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session.session_id,
        "created_at": session.created_at,
        "last_active": session.last_active,
    }


@router.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str) -> dict:
    """删除会话

    Args:
        session_id: 会话 ID

    Returns:
        操作结果
    """
    manager = _get_manager()
    success = await manager.close_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "success": True,
        "message": f"Session {session_id} closed",
    }
