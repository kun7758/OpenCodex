"""FastAPI 服务器入口"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from opennova.config import Config
from opennova.web.api import chat, health, sessions
from opennova.web.manager import WebAgentManager

_LOGGER = logging.getLogger(__name__)

# 静态文件目录
_STATIC_DIR = Path(__file__).parent.parent / "static"

# 全局 manager 实例
_manager: WebAgentManager | None = None


def get_manager() -> WebAgentManager:
    """获取全局 WebAgentManager 实例"""
    if _manager is None:
        raise RuntimeError("WebAgentManager not initialized")
    return _manager


def create_app(config: Config) -> FastAPI:
    """创建 FastAPI 应用

    Args:
        config: OpenNova 配置

    Returns:
        FastAPI 应用实例
    """
    global _manager
    _manager = WebAgentManager(config)
    manager = _manager

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """应用生命周期"""
        _LOGGER.info("Starting OpenNova Web UI...")
        await manager.start()
        yield
        _LOGGER.info("Shutting down OpenNova Web UI...")
        await manager.stop()

    app = FastAPI(
        title="OpenNova Web UI",
        description="Terminal AI coding agent with Web UI",
        version="0.4.3",
        lifespan=lifespan,
    )

    # CORS 配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册 API 路由
    app.include_router(chat.router)
    app.include_router(sessions.router)
    app.include_router(health.router)

    # 静态文件服务（前端构建产物）
    if _STATIC_DIR.exists():
        assets_dir = _STATIC_DIR / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        @app.get("/{path:path}")
        async def serve_spa(path: str) -> FileResponse:
            """SPA 路由回退"""
            # 检查是否是静态文件
            file_path = _STATIC_DIR / path
            if file_path.exists() and file_path.is_file():
                return FileResponse(str(file_path))
            # 否则返回 index.html（SPA 路由）
            return FileResponse(str(_STATIC_DIR / "index.html"))

    else:
        _LOGGER.warning(
            "Static files directory not found: %s. "
            "Web UI will not be available. "
            "Please build the frontend first.",
            _STATIC_DIR,
        )

        @app.get("/")
        async def root() -> dict:
            """API 根路径（无前端时）"""
            return {
                "message": "OpenNova Web UI API",
                "version": "0.4.3",
                "docs": "/docs",
                "note": "Frontend not built. Please build the Vue app first.",
            }

    return app


def start_server(
    config: Config,
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
) -> None:
    """启动服务器

    Args:
        config: OpenNova 配置
        host: 监听主机
        port: 监听端口
        reload: 是否启用热重载（开发模式）
    """
    import uvicorn

    app = create_app(config)

    _LOGGER.info("Starting OpenNova Web UI on http://%s:%d", host, port)
    _LOGGER.info("API docs available at http://%s:%d/docs", host, port)

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
