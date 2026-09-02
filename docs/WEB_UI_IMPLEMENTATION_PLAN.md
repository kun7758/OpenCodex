# Web UI 实现计划（方案 A：每连接独立 AgentRuntime）

## 1. 概述

### 1.1 目标
为 OpenNova 新增 Web UI 交互界面，支持：
- 浏览器访问的富文本 Markdown 渲染
- 多用户并发访问（每连接独立 AgentRuntime 实例）
- 与现有 TUI 完全独立，互不影响

### 1.2 核心设计
- **后端**：基于 FastAPI 提供 REST + WebSocket API
- **前端**：Vue 3 + TypeScript + Element Plus + Tailwind CSS
- **复用**：直接使用现有 `OpenNovaClient` SDK 作为后端核心

---

## 2. 架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        浏览器客户端                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Chat UI   │  │ Markdown    │  │  Tool Cards │         │
│  │   (Vue 3)   │  │  Renderer   │  │   Panel     │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
└─────────┼────────────────┼────────────────┼─────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────┐
│                     HTTP/WebSocket                          │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI 后端服务                          │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              WebAgentManager                          │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐     │  │
│  │  │  Session 1  │ │  Session 2  │ │  Session 3  │     │  │
│  │  │ OpenNovaClient│ │ OpenNovaClient│ │ OpenNovaClient│ │  │
│  │  └─────────────┘ └─────────────┘ └─────────────┘     │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   现有 OpenNova 核心                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              AgentRuntime (每会话独立)                 │  │
│  │  - LLM Providers (OpenAI/Anthropic/DeepSeek)         │  │
│  │  - Tool Registry (40个内置工具)                       │  │
│  │  - Session Manager (会话持久化)                       │  │
│  │  - Context Manager (上下文压缩)                       │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 并发模型

```
浏览器标签页 1 ──┐
                 │    WebSocket 1 ──► Session 1 (AgentRuntime 实例 1)
                 │
浏览器标签页 2 ──┼──► WebSocket 2 ──► Session 2 (AgentRuntime 实例 2)
                 │
浏览器标签页 3 ──┘    WebSocket 3 ──► Session 3 (AgentRuntime 实例 3)
```

**关键点**：
- 每个 WebSocket 连接对应一个独立的 `OpenNovaClient` session
- 每个 session 拥有独立的 `AgentRuntime` 实例
- 会话间完全隔离，无状态共享
- 单会话内串行执行（保持一致性）

---

## 3. 项目结构

### 3.1 新增目录

```
src/opennova/
├── web/                          # 新增：Web UI 模块
│   ├── __init__.py
│   ├── server.py                 # FastAPI 应用入口
│   ├── api/
│   │   ├── __init__.py
│   │   ├── chat.py              # WebSocket 聊天端点
│   │   ├── sessions.py          # REST 会话管理端点
│   │   └── health.py            # 健康检查端点
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py           # Pydantic 数据模型
│   └── manager.py               # WebAgentManager 管理类
│
├── static/                       # 新增：前端构建产物
│   ├── index.html
│   ├── assets/
│   │   ├── js/
│   │   └── css/
│   └── favicon.ico
│
└── cli/
    └── tui.py                   # 现有：TUI 入口（不修改）
```

### 3.2 前端项目结构（独立仓库或子目录）

```
opennova-web/                     # 前端项目（可选独立仓库）
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js
├── src/
│   ├── App.vue
│   ├── main.ts
│   ├── components/
│   │   ├── Chat/
│   │   │   ├── ChatInput.vue
│   │   │   ├── ChatMessage.vue
│   │   │   └── ChatHistory.vue
│   │   ├── Markdown/
│   │   │   ├── MarkdownRenderer.vue
│   │   │   ├── CodeBlock.vue
│   │   │   └── MermaidDiagram.vue
│   │   ├── Tools/
│   │   │   ├── ToolCard.vue
│   │   │   └── ToolProgress.vue
│   │   └── Layout/
│   │       ├── AppSidebar.vue
│   │       └── AppHeader.vue
│   ├── composables/
│   │   ├── useWebSocket.ts
│   │   └── useSession.ts
│   ├── services/
│   │   └── api.ts
│   ├── stores/
│   │   └── chat.ts
│   └── types/
│       └── index.ts
└── public/
```

---

## 4. 后端实现

### 4.1 依赖更新

```toml
# pyproject.toml
[project.optional-dependencies]
web = [
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "websockets>=12.0",
    "python-multipart>=0.0.6",
]
```

### 4.2 核心模块实现

#### 4.2.1 `src/opennova/web/manager.py`

```python
"""Web 会话管理器，管理多个独立的 OpenNovaClient 实例"""

import asyncio
import logging
from typing import Any
from uuid import uuid4

from opennova.config import Config
from opennova.sdk import OpenNovaClient, SDKEvent

_LOGGER = logging.getLogger(__name__)


class WebSession:
    """单个 Web 会话的状态"""

    def __init__(self, session_id: str, client: OpenNovaClient):
        self.session_id = session_id
        self.client = client
        self.created_at = asyncio.get_event_loop().time()
        self.last_active = self.created_at

    def touch(self):
        """更新最后活跃时间"""
        self.last_active = asyncio.get_event_loop().time()


class WebAgentManager:
    """管理多个独立的 Web 会话"""

    def __init__(self, config: Config):
        self.config = config
        self.sessions: dict[str, WebSession] = {}
        self._cleanup_task: asyncio.Task | None = None

    async def start(self):
        """启动管理器"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        _LOGGER.info("WebAgentManager started")

    async def stop(self):
        """停止管理器"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # 关闭所有会话
        for session_id in list(self.sessions.keys()):
            await self.close_session(session_id)

        _LOGGER.info("WebAgentManager stopped")

    def create_session(self) -> str:
        """创建新的 Web 会话"""
        session_id = str(uuid4())
        client = OpenNovaClient(self.config)
        session = WebSession(session_id, client)
        self.sessions[session_id] = session
        _LOGGER.info(f"Created web session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> WebSession | None:
        """获取会话"""
        session = self.sessions.get(session_id)
        if session:
            session.touch()
        return session

    async def close_session(self, session_id: str) -> bool:
        """关闭会话"""
        session = self.sessions.pop(session_id, None)
        if not session:
            return False

        await session.client.aclose()
        _LOGGER.info(f"Closed web session: {session_id}")
        return True

    def list_sessions(self) -> list[dict[str, Any]]:
        """列出所有会话"""
        return [
            {
                "session_id": sid,
                "created_at": session.created_at,
                "last_active": session.last_active,
            }
            for sid, session in self.sessions.items()
        ]

    async def _cleanup_loop(self):
        """定期清理过期会话"""
        while True:
            try:
                await asyncio.sleep(300)  # 每5分钟检查一次
                now = asyncio.get_event_loop().time()
                expired = [
                    sid
                    for sid, session in self.sessions.items()
                    if now - session.last_active > 3600  # 1小时过期
                ]
                for sid in expired:
                    await self.close_session(sid)
                    _LOGGER.info(f"Cleaned up expired session: {sid}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                _LOGGER.error(f"Error in cleanup loop: {e}")
```

#### 4.2.2 `src/opennova/web/api/chat.py`

```python
"""WebSocket 聊天端点"""

import asyncio
import json
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

    async def handle(self):
        """处理 WebSocket 连接生命周期"""
        await self.websocket.accept()

        try:
            while True:
                data = await self.websocket.receive_json()
                await self._process_message(data)
        except WebSocketDisconnect:
            _LOGGER.info(f"WebSocket disconnected: {self.session_id}")
        except Exception as e:
            _LOGGER.error(f"WebSocket error: {e}")
            await self.websocket.close(code=1011, reason=str(e))
        finally:
            if self.session_id:
                await self.manager.close_session(self.session_id)

    async def _process_message(self, data: dict[str, Any]):
        """处理客户端消息"""
        msg_type = data.get("type")

        if msg_type == "create_session":
            await self._handle_create_session()
        elif msg_type == "chat":
            await self._handle_chat(data)
        elif msg_type == "cancel":
            await self._handle_cancel()
        else:
            await self.websocket.send_json({
                "type": "error",
                "error": f"Unknown message type: {msg_type}"
            })

    async def _handle_create_session(self):
        """创建会话"""
        self.session_id = self.manager.create_session()
        await self.websocket.send_json({
            "type": "session_created",
            "session_id": self.session_id,
        })

    async def _handle_chat(self, data: dict[str, Any]):
        """处理聊天消息"""
        if not self.session_id:
            await self.websocket.send_json({
                "type": "error",
                "error": "No session created"
            })
            return

        session = self.manager.get_session(self.session_id)
        if not session:
            await self.websocket.send_json({
                "type": "error",
                "error": "Session not found"
            })
            return

        message = data.get("message", "")
        mode = data.get("mode", "act")

        try:
            async for event in session.client.stream_message(
                self.session_id, message, mode=mode, stream=True
            ):
                await self.websocket.send_json(event.to_dict())
        except Exception as e:
            await self.websocket.send_json({
                "type": "error",
                "error": str(e)
            })

    async def _handle_cancel(self):
        """取消当前运行"""
        if not self.session_id:
            return

        session = self.manager.get_session(self.session_id)
        if session:
            await session.client.cancel_run(self.session_id)
            await self.websocket.send_json({
                "type": "run_cancelled",
                "session_id": self.session_id,
            })


@router.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket, manager: WebAgentManager):
    """WebSocket 聊天端点"""
    handler = ChatWebSocketHandler(websocket, manager)
    await handler.handle()
```

#### 4.2.3 `src/opennova/web/api/sessions.py`

```python
"""REST 会话管理端点"""

from fastapi import APIRouter, HTTPException

from opennova.web.manager import WebAgentManager

router = APIRouter()


@router.get("/api/sessions")
async def list_sessions(manager: WebAgentManager):
    """列出所有会话"""
    return {"sessions": manager.list_sessions()}


@router.post("/api/sessions")
async def create_session(manager: WebAgentManager):
    """创建新会话"""
    session_id = manager.create_session()
    return {"session_id": session_id}


@router.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, manager: WebAgentManager):
    """删除会话"""
    success = await manager.close_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"success": True}
```

#### 4.2.4 `src/opennova/web/server.py`

```python
"""FastAPI 服务器入口"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from opennova.config import Config
from opennova.web.manager import WebAgentManager
from opennova.web.api import chat, sessions, health

_LOGGER = logging.getLogger(__name__)


def create_app(config: Config) -> FastAPI:
    """创建 FastAPI 应用"""
    manager = WebAgentManager(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """应用生命周期"""
        await manager.start()
        yield
        await manager.stop()

    app = FastAPI(
        title="OpenNova Web UI",
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

    # 注入 manager 到依赖
    app.dependency_overrides[WebAgentManager] = lambda: manager

    # 注册路由
    app.include_router(chat.router)
    app.include_router(sessions.router)
    app.include_router(health.router)

    # 静态文件服务（前端构建产物）
    try:
        app.mount("/assets", StaticFiles(directory="src/opennova/static/assets"), name="assets")

        @app.get("/{path:path}")
        async def serve_spa(path: str):
            """SPA 路由回退"""
            return FileResponse("src/opennova/static/index.html")
    except Exception:
        _LOGGER.warning("Static files not found, serving API only")

    return app


def start_server(config: Config, host: str = "127.0.0.1", port: int = 8000):
    """启动服务器"""
    import uvicorn

    app = create_app(config)
    uvicorn.run(app, host=host, port=port)
```

### 4.3 CLI 集成

```python
# src/opennova/main.py 中新增命令

@main.command()
@click.option("--port", "-p", default=8000, help="Port to listen on.")
@click.option("--host", "-h", default="127.0.0.1", help="Host to bind to.")
@click.pass_context
def serve(ctx: click.Context, port: int, host: str) -> None:
    """启动 Web 服务（每连接独立 AgentRuntime）"""
    from opennova.web.server import start_server

    config = load_config(ctx.obj.get("config_path"))
    click.echo(f"Starting OpenNova Web UI on http://{host}:{port}")
    click.echo("Press Ctrl+C to stop")
    start_server(config, host, port)
```

---

## 5. 前端实现

### 5.1 技术栈

- **框架**：Vue 3 + TypeScript + Composition API
- **构建**：Vite
- **UI 组件库**：Element Plus
- **样式**：Tailwind CSS
- **状态管理**：Pinia
- **Markdown**：markdown-it + highlight.js + markdown-it-katex
- **图表**：mermaid（可选）
- **数学公式**：KaTeX（可选）
- **WebSocket**：原生 WebSocket API + VueUse

### 5.2 核心组件

#### 5.2.1 WebSocket Composable

```typescript
// src/composables/useWebSocket.ts

import { ref, onMounted, onUnmounted } from 'vue'

interface SDKEvent {
  type: string
  session_id: string
  data: Record<string, any>
}

export function useWebSocket(url: string) {
  const connected = ref(false)
  const sessionId = ref<string | null>(null)
  const events = ref<SDKEvent[]>([])
  let ws: WebSocket | null = null

  const connect = () => {
    ws = new WebSocket(url)

    ws.onopen = () => {
      connected.value = true
      // 创建会话
      ws?.send(JSON.stringify({ type: 'create_session' }))
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data) as SDKEvent

      if (data.type === 'session_created') {
        sessionId.value = data.session_id
      }

      events.value.push(data)
    }

    ws.onclose = () => {
      connected.value = false
      sessionId.value = null
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }
  }

  const disconnect = () => {
    ws?.close()
    ws = null
  }

  const sendMessage = (message: string, mode: string = 'act') => {
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: 'chat',
        message,
        mode,
      }))
    }
  }

  const cancelRun = () => {
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'cancel' }))
    }
  }

  const clearEvents = () => {
    events.value = []
  }

  onMounted(() => {
    connect()
  })

  onUnmounted(() => {
    disconnect()
  })

  return {
    connected,
    sessionId,
    events,
    sendMessage,
    cancelRun,
    clearEvents,
  }
}
```

#### 5.2.2 Markdown 渲染器

```vue
<!-- src/components/Markdown/MarkdownRenderer.vue -->

<template>
  <div class="markdown-body" v-html="renderedHtml"></div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import MarkdownIt from 'markdown-it'
import hljs from 'highlight.js'
import markdownItKatex from 'markdown-it-katex'

interface Props {
  content: string
}

const props = defineProps<Props>()

const md = new MarkdownIt({
  html: true,
  linkify: true,
  typographer: true,
  highlight: function (str: string, lang: string) {
    if (lang && hljs.getLanguage(lang)) {
      try {
        return hljs.highlight(str, { language: lang }).value
      } catch (_) {}
    }
    return '' // 使用外部默认转义
  }
})

md.use(markdownItKatex)

const renderedHtml = computed(() => {
  return md.render(props.content || '')
})
</script>

<style scoped>
.markdown-body {
  /* GitHub 风格的 Markdown 样式 */
  font-size: 14px;
  line-height: 1.6;
  word-wrap: break-word;
}

.markdown-body :deep(pre) {
  background-color: #f6f8fa;
  border-radius: 6px;
  padding: 16px;
  overflow-x: auto;
}

.markdown-body :deep(code) {
  background-color: rgba(175, 184, 193, 0.2);
  padding: 0.2em 0.4em;
  border-radius: 6px;
  font-size: 85%;
}

.markdown-body :deep(pre code) {
  background-color: transparent;
  padding: 0;
  border-radius: 0;
  font-size: 100%;
}

.markdown-body :deep(table) {
  border-collapse: collapse;
  width: 100%;
}

.markdown-body :deep(th),
.markdown-body :deep(td) {
  border: 1px solid #d0d7de;
  padding: 6px 13px;
}

.markdown-body :deep(th) {
  background-color: #f6f8fa;
  font-weight: 600;
}

.markdown-body :deep(blockquote) {
  border-left: 4px solid #d0d7de;
  padding: 0 1em;
  color: #656d76;
  margin: 0;
}
</style>
```

#### 5.2.3 Chat 组件

```vue
<!-- src/components/Chat/ChatMessage.vue -->

<template>
  <div class="chat-message mb-4">
    <!-- 文本内容 -->
    <div v-if="event.type === 'text_delta'" class="prose max-w-none">
      <MarkdownRenderer :content="event.data.content" />
    </div>

    <!-- 工具开始 -->
    <div
      v-else-if="event.type === 'tool_start'"
      class="bg-gray-100 dark:bg-gray-800 rounded-lg p-3"
    >
      <div class="flex items-center gap-2 text-sm text-gray-500">
        <span class="animate-spin">&#9881;</span>
        <span>执行中: {{ event.data.tool_name }}</span>
      </div>
    </div>

    <!-- 工具结果 -->
    <div
      v-else-if="event.type === 'tool_result'"
      class="bg-gray-100 dark:bg-gray-800 rounded-lg p-3"
    >
      <div class="text-sm">
        <span :class="event.data.success ? 'text-green-500' : 'text-red-500'">
          {{ event.data.success ? '&#10003;' : '&#10007;' }}
        </span>
        <span class="ml-2">{{ event.data.tool_name }}</span>
      </div>
    </div>

    <!-- 思考过程 -->
    <div
      v-else-if="event.type === 'thought'"
      class="italic text-gray-500 text-sm"
    >
      &#128173; {{ event.data.content }}
    </div>

    <!-- 计划 -->
    <div
      v-else-if="event.type === 'plan'"
      class="border rounded-lg p-4"
    >
      <h3 class="font-semibold mb-2">&#128203; 计划</h3>
      <pre class="text-sm bg-gray-50 dark:bg-gray-900 p-3 rounded overflow-auto">{{
        JSON.stringify(event.data.plan, null, 2)
      }}</pre>
      <div class="mt-3 flex gap-2">
        <el-button type="primary" size="small" @click="$emit('approve', event.data.plan)">
          执行计划
        </el-button>
        <el-button size="small" @click="$emit('reject')">
          拒绝
        </el-button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import MarkdownRenderer from '../Markdown/MarkdownRenderer.vue'

interface SDKEvent {
  type: string
  session_id: string
  data: Record<string, any>
}

interface Props {
  event: SDKEvent
}

defineProps<Props>()

defineEmits<{
  approve: [plan: any]
  reject: []
}>()
</script>
```

#### 5.2.4 Pinia Store

```typescript
// src/stores/chat.ts

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

interface SDKEvent {
  type: string
  session_id: string
  data: Record<string, any>
}

export const useChatStore = defineStore('chat', () => {
  const connected = ref(false)
  const sessionId = ref<string | null>(null)
  const events = ref<SDKEvent[]>([])
  const isGenerating = ref(false)

  // 计算属性：获取所有消息事件
  const messages = computed(() => {
    return events.value.filter(e =>
      ['text_delta', 'tool_start', 'tool_result', 'thought', 'plan'].includes(e.type)
    )
  })

  // 计算属性：获取最新的文本内容
  const latestTextContent = computed(() => {
    const textEvents = events.value.filter(e => e.type === 'text_delta')
    return textEvents.map(e => e.data.content).join('')
  })

  const addEvent = (event: SDKEvent) => {
    events.value.push(event)

    if (event.type === 'run_start') {
      isGenerating.value = true
    } else if (['run_complete', 'run_error', 'run_cancelled'].includes(event.type)) {
      isGenerating.value = false
    }
  }

  const clearEvents = () => {
    events.value = []
  }

  const setConnected = (value: boolean) => {
    connected.value = value
  }

  const setSessionId = (id: string | null) => {
    sessionId.value = id
  }

  return {
    connected,
    sessionId,
    events,
    isGenerating,
    messages,
    latestTextContent,
    addEvent,
    clearEvents,
    setConnected,
    setSessionId,
  }
})
```

---

## 6. 实现步骤

### 6.1 阶段一：后端基础（1-2天）

1. **创建 Web 模块目录结构**
   - `src/opennova/web/` 及子目录
   - `__init__.py` 文件

2. **实现 WebAgentManager**
   - 会话生命周期管理
   - 过期会话清理

3. **实现 WebSocket 端点**
   - 连接管理
   - 消息路由
   - 错误处理

4. **实现 REST 端点**
   - 会话列表
   - 会话创建/删除

5. **更新 CLI**
   - 添加 `serve` 命令

6. **更新依赖**
   - 添加 FastAPI、uvicorn 等

### 6.2 阶段二：前端基础（2-3天）

1. **初始化前端项目**
   - Vite + Vue 3 + TypeScript
   - Element Plus 配置
   - Tailwind CSS 配置
   - Pinia 状态管理

2. **实现 WebSocket 连接**
   - `useWebSocket` composable
   - 事件处理
   - Pinia store 集成

3. **实现基础 UI**
   - 布局组件（AppSidebar、AppHeader）
   - 聊天界面（ChatHistory、ChatInput）
   - Element Plus 组件集成

4. **实现 Markdown 渲染**
   - markdown-it 集成
   - highlight.js 代码高亮
   - GFM 支持

### 6.3 阶段三：功能完善（2-3天）

1. **实现工具卡片**
   - 工具执行状态展示
   - 进度指示器

2. **实现会话管理**
   - 会话列表
   - 会话切换
   - 会话恢复

3. **实现计划审批**
   - 计划展示
   - 审批对话框

4. **实现错误处理**
   - 连接断开重连
   - 错误提示

### 6.4 阶段四：高级功能（可选，2-3天）

1. **数学公式支持**
   - KaTeX 集成

2. **图表支持**
   - Mermaid 集成

3. **文件预览**
   - 代码文件预览
   - 图片预览

4. **主题支持**
   - 亮色/暗色主题
   - 自定义主题

### 6.5 阶段五：测试与优化（1-2天）

1. **单元测试**
   - 后端 API 测试
   - 前端组件测试

2. **集成测试**
   - WebSocket 连接测试
   - 会话生命周期测试

3. **性能优化**
   - 消息批量处理
   - 虚拟滚动

4. **文档**
   - API 文档
   - 用户指南

---

## 7. 使用方式

### 7.1 启动 Web 服务

```bash
# 安装 Web 依赖
uv sync --extra web

# 启动 Web 服务
uv run opennova serve --port 8000

# 或指定主机
uv run opennova serve --host 0.0.0.0 --port 8000
```

### 7.2 访问 Web UI

浏览器打开 `http://localhost:8000`

### 7.3 API 端点

- `WebSocket /ws/chat` - 聊天 WebSocket
- `GET /api/sessions` - 列出会话
- `POST /api/sessions` - 创建会话
- `DELETE /api/sessions/{id}` - 删除会话
- `GET /api/health` - 健康检查

---

## 8. 注意事项

### 8.1 资源消耗

每个会话的 AgentRuntime 实例约占用：
- 内存：50-100MB
- 文件句柄：若干（日志、会话文件）

建议：
- 限制最大并发会话数（默认10）
- 设置会话过期时间（默认1小时）
- 监控资源使用情况

### 8.2 安全考虑

- 默认绑定 `127.0.0.1`，仅本地访问
- 如需远程访问，考虑添加认证机制
- WebSocket 连接需要验证来源

### 8.3 与 TUI 的关系

- Web UI 和 TUI 完全独立
- 可以同时运行
- 会话数据共享（都存储在 `~/.opennova/sessions/`）
- 不要同时打开同一会话

---

## 9. 文件清单

### 9.1 后端新增文件

```
src/opennova/web/
├── __init__.py
├── server.py
├── manager.py
├── api/
│   ├── __init__.py
│   ├── chat.py
│   ├── sessions.py
│   └── health.py
└── models/
    ├── __init__.py
    └── schemas.py

src/opennova/static/
├── index.html
└── assets/
    └── ...
```

### 9.2 前端项目结构（独立仓库或子目录）

```
opennova-web/
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js
├── src/
│   ├── App.vue
│   ├── main.ts
│   ├── components/
│   │   ├── Chat/
│   │   │   ├── ChatInput.vue
│   │   │   ├── ChatMessage.vue
│   │   │   └── ChatHistory.vue
│   │   ├── Markdown/
│   │   │   └── MarkdownRenderer.vue
│   │   ├── Tools/
│   │   │   ├── ToolCard.vue
│   │   │   └── ToolProgress.vue
│   │   └── Layout/
│   │       ├── AppSidebar.vue
│   │       └── AppHeader.vue
│   ├── composables/
│   │   └── useWebSocket.ts
│   ├── stores/
│   │   └── chat.ts
│   └── types/
│       └── index.ts
└── public/
```

### 9.3 修改文件

```
src/opennova/main.py          # 添加 serve 命令
pyproject.toml                 # 添加 web 依赖
```

---

## 10. 总结

本方案通过以下方式实现 Web UI：

1. **复用现有 SDK**：直接使用 `OpenNovaClient` 作为后端核心
2. **每连接独立实例**：每个 WebSocket 连接对应独立的 AgentRuntime
3. **完全独立**：不影响现有 TUI 启动方式
4. **渐进实现**：分5个阶段，逐步完善功能

预计总工时：8-13天（可根据需求裁剪）
