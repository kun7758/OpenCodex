"""OpenNova中的`sdk`模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from __future__ import annotations

import asyncio
import copy
import inspect
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from opennova.cli.tool_progress import ToolProgressTracker
from opennova.config import Config
from opennova.runtime.agent import AgentRuntime
from opennova.tools.base import ToolResult


@dataclass
class SDKEvent:
    """数据对象 `SDKEvent` 主要保存 `type`、`session_id`、`data` 字段，用于在组件之间传递或持久化这组状态。"""

    type: str
    session_id: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """把`SDKEvent`转换为可序列化字典，供事件、会话或 API 边界使用。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        return {
            "type": self.type,
            "session_id": self.session_id,
            "data": self.data,
        }


class SDKRunCancelledError(RuntimeError):
    """表示 SDK 会话中的活动运行已经取消，调用方可单独捕获它以区别普通执行失败。"""


class OpenNovaClient:
    """面向脚本和服务的无界面 SDK。每个 SDK 会话持有独立 AgentRuntime，并把内部回调规范化为可异步迭代的 SDKEvent。"""

    def __init__(
        self,
        config: Config | dict[str, Any],
        runtime_factory: Callable[[dict[str, Any]], AgentRuntime] = AgentRuntime,
    ):
        self.config = config.to_dict() if isinstance(config, Config) else copy.deepcopy(config)
        self.runtime_factory = runtime_factory
        self._sessions: dict[str, Any] = {}
        self._active_runs: dict[str, asyncio.Task[None]] = {}
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("OpenNovaClient is closed")

    def create_session(self) -> str:
        """创建会话并完成必要的初始化。

        返回：
            处理后的文本或稳定标识。
        """
        self._ensure_open()
        runtime = self.runtime_factory(self.config)
        session_id = getattr(getattr(runtime, "session_manager", None), "session_id", None)
        if not session_id:
            session_id = str(uuid.uuid4())
        self._sessions[session_id] = runtime
        return session_id

    def get_runtime(self, session_id: str) -> Any:
        """读取运行时，不改变当前对象的业务状态。

        参数：
            session_id: 目标会话的稳定标识。

        返回：
            `Any` 类型的处理结果。
        """
        self._ensure_open()
        if session_id not in self._sessions:
            raise KeyError(f"Unknown OpenNova SDK session: {session_id}")
        return self._sessions[session_id]

    def list_sessions(self) -> list[dict[str, Any]]:
        """列出 `sessions` 对应的对象，并按当前组件约定返回稳定顺序。

        返回：
            按调用约定排序的结果列表。
        """
        return [{"session_id": session_id} for session_id in self._sessions]

    def resume_session(self, session_id: str) -> str:
        """将当前写入器重新绑定到已有 session ID，加载对应消息和运行状态；后续保存继续写入原会话，不会隐式创建重复会话。

        参数：
            session_id: 目标会话的稳定标识。

        返回：
            处理后的文本或稳定标识。
        """
        self._ensure_open()
        if session_id in self._sessions:
            raise RuntimeError(f"Session {session_id} is already open")
        runtime = self.runtime_factory(self.config)
        runtime.resume_session(session_id)
        self._sessions[session_id] = runtime
        return session_id

    def fork_session(self, session_id: str) -> str:
        """复制一个已持久化会话并分配新的 session ID，使新旧时间线可以独立继续写入。

        参数：
            session_id: 目标会话的稳定标识。

        返回：
            处理后的文本或稳定标识。
        """
        source = self.get_runtime(session_id)
        flush = getattr(source, "flush_session", None)
        if callable(flush):
            flush()
        fork_id = source.session_manager.fork_session(session_id)
        runtime = self.runtime_factory(self.config)
        runtime.resume_session(fork_id)
        self._sessions[fork_id] = runtime
        return fork_id

    async def submit_message(
        self,
        session_id: str,
        message: str,
        mode: str = "act",
        stream: bool = True,
    ) -> str:
        """提交消息，并按照当前组件的约定返回结果。

        参数：
            session_id: 目标会话的稳定标识。
            message: 用户提交或组件间传递的消息。
            mode: 本次运行采用的工作模式。
            stream: 是否将模型输出以增量事件形式返回。

        返回：
            处理后的文本或稳定标识。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        final_result = ""
        async for event in self.stream_message(session_id, message, mode=mode, stream=stream):
            if event.type == "run_complete":
                final_result = str(event.data.get("result", ""))
            elif event.type == "run_error":
                raise RuntimeError(str(event.data.get("error", "OpenNova SDK run failed")))
            elif event.type == "run_cancelled":
                raise SDKRunCancelledError(str(event.data.get("reason", "Run cancelled")))
        return final_result

    async def stream_message(
        self,
        session_id: str,
        message: str,
        mode: str = "act",
        stream: bool = True,
    ) -> AsyncIterator[SDKEvent]:
        """运行一次 SDK 消息并异步产出规范事件，包括开始、文本增量、工具事件、计划、完成、错误或取消；结束时注销临时回调。

        参数：
            session_id: 目标会话的稳定标识。
            message: 用户提交或组件间传递的消息。
            mode: 本次运行采用的工作模式。
            stream: 是否将模型输出以增量事件形式返回。

        生成：
            逐项产生结果，直到数据源结束。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        runtime = self.get_runtime(session_id)
        active_run = self._active_runs.get(session_id)
        if active_run is not None and not active_run.done():
            raise RuntimeError(f"Session {session_id} already has an active run")

        queue: asyncio.Queue[SDKEvent] = asyncio.Queue()
        tool_progress = ToolProgressTracker()
        saw_canonical_tool_events = False
        unsubscribers: list[Callable[[], None]] = []

        def subscribe(event_type: str, callback: Callable[..., Any]) -> None:
            unsubscribe = runtime.register_callback(event_type, callback)
            if callable(unsubscribe):
                unsubscribers.append(unsubscribe)

        def enqueue(event_type: str, **data: Any) -> None:
            queue.put_nowait(SDKEvent(type=event_type, session_id=session_id, data=data))

        subscribe("thought", lambda thought: enqueue("thought", content=thought))

        def on_action(tool_name: str, args: dict[str, Any]) -> None:
            if saw_canonical_tool_events:
                return
            enqueue("tool_start", **tool_progress.start_tool(tool_name, args))

        def on_result(result: ToolResult) -> None:
            if saw_canonical_tool_events:
                return
            data = self._tool_result_data(result)
            data.update(tool_progress.finish_tool(result))
            enqueue("tool_result", **data)

        def on_tool_event(event: Any) -> None:
            nonlocal saw_canonical_tool_events
            saw_canonical_tool_events = True
            payload = event.to_dict() if hasattr(event, "to_dict") else dict(event)
            event_type = str(payload.pop("type"))
            enqueue(event_type, **payload)

        subscribe("action", on_action)
        subscribe("result", on_result)
        subscribe("tool_event", on_tool_event)
        subscribe(
            "stream",
            lambda chunk: enqueue("text_delta", content=getattr(chunk, "content", "") or ""),
        )
        subscribe(
            "plan",
            lambda plan, plan_file_path=None: enqueue(
                "plan",
                plan=getattr(plan, "to_dict", lambda: str(plan))(),
                plan_file_path=str(plan_file_path) if plan_file_path else None,
            ),
        )

        yield SDKEvent(
            type="run_start",
            session_id=session_id,
            data={"message": message, "mode": mode, "stream": stream},
        )

        async def run_to_queue() -> None:
            try:
                result = await runtime.run(message, mode=mode, stream=stream)
                await queue.put(SDKEvent("run_complete", session_id, {"result": result}))
            except asyncio.CancelledError:
                await queue.put(SDKEvent("run_cancelled", session_id, {"reason": "Run cancelled"}))
                raise
            except Exception as e:
                await queue.put(SDKEvent("run_error", session_id, {"error": str(e)}))

        task = asyncio.create_task(run_to_queue())
        self._active_runs[session_id] = task
        try:
            while True:
                event = await queue.get()
                yield event
                if event.type in {"run_complete", "run_error", "run_cancelled"}:
                    break
        finally:
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            if self._active_runs.get(session_id) is task:
                self._active_runs.pop(session_id, None)
            for unsubscribe in reversed(unsubscribers):
                with suppress(Exception):
                    unsubscribe()

    async def cancel_run(self, session_id: str) -> bool:
        """取消运行，并按照当前组件的约定返回结果。

        参数：
            session_id: 目标会话的稳定标识。

        返回：
            表示条件是否成立。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        self._ensure_open()
        self.get_runtime(session_id)
        task = self._active_runs.get(session_id)
        if task is None or task.done():
            return False
        runtime = self._sessions[session_id]
        cancel_runtime = getattr(runtime, "cancel_run", None)
        if not callable(cancel_runtime) or not cancel_runtime("SDK run cancelled"):
            task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        return True

    async def close_session(self, session_id: str) -> bool:
        """关闭会话，并按照当前组件的约定返回结果。

        参数：
            session_id: 目标会话的稳定标识。

        返回：
            表示条件是否成立。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        if session_id not in self._sessions:
            return False
        task = self._active_runs.get(session_id)
        if task is not None and not task.done():
            runtime = self._sessions[session_id]
            cancel_runtime = getattr(runtime, "cancel_run", None)
            if not callable(cancel_runtime) or not cancel_runtime("SDK session closed"):
                task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        self._active_runs.pop(session_id, None)

        runtime = self._sessions.pop(session_id)
        closer = getattr(runtime, "aclose", None)
        if callable(closer):
            result = closer()
            if inspect.isawaitable(result):
                await result
        else:
            closer = getattr(runtime, "close", None)
            if callable(closer):
                closer()
        return True

    async def aclose(self) -> None:
        """异步关闭当前对象持有的任务、连接和运行时资源；重复调用保持幂等。

        说明：
            执行过程中会更新当前实例维护的状态。
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        if self._closed:
            return
        for session_id in list(self._sessions):
            await self.close_session(session_id)
        self._closed = True

    async def __aenter__(self) -> OpenNovaClient:
        self._ensure_open()
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    @staticmethod
    def _tool_result_data(result: ToolResult) -> dict[str, Any]:
        return {
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "metadata": result.metadata,
        }
