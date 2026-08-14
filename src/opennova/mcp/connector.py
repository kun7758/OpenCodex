"""MCP 集成层中的连接模块，集中定义相关数据结构、边界适配和实现逻辑。"""

import asyncio
import inspect
import json
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx

from opennova.mcp.types import (
    MCPConnectionState,
    MCPMessage,
    MCPResource,
    MCPResourceContent,
    MCPServerConfig,
    MCPServerInfo,
    MCPTool,
    MCPToolResult,
    TransportType,
)
from opennova.tools.base import BaseTool, ToolRegistry, ToolResult

SUPPORTED_PROTOCOL_VERSION = "2024-11-05"


def _client_version() -> str:
    try:
        return version("opennova")
    except PackageNotFoundError:
        return "0.4.3"


class Transport(ABC):
    """封装传输层相关的状态和操作，使调用方通过稳定接口使用该能力。"""

    @abstractmethod
    async def connect(self) -> None:
        """处理连接，并按照当前组件的约定返回结果。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """处理断开连接，并按照当前组件的约定返回结果。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        pass

    @abstractmethod
    async def send(self, message: MCPMessage) -> None:
        """处理发送，并按照当前组件的约定返回结果。

        参数：
            message: 用户提交或组件间传递的消息。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        pass

    @abstractmethod
    async def receive(self) -> MCPMessage:
        """处理接收，并按照当前组件的约定返回结果。

        返回：
            `MCPMessage` 类型的处理结果。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        pass

    @abstractmethod
    def receive_stream(self) -> AsyncIterator[MCPMessage]:
        """根据当前输入和传输层的状态计算 `receive_stream`，并返回调用方需要的结果。

        生成：
            逐项产生结果，直到数据源结束。
        """
        raise NotImplementedError

    @abstractmethod
    def is_connected(self) -> bool:
        """判断`connected`条件是否成立。

        返回：
            表示条件是否成立。
        """
        pass


class StdioTransport(Transport):
    """封装标准输入输出传输层相关的状态和操作，使调用方通过稳定接口使用该能力。"""

    def __init__(self, config: MCPServerConfig):
        """初始化标准输入输出传输层，保存后续操作需要的依赖、配置和初始状态。

        参数：
            config: 控制当前组件行为的配置。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self.config = config
        self.process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._response_queue: asyncio.Queue[MCPMessage] = asyncio.Queue()
        self._request_id = 0

    async def connect(self) -> None:
        """处理连接，并按照当前组件的约定返回结果。

        说明：
            执行过程中会更新当前实例维护的状态。
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        if not self.config.command:
            raise ValueError("Command is required for stdio transport")

        env = os.environ.copy()
        env.update(self.config.env)

        self.process = await asyncio.create_subprocess_exec(
            self.config.command,
            *self.config.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        self._reader_task = asyncio.create_task(self._read_loop())

    async def disconnect(self) -> None:
        """处理断开连接，并按照当前组件的约定返回结果。

        说明：
            执行过程中会更新当前实例维护的状态。
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        if self._reader_task:
            self._reader_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._reader_task

        if self.process:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
            self.process = None

    async def _read_loop(self) -> None:
        """读取循环，并按照当前组件的约定返回结果。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        if not self.process or not self.process.stdout:
            return

        buffer = ""
        while True:
            try:
                chunk = await self.process.stdout.read(4096)
                if not chunk:
                    break

                buffer += chunk.decode("utf-8")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            message = MCPMessage.from_dict(data)
                            await self._response_queue.put(message)
                        except json.JSONDecodeError:
                            pass

            except asyncio.CancelledError:
                break
            except Exception:
                break

    async def send(self, message: MCPMessage) -> None:
        """处理发送，并按照当前组件的约定返回结果。

        参数：
            message: 用户提交或组件间传递的消息。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        if not self.process or not self.process.stdin:
            raise RuntimeError("Not connected to MCP server")

        data = message.to_dict()
        line = json.dumps(data) + "\n"
        self.process.stdin.write(line.encode("utf-8"))
        await self.process.stdin.drain()

    async def receive(self) -> MCPMessage:
        """处理接收，并按照当前组件的约定返回结果。

        返回：
            `MCPMessage` 类型的处理结果。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        return await self._response_queue.get()

    async def receive_stream(self) -> AsyncIterator[MCPMessage]:
        """根据当前输入和标准输入输出传输层的状态计算 `receive_stream`，并返回调用方需要的结果。

        生成：
            逐项产生结果，直到数据源结束。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        while True:
            message = await self._response_queue.get()
            yield message

    def is_connected(self) -> bool:
        """判断`connected`条件是否成立。

        返回：
            表示条件是否成立。
        """
        return self.process is not None and self.process.returncode is None

    def get_next_id(self) -> int:
        """读取 `next_id` 对应的数据，不改变当前对象的业务状态。

        返回：
            `int` 类型的处理结果。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self._request_id += 1
        return self._request_id


class SSETransport(Transport):
    """封装`SSETransport`相关的状态和操作，使调用方通过稳定接口使用该能力。"""

    def __init__(self, config: MCPServerConfig):
        """初始化`SSETransport`，保存后续操作需要的依赖、配置和初始状态。

        参数：
            config: 控制当前组件行为的配置。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self.config = config
        self._connected = False
        self._request_id = 0
        self._client: httpx.AsyncClient | None = None

    def _message_url(self) -> str:
        if not self.config.url:
            raise RuntimeError("No URL configured")
        if self.config.url.endswith("/sse"):
            return self.config.url[:-4] + "/messages"
        raise RuntimeError("SSE transport URL must end with /sse")

    async def connect(self) -> None:
        """处理连接，并按照当前组件的约定返回结果。

        说明：
            执行过程中会更新当前实例维护的状态。
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        if not self.config.url:
            raise ValueError("URL is required for SSE transport")
        import httpx

        self._client = httpx.AsyncClient(timeout=self.config.timeout)
        self._connected = True

    async def disconnect(self) -> None:
        """处理断开连接，并按照当前组件的约定返回结果。

        说明：
            执行过程中会更新当前实例维护的状态。
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        self._connected = False
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def send(self, message: MCPMessage) -> None:
        """处理发送，并按照当前组件的约定返回结果。

        参数：
            message: 用户提交或组件间传递的消息。

        说明：
            执行过程中会更新当前实例维护的状态。
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        import httpx

        client = self._client
        if client is None:
            client = httpx.AsyncClient(timeout=self.config.timeout)
            self._client = client
        data = message.to_dict()

        response = await client.post(
            self._message_url(),
            json=data,
            timeout=self.config.timeout,
        )
        response.raise_for_status()

    async def receive(self) -> MCPMessage:
        """处理接收，并按照当前组件的约定返回结果。

        返回：
            `MCPMessage` 类型的处理结果。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        raise NotImplementedError("SSE uses streaming only")

    async def receive_stream(self) -> AsyncIterator[MCPMessage]:
        """根据当前输入和`SSETransport`的状态计算 `receive_stream`，并返回调用方需要的结果。

        生成：
            逐项产生结果，直到数据源结束。

        说明：
            执行过程中会更新当前实例维护的状态。
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        import httpx

        if not self.config.url:
            raise RuntimeError("No URL configured")
        client = self._client
        if client is None:
            client = httpx.AsyncClient(timeout=self.config.timeout)
            self._client = client

        async with client.stream(
            "GET",
            self.config.url,
            timeout=self.config.timeout,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise RuntimeError("Invalid SSE payload from MCP server") from exc
                yield MCPMessage.from_dict(data)

    def is_connected(self) -> bool:
        """判断`connected`条件是否成立。

        返回：
            表示条件是否成立。
        """
        return self._connected

    def get_next_id(self) -> int:
        """读取 `next_id` 对应的数据，不改变当前对象的业务状态。

        返回：
            `int` 类型的处理结果。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self._request_id += 1
        return self._request_id


class MCPConnector:
    """管理单个 MCP 服务端的协议生命周期，包括初始化、能力发现、请求关联、工具调用以及资源和提示词访问。"""

    def __init__(self, config: MCPServerConfig):
        """初始化MCP连接，保存后续操作需要的依赖、配置和初始状态。

        参数：
            config: 控制当前组件行为的配置。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self.config = config
        self.transport: Transport | None = None
        self.state = MCPConnectionState.DISCONNECTED
        self.server_info: MCPServerInfo | None = None
        self.tools: dict[str, MCPTool] = {}
        self._pending_requests: dict[int, asyncio.Future[Any]] = {}
        self._request_id = 0
        self._listener_task: asyncio.Task[None] | None = None
        self._initialized = asyncio.Event()
        self.last_error: str | None = None
        self.on_tools_changed: Callable[[], Any] | None = None
        self.roots_provider: Callable[[], list[dict[str, Any]]] | None = None
        self.elicitation_handler: Callable[[dict[str, Any]], Any] | None = None

    async def connect(self) -> None:
        """处理连接，并按照当前组件的约定返回结果。

        说明：
            执行过程中会更新当前实例维护的状态。
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        if self.state == MCPConnectionState.CONNECTED:
            return

        self.state = MCPConnectionState.CONNECTING
        self._initialized = asyncio.Event()
        self.last_error = None

        try:
            if self.config.transport == TransportType.STDIO:
                transport: Transport = StdioTransport(self.config)
            elif self.config.transport == TransportType.SSE:
                transport = SSETransport(self.config)
            else:
                raise ValueError(f"Unsupported transport: {self.config.transport}")
            self.transport = transport

            await transport.connect()

            self._listener_task = asyncio.create_task(self._listen_loop())

            self.server_info = await self._initialize()

            await self._discover_tools()

            self.state = MCPConnectionState.CONNECTED

        except Exception as e:
            self.last_error = str(e)
            self.state = MCPConnectionState.ERROR
            self._fail_pending_requests(e)
            if self._listener_task:
                self._listener_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._listener_task
                self._listener_task = None
            if self.transport:
                try:
                    await self.transport.disconnect()
                finally:
                    self.transport = None
            raise RuntimeError(f"Failed to connect to MCP server: {e}") from e

    async def disconnect(self) -> None:
        """处理断开连接，并按照当前组件的约定返回结果。

        说明：
            执行过程中会更新当前实例维护的状态。
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        self._fail_pending_requests(RuntimeError("MCP server disconnected"))

        if self._listener_task:
            self._listener_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._listener_task
            self._listener_task = None

        if self.transport:
            await self.transport.disconnect()
            self.transport = None

        self.state = MCPConnectionState.DISCONNECTED
        self.tools.clear()

    async def _listen_loop(self) -> None:
        """执行 `_listen_loop` 所定义的协调步骤，必要时更新MCP连接维护的状态。

        说明：
            执行过程中会更新当前实例维护的状态。
            该操作可能等待模型服务、MCP 服务或其他异步数据源返回。
        """
        transport = self.transport
        if transport is None:
            return

        try:
            async for message in transport.receive_stream():
                if message.id is not None and message.id in self._pending_requests:
                    future = self._pending_requests.pop(message.id)
                    if future.done():
                        continue
                    if message.error:
                        future.set_exception(RuntimeError(message.error))
                    else:
                        future.set_result(message.result)
                    continue

                if message.method == "notifications/initialized":
                    self._initialized.set()
                    continue
                if message.method in {
                    "notifications/tools/list_changed",
                    "notifications/resources/list_changed",
                    "notifications/prompts/list_changed",
                }:
                    await self._handle_list_changed(message.method)
                    continue
                if message.id is not None and message.method:
                    await self._handle_server_request(message)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.last_error = str(e)
            self.state = MCPConnectionState.ERROR
            self._fail_pending_requests(e)

    def _fail_pending_requests(self, error: Exception) -> None:
        """执行 `_fail_pending_requests` 所定义的协调步骤，必要时更新MCP连接维护的状态。

        参数：
            error: 本次操作使用的错误。
        """
        for future in self._pending_requests.values():
            if not future.done():
                future.set_exception(error)
        self._pending_requests.clear()

    def _get_next_id(self) -> int:
        """读取并返回 `_get_next_id` 所表示的数据或流程，并遵守MCP连接定义的边界与状态约束。

        返回：
            `int` 类型的处理结果。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self._request_id += 1
        return self._request_id

    async def _send_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> Any:
        """发送请求，并按照当前组件的约定返回结果。

        参数：
            method: 本次操作使用的`method`。
            params: 可选的`params`。
            timeout: 可选的超时。

        返回：
            `Any` 类型的处理结果。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        transport = self.transport
        if transport is None:
            raise RuntimeError("Not connected to MCP server")

        request_id = self._get_next_id()
        message = MCPMessage(
            id=request_id,
            method=method,
            params=params,
        )

        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending_requests[request_id] = future

        try:
            await transport.send(message)
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.CancelledError:
            if not future.done():
                future.cancel()
            with suppress(Exception):
                await asyncio.shield(
                    transport.send(
                        MCPMessage(
                            method="notifications/cancelled",
                            params={"requestId": request_id, "reason": "OpenNova run cancelled"},
                        )
                    )
                )
            raise
        except TimeoutError:
            if not future.done():
                future.cancel()
            raise RuntimeError(f"Request {method} timed out") from None
        finally:
            self._pending_requests.pop(request_id, None)

    async def _initialize(self) -> MCPServerInfo:
        """处理初始化，并按照当前组件的约定返回结果。

        返回：
            `MCPServerInfo` 类型的处理结果。

        说明：
            该操作可能等待模型服务、MCP 服务或其他异步数据源返回。
        """
        result = await self._send_request(
            "initialize",
            {
                "protocolVersion": SUPPORTED_PROTOCOL_VERSION,
                "clientInfo": {
                    "name": "OpenNova",
                    "version": _client_version(),
                },
                "capabilities": {
                    "roots": {"listChanged": True},
                    "elicitation": {},
                },
            },
        )
        transport = self.transport
        if transport is None:
            raise RuntimeError("MCP transport disconnected during initialization")
        await transport.send(MCPMessage(method="notifications/initialized", params={}))
        self._initialized.set()
        return MCPServerInfo.from_dict(result, self.config.name)

    async def _discover_tools(self) -> None:
        """发现工具，并按照当前组件的约定返回结果。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        tools_data = await self._list_paginated("tools/list", "tools")
        self.tools.clear()
        for tool_data in tools_data:
            tool = MCPTool(
                name=tool_data.get("name", ""),
                description=tool_data.get("description", ""),
                input_schema=tool_data.get("inputSchema", {}),
                server_name=self.config.name,
            )
            self.tools[tool.get_full_name()] = tool

    async def _list_paginated(
        self,
        method: str,
        key: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """根据当前输入和MCP连接的状态计算 `_list_paginated`，并返回调用方需要的结果。

        参数：
            method: 本次操作使用的`method`。
            key: 本次操作使用的`key`。
            params: 可选的`params`。

        返回：
            按调用约定排序的结果列表。

        说明：
            该操作可能等待模型服务、MCP 服务或其他异步数据源返回。
        """
        items: list[dict[str, Any]] = []
        cursor: str | None = None
        seen: set[str] = set()
        while True:
            request_params = dict(params or {})
            if cursor:
                request_params["cursor"] = cursor
            result = await self._send_request(
                method,
                request_params or None,
                timeout=self.config.timeout,
            )
            page = result.get(key, []) if isinstance(result, dict) else []
            items.extend(item for item in page if isinstance(item, dict))
            next_cursor = result.get("nextCursor") if isinstance(result, dict) else None
            if not next_cursor or str(next_cursor) in seen:
                break
            cursor = str(next_cursor)
            seen.add(cursor)
        return items

    async def _handle_list_changed(self, method: str) -> None:
        if method == "notifications/tools/list_changed":
            await self._discover_tools()
            if self.on_tools_changed:
                result = self.on_tools_changed()
                if inspect.isawaitable(result):
                    await result

    async def _handle_server_request(self, message: MCPMessage) -> None:
        transport = self.transport
        if transport is None:
            return
        try:
            if message.method == "roots/list":
                roots = self.roots_provider() if self.roots_provider else []
                result: Any = {"roots": roots}
            elif message.method == "elicitation/create":
                if self.elicitation_handler is None:
                    result = {"action": "decline"}
                else:
                    result = self.elicitation_handler(message.params or {})
                    if inspect.isawaitable(result):
                        result = await result
            else:
                await transport.send(
                    MCPMessage(
                        id=message.id,
                        error={
                            "code": -32601,
                            "message": f"Method not supported: {message.method}",
                        },
                    )
                )
                return
            await transport.send(MCPMessage(id=message.id, result=result))
        except Exception as exc:
            await transport.send(
                MCPMessage(
                    id=message.id,
                    error={"code": -32603, "message": str(exc)},
                )
            )

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> MCPToolResult:
        """启动或推进 `call_tool` 所表示的数据或流程，并遵守MCP连接定义的边界与状态约束。

        参数：
            tool_name: 目标工具在注册表中的名称。
            arguments: 工具调用的结构化参数。

        返回：
            `MCPToolResult` 类型的处理结果。

        说明：
            该操作可能等待模型服务、MCP 服务或其他异步数据源返回。
        """
        if self.state != MCPConnectionState.CONNECTED:
            raise RuntimeError("Not connected to MCP server")

        try:
            result = await self._send_request(
                "tools/call",
                {
                    "name": tool_name,
                    "arguments": arguments,
                },
                timeout=self.config.timeout,
            )

            content_blocks = result.get("content", [])

            if isinstance(content_blocks, list):
                content = "\n".join(
                    block.get("text", str(block))
                    for block in content_blocks
                    if isinstance(block, dict)
                )
            else:
                content = str(content_blocks)

            is_error = result.get("isError", False)

            return MCPToolResult(
                success=not is_error,
                content=content,
                error=content if is_error else None,
            )

        except Exception as e:
            return MCPToolResult(
                success=False,
                content="",
                error=str(e),
            )

    async def list_resources(self) -> list[MCPResource]:
        """列出资源，并按当前组件约定返回稳定顺序。

        返回：
            按调用约定排序的结果列表。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        if self.state != MCPConnectionState.CONNECTED:
            raise RuntimeError("Not connected to MCP server")

        resources_data = await self._list_paginated("resources/list", "resources")
        resources = []
        for resource_data in resources_data:
            resources.append(
                MCPResource(
                    uri=resource_data.get("uri", ""),
                    name=resource_data.get("name", ""),
                    description=resource_data.get("description", ""),
                    mime_type=resource_data.get("mimeType", resource_data.get("mime_type", "")),
                    server_name=self.config.name,
                    metadata={
                        key: value
                        for key, value in resource_data.items()
                        if key not in {"uri", "name", "description", "mimeType", "mime_type"}
                    },
                )
            )
        return resources

    async def list_resource_templates(self) -> list[dict[str, Any]]:
        """列出 `resource_templates` 对应的对象，并按当前组件约定返回稳定顺序。

        返回：
            按调用约定排序的结果列表。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        return await self._list_paginated("resources/templates/list", "resourceTemplates")

    async def list_prompts(self) -> list[dict[str, Any]]:
        """列出 `prompts` 对应的对象，并按当前组件约定返回稳定顺序。

        返回：
            按调用约定排序的结果列表。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        return await self._list_paginated("prompts/list", "prompts")

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """读取提示词，不改变当前对象的业务状态。

        参数：
            name: 待查询、注册或操作对象的名称。
            arguments: 工具调用的结构化参数。

        返回：
            供后续逻辑或序列化使用的结构化字典。

        说明：
            该操作可能等待模型服务、MCP 服务或其他异步数据源返回。
        """
        result = await self._send_request(
            "prompts/get",
            {"name": name, "arguments": arguments or {}},
            timeout=self.config.timeout,
        )
        return result if isinstance(result, dict) else {}

    async def read_resource(self, uri: str) -> MCPResourceContent:
        """读取资源，并按照当前组件的约定返回结果。

        参数：
            uri: 本次操作使用的`uri`。

        返回：
            `MCPResourceContent` 类型的处理结果。

        说明：
            该操作可能等待模型服务、MCP 服务或其他异步数据源返回。
        """
        if self.state != MCPConnectionState.CONNECTED:
            raise RuntimeError("Not connected to MCP server")

        try:
            result = await self._send_request(
                "resources/read",
                {"uri": uri},
                timeout=self.config.timeout,
            )
            contents = result.get("contents", [])
            rendered_parts: list[str] = []
            metadata: dict[str, Any] = {"server_name": self.config.name, "uri": uri}

            if isinstance(contents, list):
                for block in contents:
                    if not isinstance(block, dict):
                        rendered_parts.append(str(block))
                        continue
                    if block.get("text") is not None:
                        rendered_parts.append(str(block.get("text", "")))
                    elif block.get("blob") is not None:
                        rendered_parts.append(str(block.get("blob", "")))
                    if block.get("mimeType") or block.get("mime_type"):
                        metadata["mime_type"] = block.get("mimeType", block.get("mime_type"))
            else:
                rendered_parts.append(str(contents))

            return MCPResourceContent(
                success=True,
                content="\n".join(rendered_parts),
                metadata=metadata,
            )
        except Exception as e:
            return MCPResourceContent(
                success=False,
                content="",
                error=str(e),
                metadata={"server_name": self.config.name, "uri": uri},
            )

    def get_tools(self) -> list[MCPTool]:
        """读取工具，不改变当前对象的业务状态。

        返回：
            按调用约定排序的结果列表。
        """
        return list(self.tools.values())

    def is_connected(self) -> bool:
        """判断`connected`条件是否成立。

        返回：
            表示条件是否成立。
        """
        return self.state == MCPConnectionState.CONNECTED


class MCPToolWrapper(BaseTool):
    """封装MCP工具包装器相关的状态和操作，使调用方通过稳定接口使用该能力。"""

    def __init__(self, mcp_tool: MCPTool, connector: MCPConnector):
        """初始化MCP工具包装器，保存后续操作需要的依赖、配置和初始状态。

        参数：
            mcp_tool: 本次操作使用的MCP工具。
            connector: 本次操作使用的连接。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self.mcp_tool = mcp_tool
        self.connector = connector
        self.name = mcp_tool.get_full_name()
        self.description = mcp_tool.description
        self.parameters = mcp_tool.input_schema

    def execute(self, **kwargs: Any) -> ToolResult:
        """执行MCP工具包装器对应的实际操作，校验输入并返回统一结果。

        参数：
            **kwargs: 传递给底层实现的额外关键字参数。

        返回：
            `ToolResult` 类型的处理结果。
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.async_execute(**kwargs))
        raise RuntimeError("MCP tools must be executed via async_execute inside the runtime loop")

    def get_security_context(self) -> dict[str, Any]:
        """读取安全上下文，不改变当前对象的业务状态。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        config = self.connector.config
        return {
            "kind": "mcp",
            "server": config.name,
            "tool": self.mcp_tool.name,
            "trusted": config.trusted,
            "allowed_tools": list(config.allowed_tools),
            "denied_tools": list(config.denied_tools),
            "require_confirmation": config.require_confirmation,
        }

    async def async_execute(self, **kwargs: Any) -> ToolResult:
        """执行MCP工具包装器对应的实际操作，校验输入并返回统一结果。

        参数：
            **kwargs: 传递给底层实现的额外关键字参数。

        返回：
            `ToolResult` 类型的处理结果。

        说明：
            该操作可能等待模型服务、MCP 服务或其他异步数据源返回。
        """
        from opennova.runtime.events import current_tool_context

        context = current_tool_context()
        if context and context.abort_signal:
            context.abort_signal.raise_if_cancelled()
        original_name = self.mcp_tool.name
        result = await self.connector.call_tool(original_name, kwargs)

        return ToolResult(
            success=result.success,
            output=result.content,
            error=result.error,
            metadata={
                **result.metadata,
                "mcp_server": self.connector.config.name,
                "mcp_tool": self.mcp_tool.name,
                "mcp_trusted": self.connector.config.trusted,
            },
        )


class MCPManager:
    """管理多个 MCPConnector，并把远程工具同步到当前运行时的工具命名空间。"""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        *,
        roots: list[dict[str, Any]] | None = None,
        elicitation_handler: Callable[[dict[str, Any]], Any] | None = None,
    ):
        """初始化MCP管理，保存后续操作需要的依赖、配置和初始状态。

        参数：
            tool_registry: 本次操作使用的工具注册表。
            roots: 可选的`roots`。
            elicitation_handler: 可选的`elicitation_handler`。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self.tool_registry = tool_registry
        self.connectors: dict[str, MCPConnector] = {}
        self._registered_tools_by_server: dict[str, list[str]] = {}
        self.connection_errors: dict[str, str] = {}
        self.roots = list(roots or [])
        self.elicitation_handler = elicitation_handler

    def set_roots(self, roots: list[dict[str, Any]]) -> None:
        """设置`set_roots`并保持相关派生状态同步。

        参数：
            roots: 本次操作使用的`roots`。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self.roots = list(roots)

    def set_elicitation_handler(
        self,
        handler: Callable[[dict[str, Any]], Any] | None,
    ) -> None:
        """设置`set_elicitation_handler`并保持相关派生状态同步。

        参数：
            handler: 可选的处理器。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self.elicitation_handler = handler

    def _connector_roots(self) -> list[dict[str, Any]]:
        return list(self.roots)

    async def _sync_server_tools(self, server_name: str) -> None:
        """同步服务端工具，并按照当前组件的约定返回结果。

        参数：
            server_name: 本次操作使用的`server_name`。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        connector = self.connectors.get(server_name)
        if connector is None:
            return

        previous_names = self._registered_tools_by_server.get(server_name, [])
        for tool_name in previous_names:
            self.tool_registry.unregister(tool_name)

        registered_tools: list[str] = []
        for mcp_tool in connector.get_tools():
            wrapper = MCPToolWrapper(mcp_tool, connector)
            self.tool_registry.register(wrapper)
            registered_tools.append(wrapper.name)
        self._registered_tools_by_server[server_name] = registered_tools

    async def add_server(self, config: MCPServerConfig) -> bool:
        """添加`add_server`，必要时执行去重或容量检查。

        参数：
            config: 控制当前组件行为的配置。

        返回：
            表示条件是否成立。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        if config.name in self.connectors:
            return self.connectors[config.name].is_connected()

        if not config.enabled:
            self.connection_errors.pop(config.name, None)
            return False

        connector = MCPConnector(config)
        connector.roots_provider = self._connector_roots
        connector.elicitation_handler = lambda params: (
            self.elicitation_handler(params)
            if self.elicitation_handler is not None
            else {"action": "decline"}
        )
        connector.on_tools_changed = lambda: self._sync_server_tools(config.name)

        try:
            await connector.connect()
            self.connectors[config.name] = connector
            self.connection_errors.pop(config.name, None)
            await self._sync_server_tools(config.name)

            return True

        except Exception as e:
            self.connection_errors[config.name] = str(e)
            return False

    async def remove_server(self, name: str) -> None:
        """移除移除服务端指向的数据，并清理相关索引或资源。

        参数：
            name: 待查询、注册或操作对象的名称。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        for tool_name in self._registered_tools_by_server.pop(name, []):
            self.tool_registry.unregister(tool_name)

        if name in self.connectors:
            connector = self.connectors.pop(name)
            await connector.disconnect()

    async def connect_all(self, configs: list[MCPServerConfig]) -> dict[str, bool]:
        """连接全部，并按照当前组件的约定返回结果。

        参数：
            configs: 本次操作使用的`configs`。

        返回：
            供后续逻辑或序列化使用的结构化字典。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        results = {}

        for config in configs:
            results[config.name] = await self.add_server(config)

        return results

    async def disconnect_all(self) -> None:
        """断开全部，并按照当前组件的约定返回结果。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        for name in list(self.connectors.keys()):
            await self.remove_server(name)

    def get_server_names(self) -> list[str]:
        """读取服务端名称，不改变当前对象的业务状态。

        返回：
            按调用约定排序的结果列表。
        """
        return list(self.connectors.keys())

    def get_all_tools(self) -> list[MCPTool]:
        """读取全部工具，不改变当前对象的业务状态。

        返回：
            按调用约定排序的结果列表。
        """
        tools = []
        for connector in self.connectors.values():
            tools.extend(connector.get_tools())
        return tools

    async def list_resources(self, server_name: str | None = None) -> list[MCPResource]:
        """列出资源，并按当前组件约定返回稳定顺序。

        参数：
            server_name: 可选的`server_name`。

        返回：
            按调用约定排序的结果列表。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        resources: list[MCPResource] = []
        connectors = (
            [self.connectors[server_name]]
            if server_name and server_name in self.connectors
            else list(self.connectors.values())
        )
        for connector in connectors:
            resources.extend(await connector.list_resources())
        return resources

    async def list_resource_templates(
        self,
        server_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """列出 `resource_templates` 对应的对象，并按当前组件约定返回稳定顺序。

        参数：
            server_name: 可选的`server_name`。

        返回：
            按调用约定排序的结果列表。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        templates: list[dict[str, Any]] = []
        connectors = self._selected_connectors(server_name)
        for connector in connectors:
            for template in await connector.list_resource_templates():
                templates.append({**template, "server_name": connector.config.name})
        return templates

    async def list_prompts(self, server_name: str | None = None) -> list[dict[str, Any]]:
        """列出 `prompts` 对应的对象，并按当前组件约定返回稳定顺序。

        参数：
            server_name: 可选的`server_name`。

        返回：
            按调用约定排序的结果列表。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        prompts: list[dict[str, Any]] = []
        connectors = self._selected_connectors(server_name)
        for connector in connectors:
            for prompt in await connector.list_prompts():
                prompts.append({**prompt, "server_name": connector.config.name})
        return prompts

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, str] | None = None,
        server_name: str | None = None,
    ) -> dict[str, Any]:
        """读取提示词，不改变当前对象的业务状态。

        参数：
            name: 待查询、注册或操作对象的名称。
            arguments: 工具调用的结构化参数。
            server_name: 可选的`server_name`。

        返回：
            供后续逻辑或序列化使用的结构化字典。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        connectors = self._selected_connectors(server_name)
        if not connectors:
            raise RuntimeError(
                f"MCP server not connected: {server_name}"
                if server_name
                else "No MCP servers connected"
            )
        return await connectors[0].get_prompt(name, arguments)

    def _selected_connectors(self, server_name: str | None) -> list[MCPConnector]:
        if server_name is None:
            return list(self.connectors.values())
        connector = self.connectors.get(server_name)
        return [connector] if connector is not None else []

    async def read_resource(
        self,
        uri: str,
        server_name: str | None = None,
    ) -> MCPResourceContent:
        """读取资源，并按照当前组件的约定返回结果。

        参数：
            uri: 本次操作使用的`uri`。
            server_name: 可选的`server_name`。

        返回：
            `MCPResourceContent` 类型的处理结果。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        if server_name:
            connector = self.connectors.get(server_name)
            if not connector:
                return MCPResourceContent(
                    False, "", error=f"MCP server not connected: {server_name}"
                )
            return await connector.read_resource(uri)

        last_error = ""
        for connector in self.connectors.values():
            result = await connector.read_resource(uri)
            if result.success:
                return result
            last_error = result.error or last_error
        return MCPResourceContent(False, "", error=last_error or f"MCP resource not found: {uri}")

    def get_server_for_tool(self, tool_name: str) -> MCPConnector | None:
        """读取服务端对应工具，不改变当前对象的业务状态。

        参数：
            tool_name: 目标工具在注册表中的名称。

        返回：
            `MCPConnector | None` 类型的处理结果。
        """
        for connector in self.connectors.values():
            if tool_name in connector.tools:
                return connector
        return None
