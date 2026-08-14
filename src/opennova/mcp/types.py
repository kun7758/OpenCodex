"""MCP 集成层中的`types`模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class TransportType(StrEnum):
    """枚举传输层类型允许出现的稳定取值，序列化和状态判断均使用这些值。"""

    STDIO = "stdio"
    SSE = "sse"
    WEBSOCKET = "websocket"


class MCPConnectionState(StrEnum):
    """枚举MCP连接状态允许出现的稳定取值，序列化和状态判断均使用这些值。"""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class MCPServerConfig:
    """保存MCP服务端配置所需的结构化数据，主要包含 `name`、`transport`、`command`、`args`、`url`、`env`、`timeout`、`enabled`
    等字段，便于在组件之间传递或持久化。
    """

    name: str
    transport: TransportType = TransportType.STDIO
    command: str | None = None
    args: list[str] = field(default_factory=list)
    url: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0
    enabled: bool = True
    trusted: bool = False
    allowed_tools: list[str] = field(default_factory=list)
    denied_tools: list[str] = field(default_factory=list)
    require_confirmation: bool = True

    def validate(self) -> None:
        """处理校验，并按照当前组件的约定返回结果。"""
        if not self.name or not isinstance(self.name, str):
            raise ValueError("MCP server name is required")
        if not isinstance(self.args, list):
            raise ValueError(f"MCP server {self.name}: args must be a list")
        if not isinstance(self.env, dict):
            raise ValueError(f"MCP server {self.name}: env must be a dict")
        if not isinstance(self.allowed_tools, list):
            raise ValueError(f"MCP server {self.name}: allowed_tools must be a list")
        if not isinstance(self.denied_tools, list):
            raise ValueError(f"MCP server {self.name}: denied_tools must be a list")
        if not isinstance(self.timeout, (int, float)) or self.timeout <= 0:
            raise ValueError(f"MCP server {self.name}: timeout must be a positive number")

        if self.transport == TransportType.STDIO and not self.command:
            raise ValueError(f"MCP server {self.name}: command is required for stdio transport")
        if self.transport == TransportType.SSE and not self.url:
            raise ValueError(f"MCP server {self.name}: url is required for sse transport")
        if self.transport == TransportType.WEBSOCKET:
            raise ValueError(f"MCP server {self.name}: websocket transport is not yet supported")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPServerConfig":
        """从字典恢复MCP服务端配置，并为旧数据缺失的字段补充兼容默认值。

        参数：
            data: 用于构造或恢复对象的结构化数据。

        返回：
            `'MCPServerConfig'` 类型的处理结果。
        """
        transport = TransportType(data.get("transport", "stdio"))

        config = cls(
            name=data["name"],
            transport=transport,
            command=data.get("command"),
            args=data.get("args", []),
            url=data.get("url"),
            env=data.get("env", {}),
            timeout=data.get("timeout", 30.0),
            enabled=data.get("enabled", True),
            trusted=bool(data.get("trusted", False)),
            allowed_tools=data.get("allowed_tools", []),
            denied_tools=data.get("denied_tools", []),
            require_confirmation=bool(data.get("require_confirmation", True)),
        )
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        """把MCP服务端配置转换为可序列化字典，供事件、会话或 API 边界使用。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        return {
            "name": self.name,
            "transport": self.transport.value,
            "command": self.command,
            "args": self.args,
            "url": self.url,
            "env": self.env,
            "timeout": self.timeout,
            "enabled": self.enabled,
            "trusted": self.trusted,
            "allowed_tools": self.allowed_tools,
            "denied_tools": self.denied_tools,
            "require_confirmation": self.require_confirmation,
        }


@dataclass
class MCPToolParameter:
    """描述 MCP 远程工具的单个参数，并负责生成对应的 JSON Schema 属性。"""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    default: Any = None

    def to_json_schema(self) -> dict[str, Any]:
        """把当前参数描述转换为 JSON Schema 属性；默认值、枚举、数组元素和对象属性仅在实际存在时写入。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        schema: dict[str, Any] = {
            "type": self.type,
            "description": self.description,
        }
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class MCPTool:
    """保存MCP工具所需的结构化数据，主要包含 `name`、`description`、`input_schema`、`server_name`、`annotations`
    字段，便于在组件之间传递或持久化。
    """

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    server_name: str = ""
    annotations: dict[str, Any] = field(default_factory=dict)

    def get_full_name(self) -> str:
        """读取 `full_name` 对应的数据，不改变当前对象的业务状态。

        返回：
            处理后的文本或稳定标识。
        """
        if self.server_name:
            return f"{self.server_name}_{self.name}"
        return self.name

    def to_tool_schema(self) -> dict[str, Any]:
        """把MCP工具转换为工具Schema，供对应协议或边界直接使用。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        return {
            "type": "function",
            "function": {
                "name": self.get_full_name(),
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass
class MCPToolResult:
    """保存MCP工具结果所需的结构化数据，主要包含 `success`、`content`、`error`、`metadata`、`timestamp` 字段，便于在组件之间传递或持久化。"""

    success: bool
    content: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_string(self) -> str:
        """把MCP工具结果转换为适合写入模型上下文或终端展示的文本。

        返回：
            处理后的文本或稳定标识。
        """
        if self.success:
            return self.content
        return f"Error: {self.error}\n{self.content}"


@dataclass
class MCPResource:
    """保存MCP资源所需的结构化数据，主要包含 `uri`、`name`、`description`、`mime_type`、`server_name`、`metadata`
    字段，便于在组件之间传递或持久化。
    """

    uri: str
    name: str = ""
    description: str = ""
    mime_type: str = ""
    server_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """把MCP资源转换为可序列化字典，供事件、会话或 API 边界使用。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mime_type": self.mime_type,
            "server_name": self.server_name,
            "metadata": self.metadata,
        }


@dataclass
class MCPResourceContent:
    """保存MCP资源内容所需的结构化数据，主要包含 `success`、`content`、`error`、`metadata` 字段，便于在组件之间传递或持久化。"""

    success: bool
    content: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPMessage:
    """保存MCP消息所需的结构化数据，主要包含 `jsonrpc`、`id`、`method`、`params`、`result`、`error` 字段，便于在组件之间传递或持久化。"""

    jsonrpc: str = "2.0"
    id: int | str | None = None
    method: str | None = None
    params: dict[str, Any] | None = None
    result: Any | None = None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """把MCP消息转换为可序列化字典，供事件、会话或 API 边界使用。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        data: dict[str, Any] = {"jsonrpc": self.jsonrpc}
        if self.id is not None:
            data["id"] = self.id
        if self.method:
            data["method"] = self.method
        if self.params is not None:
            data["params"] = self.params
        if self.result is not None:
            data["result"] = self.result
        if self.error is not None:
            data["error"] = self.error
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPMessage":
        """从字典恢复MCP消息，并为旧数据缺失的字段补充兼容默认值。

        参数：
            data: 用于构造或恢复对象的结构化数据。

        返回：
            `'MCPMessage'` 类型的处理结果。
        """
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            id=data.get("id"),
            method=data.get("method"),
            params=data.get("params"),
            result=data.get("result"),
            error=data.get("error"),
        )


@dataclass
class MCPServerInfo:
    """保存MCP服务端信息所需的结构化数据，主要包含 `name`、`version`、`protocol_version`、`capabilities`、`instructions`
    字段，便于在组件之间传递或持久化。
    """

    name: str
    version: str = ""
    protocol_version: str = ""
    capabilities: dict[str, Any] = field(default_factory=dict)
    instructions: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any], server_name: str) -> "MCPServerInfo":
        """从字典恢复MCP服务端信息，并为旧数据缺失的字段补充兼容默认值。

        参数：
            data: 用于构造或恢复对象的结构化数据。
            server_name: 本次操作使用的`server_name`。

        返回：
            `'MCPServerInfo'` 类型的处理结果。
        """
        return cls(
            name=server_name,
            version=data.get("serverInfo", {}).get("version", ""),
            protocol_version=data.get("protocolVersion", ""),
            capabilities=data.get("capabilities", {}),
            instructions=data.get("instructions", ""),
        )
