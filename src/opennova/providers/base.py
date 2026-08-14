"""模型服务适配层中的基础抽象模块，集中定义相关数据结构、边界适配和实现逻辑。"""

import asyncio
import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

ToolChoice = Literal["auto", "required", "none"]


class ProviderError(RuntimeError):
    """表示模型服务错误失败；调用方可以捕获该异常并转换为稳定的用户提示或 SDK 事件。"""

    code = "provider_error"

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        retryable: bool = False,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable
        self.status_code = status_code


class ProviderProtocolError(ProviderError):
    code = "provider_protocol_error"


class ProviderRateLimitError(ProviderError):
    code = "provider_rate_limit"


class ProviderTimeoutError(ProviderError):
    code = "provider_timeout"


class ProviderContextLengthError(ProviderError):
    code = "provider_context_length"


class ProviderRetryExhaustedError(ProviderError):
    code = "provider_retry_exhausted"


def parse_tool_arguments(raw: Any, *, tool_name: str, tool_call_id: str) -> dict[str, Any]:
    """解析工具参数并转换为内部使用的规范结构。

    参数：
        raw: 本次操作使用的`raw`。
        tool_name: 目标工具在注册表中的名称。
        tool_call_id: 模型工具调用与工具结果之间的关联标识。

    返回：
        供后续逻辑或序列化使用的结构化字典。
    """
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError) as exc:
        raise ProviderProtocolError(
            f"Malformed arguments for tool '{tool_name}' ({tool_call_id}): {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ProviderProtocolError(
            f"Arguments for tool '{tool_name}' ({tool_call_id}) must be a JSON object"
        )
    return parsed


def normalize_provider_error(exc: Exception, *, provider: str) -> ProviderError:
    """规范化模型服务错误，消除不同调用格式之间的差异。

    参数：
        exc: 本次操作使用的`exc`。
        provider: 负责本次模型请求的 Provider 实例。

    返回：
        `ProviderError` 类型的处理结果。
    """
    if isinstance(exc, ProviderError):
        return exc

    name = type(exc).__name__.lower()
    message = str(exc) or type(exc).__name__
    lowered = message.lower()
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)

    if status_code == 429 or "ratelimit" in name or "rate limit" in lowered:
        return ProviderRateLimitError(
            message, provider=provider, retryable=True, status_code=status_code
        )
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or "timeout" in name:
        return ProviderTimeoutError(
            message, provider=provider, retryable=True, status_code=status_code
        )
    if "context" in lowered and any(word in lowered for word in ("length", "window", "token")):
        return ProviderContextLengthError(
            message, provider=provider, retryable=False, status_code=status_code
        )
    retry_markers = ("max retries", "retries exhausted", "retry exhausted")
    if "retry" in name or any(marker in lowered for marker in retry_markers):
        return ProviderRetryExhaustedError(
            message,
            provider=provider,
            retryable=True,
            status_code=status_code,
        )
    return ProviderError(
        message,
        provider=provider,
        retryable=bool(status_code and status_code >= 500),
        status_code=status_code,
    )


class FinishReason(StrEnum):
    """枚举`FinishReason`允许出现的稳定取值，序列化和状态判断均使用这些值。"""

    STOP = "stop"
    TOOL_CALL = "tool_call"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"


@dataclass
class ToolCall:
    """模型生成的结构化工具调用，包含调用 ID、工具名称和已经解析为字典的参数。"""

    id: str
    name: str
    arguments: dict[str, Any]
    call_type: Literal["function"] = "function"


@dataclass
class ToolParameter:
    """描述一个工具参数的类型、说明、默认值、必填状态和嵌套结构，可进一步转换为 JSON Schema。"""

    type: str
    description: str = ""
    default: Any = None
    required: bool = True
    enum: list[Any] | None = None
    properties: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """把`ToolParameter`转换为可序列化字典，供事件、会话或 API 边界使用。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        schema: dict[str, Any] = {
            "type": self.type,
            "description": self.description,
        }
        if self.default is not None:
            schema["default"] = self.default
        if self.enum:
            schema["enum"] = self.enum
        if self.properties:
            schema["properties"] = self.properties
        return schema


@dataclass
class ToolSchema:
    """保存工具Schema所需的结构化数据，主要包含 `name`、`description`、`parameters` 字段，便于在组件之间传递或持久化。"""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai_format(self) -> dict[str, Any]:
        """把公共消息或工具 Schema 转换为 OpenAI API 接受的字典格式，同时保留工具调用关联信息。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class Message:
    """模型协议中的统一消息。role 区分 system、user、assistant 和 tool；tool_calls 与 tool_call_id 用于保持工具调用和结果之间的关联。"""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    token_count: int = 0
    reasoning_content: str | None = None
    is_compression_boundary: bool = False

    def to_openai_format(self) -> dict[str, Any]:
        """把公共消息或工具 Schema 转换为 OpenAI API 接受的字典格式，同时保留工具调用关联信息。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        msg: dict[str, Any] = {"role": self.role, "content": self.content}

        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.call_type,
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in self.tool_calls
            ]

        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id

        if self.name:
            msg["name"] = self.name

        if self.role == "assistant" and self.reasoning_content:
            msg["reasoning_content"] = self.reasoning_content

        return msg

    def to_anthropic_format(self) -> dict[str, Any]:
        """把公共消息转换为 Anthropic 内容块；工具结果使用 tool_result，模型工具调用使用 tool_use。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        if self.role == "tool":
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": self.tool_call_id,
                        "content": self.content,
                    }
                ],
            }

        msg: dict[str, Any] = {"role": self.role, "content": self.content}

        if self.role == "assistant" and self.tool_calls:
            content_blocks: list[dict[str, Any]] = []
            if self.content:
                content_blocks.append({"type": "text", "text": self.content})
            for tc in self.tool_calls:
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }
                )
            msg["content"] = content_blocks

        return msg

    def to_dict(self) -> dict[str, Any]:
        """把消息转换为可序列化字典，供事件、会话或 API 边界使用。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        data: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "token_count": self.token_count,
        }
        if self.tool_calls:
            data["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments, "call_type": tc.call_type}
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.name:
            data["name"] = self.name
        if self.reasoning_content:
            data["reasoning_content"] = self.reasoning_content
        if self.is_compression_boundary:
            data["is_compression_boundary"] = True
        return data

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Message":
        """从字典恢复消息，并为旧数据缺失的字段补充兼容默认值。

        参数：
            data: 用于构造或恢复对象的结构化数据。

        返回：
            `'Message'` 类型的处理结果。
        """
        tool_calls = None
        if "tool_calls" in data and data["tool_calls"]:
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    name=tc["name"],
                    arguments=tc.get("arguments", {}),
                    call_type=tc.get("call_type", "function"),
                )
                for tc in data["tool_calls"]
            ]
        timestamp = datetime.now()
        if "timestamp" in data:
            with suppress(ValueError, TypeError):
                timestamp = datetime.fromisoformat(data["timestamp"])
        return Message(
            role=data["role"],
            content=data.get("content", ""),
            tool_calls=tool_calls,
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
            reasoning_content=data.get("reasoning_content"),
            is_compression_boundary=data.get("is_compression_boundary", False),
            timestamp=timestamp,
            token_count=data.get("token_count", 0),
        )


@dataclass
class Usage:
    """保存用量所需的结构化数据，主要包含 `prompt_tokens`、`completion_tokens`、`total_tokens` 字段，便于在组件之间传递或持久化。"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class LLMResponse:
    """统一封装一次完整模型响应，包括文本、工具调用、Token 用量、结束原因和可选推理内容。"""

    content: str
    tool_calls: list[ToolCall] | None = None
    usage: Usage | None = None
    finish_reason: FinishReason = FinishReason.STOP
    model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    reasoning_content: str | None = None


@dataclass
class StreamChunk:
    """模型流式响应中的一个增量片段，可携带文本、工具调用、用量、结束原因或推理内容。"""

    content: str | None = None
    tool_call: ToolCall | None = None
    finish_reason: FinishReason | None = None
    usage: Usage | None = None
    delta: bool = True
    reasoning_content: str | None = None


class BaseLLMProvider(ABC):
    """所有模型 Provider 的统一异步接口。具体实现负责把公共 Message 和 ToolSchema 转换为厂商协议，并把普通或流式响应还原为统一的 LLMResponse。"""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        """初始化`BaseLLMProvider`，保存后续操作需要的依赖、配置和初始状态。

        参数：
            api_key: 本次操作使用的`api_key`。
            model: 本次操作使用的模型。
            base_url: 可选的`base_url`。
            **kwargs: 传递给底层实现的额外关键字参数。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.config = kwargs

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """发送一次非流式模型请求，并把厂商响应规范化为 LLMResponse；具体 Provider 负责协议转换和异常归一化。

        参数：
            messages: 按协议顺序排列的对话消息。
            tools: 可选的工具。
            **kwargs: 传递给底层实现的额外关键字参数。

        返回：
            `LLMResponse` 类型的处理结果。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        pass

    @abstractmethod
    def stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """发送流式模型请求，逐项产生 StreamChunk，并在流结束时保留工具调用、用量和结束原因。

        参数：
            messages: 按协议顺序排列的对话消息。
            tools: 可选的工具。
            **kwargs: 传递给底层实现的额外关键字参数。

        生成：
            逐项产生结果，直到数据源结束。
        """
        raise NotImplementedError

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """返回当前 Provider 使用的模型名称、上下文窗口、最大输出和工具或推理能力。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        pass

    def _build_system_prompt(self, messages: list[Message]) -> str | None:
        """根据当前可见工具、已发现能力、Skill 和工作流状态动态生成系统提示词，确保模型只调用本轮允许使用的能力。

        参数：
            messages: 按协议顺序排列的对话消息。

        返回：
            `str | None` 类型的处理结果。
        """
        parts = [
            msg.content.strip() for msg in messages if msg.role == "system" and msg.content.strip()
        ]
        return "\n\n".join(parts) or None

    def _filter_messages_for_anthropic(self, messages: list[Message]) -> list[Message]:
        """根据当前输入和`BaseLLMProvider`的状态计算 `_filter_messages_for_anthropic`，并返回调用方需要的结果。

        参数：
            messages: 按协议顺序排列的对话消息。

        返回：
            按调用约定排序的结果列表。
        """
        return [msg for msg in messages if msg.role != "system"]

    def _messages_to_anthropic(self, messages: list[Message]) -> list[dict[str, Any]]:
        """根据当前输入和`BaseLLMProvider`的状态计算 `_messages_to_anthropic`，并返回调用方需要的结果。

        参数：
            messages: 按协议顺序排列的对话消息。

        返回：
            按调用约定排序的结果列表。
        """
        serialized: list[dict[str, Any]] = []
        for message in self._filter_messages_for_anthropic(messages):
            if message.role == "tool" and not message.tool_call_id:
                raise ProviderProtocolError("Anthropic tool results require a tool_call_id")
            converted = message.to_anthropic_format()
            if message.role == "tool" and serialized:
                previous = serialized[-1]
                previous_content = previous.get("content")
                if (
                    previous.get("role") == "user"
                    and isinstance(previous_content, list)
                    and all(block.get("type") == "tool_result" for block in previous_content)
                ):
                    previous_content.extend(converted["content"])
                    continue
            serialized.append(converted)
        return serialized

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"
