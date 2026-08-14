"""模型服务适配层中的`anthropic`模块，集中定义相关数据结构、边界适配和实现逻辑。"""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from anthropic import AsyncAnthropic

from opennova.providers.base import (
    BaseLLMProvider,
    FinishReason,
    LLMResponse,
    Message,
    StreamChunk,
    ToolCall,
    ToolSchema,
    Usage,
    normalize_provider_error,
    parse_tool_arguments,
)
from opennova.providers.models import (
    MODEL_ALIASES as CANONICAL_MODEL_ALIASES,
)
from opennova.providers.models import (
    get_model_profile,
    model_capabilities_for_provider,
)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic 模型服务适配器。它负责 system 消息分离、tool_use/tool_result 内容块转换以及流式事件聚合。"""

    provider_name = "anthropic"
    SUPPORTED_MODELS = model_capabilities_for_provider(provider_name)

    # 提供常用模型别名，简化配置中的模型选择。
    MODEL_ALIASES = CANONICAL_MODEL_ALIASES

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4",
        base_url: str | None = None,
        **kwargs: Any,
    ):
        """初始化`AnthropicProvider`，保存后续操作需要的依赖、配置和初始状态。

        参数：
            api_key: 本次操作使用的`api_key`。
            model: 可选的模型。
            base_url: 可选的`base_url`。
            **kwargs: 传递给底层实现的额外关键字参数。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        resolved_model = self.MODEL_ALIASES.get(model, model)
        super().__init__(api_key, resolved_model, base_url, **kwargs)

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": kwargs.get("timeout", 120.0),
            "max_retries": kwargs.get("max_retries", 2),
        }

        self.client = AsyncAnthropic(**client_kwargs)

    def _convert_tools_to_anthropic(self, tools: list[ToolSchema]) -> list[dict[str, Any]]:
        """构造并返回 `_convert_tools_to_anthropic` 所表示的数据或流程，并遵守`AnthropicProvider`定义的边界与状态约束。

        参数：
            tools: 本次操作使用的工具。

        返回：
            按调用约定排序的结果列表。
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]

    @staticmethod
    def _anthropic_tool_choice(value: str) -> dict[str, str] | None:
        """根据当前输入和`AnthropicProvider`的状态计算 `_anthropic_tool_choice`，并返回调用方需要的结果。

        参数：
            value: 需要保存、转换或校验的值。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        choices = {
            "auto": {"type": "auto"},
            "required": {"type": "any"},
            "none": None,
        }
        if value not in choices:
            raise ValueError(f"Unsupported tool_choice: {value}")
        return choices[value]

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
        system_prompt = self._build_system_prompt(messages)
        anthropic_messages = self._messages_to_anthropic(messages)

        request_params: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }

        if system_prompt:
            request_params["system"] = system_prompt

        tool_choice = str(kwargs.get("tool_choice", "auto"))
        if tools and tool_choice != "none":
            request_params["tools"] = self._convert_tools_to_anthropic(tools)
            request_params["tool_choice"] = self._anthropic_tool_choice(tool_choice)

        if "temperature" in kwargs:
            request_params["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            request_params["top_p"] = kwargs["top_p"]

        try:
            response = await self.client.messages.create(**request_params)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise normalize_provider_error(exc, provider=self.provider_name) from exc

        text_content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_content += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                        call_type="function",
                    )
                )

        usage = Usage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

        finish_reason = FinishReason.STOP
        if tool_calls:
            finish_reason = FinishReason.TOOL_CALL
        elif response.stop_reason == "max_tokens":
            finish_reason = FinishReason.LENGTH
        elif response.stop_reason == "end_turn":
            finish_reason = FinishReason.STOP

        return LLMResponse(
            content=text_content,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            finish_reason=finish_reason,
            model=response.model,
            metadata={"response_id": response.id},
        )

    async def stream_chat(
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

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        system_prompt = self._build_system_prompt(messages)
        anthropic_messages = self._messages_to_anthropic(messages)

        request_params: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }

        if system_prompt:
            request_params["system"] = system_prompt

        tool_choice = str(kwargs.get("tool_choice", "auto"))
        if tools and tool_choice != "none":
            request_params["tools"] = self._convert_tools_to_anthropic(tools)
            request_params["tool_choice"] = self._anthropic_tool_choice(tool_choice)

        if "temperature" in kwargs:
            request_params["temperature"] = kwargs["temperature"]

        tool_call_accumulator: dict[str, dict[str, Any]] = {}

        try:
            async with self.client.messages.stream(**request_params) as stream:
                async for event in stream:
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, "text") and event.delta.text:
                            yield StreamChunk(content=event.delta.text, delta=True)

                        if hasattr(event.delta, "partial_json"):
                            block_id = event.index
                            tool_id = f"toolu_{block_id}"
                            if tool_id not in tool_call_accumulator:
                                tool_call_accumulator[tool_id] = {
                                    "id": tool_id,
                                    "name": "",
                                    "arguments": "",
                                }
                            if event.delta.partial_json:
                                tool_call_accumulator[tool_id]["arguments"] += (
                                    event.delta.partial_json
                                )

                    elif event.type == "content_block_start":
                        tool_name = getattr(event.content_block, "name", None)
                        if tool_name:
                            tool_id = str(
                                getattr(event.content_block, "id", f"toolu_{event.index}")
                            )
                            tool_call_accumulator[tool_id] = {
                                "id": tool_id,
                                "name": str(tool_name),
                                "arguments": "",
                            }

                    elif event.type == "message_stop":
                        for _tool_id, tc_data in tool_call_accumulator.items():
                            args = parse_tool_arguments(
                                tc_data["arguments"],
                                tool_name=tc_data["name"] or "unknown",
                                tool_call_id=tc_data["id"],
                            )

                            yield StreamChunk(
                                tool_call=ToolCall(
                                    id=tc_data["id"],
                                    name=tc_data["name"],
                                    arguments=args,
                                    call_type="function",
                                ),
                                delta=False,
                            )

                        final_message = await stream.get_final_message()
                        finish_reason = FinishReason.STOP
                        if tool_call_accumulator:
                            finish_reason = FinishReason.TOOL_CALL
                        elif final_message.stop_reason == "max_tokens":
                            finish_reason = FinishReason.LENGTH

                        usage = Usage(
                            prompt_tokens=final_message.usage.input_tokens,
                            completion_tokens=final_message.usage.output_tokens,
                            total_tokens=final_message.usage.input_tokens
                            + final_message.usage.output_tokens,
                        )

                        yield StreamChunk(
                            finish_reason=finish_reason,
                            usage=usage,
                            delta=False,
                        )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise normalize_provider_error(exc, provider=self.provider_name) from exc

    def get_model_info(self) -> dict[str, Any]:
        """返回当前 Provider 使用的模型名称、上下文窗口、最大输出和工具或推理能力。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        return get_model_profile(self.provider_name, self.model).to_dict()
