"""模型服务适配层中的OpenAI模块，集中定义相关数据结构、边界适配和实现逻辑。"""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

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
from opennova.providers.models import get_model_profile, model_capabilities_for_provider


class OpenAIProvider(BaseLLMProvider):
    """OpenAI 模型服务适配器。它把统一消息和工具 Schema 转成 OpenAI Chat Completions 协议，并把普通或流式响应还原为公共类型。"""

    provider_name = "openai"
    SUPPORTED_MODELS = model_capabilities_for_provider(provider_name)

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str | None = None,
        **kwargs: Any,
    ):
        """初始化`OpenAIProvider`，保存后续操作需要的依赖、配置和初始状态。

        参数：
            api_key: 本次操作使用的`api_key`。
            model: 可选的模型。
            base_url: 可选的`base_url`。
            **kwargs: 传递给底层实现的额外关键字参数。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        super().__init__(api_key, model, base_url, **kwargs)

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=kwargs.get("timeout", 120.0),
            max_retries=kwargs.get("max_retries", 2),
        )

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
        openai_messages = [msg.to_openai_format() for msg in messages]

        request_params: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
        }

        if tools:
            request_params["tools"] = [t.to_openai_format() for t in tools]
            request_params["tool_choice"] = kwargs.get("tool_choice", "auto")

        if "max_tokens" in kwargs:
            request_params["max_tokens"] = kwargs["max_tokens"]
        if "temperature" in kwargs:
            request_params["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            request_params["top_p"] = kwargs["top_p"]

        try:
            response = await self.client.chat.completions.create(**request_params)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise normalize_provider_error(exc, provider=self.provider_name) from exc

        choice = response.choices[0]

        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=parse_tool_arguments(
                        tc.function.arguments,
                        tool_name=tc.function.name,
                        tool_call_id=tc.id,
                    ),
                    call_type="function",
                )
                for tc in choice.message.tool_calls
            ]

        usage = None
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        finish_reason_map = {
            "stop": FinishReason.STOP,
            "tool_calls": FinishReason.TOOL_CALL,
            "length": FinishReason.LENGTH,
            "content_filter": FinishReason.CONTENT_FILTER,
        }

        finish_reason_raw = choice.finish_reason
        if finish_reason_raw and hasattr(finish_reason_raw, "value"):
            finish_reason_raw = finish_reason_raw.value

        return LLMResponse(
            content=choice.message.content or "",
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish_reason_map.get(
                finish_reason_raw or "stop",
                FinishReason.STOP,
            ),
            model=response.model,
            metadata={"response_id": response.id},
            reasoning_content=getattr(choice.message, "reasoning_content", None) or None,
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
        openai_messages = [msg.to_openai_format() for msg in messages]

        request_params: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if tools:
            request_params["tools"] = [t.to_openai_format() for t in tools]
            request_params["tool_choice"] = kwargs.get("tool_choice", "auto")

        if "max_tokens" in kwargs:
            request_params["max_tokens"] = kwargs["max_tokens"]
        if "temperature" in kwargs:
            request_params["temperature"] = kwargs["temperature"]

        try:
            stream = await self.client.chat.completions.create(**request_params)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise normalize_provider_error(exc, provider=self.provider_name) from exc

        tool_call_accumulator: dict[int, dict[str, Any]] = {}
        accumulated_reasoning: str = ""

        try:
            async for chunk in stream:
                if not chunk.choices:
                    if hasattr(chunk, "usage") and chunk.usage:
                        yield StreamChunk(
                            usage=Usage(
                                prompt_tokens=chunk.usage.prompt_tokens,
                                completion_tokens=chunk.usage.completion_tokens,
                                total_tokens=chunk.usage.total_tokens,
                            ),
                            delta=False,
                        )
                    continue

                choice = chunk.choices[0]

                # 从流式增量中单独收集 reasoning_content，以兼容 DeepSeek 思考模式。
                delta_reasoning = getattr(choice.delta, "reasoning_content", None) or ""
                if delta_reasoning:
                    accumulated_reasoning += delta_reasoning

                if choice.delta.content:
                    yield StreamChunk(content=choice.delta.content, delta=True)

                if choice.delta.tool_calls:
                    for tc in choice.delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_call_accumulator:
                            tool_call_accumulator[idx] = {
                                "id": tc.id,
                                "name": tc.function.name if tc.function else None,
                                "arguments": "",
                            }

                        if tc.function:
                            if tc.function.name:
                                tool_call_accumulator[idx]["name"] = tc.function.name
                            if tc.function.arguments:
                                tool_call_accumulator[idx]["arguments"] += tc.function.arguments

                if choice.finish_reason:
                    for idx, tc_data in tool_call_accumulator.items():
                        args = parse_tool_arguments(
                            tc_data["arguments"],
                            tool_name=tc_data["name"] or "unknown",
                            tool_call_id=tc_data["id"] or f"call_{idx}",
                        )

                        yield StreamChunk(
                            tool_call=ToolCall(
                                id=tc_data["id"] or f"call_{idx}",
                                name=tc_data["name"] or "unknown",
                                arguments=args,
                                call_type="function",
                            ),
                            delta=False,
                        )

                    finish_reason_map = {
                        "stop": FinishReason.STOP,
                        "tool_calls": FinishReason.TOOL_CALL,
                        "length": FinishReason.LENGTH,
                    }
                    finish_reason_raw = choice.finish_reason
                    if finish_reason_raw and hasattr(finish_reason_raw, "value"):
                        finish_reason_raw = finish_reason_raw.value
                    yield StreamChunk(
                        finish_reason=finish_reason_map.get(
                            finish_reason_raw or "stop",
                            FinishReason.STOP,
                        ),
                        delta=False,
                        reasoning_content=accumulated_reasoning or None,
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
