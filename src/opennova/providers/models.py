"""模型服务适配层中的`models`模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelProfile:
    """保存模型配置档所需的结构化数据，主要包含
    `provider`、`model`、`context_window`、`max_output_tokens`、`supports_tools`、`supports_vision`、`supports_reasoning`、`supports_structured_output`
    字段，便于在组件之间传递或持久化。
    """

    provider: str
    model: str
    context_window: int
    max_output_tokens: int
    supports_tools: bool = True
    supports_vision: bool = False
    supports_reasoning: bool = False
    supports_structured_output: bool = False

    @property
    def supports_thinking(self) -> bool:
        """计算并返回 `supports_thinking` 属性；读取该属性不会主动改变对象的业务状态。

        返回：
            表示条件是否成立。
        """
        return self.supports_reasoning

    def estimate_tokens(self, text: str) -> int:
        """估算 `tokens` 对应的数据，并按照当前组件的约定返回结果。

        参数：
            text: 需要解析、格式化或展示的文本。

        返回：
            `int` 类型的处理结果。
        """
        return max(1, (len(text.encode("utf-8")) + 3) // 4)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_PROFILES = [
    ModelProfile(
        "openai", "gpt-4o", 128_000, 16_384, supports_vision=True, supports_structured_output=True
    ),
    ModelProfile(
        "openai",
        "gpt-4o-mini",
        128_000,
        16_384,
        supports_vision=True,
        supports_structured_output=True,
    ),
    ModelProfile("openai", "gpt-4-turbo", 128_000, 4_096, supports_vision=True),
    ModelProfile("openai", "gpt-4", 8_192, 4_096),
    ModelProfile("openai", "o1-preview", 128_000, 32_768, supports_reasoning=True),
    ModelProfile("openai", "o1-mini", 128_000, 65_536, supports_reasoning=True),
    ModelProfile(
        "anthropic",
        "claude-sonnet-4-20250514",
        200_000,
        64_000,
        supports_vision=True,
        supports_reasoning=True,
    ),
    ModelProfile(
        "anthropic",
        "claude-opus-4-20250514",
        200_000,
        32_000,
        supports_vision=True,
        supports_reasoning=True,
    ),
    ModelProfile("anthropic", "claude-3-5-sonnet-20241022", 200_000, 8_192, supports_vision=True),
    ModelProfile("anthropic", "claude-3-5-haiku-20241022", 200_000, 8_192, supports_vision=True),
    ModelProfile("anthropic", "claude-3-opus-20240229", 200_000, 4_096, supports_vision=True),
    ModelProfile("anthropic", "claude-3-sonnet-20240229", 200_000, 4_096, supports_vision=True),
    ModelProfile("anthropic", "claude-3-haiku-20240307", 200_000, 4_096, supports_vision=True),
    ModelProfile("deepseek", "deepseek-chat", 64_000, 8_192),
    ModelProfile("deepseek", "deepseek-reasoner", 64_000, 8_192, supports_reasoning=True),
    ModelProfile("deepseek", "deepseek-v4-pro", 131_072, 16_384, supports_reasoning=True),
    ModelProfile("deepseek", "deepseek-v4-flash", 131_072, 8_192),
]

MODEL_PROFILES: dict[tuple[str, str], ModelProfile] = {
    (profile.provider, profile.model): profile for profile in _PROFILES
}

MODEL_ALIASES: dict[str, str] = {
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    "claude-opus-4": "claude-opus-4-20250514",
    "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3.5-haiku": "claude-3-5-haiku-20241022",
}

DEFAULT_CONTEXT_WINDOWS = {
    "openai": 8_192,
    "anthropic": 200_000,
    "deepseek": 64_000,
}


def resolve_model_name(model: str) -> str:
    """解析`resolve_model_name`的最终目标或处理结果。

    参数：
        model: 本次操作使用的模型。

    返回：
        处理后的文本或稳定标识。
    """
    return MODEL_ALIASES.get(model, model)


def get_model_profile(provider: str, model: str) -> ModelProfile:
    """读取模型配置档，不改变当前对象的业务状态。

    参数：
        provider: 负责本次模型请求的 Provider 实例。
        model: 本次操作使用的模型。

    返回：
        `ModelProfile` 类型的处理结果。
    """
    resolved = resolve_model_name(model)
    profile = MODEL_PROFILES.get((provider, resolved))
    if profile is not None:
        return profile
    return ModelProfile(
        provider=provider,
        model=resolved,
        context_window=DEFAULT_CONTEXT_WINDOWS.get(provider, 8_192),
        max_output_tokens=4_096,
    )


def model_capabilities_for_provider(provider: str) -> dict[str, dict[str, object]]:
    """读取并返回 `model_capabilities_for_provider` 所表示的数据或流程，并遵守当前模块定义的边界与状态约束。

    参数：
        provider: 负责本次模型请求的 Provider 实例。

    返回：
        供后续逻辑或序列化使用的结构化字典。
    """
    return {
        profile.model: {
            "context_window": profile.context_window,
            "max_output_tokens": profile.max_output_tokens,
            "supports_tools": profile.supports_tools,
            "supports_vision": profile.supports_vision,
            "supports_reasoning": profile.supports_reasoning,
            "supports_structured_output": profile.supports_structured_output,
        }
        for profile in MODEL_PROFILES.values()
        if profile.provider == provider
    }


def context_window_for_model(model: str, default: int = 128_000) -> int:
    """根据当前输入和当前模块的状态计算 `context_window_for_model`，并返回调用方需要的结果。

    参数：
        model: 本次操作使用的模型。
        default: 可选的默认。

    返回：
        `int` 类型的处理结果。
    """
    resolved = resolve_model_name(model)
    for profile in MODEL_PROFILES.values():
        if profile.model == resolved:
            return profile.context_window
    return default
