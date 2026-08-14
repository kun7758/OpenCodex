"""模型服务适配层中的`factory`模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from typing import Any

from opennova.providers.anthropic import AnthropicProvider
from opennova.providers.base import BaseLLMProvider
from opennova.providers.deepseek import DeepSeekProvider
from opennova.providers.openai import OpenAIProvider


class ProviderFactory:
    """根据分层配置选择并创建模型 Provider，也允许扩展代码注册新的 Provider 类型。"""

    _instance: "ProviderFactory | None" = None
    _providers: dict[str, type[BaseLLMProvider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "deepseek": DeepSeekProvider,
    }

    def __new__(cls) -> "ProviderFactory":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseLLMProvider]) -> None:
        """注册模型服务，使后续运行能够发现并调用它。

        参数：
            name: 待查询、注册或操作对象的名称。
            provider_class: 本次操作使用的`provider_class`。
        """
        cls._providers[name] = provider_class

    @classmethod
    def create_provider(
        cls,
        provider_config: dict[str, Any],
        provider_name: str | None = None,
    ) -> BaseLLMProvider:
        """创建模型服务并完成必要的初始化。

        参数：
            provider_config: 本次操作使用的模型服务配置。
            provider_name: 可选的`provider_name`。

        返回：
            `BaseLLMProvider` 类型的处理结果。
        """
        providers_config = provider_config.get("providers", {})
        default_provider = provider_name or provider_config.get("default_provider", "deepseek")

        if default_provider not in providers_config:
            raise ValueError(
                f"Provider '{default_provider}' not found in configuration. "
                f"Available: {list(providers_config.keys())}"
            )

        config = providers_config[default_provider]

        provider_type = config.get("type", default_provider)

        if provider_type not in cls._providers:
            available = list(cls._providers.keys())
            raise ValueError(
                f"Unknown provider type: '{provider_type}'. "
                f"Available providers: {available}"
            )

        provider_class = cls._providers[provider_type]

        api_key = config.get("api_key", "")
        if not api_key:
            raise ValueError(
                f"API key not found for provider '{default_provider}'. "
                "Please set it in configuration or environment variables."
            )

        model = config.get("default_model", config.get("model", ""))

        if not model:
            default_models = {
                "openai": "gpt-4o",
                "anthropic": "claude-sonnet-4",
                "deepseek": "deepseek-v4-pro",
            }
            model = default_models.get(provider_type, "")
            if not model:
                raise ValueError(f"Model not specified for provider '{default_provider}'")

        kwargs: dict[str, Any] = {}
        if "base_url" in config:
            kwargs["base_url"] = config["base_url"]
        if "timeout" in config:
            kwargs["timeout"] = config["timeout"]

        return provider_class(api_key=api_key, model=model, **kwargs)

    @classmethod
    def list_providers(cls) -> list[str]:
        """列出模型服务，并按当前组件约定返回稳定顺序。

        返回：
            按调用约定排序的结果列表。
        """
        return list(cls._providers.keys())

    @classmethod
    def reset(cls) -> None:
        """处理重置，并按照当前组件的约定返回结果。"""
        cls._instance = None
