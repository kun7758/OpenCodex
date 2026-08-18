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
        # ── 第 1 步：确定要创建哪个 Provider ──────────────────────────────
        # 从分层配置中取出全部 providers 配置段。
        providers_config = provider_config.get("providers", {})
        # 传入了 provider_name 时优先使用它（AgentRuntime 创建备用 Provider 时会传）；
        # 否则回退到配置里的 default_provider，配置缺失时默认是 "deepseek"。
        default_provider = provider_name or provider_config.get("default_provider", "deepseek")

        # 确认配置里确实存在该 Provider 的条目，缺失时抛出明确错误，避免后续 KeyError。
        if default_provider not in providers_config:
            raise ValueError(
                f"Provider '{default_provider}' not found in configuration. "
                f"Available: {list(providers_config.keys())}"
            )

        # 取出该 Provider 自己的配置段（api_key、default_model、base_url、timeout 等）。
        config = providers_config[default_provider]

        # ── 第 2 步：确定实现类型并查找对应的 Provider 类 ────────────────
        # 实现类型默认就是 Provider 名称本身；配置里也可以用 "type" 字段指定其他实现
        # （例如某个自定义服务使用 openai 兼容协议，可以 type: openai）。
        provider_type = config.get("type", default_provider)

        # 在工厂注册表（cls._providers）中查找该类型对应的 Provider 类，未注册时报错。
        if provider_type not in cls._providers:
            available = list(cls._providers.keys())
            raise ValueError(
                f"Unknown provider type: '{provider_type}'. "
                f"Available providers: {available}"
            )

        provider_class = cls._providers[provider_type]

        # ── 第 3 步：收集构造 Provider 必需的参数 ────────────────────────
        # API 密钥来自配置；配置里的 "${OPENAI_API_KEY}" 占位符在 load_config 时
        # 已经展开成环境变量的值，为空说明配置或环境变量没有准备好。
        api_key = config.get("api_key", "")
        if not api_key:
            raise ValueError(
                f"API key not found for provider '{default_provider}'. "
                "Please set it in configuration or environment variables."
            )

        # 确定使用的模型名称；配置里优先取 default_model，缺失时回退到各 Provider 的默认模型。
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

        # 收集需要透传给 Provider 构造函数的可选参数（base_url、timeout）。
        kwargs: dict[str, Any] = {}
        if "base_url" in config:
            kwargs["base_url"] = config["base_url"]
        if "timeout" in config:
            kwargs["timeout"] = config["timeout"]

        # ── 第 4 步：真正实例化 Provider 对象并返回 ──────────────────────
        # 例如 DeepSeekProvider -> OpenAIProvider.__init__，此时会创建 AsyncOpenAI 客户端。
        # 注意：这里只建立客户端，不会发起任何网络请求；真正的模型调用发生在之后
        # 调用 chat() / stream_chat() 时。返回的对象就是 AgentRuntime.llm 那个可直接使用的实例。
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
