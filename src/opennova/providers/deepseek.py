"""模型服务适配层中的DeepSeek模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from typing import Any

from opennova.providers.models import get_model_profile, model_capabilities_for_provider
from opennova.providers.openai import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek 模型服务适配器，复用 OpenAI 兼容协议并补充模型能力、上下文窗口和思考模式信息。"""

    DEFAULT_BASE_URL = "https://api.deepseek.com/v1"

    provider_name = "deepseek"
    SUPPORTED_MODELS = model_capabilities_for_provider(provider_name)

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-v4-pro",
        base_url: str | None = None,
        **kwargs: Any,
    ):
        """初始化`DeepSeekProvider`，保存后续操作需要的依赖、配置和初始状态。

        参数：
            api_key: 本次操作使用的`api_key`。
            model: 可选的模型。
            base_url: 可选的`base_url`。
            **kwargs: 传递给底层实现的额外关键字参数。
        """
        actual_base_url = base_url or self.DEFAULT_BASE_URL
        super().__init__(api_key, model, actual_base_url, **kwargs)

    def get_model_info(self) -> dict[str, Any]:
        """返回当前 Provider 使用的模型名称、上下文窗口、最大输出和工具或推理能力。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        return get_model_profile(self.provider_name, self.model).to_dict()
