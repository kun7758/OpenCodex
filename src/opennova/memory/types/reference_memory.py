"""记忆数据类型中的引用记忆模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from dataclasses import dataclass
from typing import Any

from opennova.memory.types.user_memory import UserMemory


@dataclass
class ReferenceMemory(UserMemory):
    """保存引用记忆所需的结构化数据，主要包含 `category`、`resource_type`、`url`、`title`、`snippet` 字段，便于在组件之间传递或持久化。"""

    category: str = "reference"
    resource_type: str | None = None
    url: str | None = None
    title: str | None = None
    snippet: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """把引用记忆转换为可序列化字典，供事件、会话或 API 边界使用。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        data = super().to_dict()
        data.update({
            "resource_type": self.resource_type,
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
        })
        return data
