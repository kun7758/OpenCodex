"""记忆数据类型中的基础抽象模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any, Literal, TypeVar

MemoryCategory = Literal["user", "feedback", "project", "reference"]
MEMORY_CATEGORIES: tuple[MemoryCategory, ...] = (
    "user",
    "feedback",
    "project",
    "reference",
)

MemoryType = TypeVar("MemoryType", bound="BaseMemory")


@dataclass
class BaseMemory:
    """保存基础抽象记忆所需的结构化数据，主要包含
    `id`、`category`、`content`、`created_at`、`updated_at`、`relevance`、`tags`、`metadata`
    字段，便于在组件之间传递或持久化。
    """

    id: str
    category: str = "user"
    content: str = ""  # 为 Python 3.11 提供兼容的默认实现。
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime | None = None
    relevance: float = 1.0
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """把基础抽象记忆转换为可序列化字典，供事件、会话或 API 边界使用。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        return {
            "id": self.id,
            "category": self.category,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "relevance": self.relevance,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls: type[MemoryType], data: dict[str, Any]) -> MemoryType:
        """从字典恢复基础抽象记忆，并为旧数据缺失的字段补充兼容默认值。

        参数：
            data: 用于构造或恢复对象的结构化数据。

        返回：
            `MemoryType` 类型的处理结果。
        """
        payload = dict(data)
        for field_name in ("created_at", "updated_at"):
            value = payload.get(field_name)
            if isinstance(value, str):
                payload[field_name] = datetime.fromisoformat(value)

        known_fields = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in payload.items() if key in known_fields})
