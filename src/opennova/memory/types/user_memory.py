"""记忆数据类型中的用户记忆模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from dataclasses import dataclass

from opennova.memory.types.base import BaseMemory


@dataclass
class UserMemory(BaseMemory):
    """保存用户记忆所需的结构化数据，主要包含 `category`、`feedback_type`、`action`、`tool` 字段，便于在组件之间传递或持久化。"""

    category: str = "user"
    feedback_type: str | None = None
    action: str | None = None
    tool: str | None = None
