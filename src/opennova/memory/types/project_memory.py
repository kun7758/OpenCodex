"""记忆数据类型中的项目记忆模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from dataclasses import dataclass
from typing import Any

from opennova.memory.types.user_memory import UserMemory


@dataclass
class ProjectMemory(UserMemory):
    """保存在项目目录中的长期记忆，记录项目结构、关键决策、用户偏好和会话摘要，供后续任务检索。"""

    category: str = "project"
    project_path: str | None = None
    decision: str | None = None
    reasoning: str | None = None
    context: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """把项目记忆转换为可序列化字典，供事件、会话或 API 边界使用。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        data = super().to_dict()
        data.update({
            "project_path": self.project_path,
            "decision": self.decision,
            "reasoning": self.reasoning,
            "context": self.context,
        })
        return data
