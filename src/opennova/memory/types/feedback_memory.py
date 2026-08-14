"""记忆数据类型中的`feedback_memory`模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from opennova.memory.types.user_memory import UserMemory


class FeedbackType(StrEnum):
    """枚举`FeedbackType`允许出现的稳定取值，序列化和状态判断均使用这些值。"""

    PREFERENCE = "preference"
    CORRECTION = "correction"
    APPROVAL = "approval"
    REJECTION = "rejection"
    ERROR_REPORT = "error_report"


@dataclass
class FeedbackMemory(UserMemory):
    """数据对象 `FeedbackMemory` 主要保存 `category`、`feedback_type` 字段，用于在组件之间传递或持久化这组状态。"""

    category: str = "feedback"
    feedback_type: str | None = None  # 允许的反馈类别：偏好、纠正、认可、拒绝和错误报告。

    def to_dict(self) -> dict[str, Any]:
        """把`FeedbackMemory`转换为可序列化字典，供事件、会话或 API 边界使用。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        data = super().to_dict()
        data.update({
            "feedback_type": self.feedback_type,
            "action": self.action,
            "tool": self.tool,
        })
        return data
