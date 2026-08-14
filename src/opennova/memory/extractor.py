"""记忆与上下文子系统中的提取器模块，集中定义相关数据结构、边界适配和实现逻辑。"""

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TypedDict

from opennova.memory.types.feedback_memory import FeedbackMemory, FeedbackType
from opennova.memory.types.project_memory import ProjectMemory
from opennova.memory.types.reference_memory import ReferenceMemory
from opennova.memory.types.user_memory import UserMemory
from opennova.providers.base import Message


@dataclass
class ExtractionResult:
    """数据对象 `ExtractionResult` 主要保存
    `user_memories`、`feedback_memories`、`project_memories`、`reference_memories`
    字段，用于在组件之间传递或持久化这组状态。
    """

    user_memories: list[UserMemory] = field(default_factory=list)
    feedback_memories: list[FeedbackMemory] = field(default_factory=list)
    project_memories: list[ProjectMemory] = field(default_factory=list)
    reference_memories: list[ReferenceMemory] = field(default_factory=list)


class ExtractedMemoryMap(TypedDict):
    """数据对象 `ExtractedMemoryMap` 主要保存 `user`、`feedback`、`project`、`reference` 字段，用于在组件之间传递或持久化这组状态。"""

    user: list[UserMemory]
    feedback: list[FeedbackMemory]
    project: list[ProjectMemory]
    reference: list[ReferenceMemory]


class MemoryExtractor:
    """封装记忆提取器相关的状态和操作，使调用方通过稳定接口使用该能力。"""

    # 用于从用户消息中识别偏好、引用、项目决策和反馈的模式。
    PREFERENCE_PATTERNS = [
        r"(?:i|I) (?:prefer|like|want|choose|should use) (?:\w+\s+)?(.+?)(?:\.|,|;|$)",
        r"(?:i|I) (?:don't|do not|avoid|hate|prefer not) (?:\w+\s+)?(.+?)(?:\.|,|;|$)",
        r"(?:my|our|the project) (?:\w+\s+)?(?:uses|convention|style|pattern|architecture|framework)\s+(.+?)(?:\.|,|;|$)",
    ]

    REFERENCE_PATTERNS = [
        r"(?:https?://|http://|www\.)[^\s]+",
        r"(?:github|gitlab|bitbucket|stack\s+overflow)\.(?:com|io|org)/(?:issues|pr|pull|commit|repo)",
        r"(?:documentation|docs?|reference)\.?(?:\w+|\s+)(?:at|in|for)",
    ]

    PROJECT_PATTERNS = [
        r"(?:i|I) (?:decided|chose|chose|went with) (?:\w+\s+)?(.+?)(?:for|to use|because)",
        r"(?:i|I) (?:think|thought|considered|evaluated) (?:\w+\s+)?(.+?)(?:might|should|would be)",
        r"(?:the project|this repo) (?:\w+\s+)?(?:needs|should|requires) (?:\w+\s+)?(.+?)",
    ]

    def extract_from_messages(self, messages: list[Message]) -> ExtractionResult:
        """提取来源消息，并按照当前组件的约定返回结果。

        参数：
            messages: 按协议顺序排列的对话消息。

        返回：
            `ExtractionResult` 类型的处理结果。
        """
        result = ExtractionResult()

        for message in messages:
            if message.role != "user":
                continue

            content = message.content or ""
            extracted = self._extract_from_user_message(content, message.timestamp)

            result.user_memories.extend(extracted["user"])
            result.feedback_memories.extend(extracted["feedback"])
            result.project_memories.extend(extracted["project"])
            result.reference_memories.extend(extracted["reference"])

        return result

    def _extract_from_user_message(self, content: str, timestamp: datetime) -> ExtractedMemoryMap:
        """提取来源用户消息，并按照当前组件的约定返回结果。

        参数：
            content: 需要处理、保存或分析的文本内容。
            timestamp: 本次操作使用的`timestamp`。

        返回：
            `ExtractedMemoryMap` 类型的处理结果。
        """
        extracted: ExtractedMemoryMap = {
            "user": [],
            "feedback": [],
            "project": [],
            "reference": [],
        }

        # 提取用户明确表达的偏好。
        for pattern in self.PREFERENCE_PATTERNS:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                preference = match.group(1).strip()
                if preference and len(preference) > 2:  # 过滤缺少实际语义的过短匹配。
                    extracted["user"].append(
                        UserMemory(
                            id=str(uuid.uuid4()),
                            content=preference,
                            created_at=timestamp,
                            tags=["preference"],
                        )
                    )

        # 提取用户提供的文件、链接或其他参考资源。
        for pattern in self.REFERENCE_PATTERNS:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                url_or_reference = match.group(0).rstrip('.,;:!?)"]}')
                extracted["reference"].append(
                    ReferenceMemory(
                        id=str(uuid.uuid4()),
                        content=f"Referenced: {url_or_reference}",
                        created_at=timestamp,
                        tags=["reference"],
                        url=url_or_reference if url_or_reference.startswith("http") else None,
                    )
                )

        # 提取需要跨会话保留的项目决策。
        for pattern in self.PROJECT_PATTERNS:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                decision = match.group(1).strip()
                if len(decision) > 5:  # 过滤缺少实际语义的过短匹配。
                    extracted["project"].append(
                        ProjectMemory(
                            id=str(uuid.uuid4()),
                            content=f"Decision: {decision}",
                            created_at=timestamp,
                            tags=["decision"],
                        )
                    )

        # 提取用户对结果的正向或负向反馈。
        feedback_indicators = {
            "positive": ["good", "great", "excellent", "perfect", "thanks", "helpful"],
            "negative": ["bad", "wrong", "error", "issue", "problem", "doesn't work"],
        }

        normalized = content.lower()
        for feedback_kind, indicators in feedback_indicators.items():
            for indicator in indicators:
                if not re.search(rf"(?<!\w){re.escape(indicator)}(?!\w)", normalized):
                    continue
                if feedback_kind == "positive":
                    extracted["feedback"].append(
                        FeedbackMemory(
                            id=str(uuid.uuid4()),
                            content=f"Positive feedback: {indicator}",
                            feedback_type=FeedbackType.APPROVAL,
                            created_at=timestamp,
                            tags=["feedback", "positive"],
                        )
                    )
                else:
                    extracted["feedback"].append(
                        FeedbackMemory(
                            id=str(uuid.uuid4()),
                            content=f"Negative feedback: {indicator}",
                            feedback_type=FeedbackType.REJECTION,
                            created_at=timestamp,
                            tags=["feedback", "negative"],
                        )
                    )

        return extracted

    def extract_preferences(self, content: str, context: str = "") -> list[str]:
        """提取偏好，并按照当前组件的约定返回结果。

        参数：
            content: 需要处理、保存或分析的文本内容。
            context: 本次工具调用或运行所使用的上下文。

        返回：
            按调用约定排序的结果列表。
        """
        preferences = []
        for pattern in self.PREFERENCE_PATTERNS:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                preference = match.group(1).strip()
                if preference and len(preference) > 2:
                    preferences.append(preference)
        return preferences

    def extract_references(self, content: str) -> list[str]:
        """提取引用，并按照当前组件的约定返回结果。

        参数：
            content: 需要处理、保存或分析的文本内容。

        返回：
            按调用约定排序的结果列表。
        """
        references = []
        for pattern in self.REFERENCE_PATTERNS:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                references.append(match.group(0).rstrip('.,;:!?)"]}'))
        return references

    def extract_project_context(self, content: str) -> dict[str, Any]:
        """提取项目上下文，并按照当前组件的约定返回结果。

        参数：
            content: 需要处理、保存或分析的文本内容。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        context: dict[str, list[str]] = {
            "decisions": [],
            "requirements": [],
            "architecture": [],
        }

        for pattern in self.PROJECT_PATTERNS:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                context["decisions"].append(match.group(1).strip())

        return context
