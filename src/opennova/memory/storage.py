"""记忆与上下文子系统中的存储模块，集中定义相关数据结构、边界适配和实现逻辑。"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from opennova.memory.types.base import MEMORY_CATEGORIES
from opennova.memory.types.feedback_memory import FeedbackMemory
from opennova.memory.types.project_memory import ProjectMemory
from opennova.memory.types.reference_memory import ReferenceMemory
from opennova.memory.types.user_memory import UserMemory

MEMORY_TYPES: dict[str, type[UserMemory]] = {
    "user": UserMemory,
    "feedback": FeedbackMemory,
    "project": ProjectMemory,
    "reference": ReferenceMemory,
}


class MemoryStorage:
    """封装记忆存储相关的状态和操作，使调用方通过稳定接口使用该能力。"""

    def __init__(self, memory_dir: str | None = None):
        """初始化记忆存储，保存后续操作需要的依赖、配置和初始状态。

        参数：
            memory_dir: 可选的记忆目录。

        说明：
            执行过程中会更新当前实例维护的状态。
            该操作会访问本地文件系统，路径校验和原子写入约束由所在组件负责。
        """
        if memory_dir is None:
            # 优先使用 XDG_DATA_HOME；未设置时回退到用户数据目录。
            data_home = os.environ.get("XDG_DATA_HOME")
            if data_home:
                base = Path(data_home) / "opennova"
            else:
                base = Path.home() / ".local" / "share" / "opennova"

            self.memory_dir = base / "memory"
        else:
            self.memory_dir = Path(memory_dir)

        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # 为每种记忆类型创建独立子目录，避免不同结构混存。
        self.user_dir = self.memory_dir / "user"
        self.feedback_dir = self.memory_dir / "feedback"
        self.project_dir = self.memory_dir / "project"
        self.reference_dir = self.memory_dir / "reference"

        for dir_path in [self.user_dir, self.feedback_dir, self.project_dir, self.reference_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _get_category_dir(self, category: str) -> Path:
        """读取并返回 `_get_category_dir` 所表示的数据或流程，并遵守记忆存储定义的边界与状态约束。

        参数：
            category: 用于限定数据范围的类别。

        返回：
            `Path` 类型的处理结果。
        """
        mapping = {
            "user": self.user_dir,
            "feedback": self.feedback_dir,
            "project": self.project_dir,
            "reference": self.reference_dir,
        }
        return mapping.get(category, self.memory_dir)

    @staticmethod
    def _deserialize(data: dict[str, Any], fallback_category: str) -> UserMemory | None:
        """处理反序列化，并按照当前组件的约定返回结果。

        参数：
            data: 用于构造或恢复对象的结构化数据。
            fallback_category: 本次操作使用的回退类别。

        返回：
            `UserMemory | None` 类型的处理结果。
        """
        category = data.get("category", fallback_category)
        memory_type = MEMORY_TYPES.get(category) if isinstance(category, str) else None
        if memory_type is None:
            return None
        return memory_type.from_dict(data)

    def _get_memory_file(self, memory: UserMemory) -> Path:
        """读取并返回 `_get_memory_file` 所表示的数据或流程，并遵守记忆存储定义的边界与状态约束。

        参数：
            memory: 本次操作使用的记忆。

        返回：
            `Path` 类型的处理结果。
        """
        category_dir = self._get_category_dir(memory.category)
        return category_dir / f"{memory.id}.json"

    def save(self, memory: UserMemory) -> None:
        """处理保存，并按照当前组件的约定返回结果。

        参数：
            memory: 本次操作使用的记忆。

        说明：
            该操作会访问本地文件系统，路径校验和原子写入约束由所在组件负责。
        """
        memory.updated_at = datetime.now()
        memory_file = self._get_memory_file(memory)

        memory_data = memory.to_dict()

        with open(memory_file, "w", encoding="utf-8") as f:
            json.dump(memory_data, f, indent=2, ensure_ascii=False)

    def get(self, memory_id: str, category: str) -> UserMemory | None:
        """根据当前输入和记忆存储的状态计算 `get`，并返回调用方需要的结果。

        参数：
            memory_id: 本次操作使用的`memory_id`。
            category: 用于限定数据范围的类别。

        返回：
            `UserMemory | None` 类型的处理结果。

        说明：
            该操作会访问本地文件系统，路径校验和原子写入约束由所在组件负责。
        """
        category_dir = self._get_category_dir(category)
        memory_file = category_dir / f"{memory_id}.json"

        if not memory_file.exists():
            return None

        try:
            with open(memory_file, encoding="utf-8") as f:
                data = json.load(f)
            return self._deserialize(data, category)
        except Exception:
            return None

    def list_by_category(self, category: str) -> list[UserMemory]:
        """列出 `by_category` 对应的对象，并按当前组件约定返回稳定顺序。

        参数：
            category: 用于限定数据范围的类别。

        返回：
            按调用约定排序的结果列表。

        说明：
            该操作会访问本地文件系统，路径校验和原子写入约束由所在组件负责。
        """
        category_dir = self._get_category_dir(category)
        memories = []

        for memory_file in category_dir.glob("*.json"):
            try:
                with open(memory_file, encoding="utf-8") as f:
                    data = json.load(f)
                    memory = self._deserialize(data, category)
                    if memory is not None:
                        memories.append(memory)
            except Exception:
                pass

        # 先按创建时间和相关度整理候选结果。
        memories.sort(key=lambda m: (m.created_at.timestamp(), m.relevance), reverse=True)
        return memories

    def search(self, query: str, category: str | None = None, limit: int = 10) -> list[UserMemory]:
        """处理搜索，并按照当前组件的约定返回结果。

        参数：
            query: 用于搜索、匹配或排序的查询文本。
            category: 用于限定数据范围的类别。
            limit: 最多返回或处理的条目数量。

        返回：
            按调用约定排序的结果列表。
        """
        query_lower = query.lower()

        if category:
            memories = self.list_by_category(category)
        else:
            # 未指定类别时遍历所有记忆目录。
            memories = []
            for cat in MEMORY_CATEGORIES:
                memories.extend(self.list_by_category(cat))

        # 根据查询文本过滤候选记忆。
        matches = []
        for memory in memories:
            if query_lower in memory.content.lower() or any(
                query_lower in tag.lower() for tag in memory.tags
            ):
                matches.append(memory)

        return matches[:limit] if limit else matches

    def delete(self, memory_id: str, category: str) -> bool:
        """处理删除，并按照当前组件的约定返回结果。

        参数：
            memory_id: 本次操作使用的`memory_id`。
            category: 用于限定数据范围的类别。

        返回：
            表示条件是否成立。

        说明：
            该操作会访问本地文件系统，路径校验和原子写入约束由所在组件负责。
        """
        category_dir = self._get_category_dir(category)
        memory_file = category_dir / f"{memory_id}.json"

        if memory_file.exists():
            memory_file.unlink()
            return True
        return False

    def cleanup_old_memories(self, days: int = 30) -> int:
        """清理 `old_memories` 对应的数据，并按照当前组件的约定返回结果。

        参数：
            days: 可选的`days`。

        返回：
            `int` 类型的处理结果。

        说明：
            该操作会访问本地文件系统，路径校验和原子写入约束由所在组件负责。
        """
        cutoff = datetime.now().timestamp() - (days * 86400)
        deleted = 0

        for category in MEMORY_CATEGORIES:
            category_dir = self._get_category_dir(category)
            for memory_file in category_dir.glob("*.json"):
                try:
                    with open(memory_file, encoding="utf-8") as f:
                        data = json.load(f)
                    created_at = datetime.fromisoformat(data["created_at"]).timestamp()

                    if created_at < cutoff:
                        memory_file.unlink()
                        deleted += 1
                except Exception:
                    pass

        return deleted
