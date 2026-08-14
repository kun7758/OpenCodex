"""记忆与上下文子系统中的检索模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from datetime import datetime

from opennova.memory.storage import MemoryStorage
from opennova.memory.types.user_memory import UserMemory


class MemoryRetriever:
    """封装`MemoryRetriever`相关的状态和操作，使调用方通过稳定接口使用该能力。"""

    def __init__(self, storage: MemoryStorage | None = None):
        """初始化`MemoryRetriever`，保存后续操作需要的依赖、配置和初始状态。

        参数：
            storage: 可选的存储。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        if storage is None:
            storage = MemoryStorage()
        self.storage = storage

    def calculate_relevance(self, query: str, memory: UserMemory) -> float:
        """计算相关度，并按照当前组件的约定返回结果。

        参数：
            query: 用于搜索、匹配或排序的查询文本。
            memory: 本次操作使用的记忆。

        返回：
            `float` 类型的处理结果。
        """
        query_lower = query.lower()
        content_lower = memory.content.lower()

        score = 0.0

        # 内容完全命中查询时给予最高基础分。
        if query_lower in content_lower:
            score += 0.5

        # 根据查询词与记忆文本的重叠比例增加相关度。
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())

        overlap = len(query_words & content_words)
        if query_words:
            score += (overlap / len(query_words)) * 0.3

        # 标签命中查询词时增加相关度。
        for tag in memory.tags:
            if query_lower in tag.lower():
                score += 0.2

        # 七天内创建的记忆获得时效加分。
        age_days = (datetime.now() - memory.created_at).days
        if age_days < 7:
            score += 0.1
        elif age_days < 30:
            score += 0.05

        return min(score, 1.0)

    def retrieve(
        self,
        query: str,
        category: str | None = None,
        limit: int = 10,
        min_relevance: float = 0.3,
    ) -> list[UserMemory]:
        """处理检索，并按照当前组件的约定返回结果。

        参数：
            query: 用于搜索、匹配或排序的查询文本。
            category: 用于限定数据范围的类别。
            limit: 最多返回或处理的条目数量。
            min_relevance: 可选的`min_relevance`。

        返回：
            按调用约定排序的结果列表。
        """
        # 先读取全部记忆，或按指定类别缩小候选范围。
        if category:
            memories = self.storage.list_by_category(category)
        else:
            memories = []
            for cat in ["user", "feedback", "project", "reference"]:
                memories.extend(self.storage.list_by_category(cat))

        # 为每条候选记忆计算相关度。
        scored_memories = []
        for memory in memories:
            relevance = self.calculate_relevance(query, memory)
            if relevance >= min_relevance:
                scored_memories.append((relevance, memory))

        # 按相关度从高到低排序。
        scored_memories.sort(key=lambda x: x[0], reverse=True)

        return [memory for _, memory in scored_memories[:limit]]

    def get_recent(self, days: int = 7, limit: int = 20) -> list[UserMemory]:
        """读取近期记忆，不改变当前对象的业务状态。

        参数：
            days: 可选的`days`。
            limit: 最多返回或处理的条目数量。

        返回：
            按调用约定排序的结果列表。
        """
        cutoff = datetime.now().timestamp() - (days * 86400)
        recent_memories = []

        for category in ["user", "feedback", "project", "reference"]:
            for memory in self.storage.list_by_category(category):
                if memory.created_at.timestamp() >= cutoff:
                    recent_memories.append(memory)

        # 按创建时间从新到旧排序。
        recent_memories.sort(key=lambda m: m.created_at.timestamp(), reverse=True)
        return recent_memories[:limit]

    def get_tagged(self, tags: list[str], limit: int = 10) -> list[UserMemory]:
        """读取带标签记忆，不改变当前对象的业务状态。

        参数：
            tags: 本次操作使用的`tags`。
            limit: 最多返回或处理的条目数量。

        返回：
            按调用约定排序的结果列表。
        """
        tagged_memories = []
        tags_lower = [tag.lower() for tag in tags]

        for category in ["user", "feedback", "project", "reference"]:
            for memory in self.storage.list_by_category(category):
                memory_tags_lower = [tag.lower() for tag in memory.tags]
                # 任一标签命中即可保留该记忆。
                if any(tag in memory_tags_lower for tag in tags_lower):
                    tagged_memories.append(memory)

        return tagged_memories[:limit]
