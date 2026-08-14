"""记忆与上下文子系统中的项目模块，集中定义相关数据结构、边界适配和实现逻辑。"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

MEMORY_DIR = ".opennova"
MEMORY_FILE = "memory.json"


@dataclass
class ProjectStructure:
    """数据对象 `ProjectStructure` 主要保存
    `root_path`、`total_files`、`total_dirs`、`file_types`、`last_scanned` 字段，用于在组件之间传递或持久化这组状态。
    """

    root_path: str
    total_files: int = 0
    total_dirs: int = 0
    file_types: dict[str, int] = field(default_factory=dict)
    last_scanned: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """把`ProjectStructure`转换为可序列化字典，供事件、会话或 API 边界使用。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        return {
            "root_path": self.root_path,
            "total_files": self.total_files,
            "total_dirs": self.total_dirs,
            "file_types": self.file_types,
            "last_scanned": self.last_scanned.isoformat() if self.last_scanned else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectStructure":
        """从字典恢复`ProjectStructure`，并为旧数据缺失的字段补充兼容默认值。

        参数：
            data: 用于构造或恢复对象的结构化数据。

        返回：
            `'ProjectStructure'` 类型的处理结果。
        """
        return cls(
            root_path=data.get("root_path", ""),
            total_files=data.get("total_files", 0),
            total_dirs=data.get("total_dirs", 0),
            file_types=data.get("file_types", {}),
            last_scanned=(
                datetime.fromisoformat(data["last_scanned"])
                if data.get("last_scanned")
                else None
            ),
        )


@dataclass
class DecisionRecord:
    """保存决策记录所需的结构化数据，主要包含 `id`、`description`、`reasoning`、`timestamp`、`context` 字段，便于在组件之间传递或持久化。"""

    id: str
    description: str
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """把决策记录转换为可序列化字典，供事件、会话或 API 边界使用。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        return {
            "id": self.id,
            "description": self.description,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecisionRecord":
        """从字典恢复决策记录，并为旧数据缺失的字段补充兼容默认值。

        参数：
            data: 用于构造或恢复对象的结构化数据。

        返回：
            `'DecisionRecord'` 类型的处理结果。
        """
        return cls(
            id=data["id"],
            description=data["description"],
            reasoning=data["reasoning"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            context=data.get("context", {}),
        )


@dataclass
class UserPreference:
    """保存用户偏好所需的结构化数据，主要包含 `key`、`value`、`category`、`last_used` 字段，便于在组件之间传递或持久化。"""

    key: str
    value: Any
    category: str = "general"
    last_used: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """把用户偏好转换为可序列化字典，供事件、会话或 API 边界使用。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        return {
            "key": self.key,
            "value": self.value,
            "category": self.category,
            "last_used": self.last_used.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserPreference":
        """从字典恢复用户偏好，并为旧数据缺失的字段补充兼容默认值。

        参数：
            data: 用于构造或恢复对象的结构化数据。

        返回：
            `'UserPreference'` 类型的处理结果。
        """
        return cls(
            key=data["key"],
            value=data["value"],
            category=data.get("category", "general"),
            last_used=(
                datetime.fromisoformat(data["last_used"])
                if data.get("last_used")
                else datetime.now()
            ),
        )


class ProjectMemory:
    """保存在项目目录中的长期记忆，记录项目结构、关键决策、用户偏好和会话摘要，供后续任务检索。"""

    def __init__(self, project_path: str = "."):
        """初始化项目记忆，保存后续操作需要的依赖、配置和初始状态。

        参数：
            project_path: 可选的项目路径。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self.project_path = Path(project_path).resolve()
        self.memory_path = self.project_path / MEMORY_DIR / MEMORY_FILE

        self.structure = ProjectStructure(root_path=str(self.project_path))
        self.decisions: list[DecisionRecord] = []
        self.preferences: dict[str, UserPreference] = {}
        self.session_history: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {}

        self._load()

    def _load(self) -> None:
        """处理加载，并按照当前组件的约定返回结果。

        说明：
            执行过程中会更新当前实例维护的状态。
            该操作会访问本地文件系统，路径校验和原子写入约束由所在组件负责。
        """
        if self.memory_path.exists():
            try:
                with open(self.memory_path, encoding="utf-8") as f:
                    data = json.load(f)

                self.structure = ProjectStructure.from_dict(
                    data.get("structure", {"root_path": str(self.project_path)})
                )

                self.decisions = [
                    DecisionRecord.from_dict(d) for d in data.get("decisions", [])
                ]

                self.preferences = {
                    k: UserPreference.from_dict(v)
                    for k, v in data.get("preferences", {}).items()
                }

                self.session_history = data.get("sessions", [])
                self.metadata = data.get("metadata", {})

            except Exception:
                pass

    def save(self) -> None:
        """处理保存，并按照当前组件的约定返回结果。

        说明：
            该操作会访问本地文件系统，路径校验和原子写入约束由所在组件负责。
        """
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "structure": self.structure.to_dict(),
            "decisions": [d.to_dict() for d in self.decisions],
            "preferences": {k: v.to_dict() for k, v in self.preferences.items()},
            "sessions": self.session_history[-20:],
            "metadata": self.metadata,
            "last_updated": datetime.now().isoformat(),
        }

        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def scan_project(self) -> None:
        """扫描项目，并按照当前组件的约定返回结果。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        file_types: dict[str, int] = {}
        total_files = 0
        total_dirs = 0

        ignore_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv", ".tox", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".eggs", "*.egg-info", "dist", "build"}

        for path in self.project_path.rglob("*"):
            if any(part in ignore_dirs for part in path.parts):
                continue

            if path.is_file():
                total_files += 1
                ext = path.suffix.lower() or "no_extension"
                file_types[ext] = file_types.get(ext, 0) + 1
            elif path.is_dir():
                total_dirs += 1

        self.structure = ProjectStructure(
            root_path=str(self.project_path),
            total_files=total_files,
            total_dirs=total_dirs,
            file_types=file_types,
            last_scanned=datetime.now(),
        )

        self.save()

    def add_decision(
        self,
        description: str,
        reasoning: str,
        context: dict[str, Any] | None = None,
    ) -> DecisionRecord:
        """添加`add_decision`，必要时执行去重或容量检查。

        参数：
            description: 本次操作使用的说明。
            reasoning: 本次操作使用的`reasoning`。
            context: 本次工具调用或运行所使用的上下文。

        返回：
            `DecisionRecord` 类型的处理结果。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        decision_id = f"decision_{len(self.decisions) + 1}"

        decision = DecisionRecord(
            id=decision_id,
            description=description,
            reasoning=reasoning,
            context=context or {},
        )

        self.decisions.append(decision)

        if len(self.decisions) > 50:
            self.decisions = self.decisions[-50:]

        self.save()

        return decision

    def get_relevant_decisions(self, topic: str, limit: int = 5) -> list[DecisionRecord]:
        """读取 `relevant_decisions` 对应的数据，不改变当前对象的业务状态。

        参数：
            topic: 本次操作使用的`topic`。
            limit: 最多返回或处理的条目数量。

        返回：
            按调用约定排序的结果列表。
        """
        topic_lower = topic.lower()
        relevant = []

        for decision in reversed(self.decisions):
            if (
                topic_lower in decision.description.lower()
                or topic_lower in decision.reasoning.lower()
            ):
                relevant.append(decision)

            if len(relevant) >= limit:
                break

        return relevant

    def set_preference(self, key: str, value: Any, category: str = "general") -> None:
        """设置偏好并保持相关派生状态同步。

        参数：
            key: 本次操作使用的`key`。
            value: 需要保存、转换或校验的值。
            category: 用于限定数据范围的类别。
        """
        self.preferences[key] = UserPreference(
            key=key,
            value=value,
            category=category,
        )
        self.save()

    def get_preference(self, key: str, default: Any = None) -> Any:
        """读取偏好，不改变当前对象的业务状态。

        参数：
            key: 本次操作使用的`key`。
            default: 可选的默认。

        返回：
            `Any` 类型的处理结果。
        """
        if key in self.preferences:
            pref = self.preferences[key]
            pref.last_used = datetime.now()
            return pref.value
        return default

    def record_session(
        self,
        task: str,
        success: bool,
        duration_seconds: float,
    ) -> None:
        """记录会话，供状态展示、恢复或后续决策使用。

        参数：
            task: 用户希望 Agent 完成的任务描述。
            success: 本次操作使用的成功。
            duration_seconds: 本次操作使用的`duration_seconds`。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        session = {
            "task": task[:200],
            "success": success,
            "duration_seconds": duration_seconds,
            "timestamp": datetime.now().isoformat(),
        }

        self.session_history.append(session)

        if len(self.session_history) > 20:
            self.session_history = self.session_history[-20:]

        self.save()

    def get_project_context(self) -> str:
        """读取项目上下文，不改变当前对象的业务状态。

        返回：
            处理后的文本或稳定标识。
        """
        parts = [f"Project: {self.project_path.name}"]

        if self.structure.total_files > 0:
            parts.append(f"Files: {self.structure.total_files}")

            top_types = sorted(
                self.structure.file_types.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]

            if top_types:
                type_strs = [f"{ext} ({count})" for ext, count in top_types]
                parts.append(f"Main types: {', '.join(type_strs)}")

        if self.decisions:
            parts.append(f"Key decisions: {len(self.decisions)}")

        return "\n".join(parts)

    def get_summary(self) -> dict[str, Any]:
        """读取摘要，不改变当前对象的业务状态。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        return {
            "project_path": str(self.project_path),
            "total_files": self.structure.total_files,
            "total_dirs": self.structure.total_dirs,
            "file_types": self.structure.file_types,
            "decisions_count": len(self.decisions),
            "preferences_count": len(self.preferences),
            "sessions_count": len(self.session_history),
        }

    def clear(self) -> None:
        """处理清理，并按照当前组件的约定返回结果。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self.structure = ProjectStructure(root_path=str(self.project_path))
        self.decisions.clear()
        self.preferences.clear()
        self.session_history.clear()
        self.metadata.clear()
        self.save()

    def __repr__(self) -> str:
        return (
            f"ProjectMemory(path={self.project_path.name}, "
            f"files={self.structure.total_files}, "
            f"decisions={len(self.decisions)})"
        )
