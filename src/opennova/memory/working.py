"""记忆与上下文子系统中的工作模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class ActionStatus(StrEnum):
    """枚举动作状态允许出现的稳定取值，序列化和状态判断均使用这些值。"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ActionRecord:
    """保存动作记录所需的结构化数据，主要包含
    `id`、`tool_name`、`arguments`、`status`、`result`、`error`、`timestamp`、`duration_ms`
    字段，便于在组件之间传递或持久化。
    """

    id: str
    tool_name: str
    arguments: dict[str, Any]
    status: ActionStatus = ActionStatus.PENDING
    result: str | None = None
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: int | None = None


@dataclass
class FileObservation:
    """保存文件观察记录所需的结构化数据，主要包含 `file_path`、`change_type`、`content_preview`、`timestamp`
    字段，便于在组件之间传递或持久化。
    """

    file_path: str
    change_type: str
    content_preview: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TaskState:
    """保存任务状态所需的结构化数据，主要包含 `description`、`status`、`progress`、`started_at`、`completed_at`、`error`
    字段，便于在组件之间传递或持久化。
    """

    description: str
    status: str = "pending"
    progress: float = 0.0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None


class WorkingMemory:
    """保存单次任务生命周期内的短期状态，包括动作历史、文件观察、临时决策和上下文项；任务结束后可提炼到项目记忆。"""

    def __init__(self, task: str = ""):
        """初始化工作记忆，保存后续操作需要的依赖、配置和初始状态。

        参数：
            task: 用户希望 Agent 完成的任务描述。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self.task_state = TaskState(description=task)
        self.actions: list[ActionRecord] = []
        self.observations: list[FileObservation] = []
        self.decisions: list[str] = []
        self.context_items: dict[str, Any] = {}
        self._action_counter = 0

    def set_task(self, task: str) -> None:
        """设置任务并保持相关派生状态同步。

        参数：
            task: 用户希望 Agent 完成的任务描述。
        """
        self.task_state.description = task
        self.task_state.status = "pending"
        self.task_state.progress = 0.0

    def start_task(self) -> None:
        """启动任务，并按照当前组件的约定返回结果。"""
        self.task_state.status = "running"
        self.task_state.started_at = datetime.now()

    def update_progress(self, progress: float) -> None:
        """更新进度，保持运行时状态和持久化记录一致。

        参数：
            progress: 本次操作使用的进度。
        """
        self.task_state.progress = min(1.0, max(0.0, progress))

    def complete_task(self, success: bool = True, error: str | None = None) -> None:
        """更新 `complete_task` 所表示的数据或流程，并遵守工作记忆定义的边界与状态约束。

        参数：
            success: 可选的成功。
            error: 可选的错误。
        """
        self.task_state.status = "completed" if success else "failed"
        self.task_state.completed_at = datetime.now()
        self.task_state.progress = 1.0
        if error:
            self.task_state.error = error

    def record_action(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ActionRecord:
        """记录动作，供状态展示、恢复或后续决策使用。

        参数：
            tool_name: 目标工具在注册表中的名称。
            arguments: 工具调用的结构化参数。

        返回：
            `ActionRecord` 类型的处理结果。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self._action_counter += 1
        record = ActionRecord(
            id=f"action_{self._action_counter}",
            tool_name=tool_name,
            arguments=arguments,
            timestamp=datetime.now(),
        )
        self.actions.append(record)
        return record

    def update_action(
        self,
        action_id: str,
        status: ActionStatus,
        result: str | None = None,
        error: str | None = None,
    ) -> None:
        """更新动作，保持运行时状态和持久化记录一致。

        参数：
            action_id: 本次操作使用的`action_id`。
            status: 本次操作使用的状态。
            result: 前一步执行得到的规范化结果。
            error: 可选的错误。
        """
        for action in self.actions:
            if action.id == action_id:
                action.status = status
                action.result = result
                action.error = error
                break

    def observe_file(
        self,
        file_path: str,
        change_type: str,
        content_preview: str | None = None,
    ) -> None:
        """更新 `observe_file` 所表示的数据或流程，并遵守工作记忆定义的边界与状态约束。

        参数：
            file_path: 目标文件的路径；访问范围仍受项目沙箱约束。
            change_type: 本次操作使用的变更类型。
            content_preview: 可选的内容预览。
        """
        observation = FileObservation(
            file_path=file_path,
            change_type=change_type,
            content_preview=content_preview,
        )
        self.observations.append(observation)

    def add_decision(self, decision: str) -> None:
        """添加`add_decision`，必要时执行去重或容量检查。

        参数：
            decision: 用户或策略给出的决策结果。
        """
        self.decisions.append(decision)

    def set_context(self, key: str, value: Any) -> None:
        """设置上下文并保持相关派生状态同步。

        参数：
            key: 本次操作使用的`key`。
            value: 需要保存、转换或校验的值。
        """
        self.context_items[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """读取上下文，不改变当前对象的业务状态。

        参数：
            key: 本次操作使用的`key`。
            default: 可选的默认。

        返回：
            `Any` 类型的处理结果。
        """
        return self.context_items.get(key, default)

    def get_action_history(self) -> list[dict[str, Any]]:
        """读取动作历史，不改变当前对象的业务状态。

        返回：
            按调用约定排序的结果列表。
        """
        return [
            {
                "id": a.id,
                "tool": a.tool_name,
                "args": a.arguments,
                "status": a.status.value,
                "result": a.result,
                "error": a.error,
            }
            for a in self.actions
        ]

    def get_files_modified(self) -> list[str]:
        """读取文件已修改文件，不改变当前对象的业务状态。

        返回：
            按调用约定排序的结果列表。
        """
        return list(
            {
                obs.file_path
                for obs in self.observations
                if obs.change_type in ("modified", "created", "deleted")
            }
        )

    def get_files_read(self) -> list[str]:
        """读取文件读取，不改变当前对象的业务状态。

        返回：
            按调用约定排序的结果列表。
        """
        return list(
            {obs.file_path for obs in self.observations if obs.change_type == "read"}
        )

    def get_summary(self) -> dict[str, Any]:
        """读取摘要，不改变当前对象的业务状态。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        return {
            "task": self.task_state.description,
            "status": self.task_state.status,
            "progress": self.task_state.progress,
            "total_actions": len(self.actions),
            "successful_actions": sum(
                1 for a in self.actions if a.status == ActionStatus.SUCCESS
            ),
            "failed_actions": sum(
                1 for a in self.actions if a.status == ActionStatus.FAILED
            ),
            "files_read": len(self.get_files_read()),
            "files_modified": len(self.get_files_modified()),
            "decisions_made": len(self.decisions),
        }

    def clear(self) -> None:
        """处理清理，并按照当前组件的约定返回结果。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self.task_state = TaskState(description="")
        self.actions.clear()
        self.observations.clear()
        self.decisions.clear()
        self.context_items.clear()
        self._action_counter = 0

    def __repr__(self) -> str:
        return (
            f"WorkingMemory(task={self.task_state.description[:30]}..., "
            f"actions={len(self.actions)}, "
            f"status={self.task_state.status})"
        )
