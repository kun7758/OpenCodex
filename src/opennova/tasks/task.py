"""任务管理子系统中的任务模块，集中定义相关数据结构、边界适配和实现逻辑。"""

import asyncio
import contextlib
import hashlib
import os
import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from opennova.utils.task_output import get_task_output_dir, get_task_output_path


class TaskType(StrEnum):
    """枚举任务类型允许出现的稳定取值，序列化和状态判断均使用这些值。"""

    LOCAL_BASH = "local_bash"
    LOCAL_AGENT = "local_agent"
    LOCAL_WORKFLOW = "local_workflow"
    MONITOR_MCP = "monitor_mcp"


class TaskStatus(StrEnum):
    """枚举任务状态允许出现的稳定取值，序列化和状态判断均使用这些值。"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


def is_terminal_status(status: TaskStatus) -> bool:
    """判断终端状态条件是否成立。

    参数：
        status: 本次操作使用的状态。

    返回：
        表示条件是否成立。
    """
    return status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.KILLED)


# 不同任务类型使用不同 ID 前缀，便于日志中快速识别。
TASK_ID_PREFIXES: dict[TaskType, str] = {
    TaskType.LOCAL_BASH: "b",
    TaskType.LOCAL_AGENT: "a",
    TaskType.LOCAL_WORKFLOW: "w",
    TaskType.MONITOR_MCP: "m",
}

# 任务 ID 使用不区分大小写且不易混淆的字符表。
TASK_ID_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"


def generate_task_id(task_type: TaskType) -> str:
    """生成 `task_id` 对应的数据，并按照当前组件的约定返回结果。

    参数：
        task_type: 本次操作使用的任务类型。

    返回：
        处理后的文本或稳定标识。
    """
    prefix = TASK_ID_PREFIXES.get(task_type, "x")
    random_bytes = secrets.token_bytes(8)
    suffix = "".join(TASK_ID_ALPHABET[b % len(TASK_ID_ALPHABET)] for b in random_bytes)
    return f"{prefix}{suffix}"


def generate_message_id(content: str, timestamp: str) -> str:
    """生成 `message_id` 对应的数据，并按照当前组件的约定返回结果。

    参数：
        content: 需要处理、保存或分析的文本内容。
        timestamp: 本次操作使用的`timestamp`。

    返回：
        处理后的文本或稳定标识。
    """
    digest = hashlib.sha1(f"{timestamp}:{content}".encode()).hexdigest()
    return f"msg_{digest[:12]}"


def generate_follow_up_batch_id(task_id: str, delivered_count: int) -> str:
    """生成 `follow_up_batch_id` 对应的数据，并按照当前组件的约定返回结果。

    参数：
        task_id: 目标任务的稳定标识。
        delivered_count: 本次操作使用的`delivered_count`。

    返回：
        处理后的文本或稳定标识。
    """
    return f"batch_{task_id}_{delivered_count + 1}"


@dataclass
class TaskProgressData:
    """保存任务进度数据所需的结构化数据，主要包含 `last_activity`、`token_count`、`tool_use_count`、`last_tool_name`
    字段，便于在组件之间传递或持久化。
    """

    last_activity: str | None = None
    token_count: int = 0
    tool_use_count: int = 0
    last_tool_name: str | None = None


@dataclass
class TaskUsage:
    """保存任务用量所需的结构化数据，主要包含 `total_tokens`、`tool_uses`、`duration_ms` 字段，便于在组件之间传递或持久化。"""

    total_tokens: int = 0
    tool_uses: int = 0
    duration_ms: int = 0


@dataclass
class Task:
    """保存任务所需的结构化数据，主要包含
    `id`、`type`、`description`、`status`、`tool_use_id`、`start_time`、`end_time`、`output_file`
    等字段，便于在组件之间传递或持久化。
    """

    id: str
    type: TaskType
    description: str
    status: TaskStatus = TaskStatus.PENDING
    tool_use_id: str | None = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    output_file: str = field(init=False)
    output_offset: int = 0
    notified: bool = False
    progress: TaskProgressData = field(default_factory=TaskProgressData)
    usage: TaskUsage = field(default_factory=TaskUsage)
    messages: list[dict[str, Any]] = field(default_factory=list)
    session_state: dict[str, Any] = field(default_factory=dict)
    message_queue: list[dict[str, Any]] = field(default_factory=list)
    delivered_messages: list[dict[str, Any]] = field(default_factory=list)
    follow_up_batches: list[dict[str, Any]] = field(default_factory=list)
    blocks: list[str] = field(default_factory=list)
    blocked_by: list[str] = field(default_factory=list)
    retain: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """在数据类字段初始化后规范化任务的派生状态。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self.output_file = get_task_output_path(self.id)

    def to_dict(self) -> dict[str, Any]:
        """把任务转换为可序列化字典，供事件、会话或 API 边界使用。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        return {
            "id": self.id,
            "type": self.type.value,
            "description": self.description,
            "status": self.status.value,
            "tool_use_id": self.tool_use_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "output_file": self.output_file,
            "output_offset": self.output_offset,
            "notified": self.notified,
            "progress": {
                "last_activity": self.progress.last_activity,
                "token_count": self.progress.token_count,
                "tool_use_count": self.progress.tool_use_count,
                "last_tool_name": self.progress.last_tool_name,
            },
            "usage": {
                "total_tokens": self.usage.total_tokens,
                "tool_uses": self.usage.tool_uses,
                "duration_ms": self.usage.duration_ms,
            },
            "messages": self.messages,
            "session_state": self.session_state,
            "message_queue": self.message_queue,
            "delivered_messages": self.delivered_messages,
            "follow_up_batches": self.follow_up_batches,
            "blocks": self.blocks,
            "blocked_by": self.blocked_by,
            "retain": self.retain,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        """从字典恢复任务，并为旧数据缺失的字段补充兼容默认值。

        参数：
            data: 用于构造或恢复对象的结构化数据。

        返回：
            `'Task'` 类型的处理结果。
        """
        progress = TaskProgressData(
            last_activity=data.get("progress", {}).get("last_activity"),
            token_count=data.get("progress", {}).get("token_count", 0),
            tool_use_count=data.get("progress", {}).get("tool_use_count", 0),
            last_tool_name=data.get("progress", {}).get("last_tool_name"),
        )

        usage = TaskUsage(
            total_tokens=data.get("usage", {}).get("total_tokens", 0),
            tool_uses=data.get("usage", {}).get("tool_uses", 0),
            duration_ms=data.get("usage", {}).get("duration_ms", 0),
        )

        task = cls(
            id=data["id"],
            type=TaskType(data["type"]),
            description=data["description"],
            status=TaskStatus(data.get("status", "pending")),
            tool_use_id=data.get("tool_use_id"),
            start_time=datetime.fromisoformat(data["start_time"])
            if data.get("start_time")
            else datetime.now(),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            output_offset=data.get("output_offset", 0),
            notified=data.get("notified", False),
            progress=progress,
            usage=usage,
            messages=data.get("messages", []),
            session_state=data.get("session_state", {}),
            message_queue=data.get("message_queue", []),
            delivered_messages=data.get("delivered_messages", []),
            follow_up_batches=data.get("follow_up_batches", []),
            blocks=data.get("blocks", []),
            blocked_by=data.get("blocked_by", []),
            retain=data.get("retain", True),
            metadata=data.get("metadata", {}),
        )
        # output_file 是派生字段，反序列化后需要显式恢复。
        task.output_file = data.get("output_file", get_task_output_path(task.id))
        return task

    def get_activity_description(self) -> str:
        """读取活动记录说明，不改变当前对象的业务状态。

        返回：
            处理后的文本或稳定标识。
        """
        if self.progress.last_activity:
            return self.progress.last_activity
        return f"{self.type.value}: {self.description[:50]}..."

    def update_progress(
        self, activity: str | None = None, token_count: int = 0, tool_use_count: int = 0
    ) -> None:
        """更新进度，保持运行时状态和持久化记录一致。

        参数：
            activity: 可选的活动记录。
            token_count: 可选的Token数量。
            tool_use_count: 可选的`tool_use_count`。
        """
        if activity:
            self.progress.last_activity = activity
        if token_count:
            self.progress.token_count = token_count
        if tool_use_count:
            self.progress.tool_use_count = tool_use_count

    def update_usage(self, tokens: int = 0, duration_ms: int = 0) -> None:
        """更新用量，保持运行时状态和持久化记录一致。

        参数：
            tokens: 可选的`tokens`。
            duration_ms: 可选的`duration_ms`。
        """
        if tokens:
            self.usage.total_tokens = tokens
        if duration_ms:
            self.usage.duration_ms = duration_ms

    def get_output(self, max_length: int = 10000) -> str:
        """读取输出，不改变当前对象的业务状态。

        参数：
            max_length: 允许返回的最大文本长度。

        返回：
            处理后的文本或稳定标识。

        说明：
            该操作会访问本地文件系统，路径校验和原子写入约束由所在组件负责。
        """
        if not os.path.exists(self.output_file):
            return ""

        try:
            with open(self.output_file, encoding="utf-8") as f:
                f.seek(self.output_offset)
                content = f.read(max_length)
            return content
        except Exception:
            return ""


@dataclass
class TaskResult:
    """保存任务结果所需的结构化数据，主要包含
    `task_id`、`status`、`summary`、`result`、`usage`、`worktree_path`、`worktree_branch`
    字段，便于在组件之间传递或持久化。
    """

    task_id: str
    status: TaskStatus
    summary: str
    result: str | None = None
    usage: TaskUsage | None = None
    worktree_path: str | None = None
    worktree_branch: str | None = None

    def to_notification(self) -> dict[str, Any]:
        """把任务结果转换为通知，供对应协议或边界直接使用。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "summary": self.summary,
            "result": self.result,
            "usage": {
                "total_tokens": self.usage.total_tokens if self.usage else 0,
                "tool_uses": self.usage.tool_uses if self.usage else 0,
                "duration_ms": self.usage.duration_ms if self.usage else 0,
            }
            if self.usage
            else None,
            "worktree_path": self.worktree_path,
            "worktree_branch": self.worktree_branch,
        }


@dataclass
class TaskHandle:
    """保存任务处理所需的结构化数据，主要包含 `task_id`、`cleanup` 字段，便于在组件之间传递或持久化。"""

    task_id: str
    cleanup: Callable[[], None] | None = None

    async def stop(self) -> None:
        """处理停止，并按照当前组件的约定返回结果。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        if self.cleanup:
            self.cleanup()


class TaskManager:
    """管理前台和后台任务的生命周期、依赖关系、消息队列、进度、用量和清理回调。"""

    def __init__(
        self,
        output_dir: str | Path | None = None,
        namespace: str | None = None,
    ) -> None:
        """初始化任务管理，保存后续操作需要的依赖、配置和初始状态。

        参数：
            output_dir: 可选的输出目录。
            namespace: 可选的`namespace`。

        说明：
            执行过程中会更新当前实例维护的状态。
            该操作会访问本地文件系统，路径校验和原子写入约束由所在组件负责。
        """
        self._tasks: dict[str, Task] = {}
        self._cleanup_callbacks: dict[str, Callable[[], None]] = {}
        self._async_handles: dict[str, asyncio.Task[Any]] = {}
        base_output_dir = Path(output_dir) if output_dir is not None else get_task_output_dir()
        self.namespace = namespace or secrets.token_hex(8)
        self.output_dir = (base_output_dir / self.namespace).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_task(
        self,
        task_type: TaskType,
        description: str,
        tool_use_id: str | None = None,
        retain: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> Task:
        """创建任务并完成必要的初始化。

        参数：
            task_type: 本次操作使用的任务类型。
            description: 本次操作使用的说明。
            tool_use_id: 可选的`tool_use_id`。
            retain: 可选的`retain`。
            metadata: 随主体数据传递的扩展元数据。

        返回：
            `Task` 类型的处理结果。
        """
        task_id = generate_task_id(task_type)
        task = Task(
            id=task_id,
            type=task_type,
            description=description,
            tool_use_id=tool_use_id,
            retain=retain,
            metadata=metadata or {},
        )
        task.output_file = str(self.output_dir / f"{task_id}.txt")
        self._tasks[task_id] = task
        return task

    def write_task_output(self, task_id: str, content: str, offset: int = 0) -> int:
        """写入任务输出，并按照当前组件的约定返回结果。

        参数：
            task_id: 目标任务的稳定标识。
            content: 需要处理、保存或分析的文本内容。
            offset: 可选的`offset`。

        返回：
            `int` 类型的处理结果。

        说明：
            该操作会访问本地文件系统，路径校验和原子写入约束由所在组件负责。
        """
        task = self._tasks.get(task_id)
        if task is None:
            return offset
        mode = "a" if offset == 0 else "r+"
        try:
            with open(task.output_file, mode, encoding="utf-8") as stream:
                if offset > 0:
                    stream.seek(offset)
                stream.write(content)
                return stream.tell()
        except OSError:
            return offset

    def read_task_output(
        self,
        task_id: str,
        max_length: int = 10_000,
        offset: int = 0,
    ) -> tuple[str, int]:
        """读取任务输出，并按照当前组件的约定返回结果。

        参数：
            task_id: 目标任务的稳定标识。
            max_length: 允许返回的最大文本长度。
            offset: 可选的`offset`。

        返回：
            `tuple[str, int]` 类型的处理结果。

        说明：
            该操作会访问本地文件系统，路径校验和原子写入约束由所在组件负责。
        """
        task = self._tasks.get(task_id)
        if task is None or not Path(task.output_file).exists():
            return "", offset
        try:
            with open(task.output_file, encoding="utf-8") as stream:
                stream.seek(offset)
                content = stream.read(max_length)
                return content, stream.tell()
        except OSError:
            return "", offset

    def get_task(self, task_id: str) -> Task | None:
        """读取任务，不改变当前对象的业务状态。

        参数：
            task_id: 目标任务的稳定标识。

        返回：
            `Task | None` 类型的处理结果。
        """
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> list[Task]:
        """读取全部任务，不改变当前对象的业务状态。

        返回：
            按调用约定排序的结果列表。
        """
        return list(self._tasks.values())

    def get_tasks_by_type(self, task_type: TaskType) -> list[Task]:
        """读取 `tasks_by_type` 对应的数据，不改变当前对象的业务状态。

        参数：
            task_type: 本次操作使用的任务类型。

        返回：
            按调用约定排序的结果列表。
        """
        return [t for t in self._tasks.values() if t.type == task_type]

    def get_active_tasks(self) -> list[Task]:
        """读取 `active_tasks` 对应的数据，不改变当前对象的业务状态。

        返回：
            按调用约定排序的结果列表。
        """
        return [t for t in self._tasks.values() if t.status == TaskStatus.RUNNING]

    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        end_time: datetime | None = None,
    ) -> bool:
        """更新任务状态，保持运行时状态和持久化记录一致。

        参数：
            task_id: 目标任务的稳定标识。
            status: 本次操作使用的状态。
            end_time: 可选的`end_time`。

        返回：
            表示条件是否成立。
        """
        task = self._tasks.get(task_id)
        if not task:
            return False

        task.status = status
        if is_terminal_status(status) and end_time is None:
            task.end_time = datetime.now()
        elif end_time:
            task.end_time = end_time

        return True

    def add_message(self, task_id: str, message: dict[str, Any]) -> bool:
        """添加`add_message`，必要时执行去重或容量检查。

        参数：
            task_id: 目标任务的稳定标识。
            message: 用户提交或组件间传递的消息。

        返回：
            表示条件是否成立。
        """
        task = self._tasks.get(task_id)
        if not task:
            return False
        task.messages.append(message)
        return True

    def update_task_progress(
        self,
        task_id: str,
        activity: str | None = None,
        token_count: int = 0,
        tool_use_increment: int = 0,
        last_tool_name: str | None = None,
        mark_complete: bool = False,
    ) -> bool:
        """更新任务进度，保持运行时状态和持久化记录一致。

        参数：
            task_id: 目标任务的稳定标识。
            activity: 可选的活动记录。
            token_count: 可选的Token数量。
            tool_use_increment: 可选的`tool_use_increment`。
            last_tool_name: 可选的`last_tool_name`。
            mark_complete: 可选的`mark_complete`。

        返回：
            表示条件是否成立。
        """
        task = self._tasks.get(task_id)
        if not task:
            return False

        if activity:
            task.progress.last_activity = activity
        if token_count:
            task.progress.token_count = token_count
            task.usage.total_tokens += token_count
        if tool_use_increment:
            task.progress.tool_use_count += tool_use_increment
            task.usage.tool_uses += tool_use_increment
        if last_tool_name:
            task.progress.last_tool_name = last_tool_name
        if mark_complete and task.start_time:
            task.usage.duration_ms = int((datetime.now() - task.start_time).total_seconds() * 1000)

        return True

    def set_session_state(self, task_id: str, **state: Any) -> bool:
        """设置会话状态并保持相关派生状态同步。

        参数：
            task_id: 目标任务的稳定标识。
            **state: 传递给底层实现的额外关键字参数。

        返回：
            表示条件是否成立。
        """
        task = self._tasks.get(task_id)
        if not task:
            return False
        task.session_state.update(state)
        return True

    def dequeue_messages(self, task_id: str) -> list[dict[str, Any]]:
        """释放或移除 `dequeue_messages` 所表示的数据或流程，并遵守任务管理定义的边界与状态约束。

        参数：
            task_id: 目标任务的稳定标识。

        返回：
            按调用约定排序的结果列表。
        """
        task = self._tasks.get(task_id)
        if not task:
            return []

        queued = task.message_queue.copy()
        task.message_queue.clear()
        return queued

    def mark_messages_delivered(self, task_id: str, messages: list[dict[str, Any]]) -> bool:
        """把`mark_messages_delivered`更新为目标状态，并触发必要的状态事件。

        参数：
            task_id: 目标任务的稳定标识。
            messages: 按协议顺序排列的对话消息。

        返回：
            表示条件是否成立。
        """
        task = self._tasks.get(task_id)
        if not task:
            return False

        task.delivered_messages.extend(messages)
        return True

    def record_follow_up_batch(
        self, task_id: str, messages: list[dict[str, Any]], rendered_content: str
    ) -> dict[str, Any] | None:
        """记录`record_follow_up_batch`，供状态展示、恢复或后续决策使用。

        参数：
            task_id: 目标任务的稳定标识。
            messages: 按协议顺序排列的对话消息。
            rendered_content: 本次操作使用的`rendered_content`。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        task = self._tasks.get(task_id)
        if not task:
            return None

        delivered_at = datetime.now().isoformat()
        batch = {
            "batch_id": generate_follow_up_batch_id(task_id, len(task.follow_up_batches)),
            "delivered_at": delivered_at,
            "message_count": len(messages),
            "message_ids": [
                message.get("message_id") for message in messages if message.get("message_id")
            ],
            "messages": [message.copy() for message in messages],
            "rendered_content": rendered_content,
        }
        task.follow_up_batches.append(batch)
        return batch

    def has_pending_messages(self, task_id: str) -> bool:
        """判断`pending_messages`条件是否成立。

        参数：
            task_id: 目标任务的稳定标识。

        返回：
            表示条件是否成立。
        """
        task = self._tasks.get(task_id)
        if not task:
            return False
        return bool(task.message_queue)

    def _has_dependency_path(self, start_task_id: str, target_task_id: str) -> bool:
        """校验 `_has_dependency_path` 所表示的数据或流程，并遵守任务管理定义的边界与状态约束。

        参数：
            start_task_id: 本次操作使用的`start_task_id`。
            target_task_id: 本次操作使用的`target_task_id`。

        返回：
            表示条件是否成立。
        """
        stack = [start_task_id]
        visited: set[str] = set()

        while stack:
            current_task_id = stack.pop()
            if current_task_id == target_task_id:
                return True
            if current_task_id in visited:
                continue
            visited.add(current_task_id)

            current_task = self._tasks.get(current_task_id)
            if not current_task:
                continue
            stack.extend(current_task.blocks)

        return False

    def add_dependency(
        self, prerequisite_task_id: str, dependent_task_id: str
    ) -> tuple[bool, str | None]:
        """添加`add_dependency`，必要时执行去重或容量检查。

        参数：
            prerequisite_task_id: 本次操作使用的`prerequisite_task_id`。
            dependent_task_id: 本次操作使用的`dependent_task_id`。

        返回：
            `tuple[bool, str | None]` 类型的处理结果。
        """
        prerequisite_task = self._tasks.get(prerequisite_task_id)
        dependent_task = self._tasks.get(dependent_task_id)

        if not prerequisite_task:
            return False, f"Task '{prerequisite_task_id}' not found"
        if not dependent_task:
            return False, f"Task '{dependent_task_id}' not found"
        if prerequisite_task_id == dependent_task_id:
            return False, "A task cannot depend on itself"
        if self._has_dependency_path(dependent_task_id, prerequisite_task_id):
            return False, "Dependency cycle detected"

        if dependent_task_id not in prerequisite_task.blocks:
            prerequisite_task.blocks.append(dependent_task_id)
        if prerequisite_task_id not in dependent_task.blocked_by:
            dependent_task.blocked_by.append(prerequisite_task_id)

        return True, None

    def is_task_blocked(self, task_id: str) -> bool:
        """判断任务阻止项条件是否成立。

        参数：
            task_id: 目标任务的稳定标识。

        返回：
            表示条件是否成立。
        """
        task = self._tasks.get(task_id)
        if not task:
            return False

        for blocker_id in task.blocked_by:
            blocker = self._tasks.get(blocker_id)
            if blocker and blocker.status != TaskStatus.COMPLETED:
                return True
        return False

    def get_open_blocker_ids(self, task_id: str) -> list[str]:
        """读取 `open_blocker_ids` 对应的数据，不改变当前对象的业务状态。

        参数：
            task_id: 目标任务的稳定标识。

        返回：
            按调用约定排序的结果列表。
        """
        task = self._tasks.get(task_id)
        if not task:
            return []

        open_blockers = []
        for blocker_id in task.blocked_by:
            blocker = self._tasks.get(blocker_id)
            if blocker and blocker.status != TaskStatus.COMPLETED:
                open_blockers.append(blocker_id)
        return open_blockers

    def set_cleanup_callback(self, task_id: str, callback: Callable[[], None]) -> None:
        """设置清理回调并保持相关派生状态同步。

        参数：
            task_id: 目标任务的稳定标识。
            callback: 在对应事件发生时调用的回调函数。
        """
        self._cleanup_callbacks[task_id] = callback

    def set_async_handle(self, task_id: str, handle: asyncio.Task[Any]) -> None:
        """设置异步处理并保持相关派生状态同步。

        参数：
            task_id: 目标任务的稳定标识。
            handle: 本次操作使用的处理。
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task '{task_id}' not found")
        self._async_handles[task_id] = handle

    def release_async_handle(self, task_id: str, handle: asyncio.Task[Any]) -> None:
        """释放异步处理，并按照当前组件的约定返回结果。

        参数：
            task_id: 目标任务的稳定标识。
            handle: 本次操作使用的处理。
        """
        if self._async_handles.get(task_id) is handle:
            self._async_handles.pop(task_id, None)

    async def stop_task(self, task_id: str) -> bool:
        """停止任务，并按照当前组件的约定返回结果。

        参数：
            task_id: 目标任务的稳定标识。

        返回：
            表示条件是否成立。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        task = self._tasks.get(task_id)
        if not task or task.status != TaskStatus.RUNNING:
            return False

        # 先更新状态，使随后执行的清理回调看到一致的最终状态。
        self.update_task_status(task_id, TaskStatus.KILLED)

        handle = self._async_handles.pop(task_id, None)
        if handle is not None and not handle.done():
            handle.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await handle

        # 如果任务注册了清理回调，在移除任务前执行。
        if task_id in self._cleanup_callbacks:
            with contextlib.suppress(Exception):
                self._cleanup_callbacks[task_id]()
            del self._cleanup_callbacks[task_id]

        return True

    async def aclose(self) -> None:
        """异步关闭当前对象持有的任务、连接和运行时资源；重复调用保持幂等。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        handles = list(self._async_handles.items())
        self._async_handles.clear()
        for task_id, handle in handles:
            if not handle.done():
                self.update_task_status(task_id, TaskStatus.KILLED)
                handle.cancel()
        if handles:
            await asyncio.gather(*(handle for _, handle in handles), return_exceptions=True)

        callbacks = list(self._cleanup_callbacks.values())
        self._cleanup_callbacks.clear()
        for callback in callbacks:
            with contextlib.suppress(Exception):
                callback()

    def remove_task(self, task_id: str) -> bool:
        """移除移除任务指向的数据，并清理相关索引或资源。

        参数：
            task_id: 目标任务的稳定标识。

        返回：
            表示条件是否成立。
        """
        if task_id in self._tasks:
            del self._tasks[task_id]
            self._async_handles.pop(task_id, None)
            if task_id in self._cleanup_callbacks:
                del self._cleanup_callbacks[task_id]
            return True
        return False

    def cleanup_completed_tasks(self, max_age_hours: int = 24) -> int:
        """清理 `completed_tasks` 对应的数据，并按照当前组件的约定返回结果。

        参数：
            max_age_hours: 可选的`max_age_hours`。

        返回：
            `int` 类型的处理结果。
        """
        now = datetime.now()
        to_remove = []

        for task_id, task in self._tasks.items():
            if is_terminal_status(task.status) and task.end_time:
                age = now - task.end_time
                if age.total_seconds() > max_age_hours * 3600:
                    to_remove.append(task_id)

        for task_id in to_remove:
            self.remove_task(task_id)

        return len(to_remove)

    def to_dict(self) -> dict[str, Any]:
        """把任务管理转换为可序列化字典，供事件、会话或 API 边界使用。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        return {
            "tasks": [task.to_dict() for task in self._tasks.values()],
            "count": len(self._tasks),
            "active": len(self.get_active_tasks()),
        }
