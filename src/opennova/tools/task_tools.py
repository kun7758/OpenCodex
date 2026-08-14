"""内置工具系统中的任务工具模块，集中定义相关数据结构、边界适配和实现逻辑。"""

import asyncio
from typing import Any

from opennova.tasks import Task, TaskManager, TaskStatus, TaskType
from opennova.tools.base import BaseTool, ToolResult


class TaskManagerTool(BaseTool):
    """实现任务管理工具。模型通过统一工具 Schema 调用它，执行结果使用 ToolResult 返回，并服从运行时安全策略。"""

    def __init__(
        self,
        task_manager: TaskManager | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config)
        configured = self.config.get("task_manager")
        manager = task_manager or (configured if isinstance(configured, TaskManager) else None)
        if manager is None:
            raise ValueError("A runtime-owned TaskManager is required")
        self.task_manager: TaskManager = manager


def _format_dependency_details(manager: TaskManager, task: Task) -> list[str]:
    """把依赖详细信息整理为稳定、便于展示的文本格式。

    参数：
        manager: 本次操作使用的管理。
        task: 用户希望 Agent 完成的任务描述。

    返回：
        按调用约定排序的结果列表。
    """
    details: list[str] = []

    if task.blocked_by:
        open_blockers = manager.get_open_blocker_ids(task.id)
        blocker_text = ", ".join(task.blocked_by)
        if open_blockers:
            details.append(f"blocked_by: {blocker_text} (open: {', '.join(open_blockers)})")
        else:
            details.append(f"blocked_by: {blocker_text} (open: none)")

    if task.blocks:
        details.append(f"blocks: {', '.join(task.blocks)}")

    if task.blocked_by:
        details.append(f"is_blocked: {manager.is_task_blocked(task.id)}")

    return details


class TaskCreateTool(TaskManagerTool):
    """实现任务创建工具。模型通过统一工具 Schema 调用它，执行结果使用 ToolResult 返回，并服从运行时安全策略。"""

    name = "task_create"
    description = "Create a new structured task for tracking work progress. Use this when you need to track a multi-step task, coordinate work with other tools, or want to organize complex work into smaller trackable units."

    def execute(
        self,
        subject: str,
        description: str,
        active_form: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """执行任务创建工具对应的实际操作，校验输入并返回统一结果。

        参数：
            subject: 本次操作使用的`subject`。
            description: 本次操作使用的说明。
            active_form: 可选的`active_form`。
            metadata: 随主体数据传递的扩展元数据。
            **kwargs: 传递给底层实现的额外关键字参数。

        返回：
            `ToolResult` 类型的处理结果。
        """
        try:
            manager = self.task_manager
            full_description = f"{subject}: {description}"
            task_type = TaskType.LOCAL_WORKFLOW

            task = manager.create_task(
                task_type=task_type,
                description=full_description,
                metadata={
                    "subject": subject,
                    "active_form": active_form,
                    **(metadata or {}),
                },
            )

            return ToolResult(
                success=True,
                output=f"Created task {task.id}: {subject}",
                metadata={"task_id": task.id, "task": task.to_dict()},
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class TaskListTool(TaskManagerTool):
    """实现任务列表工具。模型通过统一工具 Schema 调用它，执行结果使用 ToolResult 返回，并服从运行时安全策略。"""

    name = "task_list"
    description = "List all tasks in the task list. Use this to see all available tasks, their status, and which tasks you can work on next."

    def execute(self, **kwargs: Any) -> ToolResult:
        """执行任务列表工具对应的实际操作，校验输入并返回统一结果。

        参数：
            **kwargs: 传递给底层实现的额外关键字参数。

        返回：
            `ToolResult` 类型的处理结果。
        """
        try:
            manager = self.task_manager
            tasks = manager.get_all_tasks()

            if not tasks:
                return ToolResult(success=True, output="No tasks in the task list.")

            output_lines = ["Tasks:"]
            for task in tasks:
                status_icon = {
                    "pending": "○",
                    "running": "⟳",
                    "completed": "✓",
                    "failed": "✗",
                    "killed": "⊘",
                }.get(task.status.value, "?")
                owner = task.metadata.get("owner", "")
                dependency_details = _format_dependency_details(manager, task)

                output_lines.append(
                    f"  [{task.id}] {status_icon} {task.description[:60]}{'...' if len(task.description) > 60 else ''}"
                )
                if owner:
                    output_lines.append(f"      owner: {owner}")
                for detail in dependency_details:
                    output_lines.append(f"      {detail}")

            return ToolResult(
                success=True,
                output="\n".join(output_lines),
                metadata={"tasks": [task.to_dict() for task in tasks]},
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class TaskGetTool(TaskManagerTool):
    """实现`TaskGetTool`。模型通过统一工具 Schema 调用它，执行结果使用 ToolResult 返回，并服从运行时安全策略。"""

    name = "task_get"
    description = "Get a task by its ID from the task list. Use this to see the full description, status, dependencies, and context of a specific task before working on it."

    def execute(self, task_id: str, **kwargs: Any) -> ToolResult:
        """执行`TaskGetTool`对应的实际操作，校验输入并返回统一结果。

        参数：
            task_id: 目标任务的稳定标识。
            **kwargs: 传递给底层实现的额外关键字参数。

        返回：
            `ToolResult` 类型的处理结果。
        """
        try:
            manager = self.task_manager
            task = manager.get_task(task_id)

            if not task:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Task '{task_id}' not found",
                )

            output_lines = [
                f"Task: {task.id}",
                f"Description: {task.description}",
                f"Status: {task.status.value}",
                f"Type: {task.type.value}",
            ]

            dependency_details = _format_dependency_details(manager, task)
            if dependency_details:
                output_lines.append("")
                output_lines.append("Dependencies:")
                for detail in dependency_details:
                    output_lines.append(f"  {detail}")

            if task.start_time:
                output_lines.append(f"Started: {task.start_time.isoformat()}")
            if task.end_time:
                output_lines.append(f"Ended: {task.end_time.isoformat()}")

            if task.metadata:
                output_lines.append("\nMetadata:")
                for key, value in task.metadata.items():
                    if key != "description":
                        output_lines.append(f"  {key}: {value}")

            if task.usage and task.usage.total_tokens > 0:
                output_lines.append(
                    f"\nUsage: {task.usage.total_tokens} tokens, {task.usage.tool_uses} tool uses, {task.usage.duration_ms}ms"
                )

            return ToolResult(
                success=True,
                output="\n".join(output_lines),
                metadata={"task": task.to_dict()},
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class TaskUpdateTool(TaskManagerTool):
    """实现任务更新工具。模型通过统一工具 Schema 调用它，执行结果使用 ToolResult 返回，并服从运行时安全策略。"""

    name = "task_update"
    description = "Update a task in the task list. Mark tasks as resolved when you complete work on them. Only mark a task as completed when you have FULLY accomplished it."

    def execute(
        self,
        task_id: str,
        status: str | None = None,
        subject: str | None = None,
        description: str | None = None,
        active_form: str | None = None,
        add_blocks: list[str] | None = None,
        add_blocked_by: list[str] | None = None,
        owner: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """执行任务更新工具对应的实际操作，校验输入并返回统一结果。

        参数：
            task_id: 目标任务的稳定标识。
            status: 可选的状态。
            subject: 可选的`subject`。
            description: 可选的说明。
            active_form: 可选的`active_form`。
            add_blocks: 可选的`add_blocks`。
            add_blocked_by: 可选的`add_blocked_by`。
            owner: 可选的`owner`。
            metadata: 随主体数据传递的扩展元数据。
            **kwargs: 传递给底层实现的额外关键字参数。

        返回：
            `ToolResult` 类型的处理结果。
        """
        try:
            manager = self.task_manager
            task = manager.get_task(task_id)

            if not task:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Task '{task_id}' not found",
                )

            if status:
                try:
                    task_status = TaskStatus(status)
                    manager.update_task_status(task_id, task_status)
                except ValueError:
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"Invalid status: {status}. Must be one of: pending, running, completed, failed, killed",
                    )

            dependency_targets = []
            for dependent_task_id in add_blocks or []:
                success, error = manager.add_dependency(task_id, dependent_task_id)
                if not success:
                    return ToolResult(success=False, output="", error=error)
                dependency_targets.append(dependent_task_id)

            for prerequisite_task_id in add_blocked_by or []:
                success, error = manager.add_dependency(prerequisite_task_id, task_id)
                if not success:
                    return ToolResult(success=False, output="", error=error)
                dependency_targets.append(prerequisite_task_id)

            if subject:
                task.metadata["subject"] = subject
            if active_form:
                task.metadata["active_form"] = active_form
            if owner:
                task.metadata["owner"] = owner

            if description:
                subject_part = task.metadata.get("subject", "")
                task.description = f"{subject_part}: {description}"
                task.metadata["description"] = description

            if metadata:
                for key, value in metadata.items():
                    if value is None:
                        task.metadata.pop(key, None)
                    else:
                        task.metadata[key] = value

            return ToolResult(
                success=True,
                output=f"Updated task {task_id}",
                metadata={
                    "task": task.to_dict(),
                    "updated_dependencies": dependency_targets,
                },
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class TaskStopTool(TaskManagerTool):
    """实现任务停止工具。模型通过统一工具 Schema 调用它，执行结果使用 ToolResult 返回，并服从运行时安全策略。"""

    name = "task_stop"
    description = "Stop a running background task. Use this to terminate a task that is in the wrong direction or no longer needed. Pass the task_id from the tool's launch result."

    def execute(self, task_id: str, **kwargs: Any) -> ToolResult:
        """执行任务停止工具对应的实际操作，校验输入并返回统一结果。

        参数：
            task_id: 目标任务的稳定标识。
            **kwargs: 传递给底层实现的额外关键字参数。

        返回：
            `ToolResult` 类型的处理结果。
        """
        try:
            manager = self.task_manager
            stopped = asyncio.run(manager.stop_task(task_id))

            if stopped:
                return ToolResult(
                    success=True,
                    output=f"Stopped task {task_id}",
                )
            else:
                task = manager.get_task(task_id)
                if not task:
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"Task '{task_id}' not found",
                    )
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Task '{task_id}' is not running (status: {task.status.value})",
                )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class TaskOutputTool(TaskManagerTool):
    """实现任务输出工具。模型通过统一工具 Schema 调用它，执行结果使用 ToolResult 返回，并服从运行时安全策略。"""

    name = "task_output"
    description = "Get output from a running or completed background task. Use this to retrieve the full output from tasks that were started in the background."

    def execute(self, task_id: str, max_length: int = 10000, **kwargs: Any) -> ToolResult:
        """执行任务输出工具对应的实际操作，校验输入并返回统一结果。

        参数：
            task_id: 目标任务的稳定标识。
            max_length: 允许返回的最大文本长度。
            **kwargs: 传递给底层实现的额外关键字参数。

        返回：
            `ToolResult` 类型的处理结果。
        """
        try:
            manager = self.task_manager
            task = manager.get_task(task_id)

            if not task:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Task '{task_id}' not found",
                )

            content, offset = manager.read_task_output(
                task_id, max_length=max_length, offset=task.output_offset
            )

            if not content:
                return ToolResult(
                    success=True,
                    output=f"Task {task_id} has no output yet.",
                )

            return ToolResult(
                success=True,
                output=content,
                metadata={
                    "task_id": task_id,
                    "offset": offset,
                    "has_more": offset < task.output_offset + max_length,
                },
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
