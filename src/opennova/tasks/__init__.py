"""任务管理子系统的公共导出入口，集中暴露上层调用方需要使用的类型和函数。"""

from opennova.tasks.task import (
    Task,
    TaskHandle,
    TaskManager,
    TaskProgressData,
    TaskResult,
    TaskStatus,
    TaskType,
    generate_task_id,
    is_terminal_status,
)

__all__ = [
    "Task",
    "TaskHandle",
    "TaskManager",
    "TaskProgressData",
    "TaskResult",
    "TaskStatus",
    "TaskType",
    "generate_task_id",
    "is_terminal_status",
]
