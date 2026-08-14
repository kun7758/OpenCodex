"""终端交互层中的记忆命令模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from __future__ import annotations

from pathlib import Path

from opennova.memory.layered import LayeredMemoryManager
from opennova.tools.base import ToolResult


def handle_memory_command(project_path: str | Path, args: str) -> ToolResult:
    """处理记忆命令，协调输入校验、状态变化和结果返回。

    参数：
        project_path: 本次操作使用的项目路径。
        args: 调用方传入的位置参数或 Skill 参数文本。

    返回：
        `ToolResult` 类型的处理结果。
    """
    manager = LayeredMemoryManager(project_path)
    command, _, remainder = args.strip().partition(" ")
    command = command or "list"
    try:
        if command == "list":
            records = manager.list_records()
            if not records:
                return ToolResult(True, "No layered memories found.")
            lines = []
            for record in records:
                status = "expired" if record.expired else "active"
                lines.append(
                    f"- {record.name}: {status}, scope={record.scope}, "
                    f"provenance={record.provenance}"
                )
            return ToolResult(True, "Layered memories:\n" + "\n".join(lines))
        if command == "add":
            name, separator, content = remainder.strip().partition(" ")
            if not separator or not content.strip():
                return ToolResult(False, "", "Usage: /memory add <name> <content>")
            record = manager.add(name, content)
            return ToolResult(True, f"Created memory: {record.name}")
        if command == "delete":
            name = remainder.strip()
            if not name:
                return ToolResult(False, "", "Usage: /memory delete <name>")
            deleted = manager.delete(name)
            if not deleted:
                return ToolResult(False, "", f"Memory not found: {name}")
            return ToolResult(True, f"Deleted memory: {name}")
        return ToolResult(False, "", "Usage: /memory [list|add <name> <content>|delete <name>]")
    except (OSError, ValueError) as exc:
        return ToolResult(False, "", str(exc))
