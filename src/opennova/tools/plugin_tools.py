"""内置工具系统中的插件工具模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from __future__ import annotations

import shlex
from typing import Any

from opennova.tools.base import BaseTool, ToolResult
from opennova.tools.shell_tools import ExecuteCommandTool


class PluginCommandTool(BaseTool):
    """实现插件命令工具。模型通过统一工具 Schema 调用它，执行结果使用 ToolResult 返回，并服从运行时安全策略。"""

    def __init__(
        self,
        name: str,
        description: str,
        command: str,
        args: list[str] | None = None,
        config: dict[str, Any] | None = None,
        read_only: bool = False,
        permission: str = "command",
    ):
        super().__init__(config)
        self.name = name
        self.description = description
        self.command = command
        self.args = args or []
        self.permission = permission
        self._read_only = read_only or permission == "read"
        self._command_tool = ExecuteCommandTool(self.config)

    @property
    def command_line(self) -> str:
        return shlex.join([self.command, *self.args])

    def execute(self) -> ToolResult:
        return self._decorate_result(self._command_tool.execute(self.command_line))

    async def async_execute(self) -> ToolResult:
        """执行插件命令工具对应的实际操作，校验输入并返回统一结果。

        返回：
            `ToolResult` 类型的处理结果。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        result = await self._command_tool.async_execute(self.command_line)
        return self._decorate_result(result)

    def _decorate_result(self, result: ToolResult) -> ToolResult:
        result.metadata.update(
            {
                "plugin_tool": True,
                "returncode": result.metadata.get("exit_code"),
                "permission": self.permission,
            }
        )
        return result

    def is_read_only(self, **kwargs: Any) -> bool:
        return self._read_only

    def requires_permission(self, **kwargs: Any) -> bool:
        return not self._read_only
