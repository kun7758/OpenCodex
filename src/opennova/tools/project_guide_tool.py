"""内置工具系统中的项目项目指南工具模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from __future__ import annotations

from pathlib import Path

from opennova.memory.project_guide import ProjectGuideManager
from opennova.tools.base import BaseTool, ToolResult


class InitProjectGuideTool(BaseTool):
    """实现初始化项目项目指南工具。模型通过统一工具 Schema 调用它，执行结果使用 ToolResult 返回，并服从运行时安全策略。"""

    name = "init_project_guide"
    description = (
        "Initialize an OPENNOVA.md project guide for long-term project memory using model-driven project understanding. "
        "By default it will skip if OPENNOVA.md already exists."
    )

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "force": {
                    "type": "boolean",
                    "description": "Whether to regenerate OPENNOVA.md even if it already exists.",
                    "default": False,
                }
            },
            "required": [],
        }

    def execute(self, force: bool = False) -> ToolResult:
        runtime = self.config.get("runtime")
        if runtime is not None and hasattr(runtime, "init_project_guide_async"):
            return ToolResult(
                success=False,
                output="",
                error=(
                    "init_project_guide requires async execution in this context. "
                    "Use async_execute."
                ),
            )

        working_dir = self.config.get("working_dir", ".")
        manager = ProjectGuideManager(project_path=Path(working_dir))

        try:
            result = manager.create_or_skip(force=force)
            return ToolResult(
                success=True,
                output=result.message,
                metadata={
                    "status": result.status,
                    "file_path": str(result.path),
                    "overwritten": result.overwritten,
                    "force": force,
                },
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to initialize OPENNOVA.md: {exc}",
            )

    async def async_execute(self, force: bool = False) -> ToolResult:
        runtime = self.config.get("runtime")
        if runtime is not None and hasattr(runtime, "init_project_guide_async"):
            return await runtime.init_project_guide_async(force=force)
        return self.execute(force=force)
