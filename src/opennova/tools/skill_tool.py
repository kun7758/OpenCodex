"""内置工具系统中的Skill工具模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from typing import Any

from opennova.tools.base import BaseTool, ToolResult


class SkillTool(BaseTool):
    """实现Skill工具。模型通过统一工具 Schema 调用它，执行结果使用 ToolResult 返回，并服从运行时安全策略。"""

    name = "skill"
    description = (
        "Execute a skill within the main conversation. "
        "Available skills are listed in the system prompt with their descriptions. "
        "Skills provide specialized capabilities and domain knowledge.\n"
        "How to invoke:\n"
        "- Set 'skill' to the exact skill name from the available skills list\n"
        "- Set 'args' to optional arguments for the skill\n"
        "BLOCKING REQUIREMENT: When a skill matches the user's request, "
        "invoke this tool BEFORE generating any other response. "
        "Do not describe a skill in prose — call this tool with the skill name."
    )

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.runtime = self.config.get("runtime")

    def _get_available_skill_names(self) -> list[str]:
        if self.runtime is None:
            return []
        skill_registry = getattr(self.runtime, "skill_registry", None)
        if skill_registry is None:
            return []
        return skill_registry.list_model_invocable_skills()

    def get_parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "skill": {
                    "type": "string",
                    "description": (
                        "The skill name from the available skills list in the system prompt. "
                        "E.g., 'code_review', 'analyze_project', or 'git_helper'."
                    ),
                },
                "args": {
                    "type": "string",
                    "description": "Optional arguments passed to the skill.",
                },
            },
            "required": ["skill"],
        }

    def execute(self, skill: str, args: str = "") -> ToolResult:
        if self.runtime is None:
            return ToolResult(success=False, output="", error="Skill tool is missing runtime context")
        return self.runtime.invoke_skill(skill_name=skill, skill_args=args, caller="model")
