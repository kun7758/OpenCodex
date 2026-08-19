"""任务规划子系统中的`models`模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from dataclasses import dataclass, field
from typing import Any

from opennova.runtime.state import Plan, PlanStatus, PlanStep, StepStatus

__all__ = [
    "PlanStep",
    "Plan",
    "PlanStatus",
    "StepStatus",
    "PlanResult",
    "PlanTemplate",
]


@dataclass
class PlanResult:
    """保存计划结果所需的结构化数据，主要包含
    `plan`、`success`、`message`、`completed_steps`、`failed_steps`、`duration_seconds`、`outputs`
    字段，便于在组件之间传递或持久化。
    """

    plan: Plan
    success: bool
    message: str
    completed_steps: int = 0
    failed_steps: int = 0
    duration_seconds: float = 0.0
    outputs: dict[str, Any] = field(default_factory=dict)

    @property
    def total_steps(self) -> int:
        return len(self.plan.steps)

    @property
    def success_rate(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.completed_steps / self.total_steps


@dataclass
class PlanTemplate:
    """保存计划模板所需的结构化数据，主要包含 `name`、`description`、`steps`、`applicable_keywords` 字段，便于在组件之间传递或持久化。"""

    name: str
    description: str
    steps: list[str]
    applicable_keywords: list[str] = field(default_factory=list)

    def matches(self, task: str) -> bool:
        """处理匹配，并按照当前组件的约定返回结果。

        参数：
            task: 用户希望 Agent 完成的任务描述。

        返回：
            表示条件是否成立。
        """
        task_lower = task.lower()
        return any(kw.lower() in task_lower for kw in self.applicable_keywords)

    def create_plan(self, task: str) -> Plan:
        """创建计划并完成必要的初始化。

        参数：
            task: 用户希望 Agent 完成的任务描述。

        返回：
            `Plan` 类型的处理结果。
        """
        from opennova.runtime.state import PlanStep

        steps = [
            PlanStep(
                id=f"step_{i + 1}",
                description=step,
            )
            for i, step in enumerate(self.steps)
        ]

        return Plan(task=task, steps=steps)


COMMON_TEMPLATES: list[PlanTemplate] = [
    PlanTemplate(
        name="add_feature",
        description="向代码库添加新功能",
        steps=[
            "理解现有代码库结构",
            "确定新功能应实现的位置",
            "设计功能实现方案",
            "编写功能代码",
            "为新功能添加测试",
            "更新文档",
        ],
        applicable_keywords=["add", "implement", "create", "new feature"],
    ),
    PlanTemplate(
        name="fix_bug",
        description="修复代码库中的 Bug",
        steps=[
            "复现并理解 Bug",
            "定位有缺陷的代码",
            "分析根本原因",
            "实施修复",
            "编写测试防止回归",
            "验证修复是否生效",
        ],
        applicable_keywords=["fix", "bug", "error", "issue", "broken"],
    ),
    PlanTemplate(
        name="refactor",
        description="重构现有代码",
        steps=[
            "分析当前代码结构",
            "识别可重构的机会",
            "规划重构方案",
            "实施重构改动",
            "运行测试确保功能正常",
            "按需更新文档",
        ],
        applicable_keywords=["refactor", "clean up", "improve", "restructure"],
    ),
    PlanTemplate(
        name="write_tests",
        description="为代码编写测试",
        steps=[
            "确定需要测试的代码范围",
            "理解代码行为",
            "设计测试用例",
            "编写单元测试",
            "运行测试并验证覆盖率",
        ],
        applicable_keywords=["test", "tests", "testing", "unit test"],
    ),
]
