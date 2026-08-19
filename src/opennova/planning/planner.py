"""任务规划子系统中的规划器模块，集中定义相关数据结构、边界适配和实现逻辑。"""

import json

from opennova.planning.models import COMMON_TEMPLATES, PlanTemplate
from opennova.providers.base import BaseLLMProvider, Message
from opennova.runtime.state import Plan, PlanStep

TEMPLATE_PREFERRED_KEYWORDS = {
    "unit test",
    "unit tests",
    "write tests",
    "add tests",
    "testing",
    "generate docs",
    "documentation",
    "docstring",
    "docstrings",
}


PLANNING_PROMPT = """你是一个任务规划助手，负责将任务拆分为清晰、可执行的步骤。

请针对以下任务，生成结构化计划，格式如下：
```json
{{
    "task_summary": "任务简要描述",
    "steps": [
        {{
            "id": "step_1",
            "description": "该步骤要做什么的清晰描述",
            "tool_hint": "建议使用的工具名称（可选）"
        }}
    ]
}}
```

要求：
1. 每个步骤应为单一、聚焦的动作
2. 步骤应按逻辑执行顺序排列
3. 当明确需要某个工具时填写 tool_hint
4. 描述应简洁且可执行
5. 通常 3-7 个步骤为宜

任务：{task}

请为此任务生成计划，仅输出 JSON 对象，不要包含其他内容。
"""


class Planner:
    """把用户任务转换为可审批的结构化计划。优先使用模型生成贴合仓库的步骤，失败时使用保守的模板计划，并提供步骤选择和复杂度估算。"""

    def __init__(
        self,
        llm: BaseLLMProvider,
        use_templates: bool = True,
    ):
        """初始化规划器，保存后续操作需要的依赖、配置和初始状态。

        参数：
            llm: 本次操作使用的`llm`。
            use_templates: 可选的`use_templates`。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self.llm = llm
        self.use_templates = use_templates
        self.templates = COMMON_TEMPLATES.copy()

    def add_template(self, template: PlanTemplate) -> None:
        """添加`add_template`，必要时执行去重或容量检查。

        参数：
            template: 本次操作使用的模板。
        """
        self.templates.append(template)

    async def create_plan(self, task: str) -> Plan:
        """创建计划并完成必要的初始化。

        参数：
            task: 用户希望 Agent 完成的任务描述。

        返回：
            `Plan` 类型的处理结果。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        preferred_template = self._select_template(task)
        if preferred_template and self._should_prefer_template(task):
            return preferred_template.create_plan(task)

        llm_plan = await self._create_llm_plan(task)
        if self._is_fallback_plan(llm_plan, task) and preferred_template:
            return preferred_template.create_plan(task)

        return llm_plan

    def _select_template(self, task: str) -> PlanTemplate | None:
        """选择模板，并按照当前组件的约定返回结果。

        参数：
            task: 用户希望 Agent 完成的任务描述。

        返回：
            `PlanTemplate | None` 类型的处理结果。
        """
        if not self.use_templates:
            return None

        for template in self.templates:
            if template.matches(task):
                return template

        return None

    def _should_prefer_template(self, task: str) -> bool:
        """根据当前输入和规划器的状态计算 `_should_prefer_template`，并返回调用方需要的结果。

        参数：
            task: 用户希望 Agent 完成的任务描述。

        返回：
            表示条件是否成立。
        """
        task_lower = task.lower()
        return any(keyword in task_lower for keyword in TEMPLATE_PREFERRED_KEYWORDS)

    def _is_fallback_plan(self, plan: Plan, task: str) -> bool:
        """根据当前输入和规划器的状态计算 `_is_fallback_plan`，并返回调用方需要的结果。

        参数：
            plan: 当前要保存、展示或执行的结构化计划。
            task: 用户希望 Agent 完成的任务描述。

        返回：
            表示条件是否成立。
        """
        return (
            plan.task == task
            and len(plan.steps) == 1
            and plan.steps[0].id == "step_1"
            and plan.steps[0].description == task
        )

    async def _create_llm_plan(self, task: str) -> Plan:
        """创建`create_llm_plan`并完成必要的初始化。

        参数：
            task: 用户希望 Agent 完成的任务描述。

        返回：
            `Plan` 类型的处理结果。

        说明：
            该操作可能等待模型服务、MCP 服务或其他异步数据源返回。
        """
        prompt = PLANNING_PROMPT.format(task=task)

        messages = [
            Message(
                role="system",
                content="你是一个专业的任务规划助手，请始终以合法 JSON 格式回复。",
            ),
            Message(role="user", content=prompt),
        ]

        try:
            response = await self.llm.chat(messages, temperature=0.7)
            plan = self._parse_plan_response(response.content, task)
            return plan
        except Exception:
            return self._create_fallback_plan(task)

    def _parse_plan_response(self, response: str, task: str) -> Plan:
        """解析`parse_plan_response`并转换为内部使用的规范结构。

        参数：
            response: 本次操作使用的`response`。
            task: 用户希望 Agent 完成的任务描述。

        返回：
            `Plan` 类型的处理结果。
        """
        json_str = response

        import re

        json_patterns = [
            r"```json\s*(.*?)\s*```",
            r"```\s*(.*?)\s*```",
            r"(\{.*\})",
        ]

        for pattern in json_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                json_str = match.group(1)
                break

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return self._create_fallback_plan(task)

        steps = []
        for i, step_data in enumerate(data.get("steps", [])):
            step = PlanStep(
                id=step_data.get("id", f"step_{i + 1}"),
                description=step_data.get("description", ""),
                tool_hint=step_data.get("tool_hint"),
            )
            steps.append(step)

        if not steps:
            return self._create_fallback_plan(task)

        task_summary = data.get("task_summary", task)

        return Plan(
            task=task_summary,
            steps=steps,
        ).reindex_steps()

    def _create_fallback_plan(self, task: str) -> Plan:
        """创建回退计划并完成必要的初始化。

        参数：
            task: 用户希望 Agent 完成的任务描述。

        返回：
            `Plan` 类型的处理结果。
        """
        return Plan(
            task=task,
            steps=[
                PlanStep(
                    id="step_1",
                    description=task,
                )
            ],
        ).reindex_steps()

    def optimize_plan(self, plan: Plan) -> Plan:
        """对计划进行步骤合并优化：当计划步骤超过 3 步时，将连续使用同一 tool_hint 的步骤
        合并为一个步骤（描述用"；然后"连接），减少不必要的模型调用轮次。

        合并规则举例：
            步骤 1（tool_hint=read_file）+ 步骤 2（tool_hint=read_file）
            → 合并为一个步骤，description="步骤1描述；然后步骤2描述"

        合并完成后重新编号（reindex_steps），确保步骤 ID 连续。

        参数：
            plan: 当前要保存、展示或执行的结构化计划。

        返回：
            优化后的 `Plan` 对象，步骤数可能少于原始计划。
        """
        if len(plan.steps) <= 3:
            return plan.reindex_steps()

        optimized_steps = []
        merged = set()

        for i, step in enumerate(plan.steps):
            if i in merged:
                continue

            current_step = step

            for j in range(i + 1, len(plan.steps)):
                if j in merged:
                    continue

                next_step = plan.steps[j]

                if (
                    current_step.tool_hint
                    and next_step.tool_hint
                    and current_step.tool_hint == next_step.tool_hint
                ):
                    current_step = PlanStep(
                        id=current_step.id,
                        description=f"{current_step.description}；然后{next_step.description}",
                        tool_hint=current_step.tool_hint,
                    )
                    merged.add(j)

            optimized_steps.append(current_step)

        return Plan(task=plan.task, steps=optimized_steps).reindex_steps()

    def get_next_step(self, plan: Plan) -> PlanStep | None:
        """读取下一个步骤，不改变当前对象的业务状态。

        参数：
            plan: 当前要保存、展示或执行的结构化计划。

        返回：
            `PlanStep | None` 类型的处理结果。
        """
        return plan.get_next_step()

    def estimate_plan_complexity(self, plan: Plan) -> str:
        """估算 `plan_complexity` 对应的数据，并按照当前组件的约定返回结果。

        参数：
            plan: 当前要保存、展示或执行的结构化计划。

        返回：
            处理后的文本或稳定标识。
        """
        num_steps = len(plan.steps)

        if num_steps <= 3:
            return "simple"
        elif num_steps <= 6:
            return "medium"
        else:
            return "complex"

    def get_plan_summary(self, plan: Plan) -> str:
        """读取计划摘要，不改变当前对象的业务状态。

        参数：
            plan: 当前要保存、展示或执行的结构化计划。

        返回：
            处理后的文本或稳定标识。
        """
        lines = [f"📋 计划: {plan.task}", ""]

        status_icons = {
            "pending": "⏳",
            "running": "🔄",
            "done": "✅",
            "failed": "❌",
            "skipped": "⏭️",
        }

        for step in plan.steps:
            icon = status_icons.get(step.status.value, "❓")
            lines.append(f"  {icon} {step.id}: {step.description}")
            if step.tool_hint:
                lines.append(f"     工具: {step.tool_hint}")

        completed = sum(1 for s in plan.steps if s.status.value == "done")
        total = len(plan.steps)

        lines.append("")
        lines.append(f"进度: {completed}/{total} 步已完成")

        return "\n".join(lines)
