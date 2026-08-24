"""任务规划子系统中的规划器模块，集中定义相关数据结构、边界适配和实现逻辑。"""

import json
import logging

from opennova.planning.models import COMMON_TEMPLATES, PlanTemplate
from opennova.providers.base import BaseLLMProvider, LLMResponse, Message
from opennova.runtime.state import Plan, PlanStep
_LOGGER = logging.getLogger(__name__)

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
        """根据任务描述生成结构化计划：优先使用模板，其次使用 LLM，必要时回退到模板。

        本函数是计划生成的核心入口，实现了一个智能的三层计划生成策略：
            1. 模板优先：对于特定类型的任务（如单元测试、文档生成），直接使用预定义模板
            2. LLM 生成：对于大多数任务，调用 LLM 生成定制化的计划步骤
            3. 模板回退：如果 LLM 生成的计划质量不佳，回退到预定义模板

        参数：
            task: 用户希望 Agent 完成的任务描述，例如 "添加用户认证功能" 或 "修复登录 Bug"。

        返回：
            生成的 Plan 对象，包含：
            - task: 任务描述
            - steps: 计划步骤列表（每个步骤包含 id、description、tool_hint 等）
            - created_at: 创建时间

        异常：
            Exception: 如果 LLM 调用失败且没有匹配的模板，会返回一个单步骤的回退计划

        说明：
            - 这是异步操作，调用方应使用 `await`，并允许取消信号向下传播
            - 计划生成策略的选择取决于任务描述中的关键词匹配
            - 模板匹配是通过 _select_template() 函数实现的
            - LLM 生成是通过 _create_llm_plan() 函数实现的
            - 回退计划判断是通过 _is_fallback_plan() 函数实现的

        生成策略详解：
            1. 模板优先策略（适用于特定类型任务）：
               - 条件：找到匹配的模板 且 任务包含 TEMPLATE_PREFERRED_KEYWORDS 中的关键词
               - 关键词包括：unit test、write tests、testing、documentation、docstring 等
               - 优点：生成速度快，步骤结构稳定，避免调用 LLM 的开销
               - 示例：任务 "write unit tests for auth module" 会直接使用 write_tests 模板

            2. LLM 生成策略（适用于大多数任务）：
               - 条件：没有找到匹配的模板，或任务不包含特定关键词
               - 过程：发送 PLANNING_PROMPT 给 LLM，期望返回 JSON 格式的步骤列表
               - 优点：生成的计划更贴合具体任务，步骤更定制化
               - 示例：任务 "实现用户认证功能" 会调用 LLM 生成定制化计划

            3. 模板回退策略（适用于 LLM 生成质量不佳的情况）：
               - 条件：LLM 生成的计划是回退计划（只有一个步骤，且步骤描述与任务相同）
               - 过程：使用匹配的模板创建计划
               - 优点：避免返回无意义的单步骤计划
               - 示例：如果 LLM 对 "fix login bug" 只返回一个步骤，会回退到 fix_bug 模板

        调用示例：
            planner = Planner(llm_provider)
            plan = await planner.create_plan("添加用户认证功能")
            # 返回包含多个步骤的 Plan 对象

        与其他函数的关系：
            create_plan() → _select_template() → _should_prefer_template() → _create_llm_plan() → _is_fallback_plan()
        """
        preferred_template = self._select_template(task)
        if preferred_template and self._should_prefer_template(task):
            return preferred_template.create_plan(task)

        llm_plan = await self._create_llm_plan(task)
        _LOGGER.info("llm_plan JSON:\n" + json.dumps(llm_plan.to_dict(), indent=2, ensure_ascii=False))
        if self._is_fallback_plan(llm_plan, task) and preferred_template:
            return preferred_template.create_plan(task)

        return llm_plan

    def _select_template(self, task: str) -> PlanTemplate | None:
        """根据任务描述选择匹配的计划模板。

        本函数遍历所有可用的计划模板（包括预定义的 COMMON_TEMPLATES 和用户添加的模板），
        检查任务描述中是否包含模板的关键词，返回第一个匹配的模板。

        参数：
            task: 用户希望 Agent 完成的任务描述，例如 "添加用户认证功能" 或 "修复登录 Bug"。

        返回：
            - 如果找到匹配的 PlanTemplate，返回该模板对象
            - 如果没有找到匹配的模板，返回 None
            - 如果 self.use_templates 为 False（模板功能被禁用），返回 None

        说明：
            - 模板匹配是通过检查任务描述中是否包含模板的 applicable_keywords 实现的
            - 匹配是大小写不敏感的（会将任务和关键词都转换为小写进行比较）
            - 返回第一个匹配的模板，不保证是"最佳"匹配
            - 常见的模板包括：add_feature、fix_bug、refactor、write_tests 等

        使用场景：
            在 create_plan() 函数中，首先尝试选择模板：
            1. 如果找到匹配的模板且任务包含特定关键词（如 "unit test"、"documentation"），
               直接使用模板创建计划（避免调用 LLM）
            2. 如果找到匹配的模板但任务不包含特定关键词，先尝试使用 LLM 生成计划，
               如果 LLM 生成的计划质量不佳（回退计划），则使用模板创建计划
            3. 如果没有找到匹配的模板，只能使用 LLM 生成计划

        示例：
            # 假设任务是 "fix the login bug"
            template = self._select_template("fix the login bug")
            # 返回 fix_bug 模板，因为任务包含 "fix" 和 "bug" 关键词
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
        """判断给定的计划是否是回退计划（即质量不佳的单步骤计划）。

        回退计划是在 LLM 生成失败时创建的兜底计划，它只有一个步骤，且步骤描述与任务相同，
        没有提供任何有价值的拆分。当检测到回退计划时，create_plan() 会尝试使用模板替代。

        参数：
            plan: 当前要保存、展示或执行的结构化计划，通常由 _create_llm_plan() 生成。
            task: 用户希望 Agent 完成的任务描述，用于与计划进行比较。

        返回：
            - True：如果计划是回退计划（满足以下所有条件）：
                1. plan.task == task：计划的任务描述与输入任务相同
                2. len(plan.steps) == 1：计划只有一个步骤
                3. plan.steps[0].id == "step_1"：第一个步骤的 id 是 "step_1"
                4. plan.steps[0].description == task：第一个步骤的描述与输入任务相同
            - False：如果计划不是回退计划（即计划有多个步骤，或步骤描述与任务不同）

        说明：
            - 回退计划通常是在以下情况下生成的：
                1. LLM 调用失败（网络异常、超时等）
                2. LLM 返回的内容无法解析为有效的 JSON
                3. LLM 返回的 JSON 格式不符合预期（缺少 steps 字段等）
            - 回退计划的质量不佳，因为它没有提供任何有价值的拆分，只是将任务作为单个步骤
            - 在 create_plan() 中，如果检测到回退计划且有匹配的模板，会使用模板替代

        使用场景：
            在 create_plan() 函数中，用于判断 LLM 生成的计划是否需要被模板替代：
            llm_plan = await self._create_llm_plan(task)
            if self._is_fallback_plan(llm_plan, task) and preferred_template:
                return preferred_template.create_plan(task)

        示例：
            # 回退计划示例
            fallback_plan = Plan(task="fix login bug", steps=[PlanStep(id="step_1", description="fix login bug")])
            self._is_fallback_plan(fallback_plan, "fix login bug")  # 返回 True

            # 正常计划示例
            normal_plan = Plan(task="fix login bug", steps=[
                PlanStep(id="step_1", description="复现并理解 Bug"),
                PlanStep(id="step_2", description="定位有缺陷的代码"),
                PlanStep(id="step_3", description="实施修复"),
            ])
            self._is_fallback_plan(normal_plan, "fix login bug")  # 返回 False
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
            _LOGGER.info("planning_prompt: %s", prompt)
            response:LLMResponse = await self.llm.chat(messages, temperature=0.7)
            _LOGGER.info("LLMResponse content: %s", response.content)
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
