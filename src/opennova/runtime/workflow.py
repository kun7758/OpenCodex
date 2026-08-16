"""在任务进入 ReAct 循环前，根据用户意图选择 Plan 或 Act 工作流。"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from opennova.providers.base import BaseLLMProvider, Message, ToolSchema


class WorkflowDecision(StrEnum):
    """表示一轮用户任务应采用的执行方式。

    `PLAN` 先生成可审批的计划，`ACT` 则允许 Agent 直接回答或执行任务。
    枚举值 `plan` 和 `act` 同时是模型工具调用协议中的稳定值，不应随意更改。
    """

    PLAN = "plan"
    ACT = "act"


@dataclass(frozen=True)
class WorkflowRoutingResult:
    """保存一次工作流路由的结构化结果。

    属性：
        decision: 选中的 Plan 或 Act 工作流；路由失败或无法确定时为 `None`。
        reason: 选择该工作流的简要语义理由，可用于诊断和后续提示。
        confidence: 对路由决策的置信度，取值范围为 0 到 1。
        error: 路由失败的原因；成功得到决策时为 `None`。
    """

    decision: WorkflowDecision | None
    reason: str = ""
    confidence: float = 0.0
    error: str | None = None

    @property
    def resolved(self) -> bool:
        """判断路由是否成功得到可使用的工作流决策。

        返回：
            当 `decision` 已确定且没有记录错误时返回 `True`，否则返回 `False`。
        """
        return self.decision is not None and self.error is None


class WorkflowRouter:
    """根据用户请求和对话上下文，在 Plan 与 Act 工作流之间进行路由。

    路由时可先识别高置信度的显式本地指令；本地规则无法决定时，再要求
    LLM 通过 `select_execution_mode` 工具返回结构化决策。该类只选择工作流，
    不会执行用户任务。
    """

    TOOL_NAME = "select_execution_mode"
    SYSTEM_MESSAGE_NAME = "opennova_workflow_router"
    ROUTER_PROMPT = """你是 OpenNova 的执行工作流控制器。

请选择最符合用户语义意图和对话上下文的工作流。

- 当用户希望先获得实施计划、进行设计审查，或要求在修改项目前明确审批时，选择 `plan`。
- 当用户希望直接执行、提出问题、仅要求分析，或未要求在修改前进行审批时，选择 `act`。
- 应结合完整请求和之前的对话进行理解，不要仅依赖孤立的词语。
- 该决策会影响执行安全。不要执行用户任务，也不要使用普通文本回答。

你必须且只能调用一次 `select_execution_mode`。"""

    TOOL_SCHEMA = ToolSchema(
        name=TOOL_NAME,
        description="选择 OpenNova 应先生成等待审批的计划，还是直接执行当前任务。",
        parameters={
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": [WorkflowDecision.PLAN.value, WorkflowDecision.ACT.value],
                    "description": "当前用户回合应采用的执行工作流。",
                },
                "reason": {
                    "type": "string",
                    "description": "选择该工作流的简明语义理由。",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "对本次工作流决策的置信度，取值范围为 0 到 1。",
                },
            },
            "required": ["mode", "reason", "confidence"],
            "additionalProperties": False,
        },
    )

    def __init__(self, llm: BaseLLMProvider):
        """创建工作流路由器，并保存用于语义路由的模型 Provider。

        参数：
            llm: 用于理解完整对话并返回结构化 Plan/Act 决策的大模型 Provider。
        """
        self.llm = llm

    @staticmethod
    def route_local(task: str) -> WorkflowRoutingResult:
        """使用高置信度的显式短语，快速判断用户要求 Plan 还是 Act。

        该方法只检查当前任务文本，不读取历史对话，也不调用大模型。匹配
        到“先做计划”或“直接执行”类明确指令时，返回置信度为 1 的决策；
        没有匹配时返回未解析结果，由调用方决定是否继续进行模型路由。

        参数：
            task: 当前用户提交的原始任务文本，用于匹配明确的 Plan/Act 指令。

        返回：
            包含工作流决策、理由和置信度的路由结果；无法明确匹配时，
            `decision` 为 `None`，并在 `error` 中说明本地路由未命中。
        """
        # 统一大小写和连续空白，避免表达形式差异影响短语匹配。
        normalized = " ".join(task.strip().lower().split())
        plan_markers = (
            "只列计划",
            "只写计划",
            "先列计划",
            "先做计划",
            "等我确认后",
            "确认后再",
            "plan only",
            "do not implement",
            "wait for my approval",
        )
        act_markers = (
            "直接实现",
            "直接修改",
            "直接执行",
            "不用计划",
            "无需计划",
            "开始开发",
            "implement directly",
            "skip the plan",
        )
        if any(marker in normalized for marker in plan_markers):
            return WorkflowRoutingResult(
                WorkflowDecision.PLAN,
                "Explicit request to prepare a plan before implementation.",
                1.0,
            )
        if any(marker in normalized for marker in act_markers):
            return WorkflowRoutingResult(
                WorkflowDecision.ACT,
                "Explicit request to execute without a planning approval step.",
                1.0,
            )
        return WorkflowRoutingResult(None, error="No high-confidence local workflow match.")

    async def route(
        self,
        messages: Sequence[Message],
        task: str,
        *,
        prefer_local: bool = False,
    ) -> WorkflowRoutingResult:
        """结合当前任务和对话上下文，选择 Plan 或 Act 工作流。

        启用本地优先时，会先用 `route_local()` 识别明确指令；本地规则无法
        决定时，将路由系统提示词和历史消息发送给模型，并强制模型通过
        `select_execution_mode` 工具返回唯一的结构化决策。

        参数：
            messages: 按对话发生顺序排列的上下文消息。模型会结合这些消息
                理解用户在当前会话中是要求先规划还是直接执行。
            task: 当前用户提交的任务文本。它用于本地短语匹配；当 `messages`
                中没有用户消息时，还会作为补充用户消息发送给模型。
            prefer_local: 是否优先使用不调用模型的本地高置信度规则。为 `True`
                且本地规则命中时直接返回；为 `False` 或本地规则未命中时，
                使用大模型结合完整对话进行语义路由。

        返回：
            解析成功时返回包含 `decision`、`reason` 和 `confidence` 的结果；
            模型调用失败、工具调用数量或名称不符合协议、参数无效时，
            返回 `decision=None` 并在 `error` 中保存原因。

        说明：
            只负责选择工作流，不会在路由请求中执行用户任务。
        """
        if prefer_local:
            local = self.route_local(task)
            if local.resolved:
                return local

        # 路由提示词作为独立的系统消息放在历史对话之前，仅约束本次决策。
        routing_messages = [
            Message(
                role="system",
                content=self.ROUTER_PROMPT,
                name=self.SYSTEM_MESSAGE_NAME,
            ),
            *messages,
        ]
        # 部分调用方可能只传入系统或助手消息，此时需要补全用户任务。
        if not any(message.role == "user" for message in routing_messages):
            routing_messages.append(Message(role="user", content=f"Task: {task}"))

        try:
            response = await self.llm.chat(
                routing_messages,
                tools=[self.TOOL_SCHEMA],
                temperature=0,
                tool_choice="required",
            )
        except Exception as exc:
            return WorkflowRoutingResult(
                decision=None,
                error=f"Workflow routing failed: {type(exc).__name__}: {exc}",
            )

        # 路由响应必须严格包含一次指定工具调用，普通文本不视为有效决策。
        tool_calls = response.tool_calls or []
        if len(tool_calls) != 1 or tool_calls[0].name != self.TOOL_NAME:
            return WorkflowRoutingResult(
                decision=None,
                error="Workflow routing did not return the required control tool call.",
            )

        # 将模型返回的协议值转为内部枚举，并将置信度限制在 0 到 1。
        arguments = tool_calls[0].arguments or {}
        try:
            decision = WorkflowDecision(str(arguments.get("mode", "")))
            confidence = max(0.0, min(1.0, float(arguments.get("confidence", 0.0))))
        except (TypeError, ValueError) as exc:
            return WorkflowRoutingResult(
                decision=None,
                error=f"Workflow routing returned invalid arguments: {exc}",
            )

        return WorkflowRoutingResult(
            decision=decision,
            reason=str(arguments.get("reason", "")).strip(),
            confidence=confidence,
        )
