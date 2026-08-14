"""Agent 核心运行时中的工作流模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from opennova.providers.base import BaseLLMProvider, Message, ToolSchema


class WorkflowDecision(StrEnum):
    """枚举工作流决策允许出现的稳定取值，序列化和状态判断均使用这些值。"""

    PLAN = "plan"
    ACT = "act"


@dataclass(frozen=True)
class WorkflowRoutingResult:
    """保存工作流路由结果所需的结构化数据，主要包含 `decision`、`reason`、`confidence`、`error` 字段，便于在组件之间传递或持久化。"""

    decision: WorkflowDecision | None
    reason: str = ""
    confidence: float = 0.0
    error: str | None = None

    @property
    def resolved(self) -> bool:
        return self.decision is not None and self.error is None


class WorkflowRouter:
    """封装`WorkflowRouter`相关的状态和操作，使调用方通过稳定接口使用该能力。"""

    TOOL_NAME = "select_execution_mode"
    SYSTEM_MESSAGE_NAME = "opennova_workflow_router"
    ROUTER_PROMPT = """You are OpenNova's execution workflow controller.

Choose the workflow that best matches the user's semantic intent and the conversation context.

- Choose `plan` when the user wants an implementation plan, design review, or explicit approval
  before project changes begin.
- Choose `act` when the user wants direct execution, asks a question, requests analysis only, or
  does not require approval before changes.
- Interpret the complete request and prior conversation. Do not rely on isolated words.
- This decision controls execution safety. Do not perform the task and do not answer in prose.

You must call `select_execution_mode` exactly once."""

    TOOL_SCHEMA = ToolSchema(
        name=TOOL_NAME,
        description="Select whether OpenNova should plan for approval or act directly.",
        parameters={
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": [WorkflowDecision.PLAN.value, WorkflowDecision.ACT.value],
                    "description": "The execution workflow for the current user turn.",
                },
                "reason": {
                    "type": "string",
                    "description": "A concise semantic reason for the selected workflow.",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Confidence in the workflow decision from 0 to 1.",
                },
            },
            "required": ["mode", "reason", "confidence"],
            "additionalProperties": False,
        },
    )

    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm

    @staticmethod
    def route_local(task: str) -> WorkflowRoutingResult:
        """根据当前输入和`WorkflowRouter`的状态计算 `route_local`，并返回调用方需要的结果。

        参数：
            task: 用户希望 Agent 完成的任务描述。

        返回：
            `WorkflowRoutingResult` 类型的处理结果。
        """
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
        """处理路由，并按照当前组件的约定返回结果。

        参数：
            messages: 按协议顺序排列的对话消息。
            task: 用户希望 Agent 完成的任务描述。
            prefer_local: 可选的`prefer_local`。

        返回：
            `WorkflowRoutingResult` 类型的处理结果。

        说明：
            该操作可能等待模型服务、MCP 服务或其他异步数据源返回。
        """
        if prefer_local:
            local = self.route_local(task)
            if local.resolved:
                return local
        routing_messages = [
            Message(
                role="system",
                content=self.ROUTER_PROMPT,
                name=self.SYSTEM_MESSAGE_NAME,
            ),
            *messages,
        ]
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

        tool_calls = response.tool_calls or []
        if len(tool_calls) != 1 or tool_calls[0].name != self.TOOL_NAME:
            return WorkflowRoutingResult(
                decision=None,
                error="Workflow routing did not return the required control tool call.",
            )

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
