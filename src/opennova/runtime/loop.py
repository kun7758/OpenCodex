"""Agent 核心运行时中的循环模块，集中定义相关数据结构、边界适配和实现逻辑。"""

import asyncio
import os
import re
import traceback
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from opennova.hooks import HookManager
from opennova.memory.context import ContextCapacityError, ContextManager, MessageAddResult
from opennova.memory.working import WorkingMemory
from opennova.providers.base import (
    BaseLLMProvider,
    FinishReason,
    LLMResponse,
    Message,
    ToolCall,
    normalize_provider_error,
)
from opennova.providers.models import get_model_profile
from opennova.runtime.artifacts import ArtifactStore, ToolResultBudget
from opennova.runtime.cancellation import CancellationToken
from opennova.runtime.events import (
    ToolEvent,
    ToolEventType,
    ToolUseContext,
)
from opennova.runtime.execution import ToolExecutionEngine
from opennova.runtime.file_state import FileVersionCache
from opennova.runtime.model_policy import ProviderCircuitBreaker, RunBudget
from opennova.runtime.state import AgentState
from opennova.runtime.workflow import WorkflowDecision, WorkflowRouter, WorkflowRoutingResult
from opennova.security.guardrails import Guardrails, GuardResult, RiskLevel
from opennova.security.secrets import redact_sensitive_data
from opennova.skills.hook_adapter import register_skill_hooks
from opennova.skills.registry import SkillRegistry
from opennova.tools.base import ToolRegistry, ToolResult

PLAN_MODE_IMPLEMENTATION_TOOLS = {
    "write_file",
    "create_file",
    "edit_file",
    "multi_edit_file",
    "delete_file",
    "execute_command",
    "git_commit",
    "git_branch",
    "git_push",
    "enter_worktree",
    "exit_worktree",
    "init_project_guide",
}
BATCH_BARRIER_TOOLS = {
    "skill",
    "ask_user_question",
    "enter_plan_mode",
    "exit_plan_mode",
}
CORE_TOOL_NAMES = {
    "read_file",
    "write_file",
    "create_file",
    "edit_file",
    "multi_edit_file",
    "delete_file",
    "list_directory",
    "execute_command",
    "glob_files",
    "grep_code",
    "tool_search",
    "ask_user_question",
    "skill",
    "enter_plan_mode",
    "exit_plan_mode",
}
RUNTIME_SYSTEM_MESSAGE_NAME = "opennova_runtime"
LEGACY_RUNTIME_SYSTEM_PROMPT_PREFIX = (
    "You are an AI coding assistant that helps users with software engineering tasks."
)


@dataclass
class ParsedAction:
    """保存解析结果动作所需的结构化数据，主要包含
    `tool_name`、`arguments`、`thought`、`requires_confirmation`、`is_final`、`raw_response`、`tool_call_id`
    字段，便于在组件之间传递或持久化。
    """

    tool_name: str
    arguments: dict[str, Any]
    thought: str | None = None
    requires_confirmation: bool = False
    is_final: bool = False
    raw_response: str = ""
    tool_call_id: str | None = None


class ReActLoop:
    """实现 Agent 的 ReAct
    主循环。每轮先把上下文和可用工具交给模型，再解析模型返回的工具调用，通过统一执行管线运行工具，最后把观察结果写回上下文，直到模型给出最终答案或达到预算上限。
    """

    _SKILL_CREATOR_TRIGGER_RE = re.compile(
        r"("
        r"\b(create|write|design|build|generate|make|improve|optimi[sz]e|modify|edit)\b[\s\S]{0,80}\b(skill|skills|skill\.md)\b"
        r"|\b(skill|skills|skill\.md)\b[\s\S]{0,80}\b(create|write|design|build|generate|make|improve|optimi[sz]e|modify|edit)\b"
        r"|创建[\s\S]{0,40}(技能|skill|skills|SKILL\.md)"
        r"|写[\s\S]{0,40}(技能|skill|skills|SKILL\.md)"
        r"|设计[\s\S]{0,40}(技能|skill|skills|SKILL\.md)"
        r"|优化[\s\S]{0,40}(技能|skill|skills|SKILL\.md)"
        r"|改进[\s\S]{0,40}(技能|skill|skills|SKILL\.md)"
        r")",
        re.IGNORECASE,
    )

    def __init__(
        self,
        llm: BaseLLMProvider,
        tool_registry: ToolRegistry,
        state: AgentState,
        max_iterations: int = 500,
        stream: bool = True,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        iteration_start_callback: Callable[[list[Message]], None] | None = None,
        interaction_callback: Callable[[dict[str, Any]], Any] | None = None,
        skill_registry: SkillRegistry | None = None,
        context_manager: ContextManager | None = None,
        working_memory: WorkingMemory | None = None,
        guardrails: Guardrails | None = None,
        working_dir: str | None = None,
        hook_manager: HookManager | None = None,
        audit_logger: Any | None = None,
        cancellation_token: CancellationToken | None = None,
        file_cache: FileVersionCache | None = None,
        artifact_store: ArtifactStore | None = None,
        parallel_tool_limit: int = 4,
        per_turn_tool_result_chars: int = 160_000,
        session_id: str | None = None,
        deferred_tools_enabled: bool = True,
        token_budget: int = 0,
        cost_budget_usd: float = 0.0,
        max_output_tokens: int = 0,
        input_cost_per_million: float = 0.0,
        output_cost_per_million: float = 0.0,
        fallback_providers: list[BaseLLMProvider] | None = None,
        provider_retry_attempts: int = 1,
        provider_circuit_breaker: ProviderCircuitBreaker | None = None,
    ):
        """初始化`ReActLoop`，保存后续操作需要的依赖、配置和初始状态。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self.llm = llm
        self.tool_registry = tool_registry
        self.state = state
        self.max_iterations = max_iterations
        self.stream = stream
        self.progress_callback = progress_callback
        self.iteration_start_callback = iteration_start_callback
        self.interaction_callback = interaction_callback
        self.skill_registry = skill_registry
        self.context_manager = (
            context_manager if context_manager is not None else ContextManager(model=llm.model)
        )
        self.working_memory = working_memory
        self.guardrails = guardrails
        self.working_dir = working_dir
        self.hook_manager = hook_manager
        self.audit_logger = audit_logger
        self.cancellation_token = cancellation_token or CancellationToken()
        self.file_cache = file_cache or FileVersionCache()
        self.artifact_store = artifact_store or ArtifactStore(
            working_dir or os.getcwd(), session_id or "session"
        )
        self.execution_engine = ToolExecutionEngine(
            registry=self.tool_registry,
            cancellation_token=self.cancellation_token,
            result_budget=ToolResultBudget(
                self.artifact_store,
                per_turn_chars=per_turn_tool_result_chars,
            ),
            guard_checker=lambda action: self._check_tool_guard(action),
            confirmation_handler=lambda action, guard, context: self._confirm_warn_action(
                action, guard, context
            ),
            interaction_handler=lambda result: self._resolve_interaction(result),
            checkpoint_before=lambda action, context: self._create_checkpoint_for_action(
                action, context
            ),
            checkpoint_after=lambda action, result, metadata, context: (
                self._finalize_checkpoint_for_action(action, result, metadata, context)
            ),
            audit_handler=lambda *args, **kwargs: self._audit_tool_action(*args, **kwargs),
            result_redactor=lambda result: self._redact_tool_result_for_observation(result),
            event_handler=lambda event: self._emit_tool_event(event),
            argument_redactor=lambda arguments: self._redacted_arguments(arguments),
            hook_manager=self.hook_manager,
            working_memory=self.working_memory,
            file_observer=self._record_file_observation,
            session_id=session_id,
            run_id_provider=lambda: getattr(self, "active_run_id", ""),
            parallel_limit=parallel_tool_limit,
            file_cache=self.file_cache,
        )
        self.deferred_tools_enabled = deferred_tools_enabled
        self.fallback_providers = list(fallback_providers or [])
        self.provider_retry_attempts = max(1, provider_retry_attempts)
        self.provider_circuit_breaker = provider_circuit_breaker or ProviderCircuitBreaker()
        provider_name = str(getattr(llm, "provider_name", ""))
        model_name = str(getattr(llm, "model", "unknown"))
        self.run_budget = RunBudget(
            get_model_profile(provider_name, model_name),
            max_turns=max_iterations,
            token_budget=token_budget,
            cost_budget_usd=cost_budget_usd,
            max_output_tokens=max_output_tokens,
            input_cost_per_million=input_cost_per_million,
            output_cost_per_million=output_cost_per_million,
        )
        self._discovered_tool_names: set[str] = set()
        self.on_thought: Callable | None = None
        self.on_action: Callable | None = None
        self.on_result: Callable | None = None
        self.on_stream: Callable | None = None
        self.on_tool_event: Callable[[ToolEvent], None] | None = None
        self._current_tool_context: ToolUseContext | None = None
        self._errors: list[str] = []
        self._tool_event_sequence = 0
        self.execution_engine.reset_run()
        self.run_budget.reset()
        self._skill_listing_sent: bool = False
        self._skill_routed: bool = False
        self._project_init_routed: bool = False
        self._active_skill_allowed_tools: set[str] | None = None
        self._active_skill_model: str | None = None
        self._base_model: str = getattr(llm, "model", "")
        self._workflow_resolved: bool = True
        self._workflow_decision: WorkflowDecision | None = None
        self._workflow_routing_error: str | None = None

    @property
    def messages(self) -> list[Message]:
        """处理消息，并按照当前组件的约定返回结果。

        返回：
            按调用约定排序的结果列表。
        """
        return self.context_manager.messages

    @messages.setter
    def messages(self, messages: list[Message]) -> None:
        self.context_manager.messages = messages

    def set_context(self, messages: list[Message]) -> None:
        """设置上下文并保持相关派生状态同步。

        参数：
            messages: 按协议顺序排列的对话消息。
        """
        self.context_manager.clear()
        for message in messages:
            self.context_manager.add_message(message)
        self._restore_discovered_tools(messages)

    def add_message(self, message: Message) -> None:
        """添加`add_message`，必要时执行去重或容量检查。

        参数：
            message: 用户提交或组件间传递的消息。
        """
        result = self.context_manager.add_message(message)
        if isinstance(result, MessageAddResult) and not result:
            raise ContextCapacityError(result.reason or "Message did not fit in context")

    async def run(
        self,
        task: str,
        on_thought: Callable | None = None,
        on_action: Callable | None = None,
        on_result: Callable | None = None,
        on_stream: Callable | None = None,
        on_tool_event: Callable[[ToolEvent], None] | None = None,
        preserve_plan_state: bool = False,
        preserve_context: bool = False,
        route_workflow: bool = False,
    ) -> str:
        """执行一轮完整的 ReAct 任务：注入系统提示词和用户任务，循环请求模型、解析全部工具调用、批量执行工具并把观察结果写回上下文，直到得到最终答案或触发轮数、预算、错误及取消边界。

        参数：
            task: 用户希望 Agent 完成的任务描述。该内容会作为本轮用户消息加入模型上下文。
            on_thought: 模型产生可展示的推理内容时调用的回调。回调接收一个字符串参数；
                传入 None 时不处理推理事件。当前版本仅保存该回调，暂未在主循环中实际调用。
            on_action: Agent 准备执行工具时调用的回调。回调接收工具名称和工具参数，
                可用于在 TUI 或 SDK 中展示当前正在执行的操作。
            on_result: 工具执行完成后调用的回调。回调接收 ToolResult 对象，
                可用于展示工具的执行结果、成功状态或错误信息。
            on_stream: 模型产生流式输出片段时调用的回调。回调接收 StreamChunk 对象，
                可用于将模型回答逐段显示在界面中；传入 None 时不发送流式展示事件。
            on_tool_event: 工具生命周期事件发生时调用的回调。回调接收 ToolEvent 对象，
                可处理工具开始、权限请求、执行完成、执行失败和取消等规范化事件。
            preserve_plan_state: 是否在开始本轮运行时保留已有计划状态。为 False 时重置
                普通任务状态并清除当前计划；为 True 时只重置本轮执行状态，保留当前计划、
                计划审批状态和计划文件信息。
            preserve_context: 是否保留之前的对话上下文。为 True 时，本轮模型可以继续使用
                前面轮次的消息；为 False 时，上层运行时会在调用本方法前清空旧上下文并重新
                注入项目记忆。当前方法内部不直接处理该参数，上下文清理由 _run_act_mode 完成。
            route_workflow: 是否在执行任务前进行工作流路由。为 True 时，根据用户意图选择
                Plan 或 Act 工作流；为 False 时跳过工作流判断，并将本轮任务直接作为 Act
                工作流处理。

        返回：
            处理后的文本或稳定标识。

        说明：
            执行过程中会更新当前实例维护的状态。
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。

        概述：
            Reason：让大模型根据当前上下文思考
            Act：解析模型要求调用的工具并执行
            Observe：把工具结果写回上下文
            然后再次 Reason
            直到模型不再调用工具、给出最终回答，或者达到迭代次数、预算、错误次数、取消等终止条件。
        """

        ''' 1. 初始化本轮运行状态
        普通 TUI 任务传入 preserve_plan_state=False，所以执行：
            self.state.reset(task)
                它会把状态设置为：
                    current_task = 用户任务
                    mode = act
                    iteration = 0
                    is_complete = False
                    error_count = 0
                    run_id = 新的唯一编号
                    清除上一份计划状态
        如果是在继续修改计划，则调用 reset_execution()，只重置执行状态，保留现有计划。
        '''
        if preserve_plan_state:
            self.state.reset_execution(task)
        else:
            self.state.reset(task)
        self.active_run_id = self.state.run_id or uuid.uuid4().hex
        self.on_thought = on_thought
        self.on_action = on_action
        self.on_result = on_result
        self.on_stream = on_stream
        self.on_tool_event = on_tool_event
        self._errors = []
        self._tool_event_sequence = 0
        self._workflow_resolved = not route_workflow
        self._workflow_decision = WorkflowDecision.ACT if not route_workflow else None
        self._workflow_routing_error = None

        ''' 2. 准备模型上下文
        你的输入最终会以这种形式放进上下文：
            Task: 帮我编写一段实现文件上传和下载的python代码
        系统提示词则告诉模型：有哪些工具、当前是 Plan 还是 Act、修改文件前要读取文件、危险操作要遵守权限规则等。
        '''
        self._upsert_runtime_system_prompt()
        self._inject_skill_listing()
        self.add_message(Message(role="user", content=f"Task: {task}"))

        self._report_progress(activity=f"Started task: {task}")

        ''' 3. 判断走 Plan 还是 Act
        因为普通 TUI 调用传入 route_workflow=True，所以执行：
            workflow = await self._resolve_workflow(task)
        对于“帮我编写文件上传和下载代码”，通常会判断为 ACT，直接执行。
        如果判断为 PLAN，系统会提前制造一个工具动作：
            ParsedAction(tool_name="enter_plan_mode")
        这意味着第一次循环不需要模型再次决定，直接进入计划模式。在计划被批准前，write_file、edit_file、execute_command 等实现型工具都会被阻止。
        '''
        pending_routed_action: ParsedAction | None = None
        if route_workflow:
            workflow: WorkflowRoutingResult = await self._resolve_workflow(task)
            if workflow.decision == WorkflowDecision.PLAN:
                pending_routed_action = ParsedAction(
                    tool_name="enter_plan_mode",
                    arguments={},
                    thought=workflow.reason
                    or "The user wants a reviewable plan before implementation.",
                )
            elif not workflow.resolved:
                self.add_message(
                    Message(
                        role="user",
                        content=(
                            "OpenNova could not resolve the execution workflow for this turn. "
                            "You may answer, inspect, search, or ask for clarification, but project "
                            "modifications are blocked until the workflow is resolved. You may call "
                            "enter_plan_mode if planning is the safe choice."
                        ),
                    )
                )

        if pending_routed_action is None and self._workflow_resolved:
            pending_routed_action = self._route_task_to_project_init(task)
            if pending_routed_action is None:
                pending_routed_action = self._route_task_to_skill(task)

        try:
            ''' 4. 判断循环能否继续
            主循环有五个继续条件：
                任务尚未完成。
                当前运行没有被新任务替换。
                没超过最大迭代次数。
                Token、费用等预算没有耗尽。
                错误次数没有超过上限。
            这里的“一次迭代”通常是“一次模型决策”，不是整个用户任务。
            '''
            while (
                not self.state.is_complete
                and self.state.run_id == self.active_run_id
                and self.state.iteration < self.max_iterations
                and self.run_budget.exhausted_reason() is None
                and not self.state.has_too_many_errors()
            ):
                self._emit_iteration_start()
                self.state.increment_iteration(self.active_run_id)

                try:
                    ''' 5. _think() 请求大模型
                    如果判断走 Plan ，则当前次迭代为 思考动作/ 预动作
                    如果判断走 Act ，则当前次迭代直接执行 _think
                        _think会把以下内容发送给 Provider：
                            系统提示词
                            之前的对话
                            用户任务
                            之前的工具调用结果
                            当前可用工具的 Schema
                        模型返回的 LLMResponse 可能是最终文本，也可能包含工具调用：
                            ToolCall(
                                name="list_directory",
                                arguments={"path": "."},
                            )
                        如果主 Provider 调用失败，_think() 还会根据错误类型重试，必要时切换备用 Provider，并记录 Token 和费用。
                    '''
                    if pending_routed_action:
                        actions = [pending_routed_action]
                        pending_routed_action = None
                        response = LLMResponse(
                            content=actions[0].thought or "",
                            finish_reason=FinishReason.TOOL_CALL,
                        )
                    else:
                        response = await self._think()
                        ''' 6. 解析模型动作
                        将模型响应转换成统一的 ParsedAction。
                        假如模型返回两个工具调用：
                            read_file("pyproject.toml")
                            read_file("src/app.py")
                        就会生成两个 ParsedAction。如果模型没有调用工具，并且 finish_reason=STOP，则设置：
                            action.is_final = True
                        表示模型认为任务已经完成。
                        计划模式是一个例外：如果模型只输出计划文字，却没有调用 exit_plan_mode 提交结构化计划，循环不会结束，而是提醒模型继续研究或正式提交计划。
                        '''
                        actions = self._parse_actions(response, task)

                    if actions[0].is_final and self._plan_submission_required():
                        if response.content:
                            self.add_message(
                                Message(
                                    role="assistant",
                                    content=response.content,
                                    reasoning_content=response.reasoning_content,
                                )
                            )
                        self.add_message(
                            Message(
                                role="user",
                                content=(
                                    "Plan mode is active. Do not finish with plan text alone. "
                                    "Continue research or call exit_plan_mode with a complete "
                                    "structured plan so the user can review it."
                                ),
                            )
                        )
                        continue

                    if actions[0].is_final:
                        self.state.mark_complete(
                            actions[0].thought or response.content or "",
                            run_id=self.active_run_id,
                        )
                        self._report_progress(activity="Completed task", mark_complete=True)
                        break

                    barrier_index = self._first_batch_barrier_index(actions)
                    completed_results: list[ToolResult | None] = [None] * len(actions)
                    scheduled_actions: list[ParsedAction] = []
                    scheduled_indices: list[int] = []

                    for action_index, action in enumerate(actions):
                        if barrier_index is not None and action_index != barrier_index:
                            completed_results[action_index] = self._deferred_batch_result(
                                action, actions[barrier_index]
                            )
                            continue

                        if not action.tool_name or action.tool_name not in self.tool_registry:
                            completed_results[action_index] = ToolResult(
                                success=False,
                                output="",
                                error=(
                                    f"Unknown tool: {action.tool_name or '(empty)'}. "
                                    "Available tools: " + ", ".join(self._available_tool_names())
                                ),
                                metadata={"unknown_tool": True},
                            )
                            continue

                        scheduled_actions.append(action)
                        scheduled_indices.append(action_index)
                        if self.on_action:
                            with suppress(Exception):
                                self.on_action(
                                    action.tool_name,
                                    self._redacted_arguments(action.arguments),
                                )
                        self._report_progress(
                            activity=f"Running tool: {action.tool_name}",
                            last_tool_name=action.tool_name,
                        )

                    ''' 7. 执行工具
                    不是最终回答时，工具会交给 execution_engine
                    每个工具大致经过：
                        规范化参数
                        → 执行前 Hook
                        → 安全规则检查
                        → 必要时询问用户权限
                        → 创建修改前检查点
                        → 真正执行工具
                        → 执行后 Hook
                        → 结果脱敏
                        → 审计和工作记忆记录
                    只读且声明为并发安全的工具可以并行执行；写文件等工具仍按顺序执行。ask_user_question、enter_plan_mode 等屏障工具会单独执行，避免和其他工具同时改变状态。
                    '''
                    if scheduled_actions:
                        outcomes = await self.execution_engine.execute_many(scheduled_actions)
                        for action_index, outcome in zip(
                            scheduled_indices, outcomes, strict=True
                        ):
                            completed_results[action_index] = outcome.result

                    finalized_results = [
                        result
                        if result is not None
                        else ToolResult(
                            success=False,
                            output="",
                            error="Tool execution produced no result",
                        )
                        for result in completed_results
                    ]
                    usage_reported = False
                    for action, result in zip(actions, finalized_results, strict=True):
                        if self.on_result:
                            with suppress(Exception):
                                self.on_result(result)
                        self._report_progress(
                            activity=f"Completed tool: {action.tool_name}",
                            last_tool_name=action.tool_name,
                            tool_use_increment=(
                                0 if result.metadata.get("batch_deferred") else 1
                            ),
                            token_count=(
                                response.usage.total_tokens
                                if response.usage and not usage_reported
                                else 0
                            ),
                        )
                        usage_reported = True

                    ''' 8. 把工具结果交还给模型 
                    会向上下文加入两类消息：
                        assistant：我调用了 read_file("src/app.py")
                        tool：文件内容是……
                    这是循环成立的关键：大模型不能直接看到磁盘，它只能通过工具结果了解项目。
                    所以下一轮 _think() 看到文件内容后，才能决定下一步是 edit_file、create_file、运行测试，还是向用户提问。
                    以 `帮我编写一段实现文件上传和下载的python代码` 任务为例，可能经历：
                        第1轮：list_directory 查看项目结构
                        第2轮：read_file 阅读框架入口和依赖
                        第3轮：create_file 创建上传下载模块
                        第4轮：execute_command 运行测试
                        第5轮：模型不再调用工具，返回完成说明
                    这只是示例，具体工具顺序由模型根据每轮观察结果动态决定。
                    '''
                    if actions:
                        await self._observe_many(
                            actions,
                            finalized_results,
                            response.reasoning_content,
                        )

                except Exception as e:
                    ''' 9. 该轮运行异常
                    如果单轮内部发生普通异常，循环不会立即崩溃，而是记录错误，并把错误作为新消息交给模型，让它换一种方式重试。
                    最终可能返回四类结果：
                        模型的最终回答
                        Task incomplete: reached maximum iterations
                        Task incomplete: 预算耗尽原因
                        Task failed: too many errors
                    '''
                    self.state.increment_error(self.active_run_id)
                    error_detail = self._redacted_text(
                        f"Error in iteration {self.state.iteration}: {type(e).__name__}: {e}"
                    )
                    tb = self._redacted_text(traceback.format_exc())
                    full_error = f"{error_detail}\n\nTraceback:\n{tb}"
                    self._errors.append(full_error)
                    print(f"\n[ERROR] {full_error}\n")
                    self.add_message(
                        Message(
                            role="user",
                            content=f"An error occurred: {error_detail}. Please try a different approach.",
                        )
                    )
        except asyncio.CancelledError:
            ''' 10. 用户取消任务
            如果用户取消任务，CancelledError 不会被吞掉，而是取消运行、通知正在执行的工具，然后继续向上传递给 TUI。
            所以，ReActLoop.run() 最核心的代码关系就是：
                _think() 让模型决定下一步
                _parse_actions() 把决定变成程序能执行的动作
                execute_many() 安全地执行工具
                _observe_many() 把现实结果重新告诉模型
                while 循环让模型基于新结果继续决定
            '''
            self.cancellation_token.cancel("Run cancelled")
            self.state.cancel_run(self.active_run_id)
            self._cancel_tool_context(self.cancellation_token.reason)
            raise
        finally:
            self._clear_skill_execution_context()

        if self.state.iteration >= self.max_iterations:
            return f"Task incomplete: reached maximum iterations ({self.max_iterations})"

        budget_reason = self.run_budget.exhausted_reason()
        if budget_reason:
            return f"Task incomplete: {budget_reason}"

        if self.state.has_too_many_errors():
            error_summary = "\n\n".join(self._errors)
            return f"Task failed: too many errors ({self.state.error_count})\n\nDetailed errors:\n{error_summary}"

        return self.state.last_result or "Task completed"

    @staticmethod
    def _first_batch_barrier_index(actions: list[ParsedAction]) -> int | None:
        """读取并返回 `_first_batch_barrier_index` 所表示的数据或流程，并遵守`ReActLoop`定义的边界与状态约束。

        参数：
            actions: 同一模型回合产生的有序动作列表。

        返回：
            `int | None` 类型的处理结果。
        """
        if len(actions) <= 1:
            return None
        return next(
            (
                index
                for index, action in enumerate(actions)
                if action.tool_name in BATCH_BARRIER_TOOLS
            ),
            None,
        )

    @staticmethod
    def _deferred_batch_result(action: ParsedAction, barrier: ParsedAction) -> ToolResult:
        """根据当前输入和`ReActLoop`的状态计算 `_deferred_batch_result`，并返回调用方需要的结果。

        参数：
            action: 模型解析出的待执行动作。
            barrier: 本次操作使用的`barrier`。

        返回：
            `ToolResult` 类型的处理结果。
        """
        return ToolResult(
            success=False,
            output="",
            error=(
                f"Tool call '{action.tool_name}' was not executed because "
                f"'{barrier.tool_name}' must execute alone. Reconsider this call after "
                "observing the updated skill, user response, or workflow state."
            ),
            metadata={
                "batch_deferred": True,
                "barrier_tool": barrier.tool_name,
            },
        )

    def _report_progress(
        self,
        activity: str,
        last_tool_name: str | None = None,
        token_count: int = 0,
        tool_use_increment: int = 0,
        mark_complete: bool = False,
    ) -> None:
        """执行 `_report_progress` 所定义的协调步骤，必要时更新`ReActLoop`维护的状态。

        参数：
            activity: 本次操作使用的活动记录。
            last_tool_name: 可选的`last_tool_name`。
            token_count: 可选的Token数量。
            tool_use_increment: 可选的`tool_use_increment`。
            mark_complete: 可选的`mark_complete`。
        """
        if not self.progress_callback:
            return

        payload = {
            "activity": activity,
            "last_tool_name": last_tool_name,
            "token_count": token_count,
            "tool_use_increment": tool_use_increment,
            "iteration": self.state.iteration,
            "is_complete": mark_complete,
        }
        with suppress(Exception):
            self.progress_callback(payload)

    def _emit_iteration_start(self) -> None:
        """发布`emit_iteration_start`，通知已订阅的界面、SDK 或持久化组件。"""
        if self.iteration_start_callback:
            self.iteration_start_callback(self.messages)

    def _start_tool_context(self, action: ParsedAction) -> ToolUseContext:
        """启动工具上下文，并按照当前组件的约定返回结果。

        参数：
            action: 模型解析出的待执行动作。

        返回：
            `ToolUseContext` 类型的处理结果。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        tool = self.tool_registry.get(action.tool_name)
        self._tool_event_sequence += 1
        run_id = getattr(self, "active_run_id", None) or uuid.uuid4().hex
        tool_id = f"tool_{run_id}_{self._tool_event_sequence:04d}"
        max_result_chars = getattr(tool, "max_result_chars", None)
        self._current_tool_context = ToolUseContext(
            tool_id=tool_id,
            tool_name=action.tool_name,
            arguments=self._redacted_arguments(action.arguments),
            started_at=perf_counter(),
            max_result_chars=max_result_chars,
            abort_signal=self.cancellation_token,
        )
        return self._current_tool_context

    def _redaction_enabled(self) -> bool:
        """读取并返回 `_redaction_enabled` 所表示的数据或流程，并遵守`ReActLoop`定义的边界与状态约束。

        返回：
            表示条件是否成立。
        """
        guardrails = getattr(self, "guardrails", None)
        if not guardrails:
            return False
        policy = guardrails.secrets_policy
        return bool(policy.get("enabled", True)) and bool(policy.get("redact_tool_outputs", True))

    def _redacted_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """构造并返回 `_redacted_arguments` 所表示的数据或流程，并遵守`ReActLoop`定义的边界与状态约束。

        参数：
            arguments: 工具调用的结构化参数。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        guardrails = getattr(self, "guardrails", None)
        if not self._redaction_enabled() or not guardrails:
            return dict(arguments)
        redacted = redact_sensitive_data(
            arguments,
            scanner=guardrails.secret_scanner,
        )
        return redacted if isinstance(redacted, dict) else {}

    def _redacted_text(self, text: str) -> str:
        """根据当前输入和`ReActLoop`的状态计算 `_redacted_text`，并返回调用方需要的结果。

        参数：
            text: 需要解析、格式化或展示的文本。

        返回：
            处理后的文本或稳定标识。
        """
        guardrails = getattr(self, "guardrails", None)
        if not self._redaction_enabled() or not guardrails:
            return text
        return str(guardrails.secret_scanner.redact(text))

    def _redact_tool_result_for_observation(self, result: ToolResult) -> ToolResult:
        """根据当前输入和`ReActLoop`的状态计算 `_redact_tool_result_for_observation`，并返回调用方需要的结果。

        参数：
            result: 前一步执行得到的规范化结果。

        返回：
            `ToolResult` 类型的处理结果。
        """
        guardrails = getattr(self, "guardrails", None)
        if not self._redaction_enabled() or not guardrails:
            return result
        result.output = self._redacted_text(result.output or "")
        if result.error:
            result.error = self._redacted_text(result.error)
        redacted_metadata = redact_sensitive_data(
            result.metadata,
            scanner=guardrails.secret_scanner,
        )
        if isinstance(redacted_metadata, dict):
            result.metadata = redacted_metadata
        return result

    def _finish_tool_context(self, result: ToolResult) -> None:
        """执行 `_finish_tool_context` 所定义的协调步骤，必要时更新`ReActLoop`维护的状态。

        参数：
            result: 前一步执行得到的规范化结果。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        context = self._current_tool_context
        if not context:
            return
        elapsed = max(0.0, perf_counter() - context.started_at)
        output = result.output or ""
        diff = result.metadata.get("diff") if isinstance(result.metadata, dict) else None
        risk_level = str(
            result.metadata.get("risk_level", context.risk_level)
            if isinstance(result.metadata, dict)
            else context.risk_level
        )
        event_type: ToolEventType = (
            "tool_cancelled"
            if result.metadata.get("cancelled")
            else "tool_result"
            if result.success
            else "tool_error"
        )
        event = ToolEvent(
            type=event_type,
            tool_id=context.tool_id,
            tool_name=context.tool_name,
            arguments=dict(context.arguments),
            started_at=context.started_at,
            duration_ms=int(elapsed * 1000),
            risk_level=risk_level,
            success=result.success,
            output=output,
            error=result.error,
            diff=diff,
            collapsible=len(output) > 1200,
            metadata=dict(result.metadata or {}),
        )
        result.metadata.setdefault("tool_id", context.tool_id)
        result.metadata.setdefault("duration_ms", int(elapsed * 1000))
        self._current_tool_context = None
        self._emit_tool_event(event)

    def _cancel_tool_context(self, reason: str) -> None:
        """取消工具上下文，并按照当前组件的约定返回结果。

        参数：
            reason: 触发当前状态变化或操作的原因。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        context = self._current_tool_context
        if context is None:
            return
        elapsed = max(0.0, perf_counter() - context.started_at)
        event = ToolEvent(
            type="tool_cancelled",
            tool_id=context.tool_id,
            tool_name=context.tool_name,
            arguments=dict(context.arguments),
            started_at=context.started_at,
            duration_ms=int(elapsed * 1000),
            risk_level=context.risk_level,
            success=False,
            error=reason,
            metadata={"cancelled": True},
        )
        self._current_tool_context = None
        self._emit_tool_event(event)

    def _emit_tool_event(self, event: ToolEvent) -> None:
        if self.on_tool_event:
            with suppress(Exception):
                self.on_tool_event(event)

    async def _think(self) -> LLMResponse:
        """向主 Provider 发起一次模型推理；遇到可重试错误时按策略重试，并在需要时切换备用 Provider，同时记录预算和熔断状态。

        返回：
            `LLMResponse` 类型的处理结果。

        说明：
            执行过程中会更新当前实例维护的状态。
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        self._upsert_runtime_system_prompt()
        tools = self._available_tools()
        providers = [self.llm, *self.fallback_providers]
        last_error: Exception | None = None
        for provider_index, provider in enumerate(providers):
            if self.provider_circuit_breaker.is_open(provider):
                continue
            for attempt in range(self.provider_retry_attempts):
                try:
                    response = await self._think_with_provider(provider, tools)
                    self.provider_circuit_breaker.record_success(provider)
                    if provider_index:
                        self.llm = provider
                        self.fallback_providers = providers[provider_index + 1 :]
                        self.context_manager.model = provider.model
                    self.run_budget.record(response.usage)
                    return response
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    provider_error = normalize_provider_error(
                        exc,
                        provider=str(getattr(provider, "provider_name", "unknown")),
                    )
                    last_error = provider_error
                    self.provider_circuit_breaker.record_failure(provider)
                    if not provider_error.retryable or attempt + 1 >= self.provider_retry_attempts:
                        break
                    await asyncio.sleep(min(0.25 * (2**attempt), 2.0))
        if last_error is not None:
            raise last_error
        raise RuntimeError("No LLM provider is available")

    async def _think_with_provider(
        self,
        provider: BaseLLMProvider,
        tools: list[Any],
    ) -> LLMResponse:
        """使用指定 Provider 完成单次模型请求。流式模式会聚合文本、工具调用、用量和推理片段，最终仍返回统一 LLMResponse。

        参数：
            provider: 负责本次模型请求的 Provider 实例。
            tools: 本次操作使用的工具。

        返回：
            `LLMResponse` 类型的处理结果。

        说明：
            该操作可能等待模型服务、MCP 服务或其他异步数据源返回。
        """
        profile = get_model_profile(str(getattr(provider, "provider_name", "")), provider.model)
        max_tokens = min(self.run_budget.output_limit(), profile.max_output_tokens)

        if self.stream and self.on_stream:
            full_content = ""
            tool_calls: list[ToolCall] = []
            usage = None
            reasoning_content: str | None = None

            async for chunk in provider.stream_chat(
                self.context_manager.get_messages_for_llm(),
                tools=tools,
                temperature=0.7,
                max_tokens=max_tokens,
            ):
                self.on_stream(chunk)

                if chunk.content:
                    full_content += chunk.content
                if chunk.tool_call:
                    tool_calls.append(chunk.tool_call)
                if chunk.usage:
                    usage = chunk.usage
                if chunk.reasoning_content:
                    if reasoning_content is None:
                        reasoning_content = chunk.reasoning_content
                    else:
                        reasoning_content += chunk.reasoning_content

            response = LLMResponse(
                content=full_content,
                tool_calls=tool_calls if tool_calls else None,
                usage=usage,
                finish_reason=FinishReason.TOOL_CALL if tool_calls else FinishReason.STOP,
                model=provider.model,
                reasoning_content=reasoning_content,
            )
            return response
        else:
            response = await provider.chat(
                self.context_manager.get_messages_for_llm(),
                tools=tools,
                temperature=0.7,
                max_tokens=max_tokens,
            )
            return response

    def _available_tools(self) -> list[Any]:
        """读取并返回 `_available_tools` 所表示的数据或流程，并遵守`ReActLoop`定义的边界与状态约束。

        返回：
            按调用约定排序的结果列表。
        """
        schemas = self.tool_registry.list_tools()
        mode = getattr(getattr(self.state, "mode", None), "value", getattr(self.state, "mode", ""))
        if not self._workflow_resolved or mode == "plan":
            schemas = [
                schema for schema in schemas if schema.name not in PLAN_MODE_IMPLEMENTATION_TOOLS
            ]
        if self.deferred_tools_enabled:
            exposed = CORE_TOOL_NAMES | self._discovered_tool_names
            if self._active_skill_allowed_tools:
                exposed |= self._active_skill_allowed_tools
            schemas = [schema for schema in schemas if schema.name in exposed]
        if not self._active_skill_allowed_tools:
            return schemas
        return [schema for schema in schemas if schema.name in self._active_skill_allowed_tools]

    def _restore_discovered_tools(self, messages: list[Message]) -> None:
        """从已有快照或持久化数据恢复已发现工具。

        参数：
            messages: 按协议顺序排列的对话消息。
        """
        self._discovered_tool_names.clear()
        for message in messages:
            if message.name != "tool_search":
                continue
            self._discovered_tool_names.update(
                re.findall(r"^- ([A-Za-z0-9_-]+):", message.content, flags=re.MULTILINE)
            )

    async def _resolve_workflow(self, task: str) -> WorkflowRoutingResult:
        """解析工作流的最终目标或处理结果。

        参数：
            task: 用户希望 Agent 完成的任务描述。

        返回：
            `WorkflowRoutingResult` 类型的处理结果。

        说明：
            执行过程中会更新当前实例维护的状态。
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        result: WorkflowRoutingResult = await WorkflowRouter(self.llm).route(
            self.context_manager.get_messages_for_llm(),
            task,
            prefer_local=True,
        )
        self._workflow_resolved = result.resolved
        self._workflow_decision = result.decision
        self._workflow_routing_error = result.error
        self._upsert_runtime_system_prompt()
        return result

    def _plan_submission_required(self) -> bool:
        """读取并返回 `_plan_submission_required` 所表示的数据或流程，并遵守`ReActLoop`定义的边界与状态约束。

        返回：
            表示条件是否成立。
        """
        mode = getattr(getattr(self.state, "mode", None), "value", getattr(self.state, "mode", ""))
        approval = getattr(
            getattr(self.state, "plan_approval_status", None),
            "value",
            getattr(self.state, "plan_approval_status", ""),
        )
        return mode == "plan" and approval not in {
            "awaiting_approval",
            "approved",
            "executing",
            "completed",
        }

    def _available_tool_names(self) -> list[str]:
        return [schema.name for schema in self._available_tools()]

    def _parse_response(self, response: LLMResponse, task: str = "") -> ParsedAction:
        """解析`parse_response`并转换为内部使用的规范结构。

        参数：
            response: 本次操作使用的`response`。
            task: 用户希望 Agent 完成的任务描述。

        返回：
            `ParsedAction` 类型的处理结果。
        """
        content = response.content or ""
        action = ParsedAction(
            thought=content,
            tool_name="",
            arguments={},
            raw_response=content,
        )

        if response.tool_calls:
            tool_call = response.tool_calls[0]
            action.tool_name = tool_call.name
            action.arguments = tool_call.arguments or {}

            if self._is_dangerous_action(action.tool_name, action.arguments):
                action.requires_confirmation = True
        elif self.skill_registry:
            action = self._parse_skill_invocation(action, content)

        if (
            response.finish_reason == FinishReason.STOP
            and not response.tool_calls
            and not action.tool_name
        ):
            routed_action = self._route_task_to_project_init(task)
            if not routed_action:
                routed_action = self._route_task_to_skill(task)
            if routed_action:
                return routed_action
            action.is_final = True

        return action

    def _parse_actions(self, response: LLMResponse, task: str = "") -> list[ParsedAction]:
        """把模型响应中的每个 ToolCall 按原顺序转换为 ParsedAction，避免一次响应包含多个调用时丢失后续动作。

        参数：
            response: 本次操作使用的`response`。
            task: 用户希望 Agent 完成的任务描述。

        返回：
            按调用约定排序的结果列表。
        """
        if not response.tool_calls:
            return [self._parse_response(response, task)]

        actions: list[ParsedAction] = []
        for tool_call in response.tool_calls:
            action = ParsedAction(
                tool_name=tool_call.name,
                arguments=tool_call.arguments or {},
                thought=response.content or "",
                raw_response=response.content or "",
                tool_call_id=tool_call.id,
            )
            if self._is_dangerous_action(action.tool_name, action.arguments):
                action.requires_confirmation = True
            actions.append(action)
        return actions

    def _route_task_to_project_init(self, task: str) -> ParsedAction | None:
        """根据当前输入和`ReActLoop`的状态计算 `_route_task_to_project_init`，并返回调用方需要的结果。

        参数：
            task: 用户希望 Agent 完成的任务描述。

        返回：
            `ParsedAction | None` 类型的处理结果。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        if self._project_init_routed:
            return None
        if "init_project_guide" not in self.tool_registry:
            return None

        from opennova.memory.project_guide import ProjectGuideManager

        guide_manager = ProjectGuideManager(project_path=".")
        if guide_manager.exists():
            return None
        if not guide_manager.is_high_confidence_init_request(task):
            return None

        self._project_init_routed = True
        return ParsedAction(
            tool_name="init_project_guide",
            arguments={"force": False},
            thought=(
                "The user asked to initialize project onboarding context, "
                "so I will create OPENNOVA.md first."
            ),
        )

    def _route_task_to_skill(self, task: str) -> ParsedAction | None:
        """根据当前输入和`ReActLoop`的状态计算 `_route_task_to_skill`，并返回调用方需要的结果。

        参数：
            task: 用户希望 Agent 完成的任务描述。

        返回：
            `ParsedAction | None` 类型的处理结果。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        if self._skill_routed:
            return None
        if "skill" not in self.tool_registry:
            return None
        if not self.skill_registry or not self.skill_registry.can_model_invoke("skill-creator"):
            return None
        if not self._is_skill_creator_request(task):
            return None

        self._skill_routed = True

        return ParsedAction(
            tool_name="skill",
            arguments={"skill": "skill-creator", "args": task},
            thought="The user's request is to create or improve a skill, so I will invoke skill-creator first.",
        )

    def _is_skill_creator_request(self, task: str) -> bool:
        """读取并返回 `_is_skill_creator_request` 所表示的数据或流程，并遵守`ReActLoop`定义的边界与状态约束。

        参数：
            task: 用户希望 Agent 完成的任务描述。

        返回：
            表示条件是否成立。
        """
        return bool(self._SKILL_CREATOR_TRIGGER_RE.search(task))

    def _parse_skill_invocation(self, action: ParsedAction, content: str) -> ParsedAction:
        """解析`parse_skill_invocation`并转换为内部使用的规范结构。

        参数：
            action: 模型解析出的待执行动作。
            content: 需要处理、保存或分析的文本内容。

        返回：
            `ParsedAction` 类型的处理结果。
        """
        if not self.skill_registry:
            return action

        stripped = content.strip()
        if not stripped.lower().startswith("/skill"):
            return action

        parts = stripped.split(maxsplit=2)
        if len(parts) < 2:
            return action

        skill_name = parts[1].strip()
        skill_args = parts[2].strip() if len(parts) > 2 else ""
        if not self.skill_registry.can_model_invoke(skill_name):
            return action

        action.tool_name = "skill"
        action.arguments = {"skill": skill_name, "args": skill_args}
        return action

    def _record_file_observation(self, action: ParsedAction, result: ToolResult) -> None:
        """记录文件观察记录，供状态展示、恢复或后续决策使用。

        参数：
            action: 模型解析出的待执行动作。
            result: 前一步执行得到的规范化结果。
        """
        if not self.working_memory or not result.success:
            pass

        observed_paths: list[str] = []
        file_path = None
        if isinstance(result.metadata, dict):
            file_path = result.metadata.get("file_path")
            if isinstance(file_path, str):
                observed_paths.append(file_path)
            directory = result.metadata.get("directory")
            if isinstance(directory, str):
                observed_paths.append(directory)
        argument_path = action.arguments.get("file_path")
        if isinstance(argument_path, str):
            observed_paths.append(argument_path)
        argument_directory = action.arguments.get("directory")
        if isinstance(argument_directory, str):
            observed_paths.append(argument_directory)

        change_types = {
            "read_file": "read",
            "write_file": "modified",
            "create_file": "created",
            "delete_file": "deleted",
        }
        change_type = change_types.get(action.tool_name)
        if file_path and change_type and self.working_memory and result.success:
            preview = (result.output or result.error or "")[:200] or None
            self.working_memory.observe_file(file_path, change_type, preview)

        if self.skill_registry and observed_paths:
            cwd = self.working_dir or os.getcwd()
            self.skill_registry.discover_for_paths(observed_paths, cwd)
            self.skill_registry.activate_for_paths(observed_paths, cwd)

    async def _act(self, action: ParsedAction) -> ToolResult:
        """启动或推进 `_act` 所表示的数据或流程，并遵守`ReActLoop`定义的边界与状态约束。

        参数：
            action: 模型解析出的待执行动作。

        返回：
            `ToolResult` 类型的处理结果。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        return (await self.execution_engine.execute_one(action)).result

    def _audit_tool_action(
        self,
        action: ParsedAction,
        guard_result: GuardResult | None,
        result: ToolResult,
        *,
        confirmation_outcome: str | None,
        checkpoint_metadata: dict[str, Any],
        started_at: float,
    ) -> None:
        """执行 `_audit_tool_action` 所定义的协调步骤，必要时更新`ReActLoop`维护的状态。

        参数：
            action: 模型解析出的待执行动作。
            guard_result: 可选的安全检查结果。
            result: 前一步执行得到的规范化结果。
            confirmation_outcome: 可选的`confirmation_outcome`。
            checkpoint_metadata: 本次操作使用的`checkpoint_metadata`。
            started_at: 本次操作使用的`started_at`。
        """
        if not self.audit_logger:
            return
        checkpoint_id = None
        if checkpoint_metadata:
            checkpoint_id = checkpoint_metadata.get("checkpoint_id")
        self.audit_logger.log_tool_event(
            tool_name=action.tool_name,
            arguments=dict(action.arguments),
            guard_result=guard_result,
            result=result,
            confirmation_outcome=confirmation_outcome,
            checkpoint_id=checkpoint_id,
            duration_ms=round((perf_counter() - started_at) * 1000, 3),
        )

    def _create_checkpoint_for_action(
        self,
        action: ParsedAction,
        context: ToolUseContext | None = None,
    ) -> dict[str, Any]:
        """创建检查点对应动作并完成必要的初始化。

        参数：
            action: 模型解析出的待执行动作。
            context: 本次工具调用或运行所使用的上下文。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        if action.tool_name not in {
            "write_file",
            "create_file",
            "edit_file",
            "multi_edit_file",
            "delete_file",
        }:
            return {}

        file_path = action.arguments.get("file_path")
        if not file_path:
            return {}

        try:
            from pathlib import Path

            from opennova.checkpoints import CheckpointManager

            project_path = Path(self.working_dir or ".").resolve()
            target = Path(file_path).expanduser().resolve()
            checkpoint_id = CheckpointManager(project_path).create(
                f"Before {action.tool_name}",
                [target],
                run_id=getattr(self, "active_run_id", None),
                user_message=getattr(self.state, "task", None),
                tool_id=context.tool_id if context else None,
            )
            return {
                "checkpoint_id": checkpoint_id,
                "checkpoint_tool_id": context.tool_id if context else None,
            }
        except Exception as exc:
            return {"checkpoint_warning": str(exc)}

    def _finalize_checkpoint_for_action(
        self,
        action: ParsedAction,
        result: ToolResult,
        checkpoint_metadata: dict[str, Any],
        context: ToolUseContext,
    ) -> None:
        """执行 `_finalize_checkpoint_for_action` 所定义的协调步骤，必要时更新`ReActLoop`维护的状态。

        参数：
            action: 模型解析出的待执行动作。
            result: 前一步执行得到的规范化结果。
            checkpoint_metadata: 本次操作使用的`checkpoint_metadata`。
            context: 本次工具调用或运行所使用的上下文。
        """
        del action, context
        checkpoint_id = checkpoint_metadata.get("checkpoint_id")
        if not checkpoint_id:
            return
        try:
            from opennova.checkpoints import CheckpointManager

            checkpoint = CheckpointManager(self.working_dir or ".").finalize(checkpoint_id)
            result.metadata["checkpoint_operations"] = {
                entry.path: entry.operation for entry in checkpoint.entries
            }
        except Exception as exc:
            result.metadata["checkpoint_warning"] = str(exc)

    def _normalize_tool_arguments(
        self,
        tool: Any,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """规范化工具参数，消除不同调用格式之间的差异。

        参数：
            tool: 要注册、检查或调用的工具实例。
            arguments: 工具调用的结构化参数。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        normalizer = getattr(tool, "normalize_arguments", None)
        if not callable(normalizer):
            return arguments
        normalized = normalizer(arguments)
        return normalized if isinstance(normalized, dict) else arguments

    def _check_tool_guard(self, action: ParsedAction) -> GuardResult:
        """检查`check_tool_guard`并返回明确的校验或策略结果。

        参数：
            action: 模型解析出的待执行动作。

        返回：
            `GuardResult` 类型的处理结果。
        """
        if (
            self._active_skill_allowed_tools is not None
            and action.tool_name not in self._active_skill_allowed_tools
        ):
            return GuardResult(
                allowed=False,
                risk_level=RiskLevel.BLOCK,
                reason=(
                    f"Tool '{action.tool_name}' is not allowed by the currently active skill. "
                    f"Allowed tools: {', '.join(sorted(self._active_skill_allowed_tools))}"
                ),
            )
        if not self._workflow_resolved and action.tool_name in PLAN_MODE_IMPLEMENTATION_TOOLS:
            return GuardResult(
                allowed=False,
                risk_level=RiskLevel.BLOCK,
                reason=(
                    f"Tool '{action.tool_name}' is blocked because the execution workflow "
                    "has not been resolved for this turn."
                ),
                requires_confirmation=False,
                suggestions=[
                    "Answer without modifying files, continue inspecting, or call enter_plan_mode."
                ],
                metadata={"workflow_unresolved": True},
            )
        if (
            self._workflow_resolved
            and self._workflow_decision == WorkflowDecision.ACT
            and action.tool_name == "enter_plan_mode"
        ):
            return GuardResult(
                allowed=False,
                risk_level=RiskLevel.BLOCK,
                reason="Plan mode is disabled for this explicitly direct execution turn.",
                requires_confirmation=False,
                metadata={"workflow_decision": WorkflowDecision.ACT.value},
            )
        if (
            self._workflow_decision == WorkflowDecision.PLAN
            and action.tool_name in PLAN_MODE_IMPLEMENTATION_TOOLS
        ):
            return GuardResult(
                allowed=False,
                risk_level=RiskLevel.BLOCK,
                reason=f"Tool '{action.tool_name}' is blocked until the plan is approved.",
                requires_confirmation=False,
                metadata={"workflow_decision": WorkflowDecision.PLAN.value},
            )
        plan_mode = getattr(
            getattr(self.state, "mode", None), "value", getattr(self.state, "mode", "")
        )
        if plan_mode == "plan" and action.tool_name in PLAN_MODE_IMPLEMENTATION_TOOLS:
            return GuardResult(
                allowed=False,
                risk_level=RiskLevel.BLOCK,
                reason=(
                    f"Tool '{action.tool_name}' is blocked in plan mode. "
                    "Continue researching with read/search tools, then call exit_plan_mode "
                    "with a concrete plan and wait for user approval before implementation."
                ),
                requires_confirmation=False,
                suggestions=[
                    "Use read_file, list_directory, glob_files, grep_code, or ask_user_question to finish the plan.",
                    "Call exit_plan_mode with the proposed plan before modifying files.",
                ],
                metadata={"plan_mode_blocked": True},
            )
        if not self.guardrails:
            return GuardResult(
                allowed=True,
                risk_level=RiskLevel.SAFE,
                reason="Guardrails disabled",
                requires_confirmation=False,
            )
        tool = self.tool_registry.get(action.tool_name)
        tool_context_provider = getattr(tool, "get_security_context", None)
        tool_context = tool_context_provider() if callable(tool_context_provider) else None
        return self.guardrails.check_tool_call(
            action.tool_name,
            action.arguments,
            working_dir=self.working_dir,
            tool_context=tool_context,
        )

    async def _confirm_warn_action(
        self,
        action: ParsedAction,
        guard_result: GuardResult,
        context: ToolUseContext | None = None,
    ) -> ToolResult:
        """根据当前输入和`ReActLoop`的状态计算 `_confirm_warn_action`，并返回调用方需要的结果。

        参数：
            action: 模型解析出的待执行动作。
            guard_result: 本次操作使用的安全检查结果。
            context: 本次工具调用或运行所使用的上下文。

        返回：
            `ToolResult` 类型的处理结果。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        del context
        prompt_result = ToolResult(
            success=True,
            output=(
                f"Confirmation required: {guard_result.reason}\n"
                "Proceed only if this is intentional."
            ),
            metadata={
                "interaction_required": True,
                "interaction_type": "ask_user_question",
                "questions": [
                    {
                        "question": (
                            f"{guard_result.reason}\n"
                            f"Tool: {action.tool_name}\n"
                            "Do you want to proceed?"
                        ),
                        "header": "Confirm",
                        "options": [
                            {
                                "index": 1,
                                "label": "Proceed",
                                "description": "Execute this action now.",
                            },
                            {
                                "index": 2,
                                "label": "Cancel",
                                "description": "Skip this action and continue safely.",
                            },
                        ],
                        "multiSelect": False,
                        "free_text": False,
                        "allow_custom_answer": False,
                    }
                ],
                "prompt_payload": {
                    "question": (
                        f"{guard_result.reason}\nTool: {action.tool_name}\nDo you want to proceed?"
                    ),
                    "header": "Confirm",
                    "options": [
                        {"index": 1, "label": "Proceed", "description": "Execute this action now."},
                        {
                            "index": 2,
                            "label": "Cancel",
                            "description": "Skip this action and continue safely.",
                        },
                    ],
                    "multi_select": False,
                    "free_text": False,
                    "allow_custom_answer": False,
                },
            },
        )
        resolved = await self._resolve_interaction(prompt_result)
        if not resolved.success:
            return ToolResult(
                success=False,
                output="",
                error=resolved.error or "Confirmation failed",
                metadata={**resolved.metadata, "guard_confirmation_failed": True},
            )

        all_answers = resolved.metadata.get("all_answers", [])
        selected_answer = ""
        if all_answers:
            selected_answer = str(all_answers[0].get("answer") or "")
        if not selected_answer:
            selected_answer = str(resolved.metadata.get("answer") or "")
        if selected_answer.strip().lower() not in {"proceed", "yes", "y", "1"}:
            return ToolResult(
                success=False,
                output="Action cancelled by user confirmation policy.",
                error="User declined confirmation",
                metadata={"guard_confirmation_declined": True},
            )

        return ToolResult(success=True, output="User confirmed action")

    async def _resolve_interaction(self, result: ToolResult) -> ToolResult:
        """解析用户交互的最终目标或处理结果。

        参数：
            result: 前一步执行得到的规范化结果。

        返回：
            `ToolResult` 类型的处理结果。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        self.state.begin_interaction("tool_confirmation")
        if not self.interaction_callback:
            self.state.end_interaction()
            return ToolResult(
                success=False,
                output=result.output,
                error="Interactive response required but no interaction handler is available.",
                metadata={**result.metadata, "interaction_unresolved": True},
            )

        try:
            try:
                interaction_result = self.interaction_callback(result.metadata)
                if asyncio.iscoroutine(interaction_result):
                    interaction_result = await interaction_result
            except Exception as e:
                return ToolResult(
                    success=False,
                    output=result.output,
                    error=f"Interaction failed: {e}",
                    metadata={**result.metadata, "interaction_unresolved": True},
                )
        finally:
            self.state.end_interaction()

        if not isinstance(interaction_result, dict):
            return ToolResult(
                success=False,
                output=result.output,
                error=f"Interaction callback returned unexpected type: {type(interaction_result).__name__}",
                metadata={**result.metadata, "interaction_unresolved": True},
            )

        all_answers = interaction_result.get("all_answers", [])
        skipped = interaction_result.get("skipped", False)

        # 兼容旧版单问题回调格式，其中没有 all_answers 字段。
        if not all_answers:
            prompt_payload = result.metadata.get("prompt_payload", {})
            question = prompt_payload.get("question", "")
            if skipped:
                return ToolResult(
                    success=True,
                    output=f"Question: {question}\n"
                    "User did not provide an answer. Please make the best decision.",
                    metadata={
                        **result.metadata,
                        "interaction_required": False,
                        "skipped": True,
                        "skipped_question": question,
                    },
                )
            return ToolResult(
                success=True,
                output=f"Answer to: {question}\n{interaction_result.get('display', '')}".strip(),
                metadata={
                    **result.metadata,
                    "interaction_required": False,
                    "answers": interaction_result.get("answers", {}),
                    "answer": interaction_result.get("answer"),
                    "selected_options": interaction_result.get("selected_options", []),
                },
            )

        # 处理新版多问题交互格式。
        if skipped and all(a.get("skipped") for a in all_answers):
            questions = result.metadata.get("questions", [])
            first_q = questions[0].get("question", "") if questions else ""
            return ToolResult(
                success=True,
                output=f"Question: {first_q}\n"
                "User did not provide an answer. Please make the best decision.",
                metadata={
                    **result.metadata,
                    "interaction_required": False,
                    "skipped": True,
                    "skipped_question": first_q,
                },
            )

        # 把多项回答整理为模型容易继续理解的结构化文本。
        answer_parts = [
            f'"{a.get("question", "")}"="{a.get("answer", "(skipped)")}"' for a in all_answers
        ]
        output = (
            f"User has answered your questions: {'; '.join(answer_parts)}. "
            "You can now continue with the user's answers in mind."
        )

        return ToolResult(
            success=True,
            output=output,
            metadata={
                **result.metadata,
                "interaction_required": False,
                "answers": interaction_result.get("answers", {}),
                "all_answers": all_answers,
                "display": interaction_result.get("display", ""),
            },
        )

    async def _observe(
        self, action: ParsedAction, result: ToolResult, reasoning_content: str | None = None
    ) -> None:
        """把单个工具动作和结果包装成一组数据，复用 `_observe_many()` 写回上下文，确保单工具与批量工具调用遵循同一消息协议。

        参数：
            action: 模型解析出的待执行动作。
            result: 前一步执行得到的规范化结果。
            reasoning_content: 可选的`reasoning_content`。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        await self._observe_many([action], [result], reasoning_content)

    async def _observe_many(
        self,
        actions: list[ParsedAction],
        results: list[ToolResult],
        reasoning_content: str | None = None,
    ) -> None:
        """把模型同一回合产生的 assistant 工具调用消息和全部 tool 结果作为一个完整协议组写入上下文。成组插入可避免压缩或容量裁剪留下孤立的工具结果。

        参数：
            actions: 同一模型回合产生的有序动作列表。
            results: 与动作列表按位置一一对应的工具结果。
            reasoning_content: 可选的`reasoning_content`。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        if len(actions) != len(results):
            raise ValueError("Actions and results must have the same length")

        tool_calls = [
            ToolCall(
                id=getattr(action, "tool_call_id", None) or f"call_{self.state.iteration}_{index}",
                name=action.tool_name,
                arguments=self._redacted_arguments(action.arguments),
            )
            for index, action in enumerate(actions, start=1)
        ]
        assistant_msg = Message(
            role="assistant",
            content=actions[0].thought or "",
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
        )
        protocol_messages = [assistant_msg]
        for action, result, tool_call in zip(actions, results, tool_calls, strict=True):
            protocol_messages.append(
                Message(
                    role="tool",
                    content=result.to_string(),
                    tool_call_id=tool_call.id,
                    name=action.tool_name,
                )
            )

        add_group = getattr(self.context_manager, "add_messages_and_compress", None)
        if callable(add_group):
            insertion = await add_group(protocol_messages)
            if isinstance(insertion, MessageAddResult) and not insertion:
                raise ContextCapacityError(insertion.reason or "Tool protocol group did not fit")
        else:
            # 为测试中的轻量上下文替身保留兼容路径。
            self.add_message(assistant_msg)
            for tool_message in protocol_messages[1:-1]:
                self.add_message(tool_message)
            await self.context_manager.add_message_and_compress(protocol_messages[-1])

        for action, result in zip(actions, results, strict=True):
            if not result.metadata.get("batch_deferred"):
                self._post_observation(action, result)

    def _post_observation(self, action: ParsedAction, result: ToolResult) -> None:
        """更新 `_post_observation` 所表示的数据或流程，并遵守`ReActLoop`定义的边界与状态约束。

        参数：
            action: 模型解析出的待执行动作。
            result: 前一步执行得到的规范化结果。

        说明：
            执行过程中会更新当前实例维护的状态。
        """

        if action.tool_name == "tool_search" and result.success:
            discovered = result.metadata.get("discovered_tools", [])
            self._discovered_tool_names.update(
                str(name) for name in discovered if str(name) in self.tool_registry
            )
            self._upsert_runtime_system_prompt()

        # 用户跳过问题时，明确告诉模型可自行作出决定。
        if result.metadata.get("skipped"):
            question = result.metadata.get("skipped_question", "")
            if question:
                self.add_message(
                    Message(
                        role="user",
                        content=f"I'll let you decide on this: {question}",
                    )
                )
        # 多问题仅部分跳过时，明确列出由模型自行决定的项目。
        all_answers = result.metadata.get("all_answers", [])
        skipped_questions = [a for a in all_answers if a.get("skipped")]
        if skipped_questions and not result.metadata.get("skipped"):
            skipped_texts = [f'"{a.get("question", "")}"' for a in skipped_questions]
            self.add_message(
                Message(
                    role="user",
                    content=f"I'll let you decide on: {', '.join(skipped_texts)}",
                )
            )

        # Skill 调用成功后，把展开后的 Skill 提示词作为用户消息加入上下文，
        # 并放在工具结果之后，以保持兼容的消息顺序。
        if action.tool_name == "skill" and result.success and "skill_prompt" in result.metadata:
            skill_name = result.metadata.get("resolved_skill") or result.metadata.get(
                "skill", "unknown"
            )
            skill_prompt = result.metadata["skill_prompt"]
            if self.skill_registry:
                self.skill_registry.record_skill_usage(skill_name)
            self._apply_skill_execution_context(result.metadata)
            if self.hook_manager and isinstance(result.metadata.get("hooks"), dict):
                register_skill_hooks(
                    self.hook_manager,
                    result.metadata["hooks"],
                    skill_name=skill_name,
                    skill_root=result.metadata.get("skill_dir"),
                )
            self.add_message(
                Message(
                    role="user",
                    content=f"Invoked skill '{skill_name}':\n\n{skill_prompt}",
                )
            )

        self.state.record_action_result(
            action.tool_name,
            result.output,
            run_id=getattr(self, "active_run_id", None),
        )
        if (
            action.tool_name == "exit_plan_mode"
            and result.success
            and result.metadata.get("status") == "awaiting_approval"
        ):
            self.state.mark_complete(
                result.output or "Plan ready for approval",
                run_id=getattr(self, "active_run_id", None),
            )
        elif action.tool_name == "enter_plan_mode" and result.success:
            self._workflow_resolved = True
            self._workflow_decision = WorkflowDecision.PLAN
            self._workflow_routing_error = None
            self._upsert_runtime_system_prompt()

    def _apply_skill_execution_context(self, metadata: dict[str, Any]) -> None:
        """应用Skill执行上下文，并按照当前组件的约定返回结果。

        参数：
            metadata: 随主体数据传递的扩展元数据。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        allowed_tools = metadata.get("allowed_tools") or []
        self._active_skill_allowed_tools = set(allowed_tools) if allowed_tools else None

        model = str(metadata.get("model") or "").strip()
        if model:
            if self._active_skill_model is None:
                self._base_model = getattr(self.llm, "model", self._base_model)
            self._active_skill_model = model
            self.llm.model = model

    def _clear_skill_execution_context(self) -> None:
        """恢复 `_clear_skill_execution_context` 所表示的数据或流程，并遵守`ReActLoop`定义的边界与状态约束。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self._active_skill_allowed_tools = None
        if self._active_skill_model is not None:
            self.llm.model = self._base_model
            self._active_skill_model = None

    def _build_system_prompt(self) -> str:
        """根据当前可见工具、已发现能力、Skill 和工作流状态动态生成系统提示词，确保模型只调用本轮允许使用的能力。

        返回：
            处理后的文本或稳定标识。
        """
        tools_description = []
        for schema in self._available_tools():
            params_desc = []
            props = schema.parameters.get("properties", {})
            required = schema.parameters.get("required", [])

            for name, prop in props.items():
                req = " (required)" if name in required else ""
                params_desc.append(
                    f"    - {name}: {prop.get('description', prop.get('type', ''))}{req}"
                )

            params_str = "\n".join(params_desc) if params_desc else "    No parameters"
            tools_description.append(f"- {schema.name}: {schema.description}\n{params_str}")

        prompt = f"""You are an AI coding assistant that helps users with software engineering tasks.

You have access to the following tools:
{chr(10).join(tools_description)}
"""

        if self.deferred_tools_enabled:
            prompt += """
Only core and previously discovered tool schemas are shown. If you need Git, diagnostics,
background tasks, MCP, worktrees, web access, or another hidden capability, call tool_search with
a concise capability query. Its matches become available on the next model turn and are persisted
in the transcript.
"""

        # 把 Skill 列表直接放入系统提示词，以提高模型发现能力的稳定性。
        # 如果 Skill 只出现在 user 消息中，模型可能把它当作普通历史内容，
        # 而不是必须遵循的系统级能力说明。
        if self.skill_registry:
            model_skills = self.skill_registry.list_model_invocable_skills()
            if model_skills:
                skill_entries: list[str] = []
                for name in model_skills[:20]:
                    skill = self.skill_registry.get_skill(name)
                    if skill is None:
                        continue
                    meta = skill.metadata
                    entry = f"- {meta.name}: {meta.description}"
                    if meta.when_to_use:
                        entry += f"\n  When to use: {meta.when_to_use}"
                    if meta.argument_hint:
                        entry += f"\n  Arguments: {meta.argument_hint}"
                    skill_entries.append(entry)

                prompt += f"""
In addition to tools, you have access to specialized skills. Each skill provides
domain-specific instructions that are loaded on invocation.

Available skills:
{chr(10).join(skill_entries)}

How to invoke a skill: call the Skill tool with skill="<skill-name>" and optional args.
Example: Skill("code_review", "src/main.py")

IMPORTANT: Skill invocation is a BLOCKING REQUIREMENT. When a listed skill matches
the user's request, invoke the Skill tool BEFORE generating any other response.
Do not mention a skill in prose without calling the Skill tool.
"""

        prompt += """
Use these tools and skills to complete the user's task. When you have completed the task,
provide a summary of what was done.

Rules:
1. Always explain what you are doing before executing a tool or skill
2. If a tool fails, try to understand the error and attempt a different approach
3. Be careful with file operations - read before write when modifying existing files
4. For multi-step implementation work, maintain explicit progress with todo/progress tracking
5. If you are executing an approved plan, follow the current plan instead of silently re-planning
6. If the user asks you to plan before coding, write a plan first, make a plan first,
   or otherwise requests approval before implementation, call enter_plan_mode before any implementation or file modification tool.
   Do not modify files before exit_plan_mode has requested user approval.
7. When the task is complete, provide a clear summary
"""
        return prompt

    def _upsert_runtime_system_prompt(self) -> None:
        """执行 `_upsert_runtime_system_prompt` 所定义的协调步骤，必要时更新`ReActLoop`维护的状态。"""
        runtime_message = Message(
            role="system",
            content=self._build_system_prompt(),
            name=RUNTIME_SYSTEM_MESSAGE_NAME,
        )
        retained = [
            message
            for message in self.context_manager.messages
            if not (
                message.role == "system"
                and (
                    message.name == RUNTIME_SYSTEM_MESSAGE_NAME
                    or (
                        message.name is None
                        and message.content.startswith(LEGACY_RUNTIME_SYSTEM_PROMPT_PREFIX)
                    )
                )
            )
        ]
        self.context_manager.messages[:] = [runtime_message, *retained]

    def _is_dangerous_action(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        """校验 `_is_dangerous_action` 所表示的数据或流程，并遵守`ReActLoop`定义的边界与状态约束。

        参数：
            tool_name: 目标工具在注册表中的名称。
            arguments: 工具调用的结构化参数。

        返回：
            表示条件是否成立。
        """
        dangerous_tools = {"delete_file", "execute_command", "write_file"}
        return tool_name in dangerous_tools

    def _inject_skill_listing(self) -> None:
        """执行 `_inject_skill_listing` 所定义的协调步骤，必要时更新`ReActLoop`维护的状态。"""
        pass


async def run_simple_task(
    llm: BaseLLMProvider,
    tool_registry: ToolRegistry,
    task: str,
    max_iterations: int = 200,
    stream: bool = True,
    on_stream: Callable | None = None,
) -> str:
    """运行`run_simple_task`流程，并统一处理完成、失败和取消。

    参数：
        llm: 本次操作使用的`llm`。
        tool_registry: 本次操作使用的工具注册表。
        task: 用户希望 Agent 完成的任务描述。
        max_iterations: 允许 ReAct 循环执行的最大轮数，用于阻止模型无限调用工具。
        stream: 是否将模型输出以增量事件形式返回。
        on_stream: 模型返回流式片段时调用的回调。

    返回：
        处理后的文本或稳定标识。

    说明：
        这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
    """
    state = AgentState()
    loop = ReActLoop(
        llm=llm,
        tool_registry=tool_registry,
        state=state,
        max_iterations=max_iterations,
        stream=stream,
    )
    return await loop.run(task, on_stream=on_stream, route_workflow=True)
