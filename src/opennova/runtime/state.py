"""Agent 核心运行时中的状态模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

if TYPE_CHECKING:
    from opennova.runtime.store import RuntimeStateStore


class AgentMode(StrEnum):
    """枚举Agent模式允许出现的稳定取值，序列化和状态判断均使用这些值。"""

    PLAN = "plan"
    ACT = "act"


class StepStatus(StrEnum):
    """枚举步骤状态允许出现的稳定取值，序列化和状态判断均使用这些值。"""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"
    INTERRUPTED = "interrupted"


class PlanStatus(StrEnum):
    """枚举计划状态允许出现的稳定取值，序列化和状态判断均使用这些值。"""

    PLANNING = "planning"
    EXECUTING = "executing"
    DONE = "done"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class PlanApprovalStatus(StrEnum):
    """枚举计划审批状态允许出现的稳定取值，序列化和状态判断均使用这些值。"""

    NONE = "none"
    DRAFT = "draft"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    EXECUTING = "executing"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DISCARDED = "discarded"


@dataclass
class PlanStep:
    """保存计划步骤所需的结构化数据，主要包含 `id`、`description`、`uid`、`status`、`tool_hint`、`result_summary`、`error`
    字段，便于在组件之间传递或持久化。
    """

    id: str
    description: str
    uid: str = field(default_factory=lambda: uuid4().hex)
    status: StepStatus = StepStatus.PENDING
    tool_hint: str | None = None
    result_summary: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """把计划步骤转换为可序列化字典，供事件、会话或 API 边界使用。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        return {
            "id": self.id,
            "uid": self.uid,
            "description": self.description,
            "status": self.status.value,
            "tool_hint": self.tool_hint,
            "result_summary": self.result_summary,
            "error": self.error,
        }


@dataclass
class Plan:
    """保存计划所需的结构化数据，主要包含 `task`、`steps`、`created_at`、`status` 字段，便于在组件之间传递或持久化。"""

    task: str
    steps: list[PlanStep] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    status: PlanStatus = PlanStatus.PLANNING

    def get_next_step(self) -> PlanStep | None:
        """读取下一个步骤，不改变当前对象的业务状态。

        返回：
            `PlanStep | None` 类型的处理结果。
        """
        for step in self.steps:
            if step.status == StepStatus.PENDING:
                return step
        return None

    def mark_step_running(self, step_id: str) -> None:
        """把`mark_step_running`更新为目标状态，并触发必要的状态事件。

        参数：
            step_id: 本次操作使用的`step_id`。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        for step in self.steps:
            if step.id == step_id:
                step.status = StepStatus.RUNNING
                self.status = PlanStatus.EXECUTING
                break

    def mark_step_done(self, step_id: str, result: str | None = None) -> None:
        """把`mark_step_done`更新为目标状态，并触发必要的状态事件。

        参数：
            step_id: 本次操作使用的`step_id`。
            result: 前一步执行得到的规范化结果。
        """
        for step in self.steps:
            if step.id == step_id:
                step.status = StepStatus.DONE
                step.result_summary = result
                break

        self._update_plan_status()

    def mark_step_failed(self, step_id: str, error: str) -> None:
        """把`mark_step_failed`更新为目标状态，并触发必要的状态事件。

        参数：
            step_id: 本次操作使用的`step_id`。
            error: 本次操作使用的错误。
        """
        for step in self.steps:
            if step.id == step_id:
                step.status = StepStatus.FAILED
                step.error = error
                break

        self._update_plan_status()

    def _update_plan_status(self) -> None:
        """更新 `_update_plan_status` 所表示的数据或流程，并遵守计划定义的边界与状态约束。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        all_done = bool(self.steps) and all(
            s.status in {StepStatus.DONE, StepStatus.SKIPPED} for s in self.steps
        )
        any_failed = any(s.status == StepStatus.FAILED for s in self.steps)
        any_interrupted = any(s.status == StepStatus.INTERRUPTED for s in self.steps)

        if any_failed:
            self.status = PlanStatus.FAILED
        elif any_interrupted:
            self.status = PlanStatus.INTERRUPTED
        elif all_done:
            self.status = PlanStatus.DONE
        elif any(s.status == StepStatus.RUNNING for s in self.steps):
            self.status = PlanStatus.EXECUTING

    def to_dict(self) -> dict[str, Any]:
        """把计划转换为可序列化字典，供事件、会话或 API 边界使用。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        return {
            "task": self.task,
            "status": self.status.value,
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at.isoformat(),
        }

    def reindex_steps(self) -> "Plan":
        """根据当前输入和计划的状态计算 `reindex_steps`，并返回调用方需要的结果。

        返回：
            `'Plan'` 类型的处理结果。
        """
        for index, step in enumerate(self.steps, start=1):
            step.id = f"step_{index}"
        return self

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Plan":
        """从字典恢复计划，并为旧数据缺失的字段补充兼容默认值。

        参数：
            data: 用于构造或恢复对象的结构化数据。

        返回：
            `'Plan'` 类型的处理结果。
        """
        steps = [
            PlanStep(
                id=s["id"],
                description=s["description"],
                uid=str(s.get("uid") or uuid4().hex),
                status=StepStatus(s.get("status", "pending")),
                tool_hint=s.get("tool_hint"),
                result_summary=s.get("result_summary"),
                error=s.get("error"),
            )
            for s in data.get("steps", [])
        ]

        return cls(
            task=data["task"],
            steps=steps,
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if data.get("created_at")
                else datetime.now()
            ),
            status=PlanStatus(data.get("status", "planning")),
        ).reindex_steps()


@dataclass
class AgentState:
    """保存一次 Agent 运行及当前计划的可序列化状态。状态变更优先分发给 RuntimeStateStore，以便统一校验修订号、发布事件并持久化。"""

    current_task: str = ""  # 当前任务描述
    mode: Literal["plan", "act"] = "act"  # 工作模式：plan=生成计划，act=直接执行
    iteration: int = 0  # 当前迭代次数，ReAct 循环每轮+1
    is_complete: bool = False  # 任务是否已完成
    requires_confirmation: bool = False  # 是否需要用户确认（如计划审批）
    current_plan: Plan | None = None  # 当前计划对象，plan 模式下生成
    plan_file_path: Path | None = None  # 计划文件保存路径（.opennova/plan/xxx.md）
    plan_approval_status: PlanApprovalStatus = PlanApprovalStatus.NONE  # 计划审批状态
    error_count: int = 0  # 连续错误计数，达到 max_errors 时终止
    max_errors: int = 3  # 最大连续错误数，超过则任务失败
    last_action: str | None = None  # 最后一次工具调用名称
    last_result: str | None = None  # 最后一次工具执行结果摘要
    run_id: str | None = None  # 本次运行唯一标识，用于取消和状态跟踪
    plan_revision: int = 0  # 计划修订号，每次计划变更+1
    _store: "RuntimeStateStore | None" = field(default=None, init=False, repr=False, compare=False)  # 状态存储，用于事件分发和持久化

    def attach_store(self, store: "RuntimeStateStore") -> None:
        """执行 `attach_store` 所定义的协调步骤，必要时更新Agent状态维护的状态。

        参数：
            store: 本次操作使用的存储。
        """
        object.__setattr__(self, "_store", store)

    @property
    def store(self) -> "RuntimeStateStore | None":
        return self._store

    def _dispatch(
        self,
        action_type: str,
        *,
        expected_run_id: str | None = None,
        expected_plan_revision: int | None = None,
        **payload: Any,
    ) -> bool:
        store = self._store
        if store is None:
            return False
        from opennova.runtime.store import RuntimeAction

        store.dispatch(
            RuntimeAction(
                type=action_type,
                payload=payload,
                expected_run_id=expected_run_id,
                expected_plan_revision=expected_plan_revision,
            )
        )
        return True

    def reset(self, task: str = "") -> None:
        """处理重置，并按照当前组件的约定返回结果。

        参数：
            task: 用户希望 Agent 完成的任务描述。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        if self._dispatch("run_started", task=task, preserve_plan=False):
            return
        self.current_task = task
        self.mode = "act"
        self.iteration = 0
        self.is_complete = False
        self.requires_confirmation = False
        self.current_plan = None
        self.plan_file_path = None
        self.plan_approval_status = PlanApprovalStatus.NONE
        self.error_count = 0
        self.last_action = None
        self.last_result = None
        self.run_id = uuid4().hex

    def reset_execution(self, task: str = "") -> None:
        """执行 `reset_execution` 所定义的协调步骤，必要时更新Agent状态维护的状态。

        参数：
            task: 用户希望 Agent 完成的任务描述。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        if self._dispatch("run_started", task=task, preserve_plan=True):
            return
        self.current_task = task
        self.iteration = 0
        self.is_complete = False
        self.requires_confirmation = False
        self.error_count = 0
        self.last_action = None
        self.last_result = None
        self.run_id = uuid4().hex

    def increment_iteration(self, run_id: str | None = None) -> None:
        """执行 `increment_iteration` 所定义的协调步骤，必要时更新Agent状态维护的状态。

        参数：
            run_id: 可选的`run_id`。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        if self._dispatch("run_iteration_incremented", expected_run_id=run_id):
            return
        self.iteration += 1

    def increment_error(self, run_id: str | None = None) -> None:
        """执行 `increment_error` 所定义的协调步骤，必要时更新Agent状态维护的状态。

        参数：
            run_id: 可选的`run_id`。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        if self._dispatch("run_error_incremented", expected_run_id=run_id):
            return
        self.error_count += 1

    def has_too_many_errors(self) -> bool:
        """判断`too_many_errors`条件是否成立。

        返回：
            表示条件是否成立。
        """
        return self.error_count >= self.max_errors

    def mark_complete(self, result: str | None = None, run_id: str | None = None) -> None:
        """把`mark_complete`更新为目标状态，并触发必要的状态事件。

        参数：
            result: 前一步执行得到的规范化结果。
            run_id: 可选的`run_id`。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        if self._dispatch("run_completed", expected_run_id=run_id, result=result, success=True):
            return
        self.is_complete = True
        self.last_result = result

    def finish_run(self, result: str, *, success: bool, run_id: str | None = None) -> None:
        """执行 `finish_run` 所定义的协调步骤，必要时更新Agent状态维护的状态。

        参数：
            result: 前一步执行得到的规范化结果。
            success: 本次操作使用的成功。
            run_id: 可选的`run_id`。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        if self._dispatch(
            "run_completed",
            expected_run_id=run_id,
            result=result,
            success=success,
        ):
            return
        self.is_complete = success
        self.last_result = result

    def cancel_run(self, run_id: str | None = None) -> None:
        """取消运行，并按照当前组件的约定返回结果。

        参数：
            run_id: 可选的`run_id`。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        if self._dispatch("run_cancelled", expected_run_id=run_id):
            return
        if run_id is not None and self.run_id != run_id:
            return
        self.is_complete = False
        self.last_result = "Run cancelled"
        self.run_id = None

    def begin_interaction(self, interaction_type: str) -> None:
        self._dispatch("interaction_waiting", interaction_type=interaction_type)

    def end_interaction(self) -> None:
        self._dispatch("interaction_cleared")

    def set_mode(self, mode: Literal["plan", "act"]) -> None:
        """设置模式并保持相关派生状态同步。

        参数：
            mode: 本次运行采用的工作模式。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        if self._dispatch("mode_changed", mode=mode):
            return
        self.mode = mode

    def set_plan(self, plan: Plan) -> None:
        """设置计划并保持相关派生状态同步。

        参数：
            plan: 当前要保存、展示或执行的结构化计划。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        if self._dispatch("plan_created", plan=plan):
            return
        plan.reindex_steps()
        self.current_plan = plan
        self.mode = "plan"
        self.plan_approval_status = PlanApprovalStatus.DRAFT
        self.requires_confirmation = False

    def set_plan_file_path(self, path: str | Path, file_hash: str | None = None) -> None:
        """设置计划文件路径并保持相关派生状态同步。

        参数：
            path: 需要读取、检查或写入的路径。
            file_hash: 可选的文件摘要。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        if self._dispatch("plan_path_set", path=Path(path), file_hash=file_hash):
            return
        self.plan_file_path = Path(path)

    def mark_plan_awaiting_approval(self) -> None:
        """把`mark_plan_awaiting_approval`更新为目标状态，并触发必要的状态事件。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        if self._dispatch("plan_awaiting_approval"):
            return
        self.mode = "plan"
        self.plan_approval_status = PlanApprovalStatus.AWAITING_APPROVAL
        self.requires_confirmation = True

    def mark_plan_approved(self) -> None:
        """把`mark_plan_approved`更新为目标状态，并触发必要的状态事件。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        if self._dispatch("plan_approved"):
            return
        self.plan_approval_status = PlanApprovalStatus.APPROVED
        self.requires_confirmation = False

    def mark_plan_executing(self) -> None:
        """把`mark_plan_executing`更新为目标状态，并触发必要的状态事件。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        if self._dispatch("plan_executing"):
            return
        self.mode = "act"
        self.plan_approval_status = PlanApprovalStatus.EXECUTING
        self.requires_confirmation = False

    def clear_plan_state(self) -> None:
        """清空计划状态并恢复到可继续使用的初始状态。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        if self._dispatch("plan_cleared"):
            return
        self.current_plan = None
        self.plan_file_path = None
        self.plan_approval_status = PlanApprovalStatus.NONE
        self.requires_confirmation = False

    def mark_plan_failed(self) -> None:
        """把`mark_plan_failed`更新为目标状态，并触发必要的状态事件。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        if self._dispatch("plan_failed"):
            return
        self.mode = "act"
        self.plan_approval_status = PlanApprovalStatus.FAILED
        self.requires_confirmation = False

    def mark_plan_completed(self) -> None:
        """把`mark_plan_completed`更新为目标状态，并触发必要的状态事件。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        if self._dispatch("plan_completed"):
            return
        self.mode = "act"
        self.plan_approval_status = PlanApprovalStatus.COMPLETED
        self.requires_confirmation = False

    def mark_step_running(self, step_id: str) -> None:
        if self._dispatch("plan_step_started", step_id=step_id):
            return
        if self.current_plan:
            self.current_plan.mark_step_running(step_id)

    def mark_step_done(
        self,
        step_id: str,
        result: str | None = None,
        *,
        expected_plan_revision: int | None = None,
    ) -> None:
        if self._dispatch(
            "plan_step_completed",
            expected_plan_revision=expected_plan_revision,
            step_id=step_id,
            result=result,
        ):
            return
        if self.current_plan:
            self.current_plan.mark_step_done(step_id, result)

    def mark_step_failed(
        self,
        step_id: str,
        error: str,
        *,
        expected_plan_revision: int | None = None,
    ) -> None:
        if self._dispatch(
            "plan_step_failed",
            expected_plan_revision=expected_plan_revision,
            step_id=step_id,
            error=error,
        ):
            return
        if self.current_plan:
            self.current_plan.mark_step_failed(step_id, error)

    def requeue_interrupted_plan_steps(self) -> None:
        """执行 `requeue_interrupted_plan_steps` 所定义的协调步骤，必要时更新Agent状态维护的状态。"""
        if self._dispatch("plan_steps_requeued"):
            return
        if self.current_plan:
            for step in self.current_plan.steps:
                if step.status in {
                    StepStatus.RUNNING,
                    StepStatus.FAILED,
                    StepStatus.INTERRUPTED,
                }:
                    step.status = StepStatus.PENDING
                    step.error = None

    def record_action_result(
        self,
        action: str,
        result: str | None,
        *,
        run_id: str | None = None,
    ) -> None:
        """记录动作结果，供状态展示、恢复或后续决策使用。

        参数：
            action: 模型解析出的待执行动作。
            result: 前一步执行得到的规范化结果。
            run_id: 可选的`run_id`。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        if self._dispatch(
            "run_action_recorded", expected_run_id=run_id, action=action, result=result
        ):
            return
        self.last_action = action
        self.last_result = result

    def to_dict(self) -> dict[str, Any]:
        """把Agent状态转换为可序列化字典，供事件、会话或 API 边界使用。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        return {
            "current_task": self.current_task,
            "mode": self.mode,
            "iteration": self.iteration,
            "is_complete": self.is_complete,
            "error_count": self.error_count,
            "requires_confirmation": self.requires_confirmation,
            "plan_approval_status": self.plan_approval_status.value,
            "plan_file_path": str(self.plan_file_path) if self.plan_file_path else None,
            "current_plan": self.current_plan.to_dict() if self.current_plan else None,
            "run_id": self.run_id,
            "plan_revision": self.plan_revision,
        }
