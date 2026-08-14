"""终端交互层中的终端界面工作台模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from opennova.cli.tool_cards import ToolCardPanelState, ToolCardStore, build_tool_card_panel
from opennova.tools.todo_tools import TodoWriteTool

WorkbenchTab = Literal["context", "tasks", "activity"]
WORKBENCH_TABS: tuple[WorkbenchTab, ...] = ("context", "tasks", "activity")
LEGACY_WORKBENCH_TABS = {
    "tools": "activity",
    "plan": "tasks",
    "todos": "tasks",
}


@dataclass(frozen=True)
class ActiveFileSnapshot:
    """数据对象 `ActiveFileSnapshot` 主要保存 `path`、`activity` 字段，用于在组件之间传递或持久化这组状态。"""

    path: str
    activity: str


@dataclass(frozen=True)
class ContextWorkbenchSnapshot:
    """保存上下文工作台快照所需的结构化数据，主要包含
    `task`、`run_phase`、`current_step`、`total_messages`、`total_tokens`、`context_window`、`utilization_percent`、`compression_count`
    等字段，便于在组件之间传递或持久化。
    """

    task: str = ""
    run_phase: str = "idle"
    current_step: str = ""
    total_messages: int = 0
    total_tokens: int = 0
    context_window: int = 0
    utilization_percent: float = 0.0
    compression_count: int = 0
    has_compressed_summary: bool = False
    compression_threshold_percent: float = 55.0
    active_files: tuple[ActiveFileSnapshot, ...] = ()
    recent_decisions: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()


@dataclass
class PlanStepSnapshot:
    """保存计划步骤快照所需的结构化数据，主要包含 `id`、`description`、`status`、`result_summary`、`error` 字段，便于在组件之间传递或持久化。"""

    id: str
    description: str
    status: str
    result_summary: str = ""
    error: str = ""


@dataclass
class PlanWorkbenchSnapshot:
    """保存计划工作台快照所需的结构化数据，主要包含 `task`、`status`、`approval_status`、`plan_file_path`、`steps`
    字段，便于在组件之间传递或持久化。
    """

    task: str
    status: str
    approval_status: str
    plan_file_path: str = ""
    steps: list[PlanStepSnapshot] = field(default_factory=list)


@dataclass(frozen=True)
class TaskWorkbenchSnapshot:
    """保存任务工作台快照所需的结构化数据，主要包含 `plan`、`todos`、`completed`、`total`、`current_item`、`status_counts`
    字段，便于在组件之间传递或持久化。
    """

    plan: PlanWorkbenchSnapshot | None
    todos: tuple[dict[str, Any], ...]
    completed: int = 0
    total: int = 0
    current_item: str = ""
    status_counts: tuple[tuple[str, int], ...] = ()


@dataclass
class WorkbenchPanelState:
    """保存工作台面板状态所需的结构化数据，主要包含 `active_tab`、`tools`、`plan`、`todos`、`context`、`tasks`、`key_hint`
    字段，便于在组件之间传递或持久化。
    """

    active_tab: WorkbenchTab
    tools: ToolCardPanelState
    plan: PlanWorkbenchSnapshot | None
    todos: list[dict[str, Any]]
    context: ContextWorkbenchSnapshot | None = None
    tasks: TaskWorkbenchSnapshot | None = None
    key_hint: str = "alt+1 context  alt+2 tasks  alt+3 activity  alt+t hide"

    @property
    def activity(self) -> ToolCardPanelState:
        """处理活动记录，并按照当前组件的约定返回结果。

        返回：
            `ToolCardPanelState` 类型的处理结果。
        """
        return self.tools


def normalize_workbench_tab(tab: str) -> WorkbenchTab:
    """规范化`normalize_workbench_tab`，消除不同调用格式之间的差异。

    参数：
        tab: 本次操作使用的`tab`。

    返回：
        `WorkbenchTab` 类型的处理结果。
    """
    normalized = LEGACY_WORKBENCH_TABS.get(tab, tab)
    return cast(WorkbenchTab, normalized if normalized in WORKBENCH_TABS else "context")


def next_workbench_tab(tab: WorkbenchTab | str) -> WorkbenchTab:
    """读取并返回 `next_workbench_tab` 所表示的数据或流程，并遵守当前模块定义的边界与状态约束。

    参数：
        tab: 本次操作使用的`tab`。

    返回：
        `WorkbenchTab` 类型的处理结果。
    """
    current = normalize_workbench_tab(tab)
    index = WORKBENCH_TABS.index(current)
    return WORKBENCH_TABS[(index + 1) % len(WORKBENCH_TABS)]


def previous_workbench_tab(tab: WorkbenchTab | str) -> WorkbenchTab:
    """读取并返回 `previous_workbench_tab` 所表示的数据或流程，并遵守当前模块定义的边界与状态约束。

    参数：
        tab: 本次操作使用的`tab`。

    返回：
        `WorkbenchTab` 类型的处理结果。
    """
    current = normalize_workbench_tab(tab)
    index = WORKBENCH_TABS.index(current)
    return WORKBENCH_TABS[(index - 1) % len(WORKBENCH_TABS)]


def build_workbench_panel_state(
    *,
    agent: Any,
    tool_cards: ToolCardStore,
    active_tab: WorkbenchTab,
    last_plan: PlanWorkbenchSnapshot | None = None,
) -> WorkbenchPanelState:
    """根据当前输入和状态构造`build_workbench_panel_state`。

    参数：
        agent: 本次操作使用的Agent。
        tool_cards: 本次操作使用的`tool_cards`。
        active_tab: 本次操作使用的`active_tab`。
        last_plan: 可选的`last_plan`。

    返回：
        `WorkbenchPanelState` 类型的处理结果。
    """
    plan = _snapshot_plan(agent) or last_plan
    todos = TodoWriteTool.current_todos(getattr(agent, "state_store", None))
    return WorkbenchPanelState(
        active_tab=normalize_workbench_tab(active_tab),
        tools=build_tool_card_panel(tool_cards),
        plan=plan,
        todos=todos,
        context=_snapshot_context(agent, plan),
        tasks=snapshot_tasks(plan, todos),
    )


def _snapshot_context(
    agent: Any,
    plan: PlanWorkbenchSnapshot | None,
) -> ContextWorkbenchSnapshot:
    context_manager = getattr(agent, "context_manager", None)
    presentation = (
        context_manager.get_presentation_snapshot()
        if context_manager and hasattr(context_manager, "get_presentation_snapshot")
        else None
    )
    working_memory = getattr(agent, "working_memory", None)
    state = getattr(agent, "state", None)
    task = str(
        getattr(getattr(working_memory, "task_state", None), "description", "")
        or getattr(state, "current_task", "")
        or (plan.task if plan else "")
    )
    run_phase = _runtime_phase(agent, working_memory)
    current_step = ""
    if plan:
        active = next(
            (
                step
                for step in plan.steps
                if step.status in {"running", "in_progress", "executing", "interrupted"}
            ),
            None,
        )
        if active:
            current_step = f"{active.id} · {active.description}"

    active_files = _active_files(working_memory)
    decisions = tuple(str(item) for item in (getattr(working_memory, "decisions", []) or [])[-5:])
    sources: list[str] = []
    if presentation and presentation.total_messages:
        sources.append(f"conversation · {presentation.total_messages} messages")
    if getattr(context_manager, "system_prompt", None):
        sources.append("system instructions")
    if plan:
        sources.append(f"plan · {len(plan.steps)} steps")
    if presentation and presentation.has_compressed_summary:
        sources.append("compressed summary")
    if active_files:
        sources.append(f"working files · {len(active_files)}")
    if getattr(agent, "project_memory", None) is not None:
        sources.append("project memory")

    return ContextWorkbenchSnapshot(
        task=task,
        run_phase=run_phase,
        current_step=current_step,
        total_messages=int(getattr(presentation, "total_messages", 0)),
        total_tokens=int(getattr(presentation, "total_tokens", 0)),
        context_window=int(getattr(presentation, "context_window", 0)),
        utilization_percent=float(getattr(presentation, "utilization_percent", 0.0)),
        compression_count=int(getattr(presentation, "compression_count", 0)),
        has_compressed_summary=bool(getattr(presentation, "has_compressed_summary", False)),
        compression_threshold_percent=float(
            getattr(presentation, "compression_threshold_percent", 55.0)
        ),
        active_files=active_files,
        recent_decisions=decisions,
        sources=tuple(sources),
    )


def _runtime_phase(agent: Any, working_memory: Any) -> str:
    store = getattr(agent, "state_store", None)
    try:
        phase = store.get_state().run.phase
        return str(getattr(phase, "value", phase))
    except Exception:
        return str(getattr(getattr(working_memory, "task_state", None), "status", "idle"))


def _active_files(working_memory: Any, limit: int = 8) -> tuple[ActiveFileSnapshot, ...]:
    observations = list(getattr(working_memory, "observations", []) or [])
    latest: dict[str, str] = {}
    ordered: list[str] = []
    for observation in reversed(observations):
        path = str(getattr(observation, "file_path", "") or "")
        if not path or path in latest:
            continue
        latest[path] = str(getattr(observation, "change_type", "observed") or "observed")
        ordered.append(path)
        if len(ordered) >= limit:
            break
    return tuple(ActiveFileSnapshot(path=path, activity=latest[path]) for path in ordered)


def snapshot_tasks(
    plan: PlanWorkbenchSnapshot | None,
    todos: list[dict[str, Any]],
) -> TaskWorkbenchSnapshot:
    items: list[tuple[str, str]] = []
    if plan and plan.steps:
        items = [(step.status, step.description) for step in plan.steps]
        plan_ids = {str(getattr(step, "id", "")) for step in plan.steps}
        items.extend(
            (str(todo.get("status", "pending")), str(todo.get("content", "")))
            for todo in todos
            if todo.get("source") != "plan" and str(todo.get("id", "")) not in plan_ids
        )
    else:
        items = [
            (str(todo.get("status", "pending")), str(todo.get("content", "")))
            for todo in todos
        ]
    counts: dict[str, int] = {}
    for status, _ in items:
        counts[status] = counts.get(status, 0) + 1
    completed = sum(counts.get(status, 0) for status in ("done", "completed", "skipped"))
    current = next(
        (
            content
            for status, content in items
            if status in {"running", "in_progress", "executing", "interrupted"}
        ),
        "",
    )
    return TaskWorkbenchSnapshot(
        plan=plan,
        todos=tuple(dict(todo) for todo in todos),
        completed=completed,
        total=len(items),
        current_item=current,
        status_counts=tuple(sorted(counts.items())),
    )


def _snapshot_plan(agent: Any) -> PlanWorkbenchSnapshot | None:
    state = getattr(agent, "state", None)
    plan = getattr(state, "current_plan", None)
    if plan is None:
        return None

    approval = getattr(getattr(state, "plan_approval_status", None), "value", None)
    plan_path = getattr(state, "plan_file_path", None)
    return snapshot_plan(plan, plan_file_path=plan_path, approval_status=approval)


def snapshot_plan(
    plan: Any,
    *,
    plan_file_path: str | Path | None = None,
    approval_status: str | None = None,
) -> PlanWorkbenchSnapshot:
    """构造并返回 `snapshot_plan` 所表示的数据或流程，并遵守当前模块定义的边界与状态约束。

    参数：
        plan: 当前要保存、展示或执行的结构化计划。
        plan_file_path: 可选的计划文件路径。
        approval_status: 可选的审批状态。

    返回：
        `PlanWorkbenchSnapshot` 类型的处理结果。
    """
    steps: list[PlanStepSnapshot] = []
    for step in getattr(plan, "steps", []) or []:
        steps.append(
            PlanStepSnapshot(
                id=str(getattr(step, "id", "")),
                description=str(getattr(step, "description", "")),
                status=str(getattr(getattr(step, "status", None), "value", getattr(step, "status", ""))),
                result_summary=str(getattr(step, "result_summary", "") or ""),
                error=str(getattr(step, "error", "") or ""),
            )
        )

    return PlanWorkbenchSnapshot(
        task=str(getattr(plan, "task", "") or "(untitled plan)"),
        status=str(getattr(getattr(plan, "status", None), "value", getattr(plan, "status", "planning"))),
        approval_status=str(approval_status or "none"),
        plan_file_path=str(Path(plan_file_path)) if plan_file_path else "",
        steps=steps,
    )
