"""终端交互层中的`tool_cards`模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from opennova.runtime.events import ToolEvent


@dataclass
class ToolCard:
    """保存工具工具卡片所需的结构化数据，主要包含
    `tool_id`、`tool_name`、`status`、`output_preview`、`error`、`diff`、`collapsible`、`permission_reason`
    等字段，便于在组件之间传递或持久化。
    """

    tool_id: str
    tool_name: str
    status: str = "running"
    output_preview: str = ""
    error: str | None = None
    diff: str | None = None
    collapsible: bool = False
    permission_reason: str = ""
    cancelled: bool = False
    metadata: dict[str, Any] | None = None


@dataclass
class ToolCardViewState:
    """数据对象 `ToolCardViewState` 主要保存
    `tool_id`、`tool_name`、`status`、`expanded`、`rendered`、`diff_panel`、`approval_state`、`cancelled`
    字段，用于在组件之间传递或持久化这组状态。
    """

    tool_id: str
    tool_name: str
    status: str
    expanded: bool
    rendered: str
    diff_panel: str = ""
    approval_state: str = "none"
    cancelled: bool = False


@dataclass
class ToolCardPanelState:
    """保存工具工具卡片面板状态所需的结构化数据，主要包含 `cards`、`selected_tool_id`、`diff_panel`、`actions`
    字段，便于在组件之间传递或持久化。
    """

    cards: list[ToolCardViewState]
    selected_tool_id: str | None
    diff_panel: str = ""
    actions: dict[str, bool] | None = None


@dataclass
class ToolCardInteractionState:
    """保存工具工具卡片用户交互状态所需的结构化数据，主要包含 `selected_tool_id`、`expanded_tool_ids`、`approval_states`
    字段，便于在组件之间传递或持久化。
    """

    selected_tool_id: str | None = None
    expanded_tool_ids: set[str] | None = None
    approval_states: dict[str, str] | None = None


class ToolCardStore:
    """负责工具工具卡片存储的保存、读取和一致性维护，并隐藏具体持久化格式。"""

    def __init__(self, collapse_threshold: int = 1200):
        self.collapse_threshold = collapse_threshold
        self.cards: dict[str, ToolCard] = {}
        self.interaction = ToolCardInteractionState(
            expanded_tool_ids=set(),
            approval_states={},
        )

    def apply_event(self, event: ToolEvent) -> ToolCard:
        card = self.cards.get(event.tool_id)
        if card is None:
            card = ToolCard(tool_id=event.tool_id, tool_name=event.tool_name, metadata={})
            self.cards[event.tool_id] = card
            if self.interaction.selected_tool_id is None:
                self.interaction.selected_tool_id = event.tool_id

        if event.type == "tool_start":
            card.status = "running"
        elif event.type == "permission_request":
            card.status = "waiting_for_permission"
            card.permission_reason = str(event.metadata.get("reason", ""))
        elif event.type in {"tool_result", "tool_error"}:
            card.status = "succeeded" if event.success else "failed"
            card.error = event.error
            card.diff = event.diff
            output = event.output or ""
            card.metadata = {**(card.metadata or {}), "full_output": output}
            card.collapsible = len(output) > self.collapse_threshold
            card.output_preview = (
                output[: self.collapse_threshold] + "\n... (output collapsed)"
                if card.collapsible
                else output
            )
        elif event.type == "tool_cancelled":
            card.status = "cancelled"
            card.cancelled = True

        if event.duration_ms is not None:
            card.metadata = {**(card.metadata or {}), "duration_ms": event.duration_ms}
        card.metadata = {**(card.metadata or {}), **event.metadata}
        return card

    def cancel(self, tool_id: str) -> ToolCard:
        card = self.cards[tool_id]
        card.status = "cancelled"
        card.cancelled = True
        return card

    def get(self, tool_id: str) -> ToolCard:
        return self.cards[tool_id]

    def select_next(self) -> str | None:
        """选择下一个，并按照当前组件的约定返回结果。

        返回：
            `str | None` 类型的处理结果。
        """
        ids = list(self.cards)
        if not ids:
            self.interaction.selected_tool_id = None
            return None
        if self.interaction.selected_tool_id not in ids:
            self.interaction.selected_tool_id = ids[0]
            return ids[0]
        index = ids.index(self.interaction.selected_tool_id)
        self.interaction.selected_tool_id = ids[(index + 1) % len(ids)]
        return self.interaction.selected_tool_id

    def select_previous(self) -> str | None:
        """选择上一个，并按照当前组件的约定返回结果。

        返回：
            `str | None` 类型的处理结果。
        """
        ids = list(self.cards)
        if not ids:
            self.interaction.selected_tool_id = None
            return None
        if self.interaction.selected_tool_id not in ids:
            self.interaction.selected_tool_id = ids[0]
            return ids[0]
        index = ids.index(self.interaction.selected_tool_id)
        self.interaction.selected_tool_id = ids[(index - 1) % len(ids)]
        return self.interaction.selected_tool_id

    def toggle_expanded(self, tool_id: str | None = None) -> bool:
        """切换展开状态，并按照当前组件的约定返回结果。

        参数：
            tool_id: 可选的`tool_id`。

        返回：
            表示条件是否成立。
        """
        target = tool_id or self.interaction.selected_tool_id
        if not target:
            return False
        if self.interaction.expanded_tool_ids is None:
            self.interaction.expanded_tool_ids = set()
        expanded = self.interaction.expanded_tool_ids
        if target in expanded:
            expanded.remove(target)
            return False
        expanded.add(target)
        return True

    def apply_approval(self, tool_id: str, state: str) -> None:
        """应用审批，并按照当前组件的约定返回结果。

        参数：
            tool_id: 本次操作使用的`tool_id`。
            state: 当前 Agent 或计划状态。
        """
        if self.interaction.approval_states is None:
            self.interaction.approval_states = {}
        approvals = self.interaction.approval_states
        approvals[tool_id] = state


def render_tool_card(card: ToolCard) -> str:
    """根据当前数据渲染工具工具卡片的界面或文本表示。

    参数：
        card: 本次操作使用的工具卡片。

    返回：
        处理后的文本或稳定标识。
    """
    header = f"[{card.status}] {card.tool_name} ({card.tool_id})"
    details: list[str] = []
    metadata = card.metadata or {}
    if "duration_ms" in metadata:
        details.append(f"duration={metadata['duration_ms']}ms")
    if card.cancelled:
        details.append("cancelled=yes")
    if card.collapsible:
        details.append("collapsible=yes")
    if details:
        header = f"{header} {' '.join(details)}"

    parts = [header]
    if card.permission_reason:
        parts.append(f"permission: {card.permission_reason}")
    if card.output_preview:
        parts.append(card.output_preview)
    if card.diff:
        parts.append(card.diff.rstrip())
    if card.error:
        parts.append(f"error: {card.error}")
    return "\n".join(parts)


def render_tool_cards(store: ToolCardStore) -> str:
    """根据当前数据渲染`render_tool_cards`的界面或文本表示。

    参数：
        store: 本次操作使用的存储。

    返回：
        处理后的文本或稳定标识。
    """
    return "\n\n".join(render_tool_card(card) for card in store.cards.values())


def build_tool_card_view(card: ToolCard, expanded: bool = False) -> ToolCardViewState:
    """根据当前输入和状态构造`build_tool_card_view`。

    参数：
        card: 本次操作使用的工具卡片。
        expanded: 可选的展开状态。

    返回：
        `ToolCardViewState` 类型的处理结果。
    """
    output = card.output_preview
    if expanded and card.metadata and "full_output" in card.metadata:
        output = str(card.metadata["full_output"])

    preview_card = ToolCard(
        tool_id=card.tool_id,
        tool_name=card.tool_name,
        status=card.status,
        output_preview=output,
        error=card.error,
        diff=card.diff,
        collapsible=card.collapsible,
        permission_reason=card.permission_reason,
        cancelled=card.cancelled,
        metadata=card.metadata,
    )
    approval_state = "requested" if card.permission_reason else "none"
    return ToolCardViewState(
        tool_id=card.tool_id,
        tool_name=card.tool_name,
        status=card.status,
        expanded=expanded,
        rendered=render_tool_card(preview_card),
        diff_panel=(card.diff or "").rstrip(),
        approval_state=approval_state,
        cancelled=card.cancelled,
    )


def build_tool_card_panel(
    store: ToolCardStore,
    selected_tool_id: str | None = None,
    expanded: bool = False,
) -> ToolCardPanelState:
    """根据当前输入和状态构造`build_tool_card_panel`。

    参数：
        store: 本次操作使用的存储。
        selected_tool_id: 可选的`selected_tool_id`。
        expanded: 可选的展开状态。

    返回：
        `ToolCardPanelState` 类型的处理结果。
    """
    selected_tool_id = selected_tool_id or store.interaction.selected_tool_id or next(iter(store.cards), None)
    views: list[ToolCardViewState] = []
    selected_card: ToolCard | None = None
    for tool_id, card in store.cards.items():
        is_selected = tool_id == selected_tool_id
        if is_selected:
            selected_card = card
        is_expanded = expanded and is_selected
        is_expanded = is_expanded or tool_id in (store.interaction.expanded_tool_ids or set())
        view = build_tool_card_view(card, expanded=is_expanded)
        approval_state = (store.interaction.approval_states or {}).get(tool_id)
        if approval_state:
            view.approval_state = approval_state
        views.append(view)

    actions = {
        "approve": bool(selected_card and selected_card.permission_reason),
        "cancel": bool(selected_card and selected_card.status in {"running", "waiting_for_permission"}),
        "toggle": bool(selected_card and selected_card.collapsible),
    }
    return ToolCardPanelState(
        cards=views,
        selected_tool_id=selected_tool_id,
        diff_panel=(selected_card.diff or "").rstrip() if selected_card else "",
        actions=actions,
    )


def apply_tool_card_key(store: ToolCardStore, key: str) -> str:
    """应用 `tool_card_key` 对应的数据，并按照当前组件的约定返回结果。

    参数：
        store: 本次操作使用的存储。
        key: 本次操作使用的`key`。

    返回：
        处理后的文本或稳定标识。
    """
    selected = store.interaction.selected_tool_id
    if key in {"j", "down"}:
        selected = store.select_next()
        return f"selected:{selected}" if selected else "selected:none"
    if key in {"k", "up"}:
        selected = store.select_previous()
        return f"selected:{selected}" if selected else "selected:none"
    if key in {"enter", "space"}:
        expanded = store.toggle_expanded(selected)
        return f"{'expanded' if expanded else 'collapsed'}:{selected}"
    if key == "a" and selected:
        store.apply_approval(selected, "approved")
        return f"approval:{selected}:approved"
    if key == "d" and selected:
        store.apply_approval(selected, "denied")
        return f"approval:{selected}:denied"
    if key == "c" and selected:
        store.cancel(selected)
        return f"cancelled:{selected}"
    return "ignored"


def tool_card_key_bindings() -> list[dict[str, str]]:
    """读取并返回 `tool_card_key_bindings` 所表示的数据或流程，并遵守当前模块定义的边界与状态约束。

    返回：
        按调用约定排序的结果列表。
    """
    return [
        {"key": "j", "action": "select_next", "description": "Select next tool card"},
        {"key": "k", "action": "select_previous", "description": "Select previous tool card"},
        {"key": "enter", "action": "toggle_expanded", "description": "Expand or collapse selected card"},
        {"key": "a", "action": "approve", "description": "Approve selected permission request"},
        {"key": "d", "action": "deny", "description": "Deny selected permission request"},
        {"key": "c", "action": "cancel", "description": "Cancel selected tool"},
    ]


def build_tool_card_binding_plan(store: ToolCardStore) -> list[dict[str, object]]:
    """根据当前输入和状态构造`build_tool_card_binding_plan`。

    参数：
        store: 本次操作使用的存储。

    返回：
        按调用约定排序的结果列表。
    """
    panel = build_tool_card_panel(store)
    actions = panel.actions or {}
    always_enabled = {"select_next", "select_previous"}
    action_map = {
        "approve": "approve",
        "deny": "approve",
        "cancel": "cancel",
        "toggle_expanded": "toggle",
    }
    disabled_reasons = {
        "approve": "selected tool has no pending permission request",
        "deny": "selected tool has no pending permission request",
        "cancel": "selected tool is not running",
        "toggle_expanded": "selected tool output is not collapsible",
    }
    plan: list[dict[str, object]] = []
    for binding in tool_card_key_bindings():
        action = binding["action"]
        enabled = action in always_enabled or bool(actions.get(action_map.get(action, action)))
        item: dict[str, object] = dict(binding)
        item["enabled"] = enabled
        item["disabled_reason"] = "" if enabled else disabled_reasons.get(action, "action unavailable")
        plan.append(item)
    return plan


def render_tool_card_binding_help(store: ToolCardStore) -> str:
    """根据当前数据渲染`render_tool_card_binding_help`的界面或文本表示。

    参数：
        store: 本次操作使用的存储。

    返回：
        处理后的文本或稳定标识。
    """
    lines: list[str] = []
    for item in build_tool_card_binding_plan(store):
        status = "enabled" if item["enabled"] else f"disabled: {item['disabled_reason']}"
        lines.append(f"{item['key']} {item['action']} {status}")
    return "\n".join(lines)
