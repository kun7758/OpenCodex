"""终端交互层中的`ask_question_dialog`模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Input, ListItem, ListView, Static

CUSTOM_OPTION_ID = "__custom_text__"
CUSTOM_OPTION_LABEL = "Other / 自定义输入"


def options_with_custom_answer(options: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """读取并返回 `options_with_custom_answer` 所表示的数据或流程，并遵守当前模块定义的边界与状态约束。

    参数：
        options: 本次操作使用的`options`。

    返回：
        按调用约定排序的结果列表。
    """
    normalized = [dict(option) for option in options]
    next_index = max([int(opt.get("index") or 0) for opt in normalized] or [0]) + 1
    normalized.append(
        {
            "index": next_index,
            "label": CUSTOM_OPTION_LABEL,
            "description": "Type your own answer instead of choosing a provided option.",
            "custom": True,
            "id": CUSTOM_OPTION_ID,
        }
    )
    return normalized


def answer_from_selected_option(
    *,
    question: str,
    header: str | None,
    option: dict[str, Any],
    custom_text: str | None = None,
) -> dict[str, Any]:
    """构造并返回 `answer_from_selected_option` 所表示的数据或流程，并遵守当前模块定义的边界与状态约束。

    参数：
        question: 本次操作使用的问题。
        header: 可选的头部元数据。
        option: 本次操作使用的选项。
        custom_text: 可选的`custom_text`。

    返回：
        供后续逻辑或序列化使用的结构化字典。
    """
    if option.get("custom"):
        answer = (custom_text or "").strip()
        return {
            "question": question,
            "answer": answer or None,
            "selected_options": [],
            "custom": True,
            "skipped": not bool(answer),
            "header": header,
        }

    return {
        "question": question,
        "answer": option.get("label"),
        "selected_options": [option],
        "custom": False,
        "skipped": False,
        "header": header,
    }


def answer_from_selected_options(
    *,
    question: str,
    header: str | None,
    options: list[dict[str, Any]],
) -> dict[str, Any]:
    """构造并返回 `answer_from_selected_options` 所表示的数据或流程，并遵守当前模块定义的边界与状态约束。

    参数：
        question: 本次操作使用的问题。
        header: 可选的头部元数据。
        options: 本次操作使用的`options`。

    返回：
        供后续逻辑或序列化使用的结构化字典。
    """
    labels = [option.get("label", "") for option in options]
    return {
        "question": question,
        "answer": ", ".join(labels) if labels else None,
        "selected_options": options,
        "custom": False,
        "skipped": not bool(options),
        "header": header,
    }


class AskQuestionDialog(ModalScreen[dict[str, Any]]):
    """封装`AskQuestionDialog`相关的状态和操作，使调用方通过稳定接口使用该能力。"""

    CSS = """
    AskQuestionDialog {
        align: center middle;
    }

    #ask-card {
        width: 72;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        border: thick $accent;
        background: $surface;
    }

    #ask-title {
        margin-bottom: 1;
    }

    #ask-options {
        height: auto;
        max-height: 12;
        margin-top: 1;
        margin-bottom: 1;
    }

    #ask-custom-input {
        display: none;
        margin-top: 1;
    }

    #ask-custom-input.visible {
        display: block;
    }

    #ask-help {
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(
        self,
        *,
        question: str,
        header: str | None,
        options: list[dict[str, Any]],
        free_text: bool = False,
        multi_select: bool = False,
        allow_custom_answer: bool = True,
        progress_label: str | None = None,
    ) -> None:
        super().__init__()
        self.question = question
        self.header = header
        self.allow_custom_answer = allow_custom_answer or free_text
        if self.allow_custom_answer:
            self.options = options_with_custom_answer([] if free_text else options)
        else:
            self.options = [dict(option) for option in options]
        self.free_text = free_text
        self.multi_select = multi_select
        self.progress_label = progress_label
        self._custom_mode = free_text
        self._selected_indices: set[int] = set()

    def compose(self) -> ComposeResult:
        with Container(id="ask-card"):
            if self.progress_label:
                yield Static(f"[dim]{self.progress_label}[/dim]", markup=True)
            if self.header:
                yield Static(f"[cyan][{self.header}][/cyan]", markup=True)
            yield Static(f"[bold]{self.question}[/bold]", id="ask-title", markup=True)
            yield ListView(
                *[
                    ListItem(
                        Static(self._render_option(option, selected=False), markup=True),
                        id=f"ask-option-{i}",
                    )
                    for i, option in enumerate(self.options)
                ],
                id="ask-options",
            )
            yield Input(
                placeholder="Type your answer, then press Enter...",
                id="ask-custom-input",
            )
            yield Static(
                self._help_text(),
                id="ask-help",
            )

    def on_mount(self) -> None:
        if self.free_text:
            self._show_custom_input()
        else:
            self.query_one("#ask-options", ListView).focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        option = self.options[event.index]
        if option.get("custom"):
            self._show_custom_input()
            return
        if self.multi_select:
            if not self._selected_indices:
                self._selected_indices.add(event.index)
            self.dismiss(
                answer_from_selected_options(
                    question=self.question,
                    header=self.header,
                    options=[
                        self.options[index]
                        for index in sorted(self._selected_indices)
                        if not self.options[index].get("custom")
                    ],
                )
            )
            return
        self.dismiss(
            answer_from_selected_option(
                question=self.question,
                header=self.header,
                option=option,
            )
        )

    def on_key(self, event) -> None:
        if not self.multi_select or event.key != "space":
            return
        event.stop()
        options_view = self.query_one("#ask-options", ListView)
        index = options_view.index
        if index is None:
            return
        option = self.options[index]
        if option.get("custom"):
            self._show_custom_input()
            return
        if index in self._selected_indices:
            self._selected_indices.remove(index)
        else:
            self._selected_indices.add(index)
        self._refresh_option(index)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "ask-custom-input":
            return
        event.stop()
        custom_option = self.options[-1]
        self.dismiss(
            answer_from_selected_option(
                question=self.question,
                header=self.header,
                option=custom_option,
                custom_text=event.value,
            )
        )

    def action_cancel(self) -> None:
        self.dismiss(
            {
                "question": self.question,
                "answer": None,
                "selected_options": [],
                "custom": False,
                "skipped": True,
                "header": self.header,
            }
        )

    def _show_custom_input(self) -> None:
        self._custom_mode = True
        options = self.query_one("#ask-options", ListView)
        custom_input = self.query_one("#ask-custom-input", Input)
        options.display = False
        custom_input.add_class("visible")
        custom_input.value = ""
        custom_input.focus()

    def _refresh_option(self, index: int) -> None:
        item = self.query_one("#ask-options", ListView).children[index]
        item.query_one(Static).update(
            self._render_option(self.options[index], selected=index in self._selected_indices)
        )

    def _help_text(self) -> str:
        if self.multi_select:
            return "↑/↓ choose · Space select · Enter confirm · Esc skip"
        return "↑/↓ choose · Enter confirm · Esc skip"

    def _render_option(self, option: dict[str, Any], *, selected: bool) -> str:
        label = option.get("label", "")
        description = option.get("description", "")
        if option.get("custom"):
            prefix = "[cyan]+[/cyan]"
        elif self.multi_select:
            prefix = "[green][x][/green]" if selected else "[dim][ ][/dim]"
        else:
            prefix = f"[yellow]{option.get('index')}[/yellow]"
        if description:
            return f"{prefix} {label}\n[dim]{description}[/dim]"
        return f"{prefix} {label}"
