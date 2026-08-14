"""终端交互层中的渲染器模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

from opennova.diff.parser import ChangeType
from opennova.providers.base import StreamChunk
from opennova.runtime.state import Plan
from opennova.tools.base import ToolResult


class Renderer:
    """封装渲染器相关的状态和操作，使调用方通过稳定接口使用该能力。"""

    def __init__(self, console: Console | None = None):
        """初始化渲染器，保存后续操作需要的依赖、配置和初始状态。

        参数：
            console: 可选的控制台。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self.console = console or Console(
            force_terminal=True,
            soft_wrap=False,  # 关闭软换行，让较长输出保持终端自身的横向与纵向滚动行为。
            markup=True,
            highlight=True,
        )

    def print(self, message: Any = "", **kwargs) -> None:
        """执行 `print` 所定义的协调步骤，必要时更新渲染器维护的状态。

        参数：
            message: 用户提交或组件间传递的消息。
            **kwargs: 传递给底层实现的额外关键字参数。
        """
        self.console.print(message, **kwargs)

    def print_welcome(self) -> None:
        """读取并返回 `print_welcome` 所表示的数据或流程，并遵守渲染器定义的边界与状态约束。"""
        self.console.print(
            Panel.fit(
                "[bold cyan]OpenNova[/bold cyan] - AI Coding Agent\n\n"
                "[dim]A terminal AI coding agent with a Textual TUI[/dim]\n\n"
                "Type [bold]/help[/bold] for commands, [bold]Ctrl+D[/bold] to exit",
                title="🌟 Welcome",
                border_style="blue",
            )
        )

    def print_help(self) -> None:
        """读取并返回 `print_help` 所表示的数据或流程，并遵守渲染器定义的边界与状态约束。"""
        help_text = """
## Commands

| Command | Description |
|---------|-------------|
| `/plan <task>` | Generate a plan before executing |
| `/act <task>` | Execute directly (default mode) |
| `/tools` | List available tools |
| `/model` | Show current model info |
| `/config` | Show current configuration |
| `/history` | Show conversation history |
| `/clear` | Clear conversation (starts new session) |
| `/resume [id]` | Resume a past session |
| `/sessions` | List all saved sessions |
| `/help` | Show this help message |
| `/exit` / `/quit` | Exit OpenNova |

## Tips

- Use `Tab` for auto-suggestions
- Use `↑/↓` for history navigation
- Multi-line input is supported
"""
        self.console.print(Markdown(help_text))

    def print_thinking(self, thought: str, collapsed: bool = False) -> None:
        """读取并返回 `print_thinking` 所表示的数据或流程，并遵守渲染器定义的边界与状态约束。

        参数：
            thought: 本次操作使用的`thought`。
            collapsed: 可选的`collapsed`。
        """
        preview = thought[:200] + "..." if collapsed and len(thought) > 200 else thought

        self.console.print(
            Panel(
                preview,
                title="💭 Thinking",
                border_style="yellow",
                expand=False,
            )
        )

    def print_action(self, tool_name: str, args: dict[str, Any]) -> None:
        """读取并返回 `print_action` 所表示的数据或流程，并遵守渲染器定义的边界与状态约束。

        参数：
            tool_name: 目标工具在注册表中的名称。
            args: 调用方传入的位置参数或 Skill 参数文本。
        """
        redacted = {"content"}
        args_preview = []
        for k, v in args.items():
            if k in redacted:
                if isinstance(v, str):
                    args_preview.append(f"[dim]{k}[/dim]=<{len(v)} chars>")
                else:
                    args_preview.append(f"[dim]{k}[/dim]=<redacted>")
            else:
                v_str = str(v)
                if len(v_str) > 50:
                    v_str = v_str[:47] + "..."
                args_preview.append(f"[dim]{k}[/dim]={repr(v_str)}")

        args_str = ", ".join(args_preview)

        self.console.print(
            Panel(
                f"[bold cyan]{tool_name}[/bold cyan]({args_str})",
                title="⚙️ Tool Call",
                border_style="blue",
                expand=False,
            )
        )

    def print_result(self, result: ToolResult, max_lines: int = 20) -> None:
        """读取并返回 `print_result` 所表示的数据或流程，并遵守渲染器定义的边界与状态约束。

        参数：
            result: 前一步执行得到的规范化结果。
            max_lines: 可选的`max_lines`。
        """
        if result.success:
            style = "green"
            icon = "✅"
            title = "Success"
        else:
            style = "red"
            icon = "❌"
            title = "Error"

        output = result.output or ""

        lines = output.split("\n")
        if len(lines) > max_lines:
            output = "\n".join(lines[:max_lines]) + f"\n\n... [truncated, {len(lines) - max_lines} more lines]"

        self.console.print(
            Panel(
                output,
                title=f"{icon} {title}",
                border_style=style,
                expand=False,
            )
        )

        if result.error:
            self.console.print(f"[red bold]Error:[/red bold] {result.error}")

    def print_stream(self, chunk: StreamChunk) -> None:
        """读取并返回 `print_stream` 所表示的数据或流程，并遵守渲染器定义的边界与状态约束。

        参数：
            chunk: 本次操作使用的流式片段。
        """
        if chunk.content:
            self.console.print(chunk.content, end="", markup=False)
        if chunk.finish_reason:
            self.console.print()

    def print_plan(self, plan: Plan, show_progress: bool = True) -> None:
        """读取并返回 `print_plan` 所表示的数据或流程，并遵守渲染器定义的边界与状态约束。

        参数：
            plan: 当前要保存、展示或执行的结构化计划。
            show_progress: 可选的`show_progress`。
        """
        tree = Tree(f"📋 [bold]{plan.task}[/bold]")

        status_icons = {
            "pending": ("⏳", "yellow"),
            "running": ("🔄", "blue"),
            "done": ("✅", "green"),
            "failed": ("❌", "red"),
            "skipped": ("⏭️", "dim"),
        }

        for step in plan.steps:
            icon, color = status_icons.get(step.status.value, ("❓", "white"))
            step_text = f"[{color}]{icon}[/{color}] [{step.id}] {step.description}"

            if step.tool_hint:
                step_text += f" [dim](tool: {step.tool_hint})[/dim]"

            tree.add(step_text)

        self.console.print(tree)

        if show_progress:
            done = sum(1 for s in plan.steps if s.status.value == "done")
            total = len(plan.steps)
            self.console.print(f"\n[dim]Progress: {done}/{total} steps completed[/dim]")

    def print_tools(self, tools: list[str], descriptions: dict[str, str] | None = None) -> None:
        """读取并返回 `print_tools` 所表示的数据或流程，并遵守渲染器定义的边界与状态约束。

        参数：
            tools: 本次操作使用的工具。
            descriptions: 可选的`descriptions`。
        """
        table = Table(title="🛠️ Available Tools", show_header=True)
        table.add_column("Tool", style="cyan")
        table.add_column("Description")

        for tool in sorted(tools):
            desc = descriptions.get(tool, "") if descriptions else ""
            table.add_row(tool, desc)

        self.console.print(table)

    def print_error(self, message: str, title: str = "Error") -> None:
        """读取并返回 `print_error` 所表示的数据或流程，并遵守渲染器定义的边界与状态约束。

        参数：
            message: 用户提交或组件间传递的消息。
            title: 可选的`title`。
        """
        self.console.print(
            Panel(
                message,
                title=f"❌ {title}",
                border_style="red",
            )
        )

    def print_success(self, message: str, title: str = "Success") -> None:
        """读取并返回 `print_success` 所表示的数据或流程，并遵守渲染器定义的边界与状态约束。

        参数：
            message: 用户提交或组件间传递的消息。
            title: 可选的`title`。
        """
        self.console.print(
            Panel(
                message,
                title=f"✅ {title}",
                border_style="green",
            )
        )

    def print_warning(self, message: str, title: str = "Warning") -> None:
        """读取并返回 `print_warning` 所表示的数据或流程，并遵守渲染器定义的边界与状态约束。

        参数：
            message: 用户提交或组件间传递的消息。
            title: 可选的`title`。
        """
        self.console.print(
            Panel(
                message,
                title=f"⚠️ {title}",
                border_style="yellow",
            )
        )

    def print_info(self, message: str, title: str = "Info") -> None:
        """读取并返回 `print_info` 所表示的数据或流程，并遵守渲染器定义的边界与状态约束。

        参数：
            message: 用户提交或组件间传递的消息。
            title: 可选的`title`。
        """
        self.console.print(
            Panel(
                message,
                title=f"ℹ️ {title}",
                border_style="blue",
            )
        )

    def print_markdown(self, text: str) -> None:
        """构造并返回 `print_markdown` 所表示的数据或流程，并遵守渲染器定义的边界与状态约束。

        参数：
            text: 需要解析、格式化或展示的文本。
        """
        self.console.print(Markdown(text))

    def print_code(
        self,
        code: str,
        language: str = "python",
        line_numbers: bool = True,
        title: str | None = None,
    ) -> None:
        """读取并返回 `print_code` 所表示的数据或流程，并遵守渲染器定义的边界与状态约束。

        参数：
            code: 本次操作使用的代码。
            language: 可选的`language`。
            line_numbers: 可选的`line_numbers`。
            title: 可选的`title`。
        """
        syntax = Syntax(
            code,
            language,
            theme="monokai",
            line_numbers=line_numbers,
        )

        if title:
            self.console.print(Panel(syntax, title=title, border_style="dim"))
        else:
            self.console.print(syntax)

    def print_diff(self, diff_text: str, title: str = "Diff") -> None:
        """读取并返回 `print_diff` 所表示的数据或流程，并遵守渲染器定义的边界与状态约束。

        参数：
            diff_text: 本次操作使用的`diff_text`。
            title: 可选的`title`。
        """
        lines = []

        for line in diff_text.splitlines():
            if line.startswith("+++") or line.startswith("---"):
                lines.append(f"[bold cyan]{line}[/bold cyan]")
            elif line.startswith("@@"):
                lines.append(f"[bold blue]{line}[/bold blue]")
            elif line.startswith("+"):
                lines.append(f"[green]{line}[/green]")
            elif line.startswith("-"):
                lines.append(f"[red]{line}[/red]")
            else:
                lines.append(f"[dim]{line}[/dim]")

        content = "\n".join(lines)

        self.console.print(
            Panel(
                content,
                title=f"📝 {title}",
                border_style="magenta",
                expand=False,
            )
        )

    def print_file_tree(
        self,
        root_path: str,
        max_depth: int = 3,
        show_files: bool = True,
    ) -> None:
        """读取并返回 `print_file_tree` 所表示的数据或流程，并遵守渲染器定义的边界与状态约束。

        参数：
            root_path: 本次操作使用的根目录路径。
            max_depth: 可选的`max_depth`。
            show_files: 可选的`show_files`。
        """
        root = Path(root_path)
        tree = Tree(f"📁 [bold]{root.name}/[/bold]")

        def add_items(parent: Tree, path: Path, depth: int) -> None:
            if depth > max_depth:
                return

            try:
                items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
            except PermissionError:
                return

            for item in items:
                if item.name.startswith(".") and item.name not in (".gitignore", ".env.example"):
                    continue

                if item.is_dir():
                    branch = parent.add(f"📁 [cyan]{item.name}/[/cyan]")
                    add_items(branch, item, depth + 1)
                elif show_files:
                    ext = item.suffix.lower()
                    icon = self._get_file_icon(ext)
                    parent.add(f"{icon} {item.name}")

        add_items(tree, root, 1)
        self.console.print(tree)

    def _get_file_icon(self, ext: str) -> str:
        """读取并返回 `_get_file_icon` 所表示的数据或流程，并遵守渲染器定义的边界与状态约束。

        参数：
            ext: 本次操作使用的`ext`。

        返回：
            处理后的文本或稳定标识。
        """
        icons = {
            ".py": "🐍",
            ".js": "📜",
            ".ts": "📘",
            ".jsx": "⚛️",
            ".tsx": "⚛️",
            ".json": "📋",
            ".yaml": "⚙️",
            ".yml": "⚙️",
            ".md": "📝",
            ".txt": "📄",
            ".html": "🌐",
            ".css": "🎨",
            ".sql": "🗃️",
            ".sh": "🖥️",
            ".toml": "⚙️",
            ".ini": "⚙️",
            ".cfg": "⚙️",
        }
        return icons.get(ext, "📄")

    def print_table(
        self,
        title: str,
        headers: list[str],
        rows: list[list[Any]],
    ) -> None:
        """读取并返回 `print_table` 所表示的数据或流程，并遵守渲染器定义的边界与状态约束。

        参数：
            title: 本次操作使用的`title`。
            headers: 本次操作使用的`headers`。
            rows: 本次操作使用的`rows`。
        """
        table = Table(title=title)

        for header in headers:
            table.add_column(header)

        for row in rows:
            table.add_row(*[str(cell) for cell in row])

        self.console.print(table)

    def print_progress(
        self,
        description: str = "Processing...",
        total: int | None = None,
    ) -> Progress:
        """构造并返回 `print_progress` 所表示的数据或流程，并遵守渲染器定义的边界与状态约束。

        参数：
            description: 可选的说明。
            total: 可选的总计。

        返回：
            `Progress` 类型的处理结果。
        """
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )
        return progress

    def print_file_change(
        self,
        file_path: str,
        change_type: ChangeType,
        diff: str | None = None,
    ) -> None:
        """读取并返回 `print_file_change` 所表示的数据或流程，并遵守渲染器定义的边界与状态约束。

        参数：
            file_path: 目标文件的路径；访问范围仍受项目沙箱约束。
            change_type: 本次操作使用的变更类型。
            diff: 可选的差异。
        """
        type_colors = {
            ChangeType.CREATE: "green",
            ChangeType.MODIFY: "yellow",
            ChangeType.DELETE: "red",
        }

        type_icons = {
            ChangeType.CREATE: "✨",
            ChangeType.MODIFY: "📝",
            ChangeType.DELETE: "🗑️",
        }

        color = type_colors.get(change_type, "white")
        icon = type_icons.get(change_type, "📄")

        self.console.print(
            Panel(
                f"[{color}]{change_type.value.upper()}[/{color}]\n{file_path}",
                title=f"{icon} File Change",
                border_style=color,
            )
        )

        if diff:
            self.print_diff(diff, title=f"Changes: {Path(file_path).name}")

    def print_statistics(
        self,
        stats: dict[str, Any],
        title: str = "Statistics",
    ) -> None:
        """读取并返回 `print_statistics` 所表示的数据或流程，并遵守渲染器定义的边界与状态约束。

        参数：
            stats: 本次操作使用的统计信息。
            title: 可选的`title`。
        """
        table = Table(title=f"📊 {title}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        for key, value in stats.items():
            if isinstance(value, float):
                value = f"{value:.2f}"
            table.add_row(key.replace("_", " ").title(), str(value))

        self.console.print(table)

    def print_divider(self, title: str | None = None) -> None:
        """执行 `print_divider` 所定义的协调步骤，必要时更新渲染器维护的状态。

        参数：
            title: 可选的`title`。
        """
        if title:
            self.console.print(f"\n[bold dim]── {title} ──[/bold dim]\n")
        else:
            self.console.print("\n[dim]" + "─" * 50 + "[/dim]\n")

    def clear(self) -> None:
        """处理清理，并按照当前组件的约定返回结果。"""
        self.console.clear()
