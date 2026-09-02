"""OpenNova中的`main`模块，集中定义相关数据结构、边界适配和实现逻辑。"""

import asyncio
import sys
from pathlib import Path

import click

from opennova import __version__
from opennova.config import (
    Config,
    create_default_config,
    load_config,
    validate_config,
)
from opennova.logging_config import get_logger, setup_logging

# 模块级日志记录器，main 函数中会配置处理器
_LOGGER = get_logger(__name__)


def print_version(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    """在 Click 处理 `--version` 时输出当前版本并立即结束命令，不再进入 TUI 初始化。

    参数：
        ctx: 本次操作使用的`ctx`。
        param: 本次操作使用的`param`。
        value: 需要保存、转换或校验的值。
    """
    if not value or ctx.resilient_parsing:
        return
    click.echo(f"OpenNova v{__version__}")
    ctx.exit()


@click.group(invoke_without_command=True)
@click.option(
    "--version",
    "-v",
    is_flag=True,
    expose_value=False,
    is_eager=True,
    callback=print_version,
    help="Show version and exit.",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=False),
    help="Path to configuration file.",
)
@click.option(
    "--resume",
    "resume_mode",
    is_flag=True,
    help="Open the TUI and choose a saved session to resume.",
)
@click.option(
    "--continue",
    "continue_mode",
    is_flag=True,
    help="Open the TUI and continue the most recent saved session.",
)
@click.option(
    "--permission-mode",
    type=click.Choice(["request", "auto", "full"], case_sensitive=False),
    help="Approval mode for this run: request, auto, or full.",
)
@click.pass_context
def main(
    ctx: click.Context,
    config_path: str | None,
    resume_mode: bool,
    continue_mode: bool,
    permission_mode: str | None,
) -> None:
    """OpenNova 终端 AI 编程 Agent。

    不带子命令启动 Textual 交互界面；也可以使用下方子命令执行一次性任务、检查配置或初始化环境。
    """
    # 加载配置并初始化日志系统
    config = load_config(config_path)           # 1. 加载配置
    logging_config = config.get_logging_config() # 2. 提取日志配置
    setup_logging(logging_config)                # 3. 初始化日志系统

    _LOGGER.info("OpenNova v%s starting", __version__)
    _LOGGER.debug("Config path: %s, resume: %s, continue: %s, permission_mode: %s",
                   config_path, resume_mode, continue_mode, permission_mode)

    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path
    ctx.obj["config"] = config
    ctx.obj["resume_mode"] = resume_mode
    ctx.obj["continue_mode"] = continue_mode
    ctx.obj["permission_mode"] = permission_mode

    if ctx.invoked_subcommand is None:
        _LOGGER.info("No subcommand specified, invoking run command")
        ctx.invoke(run, task=None)


@main.command()
@click.argument("task", required=False)
@click.option("--plan", "-p", is_flag=True, help="Run in plan mode.")
@click.option("--model", "-m", "model", help="Override model to use.")
@click.option("--provider", help="Override provider to use.")
@click.option("--no-stream", is_flag=True, help="Disable streaming output.")
@click.option(
    "--tui",
    "force_tui",
    is_flag=True,
    help="Force the Textual TUI, including on Windows terminals.",
)
@click.pass_context
def run(
    ctx: click.Context,
    task: str | None,
    plan: bool,
    model: str | None,
    provider: str | None,
    no_stream: bool,
    force_tui: bool,
) -> None:
    """执行一次 Agent 任务；省略 TASK 时启动 Textual 交互界面。

    示例：

        opennova run "读取 README.md 并说明项目入口"

        opennova run --plan "为会话恢复功能制定重构计划"

        opennova run --provider deepseek -m deepseek-v4-pro "审查 src/ 目录"
    """
    _LOGGER.info("Run command invoked: task=%s, plan=%s, model=%s, provider=%s, no_stream=%s, force_tui=%s",
                  task, plan, model, provider, no_stream, force_tui)
    _LOGGER.debug("Resume mode: %s, continue mode: %s",
                   ctx.obj.get("resume_mode"), ctx.obj.get("continue_mode"))

    config = _load_and_validate_config(
        ctx.obj.get("config_path"),
        provider,
        model,
        ctx.obj.get("permission_mode"),
    )

    _LOGGER.info("Config loaded: default_provider=%s, model=%s",
                  config.get("default_provider"), model)

    resume_mode = bool(ctx.obj.get("resume_mode"))
    continue_mode = bool(ctx.obj.get("continue_mode"))

    if resume_mode and continue_mode:
        _LOGGER.error("Conflicting options: --resume and --continue cannot be used together")
        raise click.UsageError("Use only one of --resume or --continue.")
    if (resume_mode or continue_mode) and task:
        _LOGGER.error("Conflicting options: --resume/--continue cannot be used with a direct task")
        raise click.UsageError("--resume/--continue cannot be used with a direct task.")

    if task:
        _LOGGER.info("Running single task: %s", task[:100])
        asyncio.run(_run_single_task(config, task, plan, not no_stream))
    elif _use_tui_for_interactive(force_tui=force_tui):
        from opennova.cli.tui import run_tui

        startup_resume_mode = None
        if resume_mode:
            _LOGGER.info("Starting TUI in resume mode")
            startup_resume_mode = "resume"
        elif continue_mode:
            _LOGGER.info("Starting TUI in continue mode")
            startup_resume_mode = "continue"
        else:
            _LOGGER.info("Starting TUI in normal mode")
        asyncio.run(run_tui(config, startup_resume_mode=startup_resume_mode))


@main.command()
@click.argument("task")
@click.option("--edit", is_flag=True, help="Open plan in editor before execution.")
@click.pass_context
def plan(ctx: click.Context, task: str, edit: bool) -> None:
    """先为 TASK 生成结构化计划，再由用户审阅并决定是否执行。

    示例：

        opennova plan "为认证模块补充单元测试"
    """
    config = _load_and_validate_config(
        ctx.obj.get("config_path"),
        permission_mode=ctx.obj.get("permission_mode"),
    )
    asyncio.run(_run_single_task(config, task, plan_mode=True, stream=True))


@main.command("list-tools")
@click.pass_context
def list_tools(ctx: click.Context) -> None:
    """列出当前版本自带的全部工具；该命令使用无副作用检查路径，不创建 Provider 或会话。"""
    del ctx
    from opennova.runtime.bootstrap import inspect_runtime

    snapshot = inspect_runtime()

    click.echo("Available tools:\n")
    for tool_name in snapshot.tool_names:
        click.echo(f"  • {tool_name}")

    click.echo(f"\nTotal: {len(snapshot.tool_names)} tools")


@main.command()
@click.pass_context
def doctor(ctx: click.Context) -> None:
    """检查配置、Python 环境、内置工具和扩展声明，但不创建 Provider、会话，不连接 MCP，也不加载项目扩展。"""
    from opennova.runtime.bootstrap import inspect_runtime

    config = load_config(ctx.obj.get("config_path"))
    snapshot = inspect_runtime()
    project = Path.cwd()
    hooks = list((project / ".opennova" / "hooks").glob("*.py"))
    plugins = list((project / ".opennova" / "plugins").glob("*/plugin.yaml"))
    mcp_servers = config.get_mcp_servers()
    process_sandbox = config.get("security.process_sandbox", {})
    click.echo("OpenNova doctor (side-effect-free)\n")
    click.echo(f"Version: {__version__}")
    click.echo(f"Bootstrap profile: {snapshot.profile.value}")
    click.echo(f"Python encoding: {sys.getfilesystemencoding()}")
    click.echo(f"Built-in tools: {len(snapshot.tool_names)}")
    click.echo(f"Project hooks declared: {len(hooks)} (not imported)")
    click.echo(f"Project plugins declared: {len(plugins)} (not loaded)")
    click.echo(f"MCP servers configured: {len(mcp_servers)} (not connected)")
    click.echo(
        "Process sandbox: "
        f"enabled={bool(process_sandbox.get('enabled', True))} "
        f"backend={process_sandbox.get('backend', 'auto')}"
    )


@main.command()
@click.pass_context
def config_cmd(ctx: click.Context) -> None:
    """显示默认配置、全局配置、项目配置和环境变量展开后合并得到的当前配置；敏感字段会先脱敏。"""
    config = load_config(ctx.obj.get("config_path"))

    import yaml

    click.echo("Current configuration:\n")
    click.echo(yaml.dump(config.redacted_data(), default_flow_style=False, sort_keys=False))


@main.command()
def init() -> None:
    """在用户配置目录创建默认配置文件，供后续填写模型 Provider 和 API 密钥。"""
    config_path = create_default_config()
    click.echo(f"Created configuration file: {config_path}")
    click.echo("\nPlease edit the configuration file and add your API keys.")
    click.echo("\nYou can also set environment variables:")
    click.echo("  - OPENAI_API_KEY")
    click.echo("  - ANTHROPIC_API_KEY")
    click.echo("  - DEEPSEEK_API_KEY")


@main.command()
@click.option("--port", "-p", default=8000, help="Port to listen on.")
@click.option("--host", "-h", default="127.0.0.1", help="Host to bind to.")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development.")
@click.pass_context
def serve(ctx: click.Context, port: int, host: str, reload: bool) -> None:
    """启动 Web 服务（每连接独立 AgentRuntime）。

    示例：

        opennova serve --port 8000

        opennova serve --host 0.0.0.0 --port 8080
    """
    config = _load_and_validate_config(ctx.obj.get("config_path"))

    click.echo(f"Starting OpenNova Web UI on http://{host}:{port}")
    click.echo(f"API docs available at http://{host}:{port}/docs")
    click.echo("Press Ctrl+C to stop\n")

    from opennova.web.server import start_server

    start_server(config, host, port, reload)


def _use_tui_for_interactive(*, force_tui: bool, platform: str | None = None) -> bool:
    """判断无直接任务时是否启动 Textual 界面；当前产品只保留 TUI，因此始终返回真。

    参数：
        force_tui: 本次操作使用的`force_tui`。
        platform: 可选的`platform`。

    返回：
        表示条件是否成立。
    """
    return True


async def _run_single_task(
    config: Config,
    task: str,
    plan_mode: bool = False,
    stream: bool = True,
) -> None:
    """运行单个任务流程，并统一处理完成、失败和取消。

    参数：
        config: 控制当前组件行为的配置。
        task: 用户希望 Agent 完成的任务描述。
        plan_mode: 可选的计划模式。
        stream: 是否将模型输出以增量事件形式返回。

    说明：
        这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
    """
    from rich.console import Console

    from opennova.providers.base import StreamChunk
    from opennova.runtime.agent import AgentRuntime
    from opennova.runtime.state import Plan
    from opennova.tools.base import ToolResult

    _LOGGER.info("Starting single task: plan_mode=%s, stream=%s", plan_mode, stream)
    _LOGGER.debug("Task content: %s", task[:200])

    console = Console(
        force_terminal=True,
        soft_wrap=False,  # 关闭软换行，让较长输出保持终端自身的横向与纵向滚动行为。
        markup=True,
        highlight=True,
    )

    from opennova.runtime.bootstrap import RuntimeBootstrapProfile

    _LOGGER.info("Creating AgentRuntime with HEADLESS profile")

    agent = AgentRuntime(config, bootstrap_profile=RuntimeBootstrapProfile.HEADLESS)

    _LOGGER.info("AgentRuntime created successfully")

    if plan_mode:
        console.print(f"[yellow]Planning: {task}[/yellow]\n")
    else:
        console.print(f"[cyan]Task: {task}[/cyan]\n")

    def on_thought(thought: str) -> None:
        console.print(f"[dim]💭 {thought}[/dim]\n")

    def on_action(tool_name: str, args: dict) -> None:
        args_str = ", ".join(f"{k}={repr(v)[:50]}" for k, v in args.items())
        console.print(f"[blue]⚙️  {tool_name}({args_str})[/blue]")

    def on_result(result: ToolResult) -> None:
        if result.success:
            console.print("[green]✅ Done[/green]\n")
        else:
            console.print(f"[red]❌ Error: {result.error}[/red]\n")

    def on_stream(chunk: StreamChunk) -> None:
        if chunk.content:
            print(chunk.content, end="", flush=True)

    def on_plan(plan: Plan, plan_file_path: str | None = None) -> None:
        step_count = len(plan.steps)
        console.print(f"[cyan]Generated plan with {step_count} steps.[/cyan]")
        if plan_file_path:
            console.print(f"[green]Plan saved to:[/green] {plan_file_path}\n")

    agent.register_callback("thought", on_thought)
    agent.register_callback("action", on_action)
    agent.register_callback("result", on_result)
    agent.register_callback("stream", on_stream)
    if plan_mode:
        agent.register_callback("plan", on_plan)

    try:
        _LOGGER.info("Starting agent.run()")

        result = await agent.run(
            task,
            mode="plan" if plan_mode else "act",
            stream=stream,
        )

        _LOGGER.info("Agent.run() completed successfully")
        _LOGGER.debug("Result: %s", str(result)[:500])

        console.print()
        console.print("[bold]Result:[/bold]")
        console.print(result)

        if plan_mode:
            if click.confirm("Execute this saved plan now?", default=False):
                _LOGGER.info("User approved plan execution")
                agent.state.mark_plan_approved()
                execution_result = await agent.execute_approved_plan(stream=stream)
                _LOGGER.info("Plan execution completed")
                _LOGGER.debug("Execution result: %s", str(execution_result)[:500])
                console.print()
                console.print("[bold]Execution Result:[/bold]")
                console.print(execution_result)
            else:
                _LOGGER.info("User declined plan execution")
                console.print("[yellow]Plan kept for later execution.[/yellow]")

    except KeyboardInterrupt:
        _LOGGER.warning("Task interrupted by user (KeyboardInterrupt)")
        console.print("\n[yellow]Task interrupted.[/yellow]")
        sys.exit(1)
    except Exception as e:
        _LOGGER.error("Task failed with error: %s: %s", type(e).__name__, str(e), exc_info=True)
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)
    finally:
        _LOGGER.info("Closing AgentRuntime")
        await agent.aclose()
        _LOGGER.info("AgentRuntime closed")


def _load_and_validate_config(
    config_path: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    permission_mode: str | None = None,
) -> Config:
    """加载分层配置、应用命令行覆盖并执行校验，为后续创建 AgentRuntime 准备好可用的 Config。

    配置加载顺序由 load_config 完成：以内置 DEFAULT_CONFIG 为基底，依次深合并全局配置
    （~/.opennova/config.yaml）与项目配置（.opennova/config.yaml；若传入了 config_path 则用它
    替代项目配置），最后展开 ${ENV} 占位符。provider、model、permission_mode 三个参数是命令行
    选项，非 None 时会在合并结果上覆盖对应字段。校验失败时打印错误并直接结束进程。

    参数：
        config_path: 可选的配置路径。来自主命令的 --config 选项：main() 把它存入
            ctx.obj["config_path"]，run 和 plan 命令再通过 ctx.obj.get("config_path") 传入；
            为 None 时由 load_config 自动查找全局配置和项目配置。
        provider: 可选的 Provider 名称。来自 run 子命令的 --provider 选项，是 Click 解析
            命令行时得到的参数；非 None 时覆盖配置中的 default_provider。
        model: 可选的模型名称。来自 run 子命令的 --model 选项，是 Click 解析命令行时得到
            的参数；非 None 时覆盖当前默认 Provider 的 default_model。
        permission_mode: 可选的权限模式。来自主命令的 --permission-mode 选项：main() 把它
            存入 ctx.obj["permission_mode"]，run 和 plan 命令再通过 ctx.obj.get("permission_mode")
            传入；非 None 时覆盖配置中的 security.permission_mode。

    返回：
        经过校验、可直接交给 AgentRuntime 使用的 `Config` 对象；校验失败时打印错误信息并
        以 sys.exit(1) 结束进程，不会正常返回。

    说明：
        该函数会读取本地文件系统和环境变量，但不会创建 Provider 或会话。
    """
    _LOGGER.info("Loading configuration: config_path=%s, provider=%s, model=%s, permission_mode=%s",
                  config_path, provider, model, permission_mode)

    config = load_config(config_path)

    _LOGGER.debug("Default provider from config: %s", config.get("default_provider"))

    if provider:
        _LOGGER.info("Overriding default_provider: %s", provider)
        config.set("default_provider", provider)

    if model:
        current_provider = config.get("default_provider")
        providers = config.get("providers", {})
        if current_provider in providers:
            _LOGGER.info("Overriding model for provider %s: %s", current_provider, model)
            providers[current_provider]["default_model"] = model
            config.data["providers"] = providers

    if permission_mode:
        _LOGGER.info("Overriding permission_mode: %s", permission_mode)
        config.set("security.permission_mode", permission_mode.lower())

    errors = validate_config(config)
    if errors:
        _LOGGER.error("Configuration validation failed: %s", errors)
        click.echo("Configuration errors:\n", err=True)
        for error in errors:
            click.echo(f"  • {error}", err=True)
        click.echo(
            "\nRun 'opennova init' to create a configuration file, "
            "or set the appropriate API key environment variable.",
            err=True,
        )
        sys.exit(1)

    _LOGGER.info("Configuration loaded and validated successfully")

    return config


if __name__ == "__main__":
    main()
