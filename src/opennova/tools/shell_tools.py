"""内置工具系统中的Shell工具模块，集中定义相关数据结构、边界适配和实现逻辑。"""

import asyncio
import os
import shlex
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opennova.runtime.events import current_tool_context
from opennova.security.guardrails import Guardrails, GuardResult
from opennova.security.process_sandbox import (
    ProcessSandbox,
    ProcessSandboxConfig,
    ProcessSandboxError,
)
from opennova.tools.base import BaseTool, ToolResult
from opennova.utils.encoding import utf8_environment

DEFAULT_TIMEOUT = 30
MAX_OUTPUT_SIZE = 100 * 1024


@dataclass
class PreparedCommand:
    command: str
    timeout: int | float
    working_dir: str
    run_with_shell: bool
    argv: list[str]
    guard_result: GuardResult
    sandbox_metadata: dict[str, Any]
    cleanup_paths: list[str]


class ExecuteCommandTool(BaseTool):
    """实现执行命令工具。模型通过统一工具 Schema 调用它，执行结果使用 ToolResult 返回，并服从运行时安全策略。"""

    name = "execute_command"
    search_hint = "Run tests, scripts, git commands, package managers, and shell commands"
    description = (
        "Execute a local shell command. Provide arguments as an object with a required "
        '`command` string, for example {"command": "uv run pytest -q"}. '
        "Do not pass command arrays or alternate keys such as `cmd`. Commands have "
        "timeout limits and are monitored for dangerous patterns."
    )

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.default_timeout = self.config.get("command_timeout", DEFAULT_TIMEOUT)
        self.working_dir = self.config.get("working_dir", os.getcwd())
        self.strict_shell_parsing = bool(self.config.get("strict_shell_parsing", False))
        self.process_sandbox_config = self.config.get("process_sandbox", {})
        self.guardrails = Guardrails(
            sandbox_mode=bool(self.config.get("sandbox_mode", True)),
            allowed_paths=self.config.get("allowed_paths", []),
            blocked_commands=self.config.get("blocked_commands", []),
            auto_confirm_safe=bool(self.config.get("auto_confirm_safe", True)),
            allow_network=bool(self.config.get("allow_network", True)),
            permission_mode=self.config.get("permission_mode", "default"),
            always_allow_tools=self.config.get("always_allow_tools", []),
            always_deny_tools=self.config.get("always_deny_tools", []),
            always_ask_tools=self.config.get("always_ask_tools", []),
            permission_rules=self.config.get("permission_rules", []),
            strict_shell_parsing=self.strict_shell_parsing,
            network_policy=self.config.get("network_policy", {}),
            secrets_policy=self.config.get("secrets_policy", {}),
        )

    def get_parameters_schema(self) -> dict[str, Any]:
        """通过反射工具 `execute()` 方法的签名和类型注解生成 JSON Schema；没有默认值的参数会进入 required 列表。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "Required. The complete command as one single string, not an array "
                        "or object. Examples: 'uv run pytest -q', 'git status --short'. "
                        "Use shell operators like pipes or redirects only when necessary."
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Optional timeout in seconds. Omit to use the configured default."
                    ),
                    "default": None,
                },
                "working_dir": {
                    "type": "string",
                    "description": (
                        "Optional working directory for the command. Omit to use the "
                        "current project working directory."
                    ),
                    "default": None,
                },
                "capture_stderr": {
                    "type": "boolean",
                    "description": "Whether stderr should be included in the tool output.",
                    "default": True,
                },
            },
            "required": ["command"],
        }

    def normalize_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """规范化参数，消除不同调用格式之间的差异。

        参数：
            arguments: 工具调用的结构化参数。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        normalized = dict(arguments)

        command_aliases = ("cmd", "shell_command")
        if "command" not in normalized:
            for alias in command_aliases:
                if alias in normalized:
                    normalized["command"] = normalized[alias]
                    break
        for alias in command_aliases:
            normalized.pop(alias, None)

        working_dir_aliases = ("cwd", "workdir")
        if "working_dir" not in normalized:
            for alias in working_dir_aliases:
                if alias in normalized:
                    normalized["working_dir"] = normalized[alias]
                    break
        for alias in working_dir_aliases:
            normalized.pop(alias, None)

        extra_args = normalized.pop("args", None)
        command = normalized.get("command")
        if isinstance(command, list):
            command_parts = [str(part) for part in command]
        elif command is None:
            command_parts = []
        else:
            command_parts = [str(command)]

        if isinstance(extra_args, list):
            command_parts.extend(str(part) for part in extra_args)

        if len(command_parts) > 1:
            normalized["command"] = shlex.join(command_parts)
        elif command_parts:
            normalized["command"] = command_parts[0]

        return normalized

    def _prepare_command_execution(
        self,
        command: str,
        timeout: int | float | None,
        working_dir: str | None,
    ) -> PreparedCommand | ToolResult:
        """校验 `_prepare_command_execution` 所表示的数据或流程，并遵守执行命令工具定义的边界与状态约束。

        参数：
            command: 需要分析或执行的 Shell 命令。
            timeout: 可选的超时。
            working_dir: 所有相对路径和本地工具操作所基于的工作目录。

        返回：
            `PreparedCommand | ToolResult` 类型的处理结果。
        """
        resolved_timeout = timeout if timeout is not None else self.default_timeout
        if not isinstance(resolved_timeout, (int, float)) or isinstance(resolved_timeout, bool):
            return ToolResult(
                success=False,
                output="",
                error="timeout must be a number of seconds",
                metadata={"guard_blocked": True, "risk_level": "block"},
            )
        if resolved_timeout <= 0:
            return ToolResult(
                success=False,
                output="",
                error="timeout must be greater than zero",
                metadata={"guard_blocked": True, "risk_level": "block"},
            )

        work_dir = working_dir or self.working_dir

        guard_result = self.guardrails.check_command(command)
        if not guard_result.allowed:
            return ToolResult(
                success=False,
                output="",
                error=guard_result.reason,
                metadata={
                    "requires_confirmation": guard_result.requires_confirmation,
                    "risk_level": guard_result.risk_level.value,
                    "suggestions": guard_result.suggestions,
                    "guard_blocked": True,
                    **guard_result.metadata,
                },
            )

        uses_shell_features = Guardrails.command_uses_shell_features(command)
        if uses_shell_features and self.strict_shell_parsing:
            return ToolResult(
                success=False,
                output="",
                error=(
                    "Command uses shell syntax, but strict shell parsing is enabled. "
                    "Run a plain argv-style command or disable strict_shell_parsing."
                ),
                metadata={
                    "guard_blocked": True,
                    "requires_confirmation": True,
                    "risk_level": "block",
                    **guard_result.metadata,
                },
            )

        try:
            work_path = Path(work_dir).resolve()
            if not work_path.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Working directory does not exist: {work_dir}",
                )
            if not work_path.is_dir():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Working directory is not a directory: {work_dir}",
                )
            path_result = self.guardrails.check_file_path(
                str(work_path),
                "read",
                self.working_dir,
            )
            if not path_result.allowed:
                return ToolResult(
                    success=False,
                    output="",
                    error=path_result.reason,
                    metadata={
                        "guard_blocked": True,
                        "requires_confirmation": path_result.requires_confirmation,
                        "risk_level": path_result.risk_level.value,
                    },
                )
            work_dir = str(work_path)
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Invalid working directory: {e}",
            )

        run_with_shell = uses_shell_features and not self.strict_shell_parsing
        argv: list[str] | None = None
        if not run_with_shell:
            try:
                argv = shlex.split(command, posix=(os.name != "nt"))
            except ValueError as e:
                return ToolResult(success=False, output="", error=f"Invalid command syntax: {e}")
            if not argv:
                return ToolResult(success=False, output="", error="Empty command")

        try:
            sandbox = ProcessSandbox(
                ProcessSandboxConfig.from_config(
                    self.process_sandbox_config,
                    working_dir=work_dir,
                    allowed_paths=self.config.get("allowed_paths", []),
                    allow_network=bool(self.config.get("allow_network", True)),
                    tmp_dir=self.config.get("temp_dir"),
                )
            )
            sandbox_plan = sandbox.wrap(
                command=command,
                argv=argv,
                run_with_shell=run_with_shell,
                working_dir=work_dir,
                env=utf8_environment(),
            )
            final_run_with_shell = run_with_shell and not sandbox_plan.metadata.get("applied")
        except (ProcessSandboxError, ValueError) as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e),
                metadata={
                    "guard_blocked": True,
                    "requires_confirmation": guard_result.requires_confirmation,
                    "risk_level": "block",
                    **guard_result.metadata,
                },
            )

        return PreparedCommand(
            command=command,
            timeout=resolved_timeout,
            working_dir=sandbox_plan.cwd,
            run_with_shell=final_run_with_shell,
            argv=sandbox_plan.argv,
            guard_result=guard_result,
            sandbox_metadata=sandbox_plan.metadata,
            cleanup_paths=sandbox_plan.cleanup_paths,
        )

    def execute(
        self,
        command: str,
        timeout: int | None = None,
        working_dir: str | None = None,
        capture_stderr: bool = True,
    ) -> ToolResult:
        """执行执行命令工具对应的实际操作，校验输入并返回统一结果。

        参数：
            command: 需要分析或执行的 Shell 命令。
            timeout: 可选的超时。
            working_dir: 所有相对路径和本地工具操作所基于的工作目录。
            capture_stderr: 可选的`capture_stderr`。

        返回：
            `ToolResult` 类型的处理结果。
        """
        prepared = self._prepare_command_execution(command, timeout, working_dir)
        if isinstance(prepared, ToolResult):
            return prepared

        try:
            result = subprocess.run(
                prepared.command if prepared.run_with_shell else prepared.argv,
                shell=prepared.run_with_shell,
                cwd=prepared.working_dir,
                capture_output=True,
                text=True,
                timeout=prepared.timeout,
                env=utf8_environment(),
                **self._process_group_kwargs(),
            )

            stdout = result.stdout or ""
            stderr = result.stderr or ""

            output_parts = []
            if stdout:
                output_parts.append(f"stdout:\n{stdout}")
            if capture_stderr and stderr:
                output_parts.append(f"stderr:\n{stderr}")

            combined_output = "\n\n".join(output_parts) if output_parts else "(no output)"
            combined_output = self._with_sandbox_warning(
                combined_output,
                prepared.sandbox_metadata,
            )

            truncated_output = self._truncate_output(combined_output)

            success = result.returncode == 0

            if not success:
                output = f"Exit code: {result.returncode}\n\n{truncated_output}"
            else:
                output = truncated_output

            return ToolResult(
                success=success,
                output=output,
                error=None if success else f"Command failed with exit code {result.returncode}",
                metadata={
                    "command": command,
                    "exit_code": result.returncode,
                    "timeout": prepared.timeout,
                    "working_dir": prepared.working_dir,
                    "shell_fallback": prepared.run_with_shell,
                    "requires_confirmation": prepared.guard_result.requires_confirmation,
                    "risk_level": prepared.guard_result.risk_level.value,
                    "process_sandbox": prepared.sandbox_metadata,
                    **prepared.guard_result.metadata,
                },
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error=f"Command timed out after {prepared.timeout} seconds",
                metadata={
                    "command": command,
                    "timeout": prepared.timeout,
                    "process_sandbox": prepared.sandbox_metadata,
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to execute command: {e}",
            )
        finally:
            self._cleanup_paths(prepared.cleanup_paths)

    async def async_execute(
        self,
        command: str,
        timeout: int | None = None,
        working_dir: str | None = None,
        capture_stderr: bool = True,
    ) -> ToolResult:
        """执行执行命令工具对应的实际操作，校验输入并返回统一结果。

        参数：
            command: 需要分析或执行的 Shell 命令。
            timeout: 可选的超时。
            working_dir: 所有相对路径和本地工具操作所基于的工作目录。
            capture_stderr: 可选的`capture_stderr`。

        返回：
            `ToolResult` 类型的处理结果。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        prepared = self._prepare_command_execution(command, timeout, working_dir)
        if isinstance(prepared, ToolResult):
            return prepared

        process: asyncio.subprocess.Process | None = None
        try:
            context = current_tool_context()
            if context and context.abort_signal:
                context.abort_signal.raise_if_cancelled()
            group_kwargs = self._process_group_kwargs()
            if prepared.run_with_shell:
                process = await asyncio.create_subprocess_shell(
                    prepared.command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=prepared.working_dir,
                    env=utf8_environment(),
                    **group_kwargs,
                )
            else:
                process = await asyncio.create_subprocess_exec(
                    *prepared.argv,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=prepared.working_dir,
                    env=utf8_environment(),
                    **group_kwargs,
                )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=prepared.timeout,
                )
            except TimeoutError:
                await self._terminate_process_tree(process)
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Command timed out after {prepared.timeout} seconds",
                )

            stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
            stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""

            output_parts = []
            if stdout_str:
                output_parts.append(f"stdout:\n{stdout_str}")
            if capture_stderr and stderr_str:
                output_parts.append(f"stderr:\n{stderr_str}")

            combined_output = "\n\n".join(output_parts) if output_parts else "(no output)"
            combined_output = self._with_sandbox_warning(
                combined_output,
                prepared.sandbox_metadata,
            )
            truncated_output = self._truncate_output(combined_output)

            success = process.returncode == 0

            if not success:
                output = f"Exit code: {process.returncode}\n\n{truncated_output}"
            else:
                output = truncated_output

            return ToolResult(
                success=success,
                output=output,
                error=None if success else f"Command failed with exit code {process.returncode}",
                metadata={
                    "command": command,
                    "exit_code": process.returncode,
                    "timeout": prepared.timeout,
                    "working_dir": prepared.working_dir,
                    "shell_fallback": prepared.run_with_shell,
                    "requires_confirmation": prepared.guard_result.requires_confirmation,
                    "risk_level": prepared.guard_result.risk_level.value,
                    "process_sandbox": prepared.sandbox_metadata,
                    **prepared.guard_result.metadata,
                },
            )

        except asyncio.CancelledError:
            if process is not None:
                await self._terminate_process_tree(process)
            raise
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to execute command: {e}",
            )
        finally:
            self._cleanup_paths(prepared.cleanup_paths)

    async def execute_async(
        self,
        command: str,
        timeout: int | None = None,
        working_dir: str | None = None,
        capture_stderr: bool = True,
    ) -> ToolResult:
        """执行执行命令工具对应的实际操作，校验输入并返回统一结果。

        参数：
            command: 需要分析或执行的 Shell 命令。
            timeout: 可选的超时。
            working_dir: 所有相对路径和本地工具操作所基于的工作目录。
            capture_stderr: 可选的`capture_stderr`。

        返回：
            `ToolResult` 类型的处理结果。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        return await self.async_execute(
            command,
            timeout=timeout,
            working_dir=working_dir,
            capture_stderr=capture_stderr,
        )

    @staticmethod
    def _process_group_kwargs() -> dict[str, Any]:
        """启动或推进 `_process_group_kwargs` 所表示的数据或流程，并遵守执行命令工具定义的边界与状态约束。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        if os.name == "nt":
            flag = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            return {"creationflags": flag} if flag else {}
        return {"start_new_session": True}

    @staticmethod
    async def _terminate_process_tree(
        process: asyncio.subprocess.Process,
        grace_seconds: float = 1.0,
    ) -> None:
        """终止进程树结构，并按照当前组件的约定返回结果。

        参数：
            process: 本次操作使用的进程。
            grace_seconds: 可选的`grace_seconds`。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        if process.returncode is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except (LookupError, ProcessLookupError):
            return

        try:
            await asyncio.wait_for(process.wait(), timeout=grace_seconds)
            return
        except TimeoutError:
            pass

        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except (LookupError, ProcessLookupError):
            return
        await process.wait()

    @staticmethod
    def _cleanup_paths(paths: list[str]) -> None:
        for value in paths:
            try:
                Path(value).unlink(missing_ok=True)
            except OSError:
                continue

    @staticmethod
    def _with_sandbox_warning(output: str, metadata: dict[str, Any]) -> str:
        reason = metadata.get("fallback_reason")
        if (
            metadata.get("enabled")
            and not metadata.get("applied")
            and reason
            and "disabled" not in str(reason)
        ):
            return f"[process sandbox fallback: {reason}]\n\n{output}"
        return output

    def _truncate_output(self, output: str) -> str:
        """根据当前输入和执行命令工具的状态计算 `_truncate_output`，并返回调用方需要的结果。

        参数：
            output: 本次操作使用的输出。

        返回：
            处理后的文本或稳定标识。
        """
        if len(output) > MAX_OUTPUT_SIZE:
            return (
                output[: MAX_OUTPUT_SIZE // 2]
                + f"\n\n... [truncated {len(output) - MAX_OUTPUT_SIZE} bytes] ...\n\n"
                + output[-MAX_OUTPUT_SIZE // 2 :]
            )
        return output

    def is_destructive(self, **kwargs: Any) -> bool:
        """声明本次工具调用是否可能删除或覆盖数据；默认返回假。

        参数：
            **kwargs: 传递给底层实现的额外关键字参数。

        返回：
            表示条件是否成立。
        """
        return True

    def requires_permission(self, **kwargs: Any) -> bool:
        return True
