"""内置工具系统中的文件工具模块，集中定义相关数据结构、边界适配和实现逻辑。"""

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from opennova.checkpoints import CheckpointManager
from opennova.diff.engine import DiffEngine
from opennova.runtime.events import current_tool_context
from opennova.runtime.file_state import FileVersionCache
from opennova.security.sandbox import Sandbox, SandboxConfig
from opennova.security.secrets import SecretScanner
from opennova.tools.base import BaseTool, ToolResult

MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_OUTPUT_SIZE = 100 * 1024

BINARY_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".so",
    ".dll",
    ".dylib",
    ".exe",
    ".bin",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".7z",
    ".mp3",
    ".mp4",
    ".avi",
    ".mov",
    ".wav",
    ".sqlite",
    ".db",
}


def _is_binary_file(file_path: str) -> bool:
    """校验 `_is_binary_file` 所表示的数据或流程，并遵守当前模块定义的边界与状态约束。

    参数：
        file_path: 目标文件的路径；访问范围仍受项目沙箱约束。

    返回：
        表示条件是否成立。

    说明：
        该操作会访问本地文件系统，路径校验和原子写入约束由所在组件负责。
    """
    ext = Path(file_path).suffix.lower()
    if ext in BINARY_EXTENSIONS:
        return True

    if not os.path.exists(file_path):
        return False

    try:
        with open(file_path, "rb") as f:
            chunk = f.read(1024)
            return b"\x00" in chunk
    except Exception:
        return False


def _truncate_output(output: str, max_size: int = MAX_OUTPUT_SIZE) -> str:
    """根据当前输入和当前模块的状态计算 `_truncate_output`，并返回调用方需要的结果。

    参数：
        output: 本次操作使用的输出。
        max_size: 可选的`max_size`。

    返回：
        处理后的文本或稳定标识。
    """
    if len(output) > max_size:
        return (
            output[: max_size // 2]
            + f"\n\n... [truncated {len(output) - max_size} bytes] ...\n\n"
            + output[-max_size // 2 :]
        )
    return output


def _build_sandbox(tool_config: dict[str, Any] | None = None) -> Sandbox:
    """根据当前输入和状态构造`build_sandbox`。

    参数：
        tool_config: 可选的工具配置。

    返回：
        `Sandbox` 类型的处理结果。
    """
    config = tool_config or {}
    sandbox_config = SandboxConfig(
        working_dir=str(config.get("working_dir", os.getcwd())),
        allowed_paths=config.get("allowed_paths", []),
        denied_paths=config.get("denied_paths"),
        max_file_size=int(config.get("max_file_size", MAX_FILE_SIZE)),
        allow_network=bool(config.get("allow_network", False)),
        read_only=bool(config.get("read_only", False)),
        temp_dir=config.get("temp_dir"),
    )
    return Sandbox(sandbox_config)


def _create_write_checkpoint(
    tool_config: dict[str, Any],
    path: Path,
    label: str,
) -> str | None:
    """创建写入检查点并完成必要的初始化。

    参数：
        tool_config: 本次操作使用的工具配置。
        path: 需要读取、检查或写入的路径。
        label: 本次操作使用的`label`。

    返回：
        `str | None` 类型的处理结果。
    """
    if not bool(tool_config.get("checkpoint_writes", True)):
        return None
    return CheckpointManager(tool_config.get("working_dir", os.getcwd())).create(label, [path])


def _build_secret_scanner(tool_config: dict[str, Any] | None = None) -> SecretScanner:
    config = tool_config or {}
    return SecretScanner.from_config(config.get("secrets_policy", {}))


def _active_file_cache() -> FileVersionCache | None:
    context = current_tool_context()
    cache = context.read_file_cache if context else None
    return cache if isinstance(cache, FileVersionCache) else None


def _validate_observed_version(path: Path) -> ToolResult | None:
    """校验`validate_observed_version`，发现问题时返回或抛出明确错误。

    参数：
        path: 需要读取、检查或写入的路径。

    返回：
        `ToolResult | None` 类型的处理结果。
    """
    cache = _active_file_cache()
    if cache is None:
        return None
    matches, expected, current = cache.validate(path)
    if matches:
        return None
    return ToolResult(
        success=False,
        output="",
        error=(
            "File changed after it was read by this session; read it again before editing: "
            f"{path}"
        ),
        metadata={
            "stale_file": True,
            "file_path": str(path),
            "expected_version": asdict(expected) if expected else None,
            "current_version": asdict(current),
            "retry_hint": "Call read_file again, then retry with the new content.",
        },
    )


def _record_file_version(path: Path) -> None:
    cache = _active_file_cache()
    if cache is not None:
        cache.record(path)


def _redact_tool_text(
    scanner: SecretScanner,
    text: str,
    tool_config: dict[str, Any],
) -> tuple[str, int]:
    findings = scanner.scan(text)
    if not findings:
        return text, 0
    secrets_config = tool_config.get("secrets_policy", {})
    if not bool(secrets_config.get("redact_tool_outputs", True)):
        return text, len(findings)
    return scanner.redact(text), len(findings)


class ReadFileTool(BaseTool):
    """实现读取文件工具。模型通过统一工具 Schema 调用它，执行结果使用 ToolResult 返回，并服从运行时安全策略。"""

    name = "read_file"
    description = (
        "Read the contents of a file. "
        "Returns file content with line numbers. "
        "Optionally specify start_line and end_line to read a range."
    )

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.sandbox = _build_sandbox(config)
        self.secret_scanner = _build_secret_scanner(config)

    def is_read_only(self, **kwargs: Any) -> bool:
        return True

    def execute(
        self,
        file_path: str,
        start_line: int = 1,
        end_line: int = -1,
    ) -> ToolResult:
        """执行读取文件工具对应的实际操作，校验输入并返回统一结果。

        参数：
            file_path: 目标文件的路径；访问范围仍受项目沙箱约束。
            start_line: 可选的`start_line`。
            end_line: 可选的`end_line`。

        返回：
            `ToolResult` 类型的处理结果。
        """
        is_allowed, reason = self.sandbox.is_path_allowed(file_path)
        if not is_allowed:
            return ToolResult(success=False, output="", error=reason)
        resolved_path = str(Path(file_path).resolve())

        if _is_binary_file(resolved_path):
            return ToolResult(
                success=False,
                output="",
                error=f"Cannot read binary file: {file_path}",
            )

        try:
            ok, content_or_error = self.sandbox.safe_read(resolved_path)
            if not ok:
                return ToolResult(success=False, output="", error=str(content_or_error))

            raw_content = content_or_error if isinstance(content_or_error, bytes) else b""
            file_size = len(raw_content)
            text_content = raw_content.decode("utf-8", errors="replace")
            text_content, secret_findings_count = _redact_tool_text(
                self.secret_scanner,
                text_content,
                self.config,
            )
            lines = text_content.splitlines(keepends=True)

            total_lines = len(lines)

            if end_line == -1 or end_line > total_lines:
                end_line = total_lines

            start_idx = max(0, start_line - 1)
            end_idx = min(end_line, total_lines)

            selected_lines = lines[start_idx:end_idx]

            numbered_lines = [
                f"{i + start_idx + 1:6}: {line.rstrip()}"
                for i, line in enumerate(selected_lines)
            ]

            output = "\n".join(numbered_lines)

            if start_line > 1 or end_line < total_lines:
                header = f"File: {file_path} (lines {start_idx + 1}-{end_idx} of {total_lines})\n\n"
            else:
                header = f"File: {file_path} ({total_lines} lines)\n\n"

            full_output = header + _truncate_output(output)
            _record_file_version(Path(resolved_path))

            return ToolResult(
                success=True,
                output=full_output,
                metadata={
                    "file_path": resolved_path,
                    "total_lines": total_lines,
                    "lines_read": end_line - start_line + 1,
                    "file_size": file_size,
                    "secret_findings_count": secret_findings_count,
                },
            )

        except PermissionError:
            return ToolResult(
                success=False,
                output="",
                error=f"Permission denied: {file_path}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to read file: {e}",
            )


class WriteFileTool(BaseTool):
    """实现写入文件工具。模型通过统一工具 Schema 调用它，执行结果使用 ToolResult 返回，并服从运行时安全策略。"""

    name = "write_file"
    description = (
        "Write content to a file. "
        "This will overwrite the existing file if it exists. "
        "Use with caution as this operation is destructive."
    )

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.sandbox = _build_sandbox(config)
        self.secret_scanner = _build_secret_scanner(config)

    def is_destructive(self, **kwargs: Any) -> bool:
        return True

    def execute(self, file_path: str, content: str) -> ToolResult:
        """执行写入文件工具对应的实际操作，校验输入并返回统一结果。

        参数：
            file_path: 目标文件的路径；访问范围仍受项目沙箱约束。
            content: 需要处理、保存或分析的文本内容。

        返回：
            `ToolResult` 类型的处理结果。
        """
        is_allowed, reason = self.sandbox.is_path_allowed(file_path)
        if not is_allowed:
            return ToolResult(success=False, output="", error=reason)
        path = Path(file_path).resolve()

        try:
            stale_result = _validate_observed_version(path)
            if stale_result is not None:
                return stale_result
            old_content = ""
            file_existed = path.exists()
            checkpoint_id: str | None = None
            if file_existed:
                read_ok, read_result = self.sandbox.safe_read(path)
                if not read_ok:
                    return ToolResult(success=False, output="", error=str(read_result))
                old_content = (
                    read_result.decode("utf-8", errors="replace")
                    if isinstance(read_result, bytes)
                    else ""
                )
                checkpoint_id = _create_write_checkpoint(self.config, path, "before write_file")

            write_ok, write_result = self.sandbox.safe_write(path, content.encode("utf-8"))
            if not write_ok:
                return ToolResult(success=False, output="", error=str(write_result))
            _record_file_version(path)

            diff_engine = DiffEngine()
            diff_text = diff_engine.generate_diff(old_content, content, str(path))
            diff_text, secret_findings_count = _redact_tool_text(
                self.secret_scanner,
                diff_text,
                self.config,
            )

            metadata: dict[str, Any] = {
                "file_path": str(path),
                "bytes_written": len(content),
                "change_type": "modify" if file_existed else "create",
                "secret_findings_count": secret_findings_count,
            }
            if diff_text.strip():
                metadata["diff"] = diff_text
            if checkpoint_id:
                metadata["checkpoint_id"] = checkpoint_id

            return ToolResult(
                success=True,
                output=f"Successfully wrote {len(content)} bytes to {file_path}",
                metadata=metadata,
            )

        except PermissionError:
            return ToolResult(
                success=False,
                output="",
                error=f"Permission denied: {file_path}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to write file: {e}",
            )


class CreateFileTool(BaseTool):
    """实现创建文件工具。模型通过统一工具 Schema 调用它，执行结果使用 ToolResult 返回，并服从运行时安全策略。"""

    name = "create_file"
    description = (
        "Create a new file. "
        "Optionally provide initial content. "
        "Fails if file already exists."
    )

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.sandbox = _build_sandbox(config)
        self.secret_scanner = _build_secret_scanner(config)

    def is_destructive(self, **kwargs: Any) -> bool:
        return True

    def execute(self, file_path: str, content: str = "") -> ToolResult:
        """执行创建文件工具对应的实际操作，校验输入并返回统一结果。

        参数：
            file_path: 目标文件的路径；访问范围仍受项目沙箱约束。
            content: 需要处理、保存或分析的文本内容。

        返回：
            `ToolResult` 类型的处理结果。
        """
        is_allowed, reason = self.sandbox.is_path_allowed(file_path)
        if not is_allowed:
            return ToolResult(success=False, output="", error=reason)
        path = Path(file_path).resolve()

        if path.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"File already exists: {file_path}",
            )

        try:
            write_ok, write_result = self.sandbox.safe_write(path, content.encode("utf-8"))
            if not write_ok:
                return ToolResult(success=False, output="", error=str(write_result))
            _record_file_version(path)

            metadata: dict[str, Any] = {
                "file_path": str(path),
                "bytes_written": len(content),
                "change_type": "create",
            }
            if content.strip():
                diff_engine = DiffEngine()
                diff_text = diff_engine.generate_diff("", content, str(path))
                diff_text, secret_findings_count = _redact_tool_text(
                    self.secret_scanner,
                    diff_text,
                    self.config,
                )
                metadata["secret_findings_count"] = secret_findings_count
                if diff_text.strip():
                    metadata["diff"] = diff_text

            return ToolResult(
                success=True,
                output=f"Created file: {file_path}",
                metadata=metadata,
            )

        except PermissionError:
            return ToolResult(
                success=False,
                output="",
                error=f"Permission denied: {file_path}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to create file: {e}",
            )


class EditFileTool(BaseTool):
    """实现`EditFileTool`。模型通过统一工具 Schema 调用它，执行结果使用 ToolResult 返回，并服从运行时安全策略。"""

    name = "edit_file"
    search_hint = "Make a precise text replacement in an existing file"
    description = (
        "Edit a file by replacing an exact old_text string with new_text. "
        "Fails if old_text is missing or appears multiple times unless replace_all is true."
    )

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.sandbox = _build_sandbox(config)

    def execute(
        self,
        file_path: str,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
    ) -> ToolResult:
        is_allowed, reason = self.sandbox.is_path_allowed(file_path)
        if not is_allowed:
            return ToolResult(success=False, output="", error=reason)
        path = Path(file_path).resolve()

        if not old_text:
            return ToolResult(success=False, output="", error="old_text must not be empty")
        if _is_binary_file(str(path)):
            return ToolResult(success=False, output="", error=f"Cannot edit binary file: {file_path}")

        stale_result = _validate_observed_version(path)
        if stale_result is not None:
            return stale_result

        read_ok, read_result = self.sandbox.safe_read(path)
        if not read_ok:
            return ToolResult(success=False, output="", error=str(read_result))

        old_content = read_result.decode("utf-8", errors="replace") if isinstance(read_result, bytes) else ""

        # ── 换行归一化 ─────────────────────────────────────────────
        # 文件可能使用 CRLF（\r\n）或 LF（\n），而模型传入的 old_text 通常用 LF。
        # 为避免 "\r\n" != "\n" 导致匹配失败，统一转为 LF 后再做 count/replace，
        # 写回文件时根据原始风格还原换行符。
        original_crlf = "\r\n" in old_content
        normalized_content = old_content.replace("\r\n", "\n") if original_crlf else old_content
        normalized_old = old_text.replace("\r\n", "\n")

        occurrences = normalized_content.count(normalized_old)
        if occurrences == 0:
            return ToolResult(success=False, output="", error="old_text not found in file")
        if occurrences > 1 and not replace_all:
            return ToolResult(
                success=False,
                output="",
                error=f"old_text appears {occurrences} times; set replace_all=True or provide more context",
            )

        normalized_new = normalized_content.replace(normalized_old, new_text) if replace_all else normalized_content.replace(normalized_old, new_text, 1)

        # 写回前还原原始换行风格，避免意外改变文件格式。
        new_content = normalized_new.replace("\n", "\r\n") if original_crlf else normalized_new

        checkpoint_id = _create_write_checkpoint(self.config, path, "before edit_file")
        write_ok, write_result = self.sandbox.safe_write(path, new_content.encode("utf-8"))
        if not write_ok:
            return ToolResult(success=False, output="", error=str(write_result))
        _record_file_version(path)

        diff_text = DiffEngine().generate_diff(old_content, new_content, str(path))
        metadata: dict[str, Any] = {
            "file_path": str(path),
            "change_type": "edit",
            "occurrences_replaced": occurrences if replace_all else 1,
            "diff": diff_text,
        }
        if checkpoint_id:
            metadata["checkpoint_id"] = checkpoint_id
        return ToolResult(
            success=True,
            output=f"Edited file: {file_path}",
            metadata=metadata,
        )

    def is_destructive(self, **kwargs: Any) -> bool:
        return True


class MultiEditFileTool(BaseTool):
    """实现`MultiEditFileTool`。模型通过统一工具 Schema 调用它，执行结果使用 ToolResult 返回，并服从运行时安全策略。"""

    name = "multi_edit_file"
    search_hint = "Apply multiple precise replacements to one file"
    description = (
        "Apply multiple exact text replacements to one file. "
        "Each edit requires old_text and new_text, and all edits must match before writing."
    )

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.sandbox = _build_sandbox(config)

    def execute(self, file_path: str, edits: list[dict[str, Any]]) -> ToolResult:
        is_allowed, reason = self.sandbox.is_path_allowed(file_path)
        if not is_allowed:
            return ToolResult(success=False, output="", error=reason)
        path = Path(file_path).resolve()

        if not edits:
            return ToolResult(success=False, output="", error="edits must contain at least one edit")
        if _is_binary_file(str(path)):
            return ToolResult(success=False, output="", error=f"Cannot edit binary file: {file_path}")

        stale_result = _validate_observed_version(path)
        if stale_result is not None:
            return stale_result

        read_ok, read_result = self.sandbox.safe_read(path)
        if not read_ok:
            return ToolResult(success=False, output="", error=str(read_result))

        old_content = read_result.decode("utf-8", errors="replace") if isinstance(read_result, bytes) else ""

        # ── 换行归一化 ─────────────────────────────────────────────
        # 与 EditFileTool 相同的 CRLF/LF 处理：统一为 LF 做匹配，写回时还原。
        original_crlf = "\r\n" in old_content
        new_content = old_content.replace("\r\n", "\n") if original_crlf else old_content
        replaced = 0

        for index, edit in enumerate(edits, 1):
            old_text = str(edit.get("old_text", ""))
            new_text = str(edit.get("new_text", ""))
            replace_all = bool(edit.get("replace_all", False))
            if not old_text:
                return ToolResult(success=False, output="", error=f"edit {index}: old_text must not be empty")
            normalized_old = old_text.replace("\r\n", "\n")
            occurrences = new_content.count(normalized_old)
            if occurrences == 0:
                return ToolResult(success=False, output="", error=f"edit {index}: old_text not found")
            if occurrences > 1 and not replace_all:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"edit {index}: old_text appears {occurrences} times; set replace_all=True",
                )
            new_content = new_content.replace(normalized_old, new_text) if replace_all else new_content.replace(normalized_old, new_text, 1)
            replaced += occurrences if replace_all else 1

        # 写回前还原原始换行风格，避免意外改变文件格式。
        final_content = new_content.replace("\n", "\r\n") if original_crlf else new_content

        checkpoint_id = _create_write_checkpoint(self.config, path, "before multi_edit_file")
        write_ok, write_result = self.sandbox.safe_write(path, final_content.encode("utf-8"))
        if not write_ok:
            return ToolResult(success=False, output="", error=str(write_result))
        _record_file_version(path)

        diff_text = DiffEngine().generate_diff(old_content, final_content, str(path))
        metadata: dict[str, Any] = {
            "file_path": str(path),
            "change_type": "multi_edit",
            "occurrences_replaced": replaced,
            "diff": diff_text,
        }
        if checkpoint_id:
            metadata["checkpoint_id"] = checkpoint_id
        return ToolResult(
            success=True,
            output=f"Edited file: {file_path} ({replaced} replacements)",
            metadata=metadata,
        )

    def is_destructive(self, **kwargs: Any) -> bool:
        return True


class DeleteFileTool(BaseTool):
    """实现删除文件工具。模型通过统一工具 Schema 调用它，执行结果使用 ToolResult 返回，并服从运行时安全策略。"""

    name = "delete_file"
    description = (
        "Delete a file. "
        "This is a destructive operation and requires confirmation. "
        "Use with extreme caution."
    )

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.sandbox = _build_sandbox(config)

    def is_destructive(self, **kwargs: Any) -> bool:
        return True

    def execute(self, file_path: str, confirm: bool = False) -> ToolResult:
        """执行删除文件工具对应的实际操作，校验输入并返回统一结果。

        参数：
            file_path: 目标文件的路径；访问范围仍受项目沙箱约束。
            confirm: 可选的`confirm`。

        返回：
            `ToolResult` 类型的处理结果。
        """
        if not confirm:
            return ToolResult(
                success=False,
                output="Deletion not confirmed. Set confirm=True to proceed.",
                error="Deletion requires confirmation",
                metadata={"requires_confirmation": True},
            )

        is_allowed, reason = self.sandbox.is_path_allowed(file_path)
        if not is_allowed:
            return ToolResult(success=False, output="", error=reason)
        resolved_path = str(Path(file_path).resolve())

        try:
            path = Path(resolved_path)
            if not path.is_file():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Not a file: {file_path}",
                )

            stale_result = _validate_observed_version(path)
            if stale_result is not None:
                return stale_result

            ok, delete_result = self.sandbox.safe_delete(path)
            if not ok:
                return ToolResult(success=False, output="", error=str(delete_result))
            _record_file_version(path)

            return ToolResult(
                success=True,
                output=f"Deleted file: {file_path}",
                metadata={"file_path": resolved_path},
            )

        except PermissionError:
            return ToolResult(
                success=False,
                output="",
                error=f"Permission denied: {file_path}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to delete file: {e}",
            )


class ListDirectoryTool(BaseTool):
    """实现列表目录工具。模型通过统一工具 Schema 调用它，执行结果使用 ToolResult 返回，并服从运行时安全策略。"""

    name = "list_directory"
    description = (
        "List the contents of a directory. "
        "Returns files and subdirectories with their types."
    )

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.sandbox = _build_sandbox(config)

    def is_read_only(self, **kwargs: Any) -> bool:
        return True

    def execute(
        self,
        directory: str = ".",
        recursive: bool = False,
        max_depth: int = 3,
    ) -> ToolResult:
        """执行列表目录工具对应的实际操作，校验输入并返回统一结果。

        参数：
            directory: 可选的目录。
            recursive: 可选的`recursive`。
            max_depth: 可选的`max_depth`。

        返回：
            `ToolResult` 类型的处理结果。
        """
        try:
            path = Path(directory).resolve()
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Invalid path: {e}")

        if not path.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"Directory does not exist: {directory}",
            )

        if not path.is_dir():
            return ToolResult(
                success=False,
                output="",
                error=f"Not a directory: {directory}",
            )
        is_allowed, reason = self.sandbox.is_path_allowed(path)
        if not is_allowed:
            return ToolResult(success=False, output="", error=reason)

        def format_entry(entry: Path, depth: int = 0) -> str:
            """把条目整理为稳定、便于展示的文本格式。

            参数：
                entry: 本次操作使用的条目。
                depth: 可选的`depth`。

            返回：
                处理后的文本或稳定标识。
            """
            prefix = "  " * depth
            if entry.is_dir():
                return f"{prefix}📁 {entry.name}/"
            elif entry.is_symlink():
                return f"{prefix}🔗 {entry.name} -> {entry.resolve()}"
            else:
                size = entry.stat().st_size
                return f"{prefix}📄 {entry.name} ({size} bytes)"

        def list_recursive(p: Path, depth: int = 0) -> list[str]:
            """列出 `recursive` 对应的对象，并按当前组件约定返回稳定顺序。

            参数：
                p: 本次操作使用的`p`。
                depth: 可选的`depth`。

            返回：
                按调用约定排序的结果列表。
            """
            if depth > max_depth:
                return []

            entries = []
            try:
                for entry in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name)):
                    entry_allowed, _ = self.sandbox.is_path_allowed(entry)
                    if not entry_allowed:
                        continue
                    entries.append(format_entry(entry, depth))
                    if entry.is_dir() and recursive and not entry.name.startswith("."):
                        entries.extend(list_recursive(entry, depth + 1))
            except PermissionError:
                entries.append("  " * depth + "⚠️  Permission denied")

            return entries

        try:
            if recursive:
                lines = [f"Directory: {directory} (recursive, max depth: {max_depth})\n"]
                lines.extend(list_recursive(path))
            else:
                lines = [f"Directory: {directory}\n"]
                for entry in sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name)):
                    entry_allowed, _ = self.sandbox.is_path_allowed(entry)
                    if not entry_allowed:
                        continue
                    lines.append(format_entry(entry))

            output = "\n".join(lines)
            return ToolResult(
                success=True,
                output=_truncate_output(output),
                metadata={"directory": str(path), "recursive": recursive},
            )

        except PermissionError:
            return ToolResult(
                success=False,
                output="",
                error=f"Permission denied: {directory}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to list directory: {e}",
            )
