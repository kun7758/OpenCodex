"""差异与补丁子系统中的执行引擎模块，集中定义相关数据结构、边界适配和实现逻辑。"""

import difflib
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class ApplyResult:
    """数据对象 `ApplyResult` 主要保存
    `success`、`message`、`file_path`、`backup_path`、`lines_added`、`lines_removed`、`error`
    字段，用于在组件之间传递或持久化这组状态。
    """

    success: bool
    message: str
    file_path: str | None = None
    backup_path: str | None = None
    lines_added: int = 0
    lines_removed: int = 0
    error: str | None = None


@dataclass
class Hunk:
    """数据对象 `Hunk` 主要保存 `old_start`、`old_count`、`new_start`、`new_count`、`lines`
    字段，用于在组件之间传递或持久化这组状态。
    """

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str] = field(default_factory=list)

    def to_string(self) -> str:
        """把`Hunk`转换为适合写入模型上下文或终端展示的文本。

        返回：
            处理后的文本或稳定标识。
        """
        header = f"@@ -{self.old_start},{self.old_count} +{self.new_start},{self.new_count} @@"
        return "\n".join([header] + self.lines)


class DiffEngine:
    """封装差异执行引擎相关的状态和操作，使调用方通过稳定接口使用该能力。"""

    def __init__(self, backup_dir: str = ".opennova/backups"):
        """初始化差异执行引擎，保存后续操作需要的依赖、配置和初始状态。

        参数：
            backup_dir: 可选的`backup_dir`。

        说明：
            执行过程中会更新当前实例维护的状态。
            该操作会访问本地文件系统，路径校验和原子写入约束由所在组件负责。
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def generate_diff(
        self,
        original: str,
        modified: str,
        file_path: str = "file",
        context_lines: int = 3,
    ) -> str:
        """生成差异，并按照当前组件的约定返回结果。

        参数：
            original: 本次操作使用的`original`。
            modified: 本次操作使用的已修改文件。
            file_path: 目标文件的路径；访问范围仍受项目沙箱约束。
            context_lines: 可选的`context_lines`。

        返回：
            处理后的文本或稳定标识。
        """
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)

        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            n=context_lines,
        )

        return "".join(diff)

    def parse_diff(self, diff_text: str) -> list[Hunk]:
        """解析差异并转换为内部使用的规范结构。

        参数：
            diff_text: 本次操作使用的`diff_text`。

        返回：
            按调用约定排序的结果列表。
        """
        hunks = []
        current_hunk = None
        hunk_pattern = re.compile(
            r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@"
        )

        for line in diff_text.splitlines():
            match = hunk_pattern.match(line)
            if match:
                if current_hunk:
                    hunks.append(current_hunk)

                old_start = int(match.group(1))
                old_count = int(match.group(2) or "1")
                new_start = int(match.group(3))
                new_count = int(match.group(4) or "1")

                current_hunk = Hunk(
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                )
            elif current_hunk and (line.startswith("+") or line.startswith("-") or line.startswith(" ") or line.startswith("\\ ")):
                current_hunk.lines.append(line)

        if current_hunk:
            hunks.append(current_hunk)

        return hunks

    def validate_patch(self, diff_text: str) -> tuple[bool, str]:
        """校验`validate_patch`，发现问题时返回或抛出明确错误。

        参数：
            diff_text: 本次操作使用的`diff_text`。

        返回：
            `tuple[bool, str]` 类型的处理结果。
        """
        if not diff_text.strip():
            return False, "Empty diff"

        if not diff_text.startswith("---"):
            return False, "Invalid diff format: missing --- header"

        if not re.search(r"\+\+\+ b/", diff_text):
            return False, "Invalid diff format: missing +++ header"

        hunks = self.parse_diff(diff_text)

        if not hunks:
            return False, "No hunks found in diff"

        for i, hunk in enumerate(hunks):
            if hunk.old_start < 1:
                return False, f"Hunk {i + 1}: invalid old_start"

            if hunk.new_start < 1:
                return False, f"Hunk {i + 1}: invalid new_start"

        return True, ""

    def apply_patch(
        self,
        file_path: str,
        diff_text: str,
        backup: bool = True,
    ) -> ApplyResult:
        """应用 `patch` 对应的数据，并按照当前组件的约定返回结果。

        参数：
            file_path: 目标文件的路径；访问范围仍受项目沙箱约束。
            diff_text: 本次操作使用的`diff_text`。
            backup: 可选的`backup`。

        返回：
            `ApplyResult` 类型的处理结果。

        说明：
            该操作会访问本地文件系统，路径校验和原子写入约束由所在组件负责。
        """
        is_valid, error = self.validate_patch(diff_text)
        if not is_valid:
            return ApplyResult(success=False, message="Invalid patch", error=error)

        path = Path(file_path)

        if not path.exists():
            return ApplyResult(
                success=False,
                message=f"File not found: {file_path}",
                error="File not found",
            )

        try:
            original_content = path.read_text(encoding="utf-8")
        except Exception as e:
            return ApplyResult(
                success=False,
                message=f"Failed to read file: {e}",
                error=str(e),
            )

        backup_path = None
        if backup:
            backup_path = self._create_backup(file_path, original_content)

        try:
            patched_content = self._apply_diff(original_content, diff_text)

            path.write_text(patched_content, encoding="utf-8")

            lines_added, lines_removed = self._count_changes(diff_text)

            return ApplyResult(
                success=True,
                message=f"Patch applied successfully to {file_path}",
                file_path=str(path),
                backup_path=backup_path,
                lines_added=lines_added,
                lines_removed=lines_removed,
            )

        except Exception as e:
            if backup_path and Path(backup_path).exists():
                Path(path).write_text(original_content, encoding="utf-8")

            return ApplyResult(
                success=False,
                message=f"Failed to apply patch: {e}",
                error=str(e),
                backup_path=backup_path,
            )

    def _apply_diff(self, original: str, diff_text: str) -> str:
        """应用差异，并按照当前组件的约定返回结果。

        参数：
            original: 本次操作使用的`original`。
            diff_text: 本次操作使用的`diff_text`。

        返回：
            处理后的文本或稳定标识。
        """
        original_lines = original.splitlines()
        hunks = self.parse_diff(diff_text)

        result_lines = original_lines.copy()
        offset = 0

        for hunk in hunks:
            start_idx = hunk.old_start - 1 + offset
            end_idx = start_idx + hunk.old_count

            for i in range(min(hunk.old_count, len(result_lines) - start_idx)):
                if i + start_idx < len(result_lines):
                    expected = hunk.lines[i] if i < len(hunk.lines) else None
                    if expected and expected.startswith("-"):
                        actual = result_lines[start_idx + i]
                        if expected[1:].strip() != actual.strip():
                            pass

            new_lines = []
            old_lines_to_remove = 0

            for line in hunk.lines:
                if line.startswith("-"):
                    old_lines_to_remove += 1
                elif line.startswith("+") or line.startswith(" "):
                    new_lines.append(line[1:])
                elif line.startswith("\\"):
                    pass

            result_lines = result_lines[:start_idx] + new_lines + result_lines[end_idx:]
            offset += len(new_lines) - old_lines_to_remove

        return "\n".join(result_lines) + ("\n" if original.endswith("\n") else "")

    def _create_backup(self, file_path: str, content: str) -> str:
        """创建`create_backup`并完成必要的初始化。

        参数：
            file_path: 目标文件的路径；访问范围仍受项目沙箱约束。
            content: 需要处理、保存或分析的文本内容。

        返回：
            处理后的文本或稳定标识。

        说明：
            该操作会访问本地文件系统，路径校验和原子写入约束由所在组件负责。
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{Path(file_path).stem}_{timestamp}.bak"
        backup_path = self.backup_dir / backup_name

        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_text(content, encoding="utf-8")

        return str(backup_path)

    def _count_changes(self, diff_text: str) -> tuple[int, int]:
        """统计 `changes` 对应的数据，并按照当前组件的约定返回结果。

        参数：
            diff_text: 本次操作使用的`diff_text`。

        返回：
            `tuple[int, int]` 类型的处理结果。
        """
        added = 0
        removed = 0

        for line in diff_text.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                added += 1
            elif line.startswith("-") and not line.startswith("---"):
                removed += 1

        return added, removed

    def preview_diff(self, diff_text: str) -> str:
        """预览差异，并按照当前组件的约定返回结果。

        参数：
            diff_text: 本次操作使用的`diff_text`。

        返回：
            处理后的文本或稳定标识。
        """
        lines = []

        for line in diff_text.splitlines():
            if line.startswith("+++") or line.startswith("---"):
                lines.append(f"\033[1;36m{line}\033[0m")
            elif line.startswith("@@"):
                lines.append(f"\033[1;34m{line}\033[0m")
            elif line.startswith("+"):
                lines.append(f"\033[32m{line}\033[0m")
            elif line.startswith("-"):
                lines.append(f"\033[31m{line}\033[0m")
            else:
                lines.append(line)

        return "\n".join(lines)

    def reverse_diff(self, diff_text: str) -> str:
        """反转差异，并按照当前组件的约定返回结果。

        参数：
            diff_text: 本次操作使用的`diff_text`。

        返回：
            处理后的文本或稳定标识。
        """
        lines = diff_text.splitlines()
        reversed_lines = []

        for line in lines:
            if line.startswith("---"):
                reversed_lines.append(line.replace("--- ", "+++ ", 1))
            elif line.startswith("+++"):
                reversed_lines.append(line.replace("+++ ", "--- ", 1))
            elif line.startswith("-") and not line.startswith("---"):
                reversed_lines.append("+" + line[1:])
            elif line.startswith("+") and not line.startswith("+++"):
                reversed_lines.append("-" + line[1:])
            else:
                reversed_lines.append(line)

        return "\n".join(reversed_lines)
