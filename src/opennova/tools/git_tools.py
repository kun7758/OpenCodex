"""内置工具系统中的Git工具模块，集中定义相关数据结构、边界适配和实现逻辑。"""

import subprocess
from dataclasses import dataclass, field
from typing import Any

from opennova.tools.base import BaseTool, ToolResult


@dataclass
class GitStatusInfo:
    """保存Git状态信息所需的结构化数据，主要包含 `branch`、`staged`、`unstaged`、`untracked`、`has_changes`
    字段，便于在组件之间传递或持久化。
    """

    branch: str
    staged: list[str] = field(default_factory=list)
    unstaged: list[str] = field(default_factory=list)
    untracked: list[str] = field(default_factory=list)
    has_changes: bool = False


class GitCommitTool(BaseTool):
    """实现Git提交工具。模型通过统一工具 Schema 调用它，执行结果使用 ToolResult 返回，并服从运行时安全策略。"""

    name = "git_commit"
    description = "Create a git commit. Stage relevant files first using Bash(git add path/to/file), then commit with a descriptive message following this repository's commit message style."

    def execute(
        self,
        message: str | None = None,
        amend: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        """执行Git提交工具对应的实际操作，校验输入并返回统一结果。

        参数：
            message: 用户提交或组件间传递的消息。
            amend: 可选的`amend`。
            **kwargs: 传递给底层实现的额外关键字参数。

        返回：
            `ToolResult` 类型的处理结果。
        """
        try:
            if message is None:
                # 根据当前差异分析变更类型并生成提交说明草稿。
                message = self._generate_commit_message()

            # 构造参数化的 Git 提交命令。
            command = ["git", "commit", "-m", message]

            if amend:
                command.append("--amend")

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                return ToolResult(
                    success=False,
                    output="",
                    error=result.stderr or "Commit failed",
                )

            # 提交成功后读取新提交的哈希值。
            hash_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )

            commit_hash = hash_result.stdout.strip()[:8]

            return ToolResult(
                success=True,
                output=f"Created commit {commit_hash}: {message[:50]}{'...' if len(message) > 50 else ''}",
                metadata={
                    "commit_hash": commit_hash,
                    "message": message,
                },
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _generate_commit_message(self) -> str:
        """生成提交消息，并按照当前组件的约定返回结果。

        返回：
            处理后的文本或稳定标识。
        """
        try:
            # 读取 Git 差异，作为生成提交说明的依据。
            diff_result = subprocess.run(
                ["git", "diff", "--cached", "--stat"],
                capture_output=True,
                text=True,
                check=True,
            )

            # 对差异做轻量分析，不调用额外模型。
            changed_files = []
            for line in diff_result.stdout.split("\n"):
                if "|" in line:
                    parts = line.split("|")
                    if len(parts) >= 3:
                        file_path = parts[0].strip()
                        stats = parts[2].strip()
                        changed_files.append(f"{file_path} ({stats})")

            if not changed_files:
                return "Update changes"

            # 根据变更内容选择合适的提交类型。
            return f"Update code with changes to: {', '.join(changed_files[:3])}"

        except Exception:
            return "Update changes"


class GitStatusTool(BaseTool):
    """实现Git状态工具。模型通过统一工具 Schema 调用它，执行结果使用 ToolResult 返回，并服从运行时安全策略。"""

    name = "git_status"
    description = "Show the working tree status. Use this to see staged, unstaged, and untracked files in the repository."

    def execute(self, **kwargs: Any) -> ToolResult:
        """执行Git状态工具对应的实际操作，校验输入并返回统一结果。

        参数：
            **kwargs: 传递给底层实现的额外关键字参数。

        返回：
            `ToolResult` 类型的处理结果。
        """
        try:
            result = subprocess.run(
                ["git", "status", "--short"],
                capture_output=True,
                text=True,
                check=True,
            )

            status_info = self._parse_git_status(result.stdout)

            output_lines = [
                f"Branch: {status_info.branch}",
                "",
            ]

            if status_info.staged:
                output_lines.append("Staged changes:")
                for path in status_info.staged:
                    output_lines.append(f"  {path}")

            if status_info.unstaged:
                output_lines.append("Unstaged changes:")
                for path in status_info.unstaged:
                    output_lines.append(f"  {path}")

            if status_info.untracked:
                output_lines.append("Untracked files:")
                for path in status_info.untracked:
                    output_lines.append(f"  {path}")

            if not status_info.has_changes:
                output_lines.append("(No changes)")

            return ToolResult(
                success=True,
                output="\n".join(output_lines),
                metadata={
                    "branch": status_info.branch,
                    "staged": status_info.staged,
                    "unstaged": status_info.unstaged,
                    "untracked": status_info.untracked,
                    "has_changes": status_info.has_changes,
                },
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _parse_git_status(self, output: str) -> GitStatusInfo:
        """解析Git状态并转换为内部使用的规范结构。

        参数：
            output: 本次操作使用的输出。

        返回：
            `GitStatusInfo` 类型的处理结果。
        """
        # 读取当前 Git 分支名称。
        try:
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True,
                text=True,
                check=True,
            )
            branch = branch_result.stdout.strip()
        except Exception:
            branch = "unknown"

        staged = []
        unstaged = []
        untracked = []
        has_changes = False

        for line in output.split("\n"):
            if not line:
                continue

            status_code = line[:2]
            path = line[3:]

            has_changes = True

            if status_code[0] in "M":  # 工作区已修改
                if status_code[1] == "M":  # 暂存区已修改
                    staged.append(path)
                else:
                    unstaged.append(path)
            elif status_code[0] == "A":  # 新增文件
                staged.append(path)
            elif status_code[0] == "D":  # 删除文件
                unstaged.append(path)
            elif status_code[0] == "?":  # 未跟踪文件
                untracked.append(path)

        return GitStatusInfo(
            branch=branch,
            staged=staged,
            unstaged=unstaged,
            untracked=untracked,
            has_changes=has_changes,
        )


class GitDiffTool(BaseTool):
    """实现Git差异工具。模型通过统一工具 Schema 调用它，执行结果使用 ToolResult 返回，并服从运行时安全策略。"""

    name = "git_diff"
    description = "Show unstaged and staged changes. Use this to review what will be committed or examine the differences between revisions."

    def execute(
        self,
        cached: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        """执行Git差异工具对应的实际操作，校验输入并返回统一结果。

        参数：
            cached: 可选的`cached`。
            **kwargs: 传递给底层实现的额外关键字参数。

        返回：
            `ToolResult` 类型的处理结果。
        """
        try:
            command = ["git", "diff"]
            if cached:
                command.append("--cached")

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
            )

            # 限制返回给模型的日志长度，避免占满上下文。
            output = result.stdout
            if len(output) > 10000:
                output = output[:10000] + "\n... (diff truncated)"

            return ToolResult(
                success=True,
                output=output or "No changes to show.",
                metadata={
                    "cached": cached,
                },
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class GitLogTool(BaseTool):
    """实现Git日志工具。模型通过统一工具 Schema 调用它，执行结果使用 ToolResult 返回，并服从运行时安全策略。"""

    name = "git_log"
    description = "Show commit history. Use this to review recent changes, understand project evolution, or investigate when something was changed."

    def execute(
        self,
        max_count: int = 10,
        **kwargs: Any,
    ) -> ToolResult:
        """执行Git日志工具对应的实际操作，校验输入并返回统一结果。

        参数：
            max_count: 可选的最大值数量。
            **kwargs: 传递给底层实现的额外关键字参数。

        返回：
            `ToolResult` 类型的处理结果。
        """
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", f"-{max_count}"],
                capture_output=True,
                text=True,
                check=True,
            )

            if not result.stdout.strip():
                return ToolResult(
                    success=True,
                    output="No commits in history.",
                    metadata={"commits": []},
                )

            commits = result.stdout.strip().split("\n")

            output_lines = ["Recent commits:", ""]
            for commit in commits:
                output_lines.append(f"  {commit}")

            return ToolResult(
                success=True,
                output="\n".join(output_lines),
                metadata={
                    "commits": commits,
                    "count": len(commits),
                },
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class GitBranchTool(BaseTool):
    """实现Git分支工具。模型通过统一工具 Schema 调用它，执行结果使用 ToolResult 返回，并服从运行时安全策略。"""

    name = "git_branch"
    description = "List all branches. Use this to see available branches or switch between different branches."

    def execute(
        self,
        **kwargs: Any,
    ) -> ToolResult:
        """执行Git分支工具对应的实际操作，校验输入并返回统一结果。

        参数：
            **kwargs: 传递给底层实现的额外关键字参数。

        返回：
            `ToolResult` 类型的处理结果。
        """
        try:
            result = subprocess.run(
                ["git", "branch", "-a"],
                capture_output=True,
                text=True,
                check=True,
            )

            if not result.stdout.strip():
                return ToolResult(
                    success=True,
                    output="No branches found.",
                    metadata={"branches": []},
                )

            branches = result.stdout.strip().split("\n")

            output_lines = ["Branches:", ""]
            current_branch = ""
            for branch in branches:
                branch_clean = branch.replace("*", "").strip()
                if "*" in branch:
                    current_branch = branch_clean
                    output_lines.append(f"  * {branch_clean} (current)")
                else:
                    output_lines.append(f"    {branch_clean}")

            return ToolResult(
                success=True,
                output="\n".join(output_lines),
                metadata={
                    "branches": [b.replace("*", "").strip() for b in branches],
                    "current": current_branch,
                },
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
