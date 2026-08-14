"""差异与补丁子系统中的变更集模块，集中定义相关数据结构、边界适配和实现逻辑。"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from opennova.diff.engine import ApplyResult, DiffEngine
from opennova.diff.parser import ChangeType, FileChange


@dataclass
class ChangeResult:
    """保存变更结果所需的结构化数据，主要包含 `success`、`message`、`applied_changes`、`failed_changes`、`backup_dir`
    字段，便于在组件之间传递或持久化。
    """

    success: bool
    message: str
    applied_changes: list[ApplyResult] = field(default_factory=list)
    failed_changes: list[tuple[FileChange, str]] = field(default_factory=list)
    backup_dir: str | None = None

    @property
    def total_changes(self) -> int:
        """计算并返回 `total_changes` 属性；读取该属性不会主动改变对象的业务状态。

        返回：
            `int` 类型的处理结果。
        """
        return len(self.applied_changes) + len(self.failed_changes)

    @property
    def success_count(self) -> int:
        """处理成功数量，并按照当前组件的约定返回结果。

        返回：
            `int` 类型的处理结果。
        """
        return len(self.applied_changes)

    @property
    def failure_count(self) -> int:
        """处理失败数量，并按照当前组件的约定返回结果。

        返回：
            `int` 类型的处理结果。
        """
        return len(self.failed_changes)


@dataclass
class ChangeSet:
    """保存变更设置所需的结构化数据，主要包含 `task`、`changes`、`created_at`、`metadata`、`_engine` 字段，便于在组件之间传递或持久化。"""

    task: str
    changes: list[FileChange] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    _engine: DiffEngine | None = field(default=None, repr=False)

    def __post_init__(self):
        """在数据类字段初始化后规范化变更设置的派生状态。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        if self._engine is None:
            self._engine = DiffEngine()

    def add_change(self, change: FileChange) -> None:
        """添加`add_change`，必要时执行去重或容量检查。

        参数：
            change: 本次操作使用的变更。
        """
        self.changes.append(change)

    def add_new_file(self, file_path: str, content: str) -> None:
        """添加`add_new_file`，必要时执行去重或容量检查。

        参数：
            file_path: 目标文件的路径；访问范围仍受项目沙箱约束。
            content: 需要处理、保存或分析的文本内容。
        """
        change = FileChange(
            file_path=file_path,
            change_type=ChangeType.CREATE,
            new_content=content,
        )
        self.add_change(change)

    def add_modification(
        self,
        file_path: str,
        original: str,
        new_content: str,
    ) -> None:
        """添加`add_modification`，必要时执行去重或容量检查。

        参数：
            file_path: 目标文件的路径；访问范围仍受项目沙箱约束。
            original: 本次操作使用的`original`。
            new_content: 本次操作使用的`new_content`。
        """
        diff = self._engine.generate_diff(original, new_content, file_path)
        change = FileChange(
            file_path=file_path,
            change_type=ChangeType.MODIFY,
            original_content=original,
            new_content=new_content,
            diff=diff,
        )
        self.add_change(change)

    def add_deletion(self, file_path: str) -> None:
        """添加`add_deletion`，必要时执行去重或容量检查。

        参数：
            file_path: 目标文件的路径；访问范围仍受项目沙箱约束。
        """
        change = FileChange(
            file_path=file_path,
            change_type=ChangeType.DELETE,
        )
        self.add_change(change)

    def get_preview(self) -> str:
        """读取预览，不改变当前对象的业务状态。

        返回：
            处理后的文本或稳定标识。
        """
        lines = [f"ChangeSet for: {self.task}", "=" * 50, ""]

        for i, change in enumerate(self.changes, 1):
            lines.append(f"[{i}] {change.change_type.value.upper()}: {change.file_path}")

            if change.change_type == ChangeType.CREATE:
                lines.append(f"    New file ({len(change.new_content or '')} bytes)")
            elif change.change_type == ChangeType.MODIFY:
                added, removed = change.get_lines_changed()
                lines.append(f"    +{added} lines, -{removed} lines")
            elif change.change_type == ChangeType.DELETE:
                lines.append("    File will be deleted")

            lines.append("")

        return "\n".join(lines)

    def get_diff_preview(self) -> str:
        """读取差异预览，不改变当前对象的业务状态。

        返回：
            处理后的文本或稳定标识。
        """
        previews = []

        for change in self.changes:
            if change.diff:
                preview = self._engine.preview_diff(change.diff)
                previews.append(f"--- {change.file_path} ---\n{preview}")

        return "\n\n".join(previews)

    def apply(self, backup: bool = True) -> ChangeResult:
        """更新 `apply` 所表示的数据或流程，并遵守变更设置定义的边界与状态约束。

        参数：
            backup: 可选的`backup`。

        返回：
            `ChangeResult` 类型的处理结果。
        """
        applied = []
        failed = []
        backup_dir = None

        for change in self.changes:
            result = self._apply_single_change(change, backup)

            if result.success:
                applied.append(result)
                if result.backup_path and not backup_dir:
                    backup_dir = str(Path(result.backup_path).parent)
            else:
                failed.append((change, result.error or result.message))

                if applied and backup:
                    self._rollback(applied)

                break

        return ChangeResult(
            success=len(failed) == 0,
            message=f"Applied {len(applied)}/{len(self.changes)} changes",
            applied_changes=applied,
            failed_changes=failed,
            backup_dir=backup_dir,
        )

    def _apply_single_change(
        self,
        change: FileChange,
        backup: bool,
    ) -> ApplyResult:
        """应用单个变更，并按照当前组件的约定返回结果。

        参数：
            change: 本次操作使用的变更。
            backup: 本次操作使用的`backup`。

        返回：
            `ApplyResult` 类型的处理结果。
        """
        path = Path(change.file_path)

        if change.change_type == ChangeType.CREATE:
            return self._apply_create(path, change.new_content or "", backup)
        elif change.change_type == ChangeType.MODIFY:
            return self._apply_modify(path, change.diff or "", backup)
        elif change.change_type == ChangeType.DELETE:
            return self._apply_delete(path, backup)

        return ApplyResult(success=False, message="Unknown change type")

    def _apply_create(
        self,
        path: Path,
        content: str,
        backup: bool,
    ) -> ApplyResult:
        """应用创建，并按照当前组件的约定返回结果。

        参数：
            path: 需要读取、检查或写入的路径。
            content: 需要处理、保存或分析的文本内容。
            backup: 本次操作使用的`backup`。

        返回：
            `ApplyResult` 类型的处理结果。

        说明：
            该操作会访问本地文件系统，路径校验和原子写入约束由所在组件负责。
        """
        try:
            if path.exists():
                return ApplyResult(
                    success=False,
                    message=f"File already exists: {path}",
                    error="File exists",
                )

            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

            return ApplyResult(
                success=True,
                message=f"Created file: {path}",
                file_path=str(path),
            )

        except Exception as e:
            return ApplyResult(
                success=False,
                message=f"Failed to create file: {e}",
                error=str(e),
            )

    def _apply_modify(
        self,
        path: Path,
        diff: str,
        backup: bool,
    ) -> ApplyResult:
        """应用修改，并按照当前组件的约定返回结果。

        参数：
            path: 需要读取、检查或写入的路径。
            diff: 本次操作使用的差异。
            backup: 本次操作使用的`backup`。

        返回：
            `ApplyResult` 类型的处理结果。
        """
        return self._engine.apply_patch(str(path), diff, backup)

    def _apply_delete(self, path: Path, backup: bool) -> ApplyResult:
        """应用删除，并按照当前组件的约定返回结果。

        参数：
            path: 需要读取、检查或写入的路径。
            backup: 本次操作使用的`backup`。

        返回：
            `ApplyResult` 类型的处理结果。

        说明：
            该操作会访问本地文件系统，路径校验和原子写入约束由所在组件负责。
        """
        try:
            if not path.exists():
                return ApplyResult(
                    success=False,
                    message=f"File not found: {path}",
                    error="File not found",
                )

            backup_path = None
            if backup:
                content = path.read_text(encoding="utf-8")
                backup_path = self._engine._create_backup(str(path), content)

            path.unlink()

            return ApplyResult(
                success=True,
                message=f"Deleted file: {path}",
                file_path=str(path),
                backup_path=backup_path,
            )

        except Exception as e:
            return ApplyResult(
                success=False,
                message=f"Failed to delete file: {e}",
                error=str(e),
            )

    def _rollback(self, applied: list[ApplyResult]) -> None:
        """处理回滚，并按照当前组件的约定返回结果。

        参数：
            applied: 本次操作使用的`applied`。

        说明：
            该操作会访问本地文件系统，路径校验和原子写入约束由所在组件负责。
        """
        for result in reversed(applied):
            if result.backup_path:
                backup = Path(result.backup_path)
                if backup.exists():
                    content = backup.read_text(encoding="utf-8")
                    Path(result.file_path).write_text(content, encoding="utf-8")

    def to_dict(self) -> dict[str, Any]:
        """把变更设置转换为可序列化字典，供事件、会话或 API 边界使用。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        return {
            "task": self.task,
            "created_at": self.created_at.isoformat(),
            "changes": [
                {
                    "file_path": c.file_path,
                    "change_type": c.change_type.value,
                    "diff": c.diff,
                    "new_content": c.new_content,
                }
                for c in self.changes
            ],
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """把变更设置转换为JSON，供对应协议或边界直接使用。

        返回：
            处理后的文本或稳定标识。
        """
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChangeSet":
        """从字典恢复变更设置，并为旧数据缺失的字段补充兼容默认值。

        参数：
            data: 用于构造或恢复对象的结构化数据。

        返回：
            `'ChangeSet'` 类型的处理结果。
        """
        changes = []
        for c in data.get("changes", []):
            change = FileChange(
                file_path=c["file_path"],
                change_type=ChangeType(c["change_type"]),
                diff=c.get("diff"),
                new_content=c.get("new_content"),
            )
            changes.append(change)

        return cls(
            task=data["task"],
            changes=changes,
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ChangeSet":
        """从JSON恢复变更设置，并校验必要字段。

        参数：
            json_str: 本次操作使用的`json_str`。

        返回：
            `'ChangeSet'` 类型的处理结果。
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __len__(self) -> int:
        return len(self.changes)

    def __iter__(self):
        return iter(self.changes)

    def __repr__(self) -> str:
        return f"ChangeSet(task={self.task!r}, changes={len(self.changes)})"
