"""终端交互层中的检查点命令模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from __future__ import annotations

from pathlib import Path

from opennova.checkpoints import CheckpointManager
from opennova.tools.base import ToolResult
from opennova.transcript import extract_checkpoint_index, resolve_checkpoint_diff_from_session


def handle_checkpoint_command(project_path: str | Path, args: str) -> ToolResult:
    """处理检查点命令，协调输入校验、状态变化和结果返回。

    参数：
        project_path: 本次操作使用的项目路径。
        args: 调用方传入的位置参数或 Skill 参数文本。

    返回：
        `ToolResult` 类型的处理结果。
    """
    manager = CheckpointManager(project_path)
    tokens = (args or "list").split()
    command = tokens[0] if tokens else "list"

    try:
        if command == "list":
            checkpoints = manager.list_checkpoints()
            output = (
                "\n".join(
                    f"{checkpoint.id[:8]} {checkpoint.label} files={len(checkpoint.files)}"
                    for checkpoint in checkpoints
                )
                or "No checkpoints found."
            )
            return ToolResult(success=True, output=output, metadata={"checkpoints": checkpoints})

        if command in {"diff", "restore", "rewind"}:
            if command == "diff" and len(tokens) == 4 and tokens[1] == "--from-transcript":
                transcript_path = tokens[2]
                checkpoint_id = tokens[3]
                for item in extract_checkpoint_index(transcript_path):
                    if item["checkpoint_id"].startswith(checkpoint_id):
                        return ToolResult(
                            success=True, output=item["diff"], metadata={"checkpoint": item}
                        )
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Checkpoint not found in transcript: {checkpoint_id}",
                )
            if command == "diff" and len(tokens) == 4 and tokens[1] == "--session":
                session_id = tokens[2]
                checkpoint_id = tokens[3]
                export_dir = Path(project_path) / ".opennova" / "exports"
                diff = resolve_checkpoint_diff_from_session(export_dir, session_id, checkpoint_id)
                if diff:
                    return ToolResult(
                        success=True,
                        output=diff,
                        metadata={
                            "session_id": session_id,
                            "checkpoint_id": checkpoint_id,
                            "export_dir": str(export_dir),
                        },
                    )
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Checkpoint not found in session transcript: {session_id} {checkpoint_id}",
                )
            preview = command == "restore" and len(tokens) == 3 and tokens[1] == "--preview"
            force = command == "restore" and len(tokens) == 3 and tokens[1] == "--force"
            rewind_apply = command == "rewind" and len(tokens) == 3 and tokens[1] == "--apply"
            rewind_force = command == "rewind" and len(tokens) == 3 and tokens[1] == "--force"
            if (
                len(tokens) != 2
                and not preview
                and not force
                and not rewind_apply
                and not rewind_force
            ):
                raise ValueError(
                    "Usage: /checkpoint [list|diff <id>|diff --session <session> <id>|"
                    "diff --from-transcript <path> <id>|restore [--preview|--force] <id>|"
                    "rewind [--apply|--force] <id>]"
                )

            checkpoint_id = (
                tokens[2] if preview or force or rewind_apply or rewind_force else tokens[1]
            )
            if command == "diff":
                return ToolResult(success=True, output=_diff_checkpoint(manager, checkpoint_id))
            if preview or (command == "rewind" and not rewind_apply and not rewind_force):
                return ToolResult(
                    success=True,
                    output=_diff_checkpoint(manager, checkpoint_id),
                    metadata={
                        "preview": True,
                        "rewind": command == "rewind",
                        "checkpoint_id": checkpoint_id,
                    },
                )
            manager.restore(checkpoint_id, force=force or rewind_force)
            verb = "Rewound" if command == "rewind" else "Restored"
            return ToolResult(success=True, output=f"{verb} checkpoint: {checkpoint_id}")
    except Exception as exc:
        return ToolResult(success=False, output="", error=str(exc))

    return ToolResult(
        success=False,
        output="",
        error=(
            "Usage: /checkpoint [list|diff <id>|diff --session <session> <id>|"
            "diff --from-transcript <path> <id>|restore [--preview|--force] <id>|"
            "rewind [--apply|--force] <id>]"
        ),
    )


def _diff_checkpoint(manager: CheckpointManager, checkpoint_id: str) -> str:
    return manager.diff(checkpoint_id)
