"""差异与补丁子系统的公共导出入口，集中暴露上层调用方需要使用的类型和函数。"""

from opennova.diff.changeset import ChangeResult, ChangeSet
from opennova.diff.engine import ApplyResult, DiffEngine, Hunk
from opennova.diff.parser import ChangeType, DiffParser, FileChange

__all__ = [
    "DiffEngine",
    "ApplyResult",
    "Hunk",
    "DiffParser",
    "FileChange",
    "ChangeType",
    "ChangeSet",
    "ChangeResult",
]
