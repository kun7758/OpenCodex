"""Agent 核心运行时中的文件状态模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FileVersion:
    """保存文件版本所需的结构化数据，主要包含 `path`、`exists`、`mtime_ns`、`size`、`content_hash` 字段，便于在组件之间传递或持久化。"""

    path: str
    exists: bool
    mtime_ns: int | None
    size: int
    content_hash: str


class FileVersionCache:
    """封装文件版本缓存相关的状态和操作，使调用方通过稳定接口使用该能力。"""

    def __init__(self) -> None:
        self._versions: dict[str, FileVersion] = {}

    @staticmethod
    def canonical_path(path: str | Path) -> str:
        return str(Path(path).expanduser().resolve())

    @classmethod
    def snapshot(cls, path: str | Path) -> FileVersion:
        canonical = cls.canonical_path(path)
        target = Path(canonical)
        if not target.exists():
            return FileVersion(canonical, False, None, 0, hashlib.sha256(b"").hexdigest())
        stat = target.stat()
        content_hash = hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else ""
        return FileVersion(canonical, True, stat.st_mtime_ns, stat.st_size, content_hash)

    def record(self, path: str | Path) -> FileVersion:
        version = self.snapshot(path)
        self._versions[version.path] = version
        return version

    def get(self, path: str | Path) -> FileVersion | None:
        return self._versions.get(self.canonical_path(path))

    def validate(self, path: str | Path) -> tuple[bool, FileVersion | None, FileVersion]:
        """处理校验，并按照当前组件的约定返回结果。

        参数：
            path: 需要读取、检查或写入的路径。

        返回：
            `tuple[bool, FileVersion | None, FileVersion]` 类型的处理结果。
        """
        canonical = self.canonical_path(path)
        expected = self._versions.get(canonical)
        current = self.snapshot(canonical)
        return expected is None or expected == current, expected, current

    def discard(self, path: str | Path) -> None:
        self._versions.pop(self.canonical_path(path), None)

    def clear(self) -> None:
        self._versions.clear()

    def to_dict(self) -> dict[str, FileVersion]:
        return dict(self._versions)
