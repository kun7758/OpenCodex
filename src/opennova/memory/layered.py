"""记忆与上下文子系统中的分层记忆模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

MEMORY_DIR = ".opennova/memory"
DEFAULT_LAYERED_MEMORY_MAX_CHARS = 5000
_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class MemoryRecord:
    """保存记忆记录所需的结构化数据，主要包含 `name`、`path`、`content`、`provenance`、`scope`、`created_at`、`expires_at`
    字段，便于在组件之间传递或持久化。
    """

    name: str
    path: Path
    content: str
    provenance: str = "manual"
    scope: str = "project"
    created_at: str | None = None
    expires_at: str | None = None

    @property
    def expired(self) -> bool:
        if not self.expires_at:
            return False
        try:
            value = self.expires_at.replace("Z", "+00:00")
            expires = datetime.fromisoformat(value)
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=UTC)
            return expires <= datetime.now(UTC)
        except ValueError:
            return False


class LayeredMemoryManager:
    """统一管理带来源、作用域和过期时间的分层记忆，并在写入时进行规范化去重。"""

    def __init__(self, project_path: str | Path = "."):
        self.project_path = Path(project_path).resolve()
        self.memory_dir = self.project_path / MEMORY_DIR

    def list_records(self, *, include_expired: bool = True) -> list[MemoryRecord]:
        """列出 `records` 对应的对象，并按当前组件约定返回稳定顺序。

        参数：
            include_expired: 可选的`include_expired`。

        返回：
            按调用约定排序的结果列表。
        """
        if not self.memory_dir.exists():
            return []
        records: list[MemoryRecord] = []
        for path in sorted(self.memory_dir.rglob("*.md")):
            if not path.is_file() or any(
                part.startswith(".") for part in path.relative_to(self.memory_dir).parts
            ):
                continue
            record = self._read_record(path)
            if record.content and (include_expired or not record.expired):
                records.append(record)
        return records

    def add(
        self,
        name: str,
        content: str,
        *,
        provenance: str = "user",
        scope: str = "project",
        expires_at: str | None = None,
        overwrite: bool = False,
    ) -> MemoryRecord:
        """构造并返回 `add` 所表示的数据或流程，并遵守分层记忆记忆管理定义的边界与状态约束。

        参数：
            name: 待查询、注册或操作对象的名称。
            content: 需要处理、保存或分析的文本内容。
            provenance: 可选的`provenance`。
            scope: 可选的`scope`。
            expires_at: 可选的`expires_at`。
            overwrite: 可选的`overwrite`。

        返回：
            `MemoryRecord` 类型的处理结果。

        说明：
            该操作会访问本地文件系统，路径校验和原子写入约束由所在组件负责。
        """
        normalized_name = name[:-3] if name.endswith(".md") else name
        if not _SAFE_NAME.fullmatch(normalized_name):
            raise ValueError("Memory name may contain only letters, numbers, '.', '_' and '-'")
        if not content.strip():
            raise ValueError("Memory content cannot be empty")
        if scope not in {"project", "user", "session"}:
            raise ValueError("Memory scope must be project, user, or session")
        path = self.memory_dir / f"{normalized_name}.md"
        if path.exists() and not overwrite:
            raise FileExistsError(f"Memory already exists: {normalized_name}")
        if expires_at:
            self._validate_expiry(expires_at)

        metadata = {
            "provenance": provenance.strip() or "user",
            "scope": scope,
            "created_at": datetime.now(UTC).isoformat(),
        }
        if expires_at:
            metadata["expires_at"] = expires_at
        rendered = (
            f"---\n{yaml.safe_dump(metadata, sort_keys=False).strip()}\n---\n{content.strip()}\n"
        )
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".md.tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(path)
        return self._read_record(path)

    def delete(self, name: str) -> bool:
        """处理删除，并按照当前组件的约定返回结果。

        参数：
            name: 待查询、注册或操作对象的名称。

        返回：
            表示条件是否成立。

        说明：
            该操作会访问本地文件系统，路径校验和原子写入约束由所在组件负责。
        """
        normalized_name = name[:-3] if name.endswith(".md") else name
        if not _SAFE_NAME.fullmatch(normalized_name):
            raise ValueError("Invalid memory name")
        path = self.memory_dir / f"{normalized_name}.md"
        if not path.exists():
            return False
        path.unlink()
        return True

    def load_for_context(
        self,
        max_chars: int = DEFAULT_LAYERED_MEMORY_MAX_CHARS,
        exclude_hashes: set[str] | None = None,
        scopes: set[str] | None = None,
    ) -> str | None:
        """从配置、文件或持久化记录中加载对应上下文。

        参数：
            max_chars: 可选的最大值字符数。
            exclude_hashes: 可选的`exclude_hashes`。
            scopes: 可选的`scopes`。

        返回：
            `str | None` 类型的处理结果。
        """
        seen = set(exclude_hashes or set())
        allowed_scopes = scopes or {"project", "user"}
        parts: list[str] = []
        for record in self.list_records(include_expired=False):
            if record.scope not in allowed_scopes:
                continue
            unique_paragraphs: list[str] = []
            for paragraph in re.split(r"\n\s*\n", record.content):
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                digest = self.content_hash(paragraph)
                if digest in seen:
                    continue
                seen.add(digest)
                unique_paragraphs.append(paragraph)
            if not unique_paragraphs:
                continue
            rel_path = record.path.relative_to(self.project_path).as_posix()
            label = (
                f"### Memory file: {rel_path} "
                f"[scope={record.scope}, provenance={record.provenance}]"
            )
            parts.append(f"{label}\n" + "\n\n".join(unique_paragraphs))

        if not parts:
            return None
        content = "\n\n".join(parts).strip()
        if len(content) <= max_chars:
            return content
        return (
            content[:max_chars].rstrip()
            + "\n\n[... .opennova/memory content truncated for context budget ...]"
        )

    def _read_record(self, path: Path) -> MemoryRecord:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        metadata: dict[str, Any] = {}
        match = _FRONTMATTER.match(text)
        if match:
            loaded = yaml.safe_load(match.group(1)) or {}
            metadata = loaded if isinstance(loaded, dict) else {}
            text = text[match.end() :].strip()
        return MemoryRecord(
            name=path.stem,
            path=path,
            content=text,
            provenance=str(metadata.get("provenance") or "manual"),
            scope=str(metadata.get("scope") or "project"),
            created_at=self._optional_string(metadata.get("created_at")),
            expires_at=self._optional_string(metadata.get("expires_at")),
        )

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        if value is None:
            return None
        return value.isoformat() if hasattr(value, "isoformat") else str(value)

    @staticmethod
    def _validate_expiry(value: str) -> None:
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("expires_at must be an ISO-8601 timestamp") from exc

    @staticmethod
    def content_hash(text: str) -> str:
        """根据当前输入和分层记忆记忆管理的状态计算 `content_hash`，并返回调用方需要的结果。

        参数：
            text: 需要解析、格式化或展示的文本。

        返回：
            处理后的文本或稳定标识。
        """
        normalized = " ".join(text.split()).casefold().encode("utf-8")
        return hashlib.sha256(normalized).hexdigest()

    @classmethod
    def paragraph_hashes(cls, text: str) -> set[str]:
        """读取并返回 `paragraph_hashes` 所表示的数据或流程，并遵守分层记忆记忆管理定义的边界与状态约束。

        参数：
            text: 需要解析、格式化或展示的文本。

        返回：
            `set[str]` 类型的处理结果。
        """
        return {
            cls.content_hash(paragraph)
            for paragraph in re.split(r"\n\s*\n", text)
            if paragraph.strip()
        }
