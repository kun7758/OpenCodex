"""安全控制子系统中的工作区信任模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from __future__ import annotations

import hashlib
import json
import os
import time
import unicodedata
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

TRUST_STORE_VERSION = 1


def canonical_workspace_path(project_path: str | Path) -> str:
    """读取并返回 `canonical_workspace_path` 所表示的数据或流程，并遵守当前模块定义的边界与状态约束。

    参数：
        project_path: 本次操作使用的项目路径。

    返回：
        处理后的文本或稳定标识。
    """
    resolved = Path(project_path).expanduser().resolve()
    return unicodedata.normalize("NFC", os.path.normcase(str(resolved)))


def workspace_identity(project_path: str | Path) -> str:
    """构造并返回 `workspace_identity` 所表示的数据或流程，并遵守当前模块定义的边界与状态约束。

    参数：
        project_path: 本次操作使用的项目路径。

    返回：
        处理后的文本或稳定标识。
    """
    canonical = canonical_workspace_path(project_path)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def digest_paths(project_path: str | Path, paths: Sequence[str | Path]) -> str:
    """根据当前输入和当前模块的状态计算 `digest_paths`，并返回调用方需要的结果。

    参数：
        project_path: 本次操作使用的项目路径。
        paths: 本次操作使用的`paths`。

    返回：
        处理后的文本或稳定标识。
    """
    root = Path(project_path).resolve()
    digest = hashlib.sha256()
    normalized: list[tuple[str, Path]] = []
    for value in paths:
        path = Path(value).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"Trusted extension path is outside workspace: {value}") from exc
        normalized.append((relative, path))

    for relative, path in sorted(normalized):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        if not path.is_file():
            digest.update(b"missing\0")
            continue
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class WorkspaceTrustStore:
    """负责工作区信任存储的保存、读取和一致性维护，并隐藏具体持久化格式。"""

    def __init__(self, path: str | Path | None = None):
        self.path = (
            Path(path).expanduser()
            if path is not None
            else Path.home() / ".opennova" / "trust" / "workspaces.json"
        )

    def trust_hooks(self, project_path: str | Path, digest: str) -> None:
        """信任Hook，并按照当前组件的约定返回结果。

        参数：
            project_path: 本次操作使用的项目路径。
            digest: 本次操作使用的内容摘要。
        """
        workspace = self._workspace_record(project_path, create=True)
        workspace["hooks"] = {"digest": digest, "trusted_at": time.time()}
        self._save()

    def untrust_hooks(self, project_path: str | Path) -> None:
        """取消信任Hook，并按照当前组件的约定返回结果。

        参数：
            project_path: 本次操作使用的项目路径。
        """
        workspace = self._workspace_record(project_path, create=False)
        workspace.pop("hooks", None)
        self._save()

    def hooks_are_trusted(self, project_path: str | Path, digest: str) -> bool:
        """读取并返回 `hooks_are_trusted` 所表示的数据或流程，并遵守工作区信任存储定义的边界与状态约束。

        参数：
            project_path: 本次操作使用的项目路径。
            digest: 本次操作使用的内容摘要。

        返回：
            表示条件是否成立。
        """
        workspace = self._workspace_record(project_path, create=False)
        record = workspace.get("hooks", {})
        return bool(digest and record.get("digest") == digest)

    def trust_plugin(self, project_path: str | Path, name: str, digest: str) -> None:
        """信任插件，并按照当前组件的约定返回结果。

        参数：
            project_path: 本次操作使用的项目路径。
            name: 待查询、注册或操作对象的名称。
            digest: 本次操作使用的内容摘要。
        """
        workspace = self._workspace_record(project_path, create=True)
        plugins = workspace.setdefault("plugins", {})
        plugins[name] = {"digest": digest, "trusted_at": time.time()}
        self._save()

    def untrust_plugin(self, project_path: str | Path, name: str) -> None:
        """取消信任插件，并按照当前组件的约定返回结果。

        参数：
            project_path: 本次操作使用的项目路径。
            name: 待查询、注册或操作对象的名称。
        """
        workspace = self._workspace_record(project_path, create=False)
        plugins = workspace.get("plugins", {})
        if isinstance(plugins, dict):
            plugins.pop(name, None)
        self._save()

    def plugin_is_trusted(
        self,
        project_path: str | Path,
        name: str,
        digest: str,
    ) -> bool:
        """读取并返回 `plugin_is_trusted` 所表示的数据或流程，并遵守工作区信任存储定义的边界与状态约束。

        参数：
            project_path: 本次操作使用的项目路径。
            name: 待查询、注册或操作对象的名称。
            digest: 本次操作使用的内容摘要。

        返回：
            表示条件是否成立。
        """
        workspace = self._workspace_record(project_path, create=False)
        plugins = workspace.get("plugins", {})
        if not isinstance(plugins, dict):
            return False
        record = plugins.get(name, {})
        return bool(digest and isinstance(record, dict) and record.get("digest") == digest)

    def plugin_record(self, project_path: str | Path, name: str) -> dict[str, Any]:
        """读取并返回 `plugin_record` 所表示的数据或流程，并遵守工作区信任存储定义的边界与状态约束。

        参数：
            project_path: 本次操作使用的项目路径。
            name: 待查询、注册或操作对象的名称。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        workspace = self._workspace_record(project_path, create=False)
        plugins = workspace.get("plugins", {})
        if not isinstance(plugins, dict) or not isinstance(plugins.get(name), dict):
            return {}
        return dict(plugins[name])

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": TRUST_STORE_VERSION, "workspaces": {}}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"version": TRUST_STORE_VERSION, "workspaces": {}}
        if not isinstance(payload, dict) or payload.get("version") != TRUST_STORE_VERSION:
            return {"version": TRUST_STORE_VERSION, "workspaces": {}}
        if not isinstance(payload.get("workspaces"), dict):
            payload["workspaces"] = {}
        return payload

    def _workspace_record(
        self,
        project_path: str | Path,
        *,
        create: bool,
    ) -> dict[str, Any]:
        payload = self._load()
        self._payload = payload
        workspaces = payload.setdefault("workspaces", {})
        identity = workspace_identity(project_path)
        record = workspaces.get(identity)
        if not isinstance(record, dict):
            if not create:
                return {}
            record = {
                "path": canonical_workspace_path(project_path),
                "plugins": {},
            }
            workspaces[identity] = record
        return record

    def _save(self) -> None:
        payload = getattr(
            self,
            "_payload",
            {"version": TRUST_STORE_VERSION, "workspaces": {}},
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        with suppress(OSError):
            temporary.chmod(0o600)
        os.replace(temporary, self.path)
