"""OpenNova中的插件模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from opennova.hooks import HookManager
from opennova.security.workspace_trust import WorkspaceTrustStore, digest_paths
from opennova.skills.base import SkillSource


@dataclass
class PluginTestReport:
    """保存插件验证报告所需的结构化数据，主要包含 `name`、`success`、`errors`、`warnings` 字段，便于在组件之间传递或持久化。"""

    name: str
    success: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class PluginPolicy:
    """保存插件策略所需的结构化数据，主要包含 `require_signature`、`allow_hooks`、`allow_mcp` 字段，便于在组件之间传递或持久化。"""

    require_signature: bool = False
    allow_hooks: bool = True
    allow_mcp: bool = True

    @classmethod
    def strict(cls) -> PluginPolicy:
        return cls(require_signature=True, allow_hooks=False, allow_mcp=False)


@dataclass
class PluginManifest:
    """保存插件插件清单所需的结构化数据，主要包含
    `name`、`root`、`description`、`enabled`、`signature`、`signature_verified`、`digest`、`commands`
    等字段，便于在组件之间传递或持久化。
    """

    name: str
    root: Path
    description: str = ""
    enabled: bool = True
    signature: str = ""
    signature_verified: bool = False
    digest: str = ""
    commands: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    skills: list[Path] = field(default_factory=list)
    mcp_servers: list[dict[str, Any]] = field(default_factory=list)
    hooks: list[Path] = field(default_factory=list)

    @classmethod
    def from_file(cls, manifest_path: str | Path, project_path: str | Path = ".") -> PluginManifest:
        manifest_file = Path(manifest_path).resolve()
        project_root = Path(project_path).resolve()
        plugin_root = manifest_file.parent.resolve()
        plugins_root = project_root / ".opennova" / "plugins"

        plugin_root.relative_to(plugins_root.resolve())

        data = yaml.safe_load(manifest_file.read_text(encoding="utf-8")) or {}
        name = str(data.get("name") or plugin_root.name)

        skills = [
            cls._resolve_inside_plugin(plugin_root, item, "skill path")
            for item in data.get("skills", [])
        ]
        hooks = [
            cls._resolve_inside_plugin(plugin_root, item, "hook path")
            for item in data.get("hooks", [])
        ]

        manifest = cls(
            name=name,
            root=plugin_root,
            description=str(data.get("description", "")),
            enabled=bool(data.get("enabled", True)),
            signature=str(data.get("signature", "")),
            commands=list(data.get("commands", []) or []),
            tools=list(data.get("tools", []) or []),
            skills=skills,
            mcp_servers=list(data.get("mcp_servers", []) or []),
            hooks=hooks,
        )
        manifest.digest = manifest.content_digest()
        return manifest

    def content_digest(self) -> str:
        """根据当前输入和插件插件清单的状态计算 `content_digest`，并返回调用方需要的结果。

        返回：
            处理后的文本或稳定标识。
        """
        paths: list[Path] = []
        for path in self.root.rglob("*"):
            if path.is_dir():
                continue
            resolved = path.resolve()
            try:
                resolved.relative_to(self.root)
            except ValueError as exc:
                raise ValueError(f"Plugin content is outside plugin directory: {path}") from exc
            paths.append(resolved)
        return digest_paths(self.root, paths)

    @staticmethod
    def _resolve_inside_plugin(plugin_root: Path, value: str, label: str) -> Path:
        path = (plugin_root / value).resolve()
        try:
            path.relative_to(plugin_root)
        except ValueError as exc:
            raise ValueError(f"{label} is outside plugin directory: {value}") from exc
        return path


class PluginManager:
    """集中管理插件管理的生命周期和共享状态，向上层提供一致的查询与变更入口。"""

    def __init__(
        self,
        project_path: str | Path = ".",
        *,
        trust_path: str | Path | None = None,
    ):
        self.project_path = Path(project_path).resolve()
        self.plugins_dir = self.project_path / ".opennova" / "plugins"
        self.legacy_trust_path = self.plugins_dir / "trusted.json"
        self.trust_store = WorkspaceTrustStore(trust_path)
        self.trust_path = self.trust_store.path
        self.plugins: list[PluginManifest] = []
        self.errors: dict[str, str] = {}
        self.trust_warnings: dict[str, str] = {}
        self.commands: list[dict[str, Any]] = []
        self.trusted_plugins: set[str] = set()
        self.skill_sources: list[SkillSource] = []
        self._applied_mcp_entries: list[tuple[list[Any], dict[str, Any]]] = []
        self._active_mcp_server_names: set[str] = set()

    def trust_plugin(self, name: str) -> None:
        """信任插件，并按照当前组件的约定返回结果。

        参数：
            name: 待查询、注册或操作对象的名称。
        """
        manifest = next((item for item in self.plugins if item.name == name), None)
        if manifest is None:
            for manifest_path in self.discover_manifests():
                candidate = PluginManifest.from_file(manifest_path, project_path=self.project_path)
                if candidate.name == name:
                    manifest = candidate
                    break
        if manifest is None:
            raise ValueError(f"Plugin not found: {name}")
        self.trust_store.trust_plugin(self.project_path, name, manifest.digest)
        self.trusted_plugins.add(name)

    def untrust_plugin(self, name: str) -> None:
        """取消信任插件，并按照当前组件的约定返回结果。

        参数：
            name: 待查询、注册或操作对象的名称。
        """
        self.trust_store.untrust_plugin(self.project_path, name)
        self.trusted_plugins.discard(name)

    def is_trusted(self, name: str, digest: str | None = None) -> bool:
        """判断`trusted`条件是否成立。

        参数：
            name: 待查询、注册或操作对象的名称。
            digest: 可选的内容摘要。

        返回：
            表示条件是否成立。
        """
        if digest is None:
            manifest = next((item for item in self.plugins if item.name == name), None)
            digest = manifest.digest if manifest else ""
        return self.trust_store.plugin_is_trusted(self.project_path, name, digest)

    def discover_manifests(self) -> list[Path]:
        """发现 `manifests` 对应的数据，并按照当前组件的约定返回结果。

        返回：
            按调用约定排序的结果列表。
        """
        if not self.plugins_dir.exists():
            return []
        return sorted(self.plugins_dir.glob("*/plugin.yaml"))

    def load_enabled_plugins(
        self,
        config: dict[str, Any],
        hook_manager: HookManager | None = None,
    ) -> list[PluginManifest]:
        """从配置、文件或持久化记录中加载启用状态插件。

        参数：
            config: 控制当前组件行为的配置。
            hook_manager: 运行工具前后 Hook 的管理器。

        返回：
            按调用约定排序的结果列表。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self._remove_mcp_contributions()
        self.plugins = []
        self.errors = {}
        self.trust_warnings = {}
        self.commands = []
        self.skill_sources = []
        self.trusted_plugins = set()
        self._active_mcp_server_names = set()
        if hook_manager:
            hook_manager.clear_source("plugin:", prefix=True)

        for manifest_path in self.discover_manifests():
            plugin_name = manifest_path.parent.name
            try:
                manifest = PluginManifest.from_file(manifest_path, project_path=self.project_path)
                if not manifest.enabled:
                    continue
                trust_record = self.trust_store.plugin_record(self.project_path, manifest.name)
                if trust_record and trust_record.get("digest") != manifest.digest:
                    self.trust_warnings[manifest.name] = (
                        "Plugin content changed after trust was granted; active contributions disabled"
                    )
                if self.is_trusted(manifest.name, manifest.digest):
                    self.trusted_plugins.add(manifest.name)
                    self._apply_manifest(manifest, config=config, hook_manager=hook_manager)
                self.plugins.append(manifest)
            except Exception as exc:
                self.errors[plugin_name] = str(exc)

        return self.plugins

    def build_tools(self, config: dict[str, Any] | None = None) -> list[Any]:
        """根据当前输入和状态构造`build_tools`。

        参数：
            config: 控制当前组件行为的配置。

        返回：
            按调用约定排序的结果列表。
        """
        from opennova.tools.plugin_tools import PluginCommandTool

        tools: list[Any] = []
        for manifest in self.plugins:
            if not self.is_trusted(manifest.name, manifest.digest):
                continue
            for tool_data in manifest.tools:
                name = str(tool_data.get("name", "")).strip()
                error_key = f"{manifest.name}.{name or 'tool'}"
                error = self._validate_tool_manifest(tool_data)
                if error:
                    self.errors[error_key] = error
                    continue
                command = str(tool_data.get("command", "")).strip()
                permission = self._tool_permission(tool_data)
                tools.append(
                    PluginCommandTool(
                        name=name,
                        description=str(tool_data.get("description", f"Plugin tool: {name}")),
                        command=command,
                        args=[str(item) for item in tool_data.get("args", [])],
                        config=config,
                        read_only=bool(tool_data.get("read_only", False)),
                        permission=permission,
                    )
                )
        return tools

    def get_skill_sources(self) -> list[SkillSource]:
        """读取Skill来源，不改变当前对象的业务状态。

        返回：
            按调用约定排序的结果列表。
        """
        return list(self.skill_sources)

    def get_active_mcp_server_names(self) -> set[str]:
        """读取 `active_mcp_server_names` 对应的数据，不改变当前对象的业务状态。

        返回：
            `set[str]` 类型的处理结果。
        """
        return set(self._active_mcp_server_names)

    def build_lockfile(self) -> dict[str, Any]:
        """根据当前输入和状态构造`build_lockfile`。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        plugins: list[dict[str, Any]] = []
        for manifest in self.plugins:
            plugins.append(
                {
                    "name": manifest.name,
                    "description": manifest.description,
                    "path": str(manifest.root.relative_to(self.project_path)),
                    "enabled": manifest.enabled,
                    "signature": manifest.signature,
                    "signature_verified": manifest.signature_verified,
                    "digest": manifest.digest,
                    "trusted": self.is_trusted(manifest.name, manifest.digest),
                    "commands": [dict(command) for command in manifest.commands],
                    "tools": [
                        {
                            "name": str(tool.get("name", "")),
                            "permission": self._tool_permission(tool),
                            "read_only": bool(tool.get("read_only", False)),
                        }
                        for tool in manifest.tools
                    ],
                    "skills": [str(skill.relative_to(manifest.root)) for skill in manifest.skills],
                    "hooks": [str(hook.relative_to(manifest.root)) for hook in manifest.hooks],
                    "mcp_servers": [
                        str(server.get("name", ""))
                        for server in manifest.mcp_servers
                        if isinstance(server, dict)
                    ],
                }
            )
        return {"version": 2, "plugins": plugins}

    def test_plugin(self, name: str) -> PluginTestReport:
        """验证 `plugin` 场景下的返回值、状态变化和副作用符合预期。

        参数：
            name: 待查询、注册或操作对象的名称。

        返回：
            `PluginTestReport` 类型的处理结果。
        """
        manifest = next((plugin for plugin in self.plugins if plugin.name == name), None)
        if manifest is None:
            return PluginTestReport(name=name, success=False, errors=[f"Plugin not found: {name}"])

        errors: list[str] = []
        for tool in manifest.tools:
            error = self._validate_tool_manifest(tool)
            if error:
                tool_name = str(tool.get("name") or "tool")
                errors.append(f"{tool_name}: {error}")
        for hook in manifest.hooks:
            if not hook.exists():
                errors.append(f"Hook path does not exist: {hook}")
        for skill in manifest.skills:
            if not skill.exists():
                errors.append(f"Skill path does not exist: {skill}")

        return PluginTestReport(name=name, success=not errors, errors=errors)

    def audit_permissions(self) -> list[dict[str, Any]]:
        """读取并返回 `audit_permissions` 所表示的数据或流程，并遵守插件管理定义的边界与状态约束。

        返回：
            按调用约定排序的结果列表。
        """
        audits: list[dict[str, Any]] = []
        for manifest in self.plugins:
            risks: list[str] = []
            for tool in manifest.tools:
                permission = self._tool_permission(tool)
                if permission != "read":
                    risks.append(f"tool:{tool.get('name', 'tool')}:{permission}")
            if manifest.hooks:
                risks.append(f"hooks:{len(manifest.hooks)}")
            for server in manifest.mcp_servers:
                if isinstance(server, dict):
                    risks.append(f"mcp:{server.get('name', 'server')}")
            audits.append(
                {
                    "name": manifest.name,
                    "trusted": self.is_trusted(manifest.name, manifest.digest),
                    "signature": manifest.signature,
                    "signature_verified": manifest.signature_verified,
                    "digest": manifest.digest,
                    "risks": risks,
                }
            )
        return audits

    def audit_policy(self, policy: PluginPolicy) -> list[dict[str, Any]]:
        """读取并返回 `audit_policy` 所表示的数据或流程，并遵守插件管理定义的边界与状态约束。

        参数：
            policy: 本次操作使用的策略。

        返回：
            按调用约定排序的结果列表。
        """
        reports: list[dict[str, Any]] = []
        for manifest in self.plugins:
            violations: list[str] = []
            if policy.require_signature:
                if not manifest.signature:
                    violations.append("missing-signature")
                elif not manifest.signature_verified:
                    violations.append("unverified-signature")
            if not policy.allow_hooks and manifest.hooks:
                violations.append("hooks-disallowed")
            if not policy.allow_mcp and manifest.mcp_servers:
                violations.append("mcp-disallowed")
            reports.append(
                {
                    "name": manifest.name,
                    "trusted": self.is_trusted(manifest.name, manifest.digest),
                    "violations": violations,
                }
            )
        return reports

    def startup_warnings(
        self,
        lockfile: dict[str, Any] | None = None,
        policy: PluginPolicy | None = None,
    ) -> list[dict[str, str]]:
        """读取并返回 `startup_warnings` 所表示的数据或流程，并遵守插件管理定义的边界与状态约束。

        参数：
            lockfile: 可选的锁定文件。
            policy: 可选的策略。

        返回：
            按调用约定排序的结果列表。
        """
        warnings: list[dict[str, str]] = []
        if self.legacy_trust_path.exists():
            warnings.append(
                {
                    "type": "trust",
                    "plugin": "*",
                    "message": "legacy name-only trust records are ignored; re-trust each plugin",
                }
            )
        warnings.extend(
            {
                "type": "trust",
                "plugin": name,
                "message": message,
            }
            for name, message in sorted(self.trust_warnings.items())
        )
        if lockfile:
            drift = self.compare_lockfile(lockfile)
            for item in drift["changed"]:
                warnings.append(
                    {
                        "type": "drift",
                        "plugin": str(item["name"]),
                        "message": "; ".join(str(change) for change in item["changes"]),
                    }
                )
            for key in ("added", "removed"):
                for item in drift[key]:
                    warnings.append(
                        {
                            "type": "drift",
                            "plugin": str(item["name"]),
                            "message": f"plugin {key[:-1]}",
                        }
                    )
        if policy:
            for report in self.audit_policy(policy):
                if report["violations"]:
                    warnings.append(
                        {
                            "type": "policy",
                            "plugin": str(report["name"]),
                            "message": ",".join(str(item) for item in report["violations"]),
                        }
                    )
        return warnings

    def compare_lockfile(self, lockfile: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        """比较锁定文件，并按照当前组件的约定返回结果。

        参数：
            lockfile: 本次操作使用的锁定文件。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        current = {plugin["name"]: plugin for plugin in self.build_lockfile().get("plugins", [])}
        locked = {plugin["name"]: plugin for plugin in lockfile.get("plugins", [])}

        added = [{"name": name} for name in sorted(set(current) - set(locked))]
        removed = [{"name": name} for name in sorted(set(locked) - set(current))]
        changed: list[dict[str, Any]] = []
        for name in sorted(set(current) & set(locked)):
            changes = self._plugin_lock_changes(locked[name], current[name])
            if changes:
                changed.append({"name": name, "changes": changes})
        return {"added": added, "removed": removed, "changed": changed}

    def _plugin_lock_changes(
        self,
        locked: dict[str, Any],
        current: dict[str, Any],
    ) -> list[str]:
        changes: list[str] = []
        scalar_fields = ("description", "enabled", "signature")
        for field_name in scalar_fields:
            if locked.get(field_name) != current.get(field_name):
                changes.append(f"{field_name} changed")

        list_fields = ("commands", "skills", "hooks", "mcp_servers")
        for field_name in list_fields:
            if locked.get(field_name, []) != current.get(field_name, []):
                changes.append(f"{field_name} changed")

        locked_tools = {tool.get("name"): tool for tool in locked.get("tools", [])}
        current_tools = {tool.get("name"): tool for tool in current.get("tools", [])}
        for tool_name in sorted(set(locked_tools) | set(current_tools)):
            if tool_name not in locked_tools:
                changes.append(f"tool added: {tool_name}")
                continue
            if tool_name not in current_tools:
                changes.append(f"tool removed: {tool_name}")
                continue
            if locked_tools[tool_name].get("permission") != current_tools[tool_name].get(
                "permission"
            ):
                changes.append(f"tool {tool_name} permission changed")
        for field_name in ("digest", "trusted", "signature_verified"):
            if locked.get(field_name) != current.get(field_name):
                changes.append(f"{field_name} changed")
        return changes

    def _validate_tool_manifest(self, tool_data: dict[str, Any]) -> str | None:
        """校验工具插件清单，发现问题时返回或抛出明确错误。

        参数：
            tool_data: 本次操作使用的工具数据。

        返回：
            `str | None` 类型的处理结果。
        """
        for field_name in ("name", "description", "command"):
            if not str(tool_data.get(field_name, "")).strip():
                return f"Plugin tool missing required field: {field_name}"
        args = tool_data.get("args", [])
        if args and not (isinstance(args, list) and all(isinstance(item, str) for item in args)):
            return "Plugin tool args must be a list of strings"
        permission = self._tool_permission(tool_data)
        if permission not in {"read", "edit", "command"}:
            return "Plugin tool permission must be one of: read, edit, command"
        return None

    def _tool_permission(self, tool_data: dict[str, Any]) -> str:
        """根据当前输入和插件管理的状态计算 `_tool_permission`，并返回调用方需要的结果。

        参数：
            tool_data: 本次操作使用的工具数据。

        返回：
            处理后的文本或稳定标识。
        """
        if tool_data.get("permission"):
            return str(tool_data["permission"])
        if tool_data.get("read_only"):
            return "read"
        return "command"

    def _apply_manifest(
        self,
        manifest: PluginManifest,
        config: dict[str, Any],
        hook_manager: HookManager | None,
    ) -> None:
        for skill_dir in manifest.skills:
            self.skill_sources.append(
                SkillSource(
                    root=skill_dir,
                    plugin_name=manifest.name,
                    source_type="plugin",
                    loaded_from="plugin",
                )
            )

        mcp_config = config.setdefault("mcp", {})
        mcp_servers = mcp_config.setdefault("servers", [])
        existing_names = {server.get("name") for server in mcp_servers if isinstance(server, dict)}
        for server in manifest.mcp_servers:
            server_name = str(server.get("name", "")).strip()
            if server_name and server_name not in existing_names:
                contribution = dict(server)
                mcp_servers.append(contribution)
                self._applied_mcp_entries.append((mcp_servers, contribution))
                self._active_mcp_server_names.add(server_name)
                existing_names.add(server_name)

        if hook_manager:
            for hook_path in manifest.hooks:
                hook_manager.load_hook_file(
                    hook_path,
                    module_prefix=f"opennova_plugin_hook_{manifest.name}",
                    source=f"plugin:{manifest.name}",
                )

        for command in manifest.commands:
            command_entry = dict(command)
            command_entry.setdefault("plugin", manifest.name)
            self.commands.append(command_entry)

    def _remove_mcp_contributions(self) -> None:
        """释放或移除 `_remove_mcp_contributions` 所表示的数据或流程，并遵守插件管理定义的边界与状态约束。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        for servers, contribution in self._applied_mcp_entries:
            servers[:] = [server for server in servers if server is not contribution]
        self._applied_mcp_entries = []
