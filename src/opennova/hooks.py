"""OpenNova中的Hook模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opennova.security.workspace_trust import digest_paths
from opennova.tools.base import ToolResult

HookCallback = Callable[[dict[str, Any]], dict[str, Any] | ToolResult | None]


@dataclass
class HookRegistration:
    """数据对象 `HookRegistration` 主要保存 `callback`、`source`、`once`、`session_scoped`
    字段，用于在组件之间传递或持久化这组状态。
    """

    callback: HookCallback
    source: str = "project"
    once: bool = False
    session_scoped: bool = False


class HookManager:
    """集中管理Hook管理的生命周期和共享状态，向上层提供一致的查询与变更入口。"""

    SUPPORTED_EVENTS = {
        "session_start",
        "pre_tool_use",
        "post_tool_use",
        "pre_compact",
        "post_compact",
    }

    def __init__(self, project_path: str | Path = "."):
        self.project_path = Path(project_path).resolve()
        self.hooks_dir = self.project_path / ".opennova" / "hooks"
        self._callbacks: dict[str, list[HookRegistration]] = {
            event: [] for event in self.SUPPORTED_EVENTS
        }

    def register(
        self,
        event: str,
        callback: HookCallback,
        *,
        source: str = "project",
        once: bool = False,
        session_scoped: bool = False,
    ) -> None:
        """处理注册，并按照当前组件的约定返回结果。

        参数：
            event: 需要处理或发布的运行时事件。
            callback: 在对应事件发生时调用的回调函数。
            source: 数据、插件或 Hook 的来源。
            once: 可选的`once`。
            session_scoped: 可选的`session_scoped`。
        """
        if event not in self.SUPPORTED_EVENTS:
            raise ValueError(f"Unsupported hook event: {event}")
        self._callbacks[event].append(
            HookRegistration(
                callback=callback,
                source=source,
                once=once,
                session_scoped=session_scoped,
            )
        )

    def register_session_hook(
        self,
        event: str,
        callback: HookCallback,
        *,
        source: str,
        once: bool = False,
    ) -> None:
        """注册会话Hook，使后续运行能够发现并调用它。

        参数：
            event: 需要处理或发布的运行时事件。
            callback: 在对应事件发生时调用的回调函数。
            source: 数据、插件或 Hook 的来源。
            once: 可选的`once`。
        """
        self.register(
            event,
            callback,
            source=source,
            once=once,
            session_scoped=True,
        )

    def clear_session_hooks(self, source: str | None = None) -> int:
        """清空会话Hook并恢复到可继续使用的初始状态。

        参数：
            source: 数据、插件或 Hook 的来源。

        返回：
            `int` 类型的处理结果。
        """
        cleared = 0
        for event, registrations in self._callbacks.items():
            kept: list[HookRegistration] = []
            for registration in registrations:
                is_session = registration.session_scoped
                matches_source = source is None or registration.source == source
                if is_session and matches_source:
                    cleared += 1
                    continue
                kept.append(registration)
            self._callbacks[event] = kept
        return cleared

    def clear_source(self, source: str, *, prefix: bool = False) -> int:
        """清空来源并恢复到可继续使用的初始状态。

        参数：
            source: 数据、插件或 Hook 的来源。
            prefix: 可选的`prefix`。

        返回：
            `int` 类型的处理结果。
        """
        cleared = 0
        for event, registrations in self._callbacks.items():
            kept: list[HookRegistration] = []
            for registration in registrations:
                matches = (
                    registration.source.startswith(source)
                    if prefix
                    else registration.source == source
                )
                if matches:
                    cleared += 1
                    continue
                kept.append(registration)
            self._callbacks[event] = kept
        return cleared

    def project_hook_paths(self) -> list[Path]:
        """读取并返回 `project_hook_paths` 所表示的数据或流程，并遵守Hook管理定义的边界与状态约束。

        返回：
            按调用约定排序的结果列表。
        """
        if not self.hooks_dir.exists():
            return []
        return sorted(path.resolve() for path in self.hooks_dir.glob("*.py") if path.is_file())

    def project_hooks_digest(self) -> str:
        """读取并返回 `project_hooks_digest` 所表示的数据或流程，并遵守Hook管理定义的边界与状态约束。

        返回：
            处理后的文本或稳定标识。
        """
        paths = self.project_hook_paths()
        return digest_paths(self.project_path, paths) if paths else ""

    def load_project_hooks(self) -> int:
        """从配置、文件或持久化记录中加载项目Hook。

        返回：
            `int` 类型的处理结果。
        """
        paths = self.project_hook_paths()
        if not paths:
            return 0

        self.clear_source("workspace-hooks")
        loaded = 0
        for path in paths:
            loaded += self.load_hook_file(
                path,
                module_prefix="opennova_project_hook",
                source="workspace-hooks",
            )
        return loaded

    def load_hook_file(
        self,
        path: str | Path,
        module_prefix: str = "opennova_hook",
        *,
        source: str = "project",
    ) -> int:
        """从配置、文件或持久化记录中加载Hook文件。

        参数：
            path: 需要读取、检查或写入的路径。
            module_prefix: 可选的`module_prefix`。
            source: 数据、插件或 Hook 的来源。

        返回：
            `int` 类型的处理结果。
        """
        hook_path = Path(path).resolve()
        try:
            hook_path.relative_to(self.project_path)
        except ValueError as exc:
            raise ValueError(f"Hook path is outside project directory: {path}") from exc
        module_name = f"{module_prefix}_{hook_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, hook_path)
        if not spec or not spec.loader:
            return 0
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded = 0
        for event in self.SUPPORTED_EVENTS:
            callback = getattr(module, event, None)
            if callable(callback):
                self.register(event, callback, source=source)
                loaded += 1
        return loaded

    def run_pre_tool_use(self, event: dict[str, Any]) -> dict[str, Any] | ToolResult:
        """运行`run_pre_tool_use`流程，并统一处理完成、失败和取消。

        参数：
            event: 需要处理或发布的运行时事件。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        return self._run_event("pre_tool_use", event)

    def run_post_tool_use(self, event: dict[str, Any]) -> dict[str, Any] | ToolResult:
        """运行`run_post_tool_use`流程，并统一处理完成、失败和取消。

        参数：
            event: 需要处理或发布的运行时事件。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        return self._run_event("post_tool_use", event)

    def _run_event(self, event_name: str, event: dict[str, Any]) -> dict[str, Any] | ToolResult:
        current = event
        registrations = list(self._callbacks.get(event_name, []))
        to_remove: list[HookRegistration] = []
        for registration in registrations:
            result = registration.callback(current)
            if isinstance(result, ToolResult):
                if registration.once and result.success:
                    to_remove.append(registration)
                self._remove_registrations(event_name, to_remove)
                return result
            if isinstance(result, dict):
                current = result
                if registration.once:
                    to_remove.append(registration)
        self._remove_registrations(event_name, to_remove)
        return current

    def _remove_registrations(
        self,
        event_name: str,
        registrations: list[HookRegistration],
    ) -> None:
        if not registrations:
            return
        active = self._callbacks.get(event_name, [])
        self._callbacks[event_name] = [item for item in active if item not in registrations]
