"""Skill 扩展子系统中的`hook_adapter`模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from opennova.tools.base import ToolResult


def is_valid_hook_config(value: Any) -> bool:
    """判断有效性Hook配置条件是否成立。

    参数：
        value: 需要保存、转换或校验的值。

    返回：
        表示条件是否成立。
    """
    if not isinstance(value, dict):
        return False
    for event_name, matchers in value.items():
        if not isinstance(event_name, str) or not isinstance(matchers, list):
            return False
        for matcher in matchers:
            if not isinstance(matcher, dict):
                return False
            if "matcher" in matcher and not isinstance(matcher.get("matcher"), str):
                return False
            hooks = matcher.get("hooks")
            if not isinstance(hooks, list):
                return False
            for hook in hooks:
                if not isinstance(hook, dict):
                    return False
    return True


def make_declarative_hook_callback(
    event_name: str,
    matcher: str,
    hook_definition: dict[str, Any],
) -> tuple[callable, bool]:
    """构造并返回 `make_declarative_hook_callback` 所表示的数据或流程，并遵守当前模块定义的边界与状态约束。

    参数：
        event_name: 本次操作使用的`event_name`。
        matcher: 本次操作使用的`matcher`。
        hook_definition: 本次操作使用的`hook_definition`。

    返回：
        `tuple[callable, bool]` 类型的处理结果。
    """

    once = bool(hook_definition.get("once", False))
    add_metadata = hook_definition.get("add_metadata")
    set_arguments = hook_definition.get("set_arguments")
    block_message = hook_definition.get("block")

    def callback(event: dict[str, Any]) -> dict[str, Any] | ToolResult:
        if event_name in {"pre_tool_use", "post_tool_use"} and matcher and event.get("tool_name") != matcher:
            return event

        updated = deepcopy(event)
        if isinstance(add_metadata, dict):
            updated.setdefault("metadata", {}).update(deepcopy(add_metadata))
        if isinstance(set_arguments, dict):
            updated.setdefault("arguments", {}).update(deepcopy(set_arguments))
        if isinstance(block_message, str) and block_message.strip():
            return ToolResult(success=False, output="", error=block_message.strip())
        return updated

    return callback, once


def register_skill_hooks(
    hook_manager: Any,
    hooks: dict[str, Any],
    *,
    skill_name: str,
    skill_root: str | None = None,
) -> int:
    """注册SkillHook，使后续运行能够发现并调用它。

    参数：
        hook_manager: 运行工具前后 Hook 的管理器。
        hooks: 本次操作使用的Hook。
        skill_name: 本次操作使用的`skill_name`。
        skill_root: 可选的Skill根目录。

    返回：
        `int` 类型的处理结果。
    """
    del skill_root
    if not hooks:
        return 0

    source = f"skill:{skill_name}"
    registered = 0
    for event_name, matchers in hooks.items():
        if event_name not in hook_manager.SUPPORTED_EVENTS or not isinstance(matchers, list):
            continue
        for matcher_config in matchers:
            if not isinstance(matcher_config, dict):
                continue
            matcher = str(matcher_config.get("matcher") or "")
            hook_entries = matcher_config.get("hooks")
            if not isinstance(hook_entries, list):
                continue
            for hook_definition in hook_entries:
                if not isinstance(hook_definition, dict):
                    continue
                callback, once = make_declarative_hook_callback(event_name, matcher, hook_definition)
                hook_manager.register_session_hook(event_name, callback, source=source, once=once)
                registered += 1
    return registered
