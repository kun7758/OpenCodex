"""Agent 核心运行时中的启动配置模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from opennova.tools.catalog import builtin_tool_names


class RuntimeBootstrapProfile(StrEnum):
    """枚举运行时启动配置配置档允许出现的稳定取值，序列化和状态判断均使用这些值。"""

    INSPECT = "inspect"
    BARE = "bare"
    INTERACTIVE = "interactive"
    HEADLESS = "headless"


@dataclass(frozen=True)
class RuntimeInspectionSnapshot:
    """保存运行时检查快照快照所需的结构化数据，主要包含 `profile`、`tool_names` 字段，便于在组件之间传递或持久化。"""

    profile: RuntimeBootstrapProfile
    tool_names: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeBootstrapPolicy:
    """保存运行时启动配置策略所需的结构化数据，主要包含
    `create_provider`、`create_session`、`load_extensions`、`load_skills`、`connect_mcp`
    字段，便于在组件之间传递或持久化。
    """

    create_provider: bool
    create_session: bool
    load_extensions: bool
    load_skills: bool
    connect_mcp: bool


BOOTSTRAP_POLICIES = {
    RuntimeBootstrapProfile.INSPECT: RuntimeBootstrapPolicy(False, False, False, False, False),
    RuntimeBootstrapProfile.BARE: RuntimeBootstrapPolicy(True, True, False, False, False),
    RuntimeBootstrapProfile.INTERACTIVE: RuntimeBootstrapPolicy(True, True, True, True, True),
    RuntimeBootstrapProfile.HEADLESS: RuntimeBootstrapPolicy(True, True, True, True, True),
}


def bootstrap_policy(profile: RuntimeBootstrapProfile | str) -> RuntimeBootstrapPolicy:
    """读取并返回 `bootstrap_policy` 所表示的数据或流程，并遵守当前模块定义的边界与状态约束。

    参数：
        profile: 本次操作使用的配置档。

    返回：
        `RuntimeBootstrapPolicy` 类型的处理结果。
    """
    return BOOTSTRAP_POLICIES[RuntimeBootstrapProfile(profile)]


def inspect_runtime() -> RuntimeInspectionSnapshot:
    """构造并返回 `inspect_runtime` 所表示的数据或流程，并遵守当前模块定义的边界与状态约束。

    返回：
        `RuntimeInspectionSnapshot` 类型的处理结果。
    """
    return RuntimeInspectionSnapshot(
        profile=RuntimeBootstrapProfile.INSPECT,
        tool_names=tuple(builtin_tool_names()),
    )
