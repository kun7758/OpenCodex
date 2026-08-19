"""Agent 核心运行时中的启动配置模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from opennova.tools.catalog import builtin_tool_names


class RuntimeBootstrapProfile(StrEnum):
    """枚举运行时启动配置配置档允许出现的稳定取值，序列化和状态判断均使用这些值。"""

    INSPECT = "inspect"  # 检查模式：仅用于list-tools/doctor等命令，不创建Provider、Session，不加载任何扩展
    BARE = "bare"  # 裸模式：创建Provider和Session，但不加载扩展、技能和MCP
    INTERACTIVE = "interactive"  # 交互模式：全部启用，用于TUI交互界面
    HEADLESS = "headless"  # 无头模式：全部启用，用于一次性任务(run命令)


@dataclass(frozen=True)
class RuntimeInspectionSnapshot:
    """保存运行时检查快照快照所需的结构化数据，主要包含 `profile`、`tool_names` 字段，便于在组件之间传递或持久化。"""

    profile: RuntimeBootstrapProfile  # 使用的启动配置档
    tool_names: tuple[str, ...]  # 可用工具名称列表


@dataclass(frozen=True)
class RuntimeBootstrapPolicy:
    """保存运行时启动配置策略所需的结构化数据，主要包含
    `create_provider`、`create_session`、`load_extensions`、`load_skills`、`connect_mcp`
    字段，便于在组件之间传递或持久化。
    """

    create_provider: bool  # 是否创建LLM Provider实例
    create_session: bool  # 是否创建会话管理器
    load_extensions: bool  # 是否加载扩展（插件和Python钩子）
    load_skills: bool  # 是否扫描和加载SKILL.md技能
    connect_mcp: bool  # 是否连接配置的MCP服务器


# 各配置档对应的启动策略：INSPECT仅检查、BARE仅核心、INTERACTIVE和HEADLESS全部启用
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
