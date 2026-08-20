"""Agent 核心运行时中的Agent模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import logging
import os
import re
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from opennova.config import Config
from opennova.hooks import HookManager
from opennova.memory.context import ContextManager
from opennova.memory.project import ProjectMemory
from opennova.memory.working import WorkingMemory
from opennova.planning.planner import Planner
from opennova.plugins import PluginManager
from opennova.providers.base import Message, StreamChunk
from opennova.providers.factory import ProviderFactory
from opennova.runtime.artifacts import ArtifactStore
from opennova.runtime.bootstrap import RuntimeBootstrapProfile, bootstrap_policy
from opennova.runtime.cancellation import CancellationToken, RunHandle
from opennova.runtime.event_bus import RuntimeEventBus
from opennova.runtime.events import ToolEvent
from opennova.runtime.file_state import FileVersionCache
from opennova.runtime.loop import ReActLoop
from opennova.runtime.model_policy import ProviderCircuitBreaker
from opennova.runtime.state import (
    AgentState,
    Plan,
    PlanApprovalStatus,
    PlanStatus,
    PlanStep,
    StepStatus,
)
from opennova.runtime.store import RuntimeAction, RuntimeStateStore, StateChanged
from opennova.security.guardrails import PermissionMode
from opennova.security.workspace_trust import WorkspaceTrustStore
from opennova.tasks import TaskManager
from opennova.tools.base import BaseTool, ToolRegistry, ToolResult

if TYPE_CHECKING:
    from opennova.mcp.connector import MCPManager
    from opennova.mcp.types import MCPServerConfig
    from opennova.skills.registry import SkillRegistry

_LOGGER = logging.getLogger(__name__)


class AgentRuntime:
    """OpenNova 的总装配器。它为一次会话创建模型 Provider、工具注册表、ReAct 循环、状态存储、记忆、会话、安全策略、Skill、插件和 MCP 连接，并向 TUI 与
    SDK 提供统一入口。
    """

    def __init__(
        self,
        config: Config | dict[str, Any],
        register_default_tools: bool = True,
        enable_mcp: bool = True,
        enable_skills: bool = True,
        bootstrap_profile: RuntimeBootstrapProfile | str = RuntimeBootstrapProfile.INTERACTIVE,
    ):
        """初始化Agent运行时，保存后续操作需要的依赖、配置和初始状态。

        参数：
            config: 控制当前组件行为的配置。
            register_default_tools: 是否注册项目自带的内置工具。
            enable_mcp: 是否初始化并连接配置中的 MCP 服务。
            enable_skills: 是否扫描并加载可用 Skill。
            bootstrap_profile: 控制运行时启动范围的配置档，例如交互模式或无界面模式。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self.bootstrap_profile = RuntimeBootstrapProfile(bootstrap_profile)  # 启动配置档，控制运行时启动范围
        policy = bootstrap_policy(self.bootstrap_profile)
        if not policy.create_provider or not policy.create_session:
            raise ValueError(
                "The inspect profile is side-effect free; use inspect_runtime() instead of "
                "constructing AgentRuntime."
            )
        self.config = config.to_dict() if isinstance(config, Config) else copy.deepcopy(config)  # 配置字典，包含所有运行时配置
        self.state = AgentState()  # 代理状态，存储计划、步骤等运行时状态
        self._closed = False  # 关闭标志，标记运行时是否已关闭
        self._active_run_handle: RunHandle | None = None  # 活动运行句柄，用于取消和跟踪当前运行
        self.tool_registry = ToolRegistry()  # 工具注册表，管理所有可用工具
        self._plugin_tool_names: set[str] = set()  # 插件工具名称集合，记录已注册的插件工具
        self._plugin_mcp_server_names: set[str] = set()  # 插件MCP服务器名称集合，记录插件提供的MCP服务器
        self.task_manager = TaskManager()  # 任务管理器，管理后台任务
        self.register_default_tools = register_default_tools  # 是否注册默认内置工具
        self.enable_mcp = enable_mcp and policy.connect_mcp  # 是否启用MCP连接
        self.enable_skills = enable_skills and policy.load_skills  # 是否启用技能加载
        self.enable_extensions = policy.load_extensions  # 是否启用扩展（插件和钩子）

        agent_config = self.config.get("agent", {})
        self.max_iterations = agent_config.get("max_iterations", 20)  # 最大迭代次数，ReAct循环上限
        self.show_thinking = agent_config.get("show_thinking", True)  # 是否显示思考过程
        self.auto_confirm = agent_config.get("auto_confirm", False)  # 是否自动确认工具调用
        self.security_config = self.config.get("security", {})  # 安全配置字典

        self.project_path = Path.cwd().resolve()  # 项目根路径
        self.llm = ProviderFactory.create_provider(self.config)  # 主LLM提供商实例
        self.fallback_providers = [  # 备用提供商列表，主提供商失败时使用
            ProviderFactory.create_provider(self.config, provider_name=str(provider_name))
            for provider_name in agent_config.get("fallback_providers", [])
            if str(provider_name) != self.config.get("default_provider")
        ]
        self.provider_circuit_breaker = ProviderCircuitBreaker(  # 提供商断路器，处理失败重试和冷却
            failure_threshold=int(agent_config.get("provider_failure_threshold", 3)),
            cooldown_seconds=float(agent_config.get("provider_cooldown_seconds", 30.0)),
        )
        self.project_memory = ProjectMemory(project_path=str(self.project_path))  # 项目记忆，持久化项目级知识
        self.working_memory = WorkingMemory()  # 工作记忆，当前会话的临时记忆
        self.hook_manager = HookManager(project_path=self.project_path)  # 钩子管理器，管理Python钩子脚本
        self.workspace_trust_store = WorkspaceTrustStore()  # 工作区信任存储，管理插件和钩子的信任状态
        self.workspace_hook_digest = (  # 工作区钩子摘要，用于检测钩子文件变更
            self.hook_manager.project_hooks_digest() if self.enable_extensions else ""
        )
        self.workspace_hooks_trusted = self.workspace_trust_store.hooks_are_trusted(  # 工作区钩子是否受信任
            self.project_path,
            self.workspace_hook_digest,
        )
        self.extension_warnings: list[str] = []  # 扩展警告列表，记录未加载扩展的原因
        if self.workspace_hooks_trusted:
            self.hook_manager.load_project_hooks()
        elif self.workspace_hook_digest:
            self.extension_warnings.append(
                "Project hooks are present but were not loaded because this workspace digest "
                "is not trusted. Use /hooks trust to enable them."
            )
        self.plugin_manager = PluginManager(  # 插件管理器，管理项目插件
            project_path=self.project_path,
            trust_path=self.workspace_trust_store.path,
        )
        if self.enable_extensions:
            self.plugin_manager.load_enabled_plugins(
                config=self.config,
                hook_manager=self.hook_manager,
            )
        self._plugin_mcp_server_names = self.plugin_manager.get_active_mcp_server_names()  # 插件提供的MCP服务器名称

        # 读取上下文压缩阈值、保留消息数和工具结果上限。
        compression_config = agent_config.get("compression", {})
        self.context_manager = ContextManager(  # 上下文管理器，管理消息窗口和压缩
            model=self.llm.model,
            max_tool_result_tokens=compression_config.get("max_tool_result_tokens", 8000),
        )
        self.context_manager.compression_threshold = compression_config.get("threshold", 0.55)  # 压缩阈值
        self.context_manager.keep_last_pairs = compression_config.get("keep_last_pairs", 6)  # 保留最近消息对数

        # 把使用当前 Provider 的摘要器注入上下文管理器。
        from opennova.memory.compressor import ContextCompressor

        self.context_manager.set_compressor(ContextCompressor(llm_provider=self.llm))  # 注入压缩器

        from opennova.session import SessionManager

        self.session_manager = SessionManager(  # 会话管理器，管理会话持久化
            project_path=str(self.project_path),
            persistence_config=self.config.get("session", {}).get("persistence", {}),
        )
        self.session_manager.start_session()  # 启动新会话
        self.file_version_cache = FileVersionCache()  # 文件版本缓存，检测文件变更
        self.artifact_store = ArtifactStore(  # 工件存储，存储大型工具输出
            self.project_path,
            str(self.session_manager.session_id or "session"),
        )
        self.session_transcript: list[dict[str, Any]] = []  # 会话转录，记录完整对话历史
        self.state_store = RuntimeStateStore(  # 状态存储，持久化运行时状态
            self.state,
            session_id=str(self.session_manager.session_id or ""),
        )
        self._state_persistence_ready = False  # 状态持久化就绪标志

        self.loop: ReActLoop | None = None  # ReAct循环实例，执行模型/工具迭代
        self.events = RuntimeEventBus()  # 事件总线，发布运行时事件
        self.tool_events: list[dict[str, Any]] = []  # 工具事件列表，记录工具调用历史
        self.planner = Planner(self.llm)  # 计划器，生成执行计划

        self.mcp_manager: MCPManager | None = None  # MCP管理器，管理MCP连接
        self._mcp_server_configs: list[MCPServerConfig] = []  # MCP服务器配置列表
        self._mcp_config_errors: dict[str, str] = {}  # MCP配置错误字典
        self._mcp_connection_results: dict[str, bool] = {}  # MCP连接结果字典
        self.skill_registry: SkillRegistry | None = None  # 技能注册表，管理可用技能
        from opennova.security.audit import SecurityAuditLogger
        from opennova.security.guardrails import Guardrails
        from opennova.security.permissions import PermissionStore

        self.permission_store = PermissionStore(  # 权限存储，管理工具权限规则
            Path(os.getcwd()) / ".opennova" / "permissions.json"
        )
        audit_config = self.security_config.get("audit", {})
        self.security_audit_logger = SecurityAuditLogger(  # 安全审计日志记录器
            path=audit_config.get("path", ".opennova/audit/security.jsonl"),
            enabled=audit_config.get("enabled", True),
            max_arg_chars=audit_config.get("max_arg_chars", 500),
            session_id=self.session_manager.session_id,
            secrets_policy=self.security_config.get("secrets", {}),
        )
        self.guardrails = Guardrails(  # 护栏，执行安全策略和权限检查
            sandbox_mode=self.security_config.get("sandbox_mode", True),
            allowed_paths=self.security_config.get("allowed_paths", []),
            blocked_commands=self.security_config.get("blocked_commands", []),
            auto_confirm_safe=self.security_config.get("auto_confirm_safe", True),
            allow_network=self.security_config.get("allow_network", True),
            strict_shell_parsing=self.security_config.get("strict_shell_parsing", False),
            permission_mode=self.security_config.get("permission_mode", "default"),
            always_allow_tools=self.security_config.get("always_allow_tools", []),
            always_deny_tools=self.security_config.get("always_deny_tools", []),
            always_ask_tools=self.security_config.get("always_ask_tools", []),
            permission_rules=self.security_config.get("permission_rules", []),
            network_policy=self.security_config.get("network", {}),
            secrets_policy=self.security_config.get("secrets", {}),
            permission_store=self.permission_store,
        )
        self.security_audit_logger.permission_mode = self.get_permission_mode().value  # 同步权限模式到审计日志

        if register_default_tools:
            self._register_builtin_tools()
            self._register_plugin_tools()

        if enable_skills:
            self._init_skills()

        if enable_mcp:
            self._init_mcp()

        self._state_persistence_ready = True
        self.state_store.subscribe(
            lambda snapshot: snapshot.revision,
            self._on_runtime_state_changed,
        )

    def _on_runtime_state_changed(self, revision: int, event: StateChanged) -> None:
        """响应`runtime_state_changed`事件，并把变化同步到相关状态、界面或持久化记录。

        参数：
            revision: 本次操作使用的修订。
            event: 需要处理或发布的运行时事件。
        """
        self._emit("state_changed", self.state_store.get_state(), event)
        if not getattr(self, "_state_persistence_ready", False):
            return
        try:
            journal_event = next(
                (
                    item
                    for item in reversed(self.state_store.recent_events())
                    if item.event_id == event.event_id
                ),
                None,
            )
            if journal_event is not None:
                self.session_manager.append_runtime_event(
                    journal_event,
                    durable=event.critical,
                )
            if self.session_manager.needs_runtime_snapshot:
                self._save_session_messages()
        except Exception as exc:
            _LOGGER.warning("Failed to append runtime state event: %s", exc)
            with suppress(Exception):
                self._emit("diagnostic", "runtime_event_persistence_failed", str(exc))

    def _register_builtin_tools(self) -> None:
        """构造共享的工具配置并注册文件、Shell、搜索、诊断、任务、Agent、交互、计划、网络、Git、Skill、MCP 和工作树工具。"""
        security_config = self.config.get("security", {})
        tool_config = {
            "command_timeout": security_config.get("command_timeout", 30),
            "working_dir": os.getcwd(),
            "sandbox_mode": security_config.get("sandbox_mode", True),
            "allowed_paths": security_config.get("allowed_paths", []),
            "blocked_commands": security_config.get("blocked_commands", []),
            "auto_confirm_safe": security_config.get("auto_confirm_safe", True),
            "allow_network": security_config.get("allow_network", True),
            "strict_shell_parsing": security_config.get("strict_shell_parsing", False),
            "permission_mode": security_config.get("permission_mode", "default"),
            "always_allow_tools": security_config.get("always_allow_tools", []),
            "always_deny_tools": security_config.get("always_deny_tools", []),
            "always_ask_tools": security_config.get("always_ask_tools", []),
            "permission_rules": security_config.get("permission_rules", []),
            "network_policy": security_config.get("network", {}),
            "secrets_policy": security_config.get("secrets", {}),
            "process_sandbox": security_config.get("process_sandbox", {}),
            "temp_dir": security_config.get("process_sandbox", {}).get("tmp_dir"),
            "read_only": security_config.get("read_only", False),
            "max_file_size": security_config.get("max_file_size", 100 * 1024 * 1024),
            "checkpoint_writes": False,
        }

        from opennova.tools.agent_tools import AgentTool, SendMessageTool
        from opennova.tools.ask_question_tool import AskUserQuestionTool
        from opennova.tools.diagnostics_tools import (
            PythonDefinitionTool,
            PythonDiagnosticsTool,
            PythonReferencesTool,
            PythonSymbolsTool,
        )
        from opennova.tools.file_tools import (
            CreateFileTool,
            DeleteFileTool,
            EditFileTool,
            ListDirectoryTool,
            MultiEditFileTool,
            ReadFileTool,
            WriteFileTool,
        )
        from opennova.tools.git_tools import (
            GitBranchTool,
            GitCommitTool,
            GitDiffTool,
            GitLogTool,
            GitStatusTool,
        )
        from opennova.tools.mcp_resource_tools import ListMCPResourcesTool, ReadMCPResourceTool
        from opennova.tools.plan_mode_tools import (
            EnterPlanModeTool,
            ExitPlanModeTool,
        )
        from opennova.tools.project_guide_tool import InitProjectGuideTool
        from opennova.tools.search_tools import GlobFilesTool, GrepCodeTool
        from opennova.tools.shell_tools import ExecuteCommandTool
        from opennova.tools.skill_tool import SkillTool
        from opennova.tools.task_tools import (
            TaskCreateTool,
            TaskGetTool,
            TaskListTool,
            TaskOutputTool,
            TaskStopTool,
            TaskUpdateTool,
        )
        from opennova.tools.todo_tools import TodoWriteTool
        from opennova.tools.tool_search import ToolSearchTool
        from opennova.tools.web_tools import WebFetchTool, WebSearchTool
        from opennova.tools.worktree_tools import EnterWorktreeTool, ExitWorktreeTool

        # 注册文件、Shell、搜索和 Python 诊断工具。
        self.tool_registry.register(ReadFileTool(config=tool_config))
        self.tool_registry.register(WriteFileTool(config=tool_config))
        self.tool_registry.register(CreateFileTool(config=tool_config))
        self.tool_registry.register(EditFileTool(config=tool_config))
        self.tool_registry.register(MultiEditFileTool(config=tool_config))
        self.tool_registry.register(DeleteFileTool(config=tool_config))
        self.tool_registry.register(ListDirectoryTool(config=tool_config))
        self.tool_registry.register(ExecuteCommandTool(config=tool_config))
        self.tool_registry.register(GlobFilesTool(config=tool_config))
        self.tool_registry.register(GrepCodeTool(config=tool_config))
        self.tool_registry.register(ToolSearchTool(self.tool_registry))
        self.tool_registry.register(PythonDiagnosticsTool(config=tool_config))
        self.tool_registry.register(PythonSymbolsTool(config=tool_config))
        self.tool_registry.register(PythonDefinitionTool(config=tool_config))
        self.tool_registry.register(PythonReferencesTool(config=tool_config))

        # 注册任务生命周期和待办项管理工具。
        self.tool_registry.register(TaskCreateTool(self.task_manager))
        self.tool_registry.register(TaskListTool(self.task_manager))
        self.tool_registry.register(TaskGetTool(self.task_manager))
        self.tool_registry.register(TaskUpdateTool(self.task_manager))
        self.tool_registry.register(TaskStopTool(self.task_manager))
        self.tool_registry.register(TaskOutputTool(self.task_manager))
        self.tool_registry.register(TodoWriteTool(config={"state_store": self.state_store}))

        # 注册子 Agent 创建与消息发送工具。
        self.tool_registry.register(
            AgentTool(config={"runtime": self, "task_manager": self.task_manager})
        )
        self.tool_registry.register(SendMessageTool(config={"task_manager": self.task_manager}))

        # 注册向用户提出结构化问题的交互工具。
        self.tool_registry.register(AskUserQuestionTool())
        self.tool_registry.register(SkillTool(config={"runtime": self}))

        # 注册进入和退出计划模式的工具。
        self.tool_registry.register(
            EnterPlanModeTool(config={"state": self.state, "runtime": self})
        )
        self.tool_registry.register(ExitPlanModeTool(config={"state": self.state, "runtime": self}))

        # 注册网络搜索与页面获取工具。
        self.tool_registry.register(WebSearchTool(config=tool_config))
        self.tool_registry.register(WebFetchTool(config=tool_config))
        self.tool_registry.register(
            InitProjectGuideTool(config={"working_dir": os.getcwd(), "runtime": self})
        )
        self.tool_registry.register(ListMCPResourcesTool(config={"runtime": self}))
        self.tool_registry.register(ReadMCPResourceTool(config={"runtime": self}))

        # 注册 Git 状态、差异、日志、分支和提交工具。
        self.tool_registry.register(GitCommitTool())
        self.tool_registry.register(GitStatusTool())
        self.tool_registry.register(GitDiffTool())
        self.tool_registry.register(GitLogTool())
        self.tool_registry.register(GitBranchTool())
        self.tool_registry.register(EnterWorktreeTool(config=tool_config))
        self.tool_registry.register(ExitWorktreeTool(config=tool_config))

    def _register_plugin_tools(self) -> None:
        """注册插件工具，使后续运行能够发现并调用它。"""
        for name in self._plugin_tool_names:
            self.tool_registry.unregister(name)
        self._plugin_tool_names.clear()

        security_config = self.config.get("security", {})
        tool_config = {
            "command_timeout": security_config.get("command_timeout", 30),
            "working_dir": os.getcwd(),
            "sandbox_mode": security_config.get("sandbox_mode", True),
            "allowed_paths": security_config.get("allowed_paths", []),
            "blocked_commands": security_config.get("blocked_commands", []),
            "allow_network": security_config.get("allow_network", True),
            "strict_shell_parsing": security_config.get("strict_shell_parsing", False),
            "permission_mode": security_config.get("permission_mode", "default"),
            "process_sandbox": security_config.get("process_sandbox", {}),
            "temp_dir": security_config.get("process_sandbox", {}).get("tmp_dir"),
        }
        for tool in self.plugin_manager.build_tools(config=tool_config):
            if self.tool_registry.has_tool(tool.name):
                self.plugin_manager.errors[f"tool:{tool.name}"] = (
                    f"Plugin tool conflicts with an existing tool: {tool.name}"
                )
                continue
            self.tool_registry.register(tool)
            self._plugin_tool_names.add(tool.name)

    def _init_skills(self) -> None:
        """执行 `_init_skills` 所定义的协调步骤，必要时更新Agent运行时维护的状态。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        from opennova.skills.examples import get_builtin_skill_dirs
        from opennova.skills.registry import SkillRegistry

        skills_config = self.config.get("skills", {})
        if not skills_config.get("enabled", True):
            return

        registry = SkillRegistry()
        self.skill_registry = registry

        configured_dirs = [Path(path) for path in skills_config.get("dirs", [])]
        skill_dirs: list[str | Path] = [*get_builtin_skill_dirs(), *configured_dirs]
        excluded = skills_config.get("exclude", [])

        registry.load_all(
            directories=skill_dirs,
            sources=self.plugin_manager.get_skill_sources(),
            excluded=excluded,
        )

    def _init_mcp(self) -> None:
        """执行 `_init_mcp` 所定义的协调步骤，必要时更新Agent运行时维护的状态。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        from opennova.mcp.connector import MCPManager

        mcp_config = self.config.get("mcp", {})
        if not mcp_config.get("enabled", True):
            return

        project_path = getattr(self, "project_path", Path.cwd().resolve())
        self.mcp_manager = MCPManager(
            self.tool_registry,
            roots=[
                {
                    "uri": project_path.as_uri(),
                    "name": project_path.name,
                }
            ],
        )
        self._reload_mcp_server_configs()

    def _reload_mcp_server_configs(self) -> None:
        """执行 `_reload_mcp_server_configs` 所定义的协调步骤，必要时更新Agent运行时维护的状态。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        from opennova.mcp.types import MCPServerConfig

        self._mcp_server_configs = []
        self._mcp_config_errors = {}
        self._mcp_connection_results = {}

        mcp_config = self.config.get("mcp", {})
        servers = mcp_config.get("servers", [])
        for index, server_data in enumerate(servers):
            if not isinstance(server_data, dict):
                self._mcp_config_errors[f"server[{index}]"] = "MCP server config must be a mapping"
                continue
            server_name = server_data.get("name", f"server[{index}]")
            try:
                server_config = MCPServerConfig.from_dict(server_data)
                self._mcp_server_configs.append(server_config)
            except Exception as exc:
                self._mcp_config_errors[server_name] = str(exc)

    async def refresh_plugin_contributions(self) -> None:
        """更新 `refresh_plugin_contributions` 所表示的数据或流程，并遵守Agent运行时定义的边界与状态约束。

        说明：
            执行过程中会更新当前实例维护的状态。
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        previous_mcp_names = set(self._plugin_mcp_server_names)
        self.plugin_manager.load_enabled_plugins(
            config=self.config,
            hook_manager=self.hook_manager,
        )
        self._plugin_mcp_server_names = self.plugin_manager.get_active_mcp_server_names()

        if self.register_default_tools:
            self._register_plugin_tools()
        else:
            for name in self._plugin_tool_names:
                self.tool_registry.unregister(name)
            self._plugin_tool_names.clear()

        if self.enable_skills:
            self._init_skills()

        if self.mcp_manager is not None:
            for server_name in previous_mcp_names:
                await self.mcp_manager.remove_server(server_name)
            self._reload_mcp_server_configs()

    async def connect_mcp_servers(self) -> dict[str, bool]:
        """连接MCP服务端，并按照当前组件的约定返回结果。

        返回：
            供后续逻辑或序列化使用的结构化字典。

        说明：
            执行过程中会更新当前实例维护的状态。
            该操作可能等待模型服务、MCP 服务或其他异步数据源返回。
        """
        if not self.mcp_manager or not self._mcp_server_configs:
            return {}

        self._mcp_connection_results = await self.mcp_manager.connect_all(self._mcp_server_configs)
        return self._mcp_connection_results

    async def _ensure_mcp_ready(self) -> dict[str, bool]:
        """根据当前输入和Agent运行时的状态计算 `_ensure_mcp_ready`，并返回调用方需要的结果。

        返回：
            供后续逻辑或序列化使用的结构化字典。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        mcp_manager = getattr(self, "mcp_manager", None)
        mcp_server_configs = getattr(self, "_mcp_server_configs", [])
        if not mcp_manager or not mcp_server_configs:
            return {}

        connected_servers = set(mcp_manager.get_server_names())
        enabled_configs = [config for config in mcp_server_configs if config.enabled]
        if enabled_configs and all(config.name in connected_servers for config in enabled_configs):
            return {config.name: True for config in enabled_configs}

        return await self.connect_mcp_servers()

    async def disconnect_mcp_servers(self) -> None:
        """断开MCP服务端，并按照当前组件的约定返回结果。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        if self.mcp_manager:
            await self.mcp_manager.disconnect_all()

    async def aclose(self) -> None:
        """异步关闭当前对象持有的任务、连接和运行时资源；重复调用保持幂等。

        说明：
            执行过程中会更新当前实例维护的状态。
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        if self._closed:
            return
        self._closed = True

        active_handle = getattr(self, "_active_run_handle", None)
        if (
            active_handle is not None
            and not active_handle.done
            and active_handle.task is not asyncio.current_task()
        ):
            active_handle.cancel("Runtime closed")
            with suppress(asyncio.CancelledError):
                await active_handle.wait()

        with suppress(Exception):
            await self.task_manager.aclose()
        with suppress(Exception):
            await self.disconnect_mcp_servers()
        with suppress(Exception):
            self.flush_session()

        close_target: Any = self.llm
        closer = getattr(close_target, "aclose", None)
        if not callable(closer):
            close_target = getattr(self.llm, "client", None)
            closer = getattr(close_target, "close", None)
        if callable(closer):
            with suppress(Exception):
                result = closer()
                if inspect.isawaitable(result):
                    await result

        self.events.clear()

    def cancel_run(self, reason: str = "Run cancelled") -> bool:
        """取消运行，并按照当前组件的约定返回结果。

        参数：
            reason: 触发当前状态变化或操作的原因。

        返回：
            表示条件是否成立。
        """
        handle: RunHandle | None = getattr(self, "_active_run_handle", None)
        if handle is None or handle.done:
            return False
        return bool(handle.cancel(reason))

    def create_child_runtime(self) -> AgentRuntime:
        """创建继承父级配置和功能开关的独立运行时，供子 Agent 使用，同时保持工具状态和会话状态隔离。

        返回：
            `AgentRuntime` 类型的处理结果。
        """
        child = AgentRuntime(
            config=copy.deepcopy(self.config),
            register_default_tools=self.register_default_tools,
            enable_mcp=self.enable_mcp,
            enable_skills=self.enable_skills,
            bootstrap_profile=self.bootstrap_profile,
        )
        child.auto_confirm = self.auto_confirm
        if child.get_permission_mode() != self.get_permission_mode():
            child.set_permission_mode(self.get_permission_mode())
        return child

    def get_permission_mode(self) -> PermissionMode:
        """读取权限模式，不改变当前对象的业务状态。

        返回：
            `PermissionMode` 类型的处理结果。
        """
        guardrails = getattr(self, "guardrails", None)
        if guardrails is None:
            return PermissionMode.AUTO
        return PermissionMode.normalize(guardrails.effective_permission_mode)

    def set_permission_mode(self, mode: str | PermissionMode) -> PermissionMode:
        """设置权限模式并保持相关派生状态同步。

        参数：
            mode: 本次运行采用的工作模式。

        返回：
            `PermissionMode` 类型的处理结果。
        """
        canonical = PermissionMode.normalize(mode)
        self.guardrails.set_permission_mode(canonical)
        self.security_config["permission_mode"] = canonical.value

        config_setter = getattr(self.config, "set", None)
        if callable(config_setter):
            config_setter("security.permission_mode", canonical.value)
        elif isinstance(self.config, dict):
            self.config.setdefault("security", {})["permission_mode"] = canonical.value

        if self.tool_registry.has_tool("execute_command"):
            command_tool = self.tool_registry.get("execute_command")
            command_guardrails = getattr(command_tool, "guardrails", None)
            if command_guardrails is not None:
                command_guardrails.set_permission_mode(canonical)
            command_tool.config["permission_mode"] = canonical.value

        self.security_audit_logger.permission_mode = canonical.value
        self._emit("permission_mode_changed", canonical)
        return canonical

    def register_tool(self, tool: BaseTool) -> None:
        """注册工具，使后续运行能够发现并调用它。

        参数：
            tool: 要注册、检查或调用的工具实例。
        """
        self.tool_registry.register(tool)

    def register_callback(
        self,
        event: str,
        callback: Callable[..., Any],
    ) -> Callable[[], None]:
        """注册回调，使后续运行能够发现并调用它。

        参数：
            event: 需要处理或发布的运行时事件。
            callback: 在对应事件发生时调用的回调函数。

        返回：
            `Callable[[], None]` 类型的处理结果。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        event_bus = getattr(self, "events", None)
        if isinstance(event_bus, RuntimeEventBus):
            return event_bus.subscribe(event, callback)
        callbacks = getattr(self, "_callbacks", None)
        if callbacks is None:
            callbacks = {}
            self._callbacks = callbacks
        callbacks[event] = callback

        def unsubscribe() -> None:
            if callbacks.get(event) is callback:
                callbacks.pop(event, None)

        return unsubscribe

    def _emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        """执行 `_emit` 所定义的协调步骤，必要时更新Agent运行时维护的状态。

        参数：
            event: 需要处理或发布的运行时事件。
            *args: 传递给底层实现的额外位置参数。
            **kwargs: 传递给底层实现的额外关键字参数。
        """
        event_bus = getattr(self, "events", None)
        if event_bus is not None:
            event_bus.publish(event, *args, **kwargs)
            return
        callback = getattr(self, "_callbacks", {}).get(event)
        if callback:
            callback(*args, **kwargs)

    async def run(
        self,
        task: str,
        mode: Literal["plan", "act"] = "act",
        stream: bool = True,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> str:
        """Agent 运行入口：接收任务描述，根据 mode 执行 Plan 或 Act 工作流。

        流程：
            1. 校验运行时状态（未关闭、无并发运行）
            2. 创建 RunHandle 用于取消和状态跟踪
            3. 若 task 是计划审批确认，直接进入 execute_approved_plan
            4. 重置 Agent 状态，按 mode 分流：
               - "plan": 生成结构化计划，返回计划摘要供用户审批
               - "act":  执行 ReAct 循环（思考→工具调用→观察），返回最终结果

        参数：
            task: 用户提交的任务描述，如 "实现文件上传功能" 或 "y"（计划审批确认）
            mode: 工作模式，"plan" 生成计划供审批，"act" 直接执行任务
            stream: 是否流式输出模型回复，为 True 时通过 stream 回调推送增量内容
            progress_callback: 进度回调，每次工具执行完成时调用，接收包含 tool_name、success 等的字典

        返回：
            str: 任务执行结果文本
                 - plan 模式：返回计划摘要，用户需调用 execute_approved_plan() 执行
                 - act 模式：返回最终执行结果或错误信息
        """
        if getattr(self, "_closed", False):
            raise RuntimeError("AgentRuntime is closed")
        current_task = asyncio.current_task()
        active_handle = getattr(self, "_active_run_handle", None)
        if active_handle is not None and not active_handle.done:
            raise RuntimeError("AgentRuntime already has an active run")
        handle = RunHandle(run_id=uuid4().hex, task=current_task)
        self._active_run_handle = handle
        try:
            if mode != "plan" and self._is_plan_execution_approval(task):
                self.state.mark_plan_approved()
                return await self.execute_approved_plan(stream=stream)

            self.state.reset(task)
            self.state.set_mode(mode)

            if mode == "plan":
                return await self._run_plan_mode(task, stream=stream)
            return await self._run_act_mode(
                task,
                stream=stream,
                progress_callback=progress_callback,
            )
        except asyncio.CancelledError:
            handle.token.cancel("Run cancelled")
            self.state.cancel_run(self.state.run_id)
            raise
        finally:
            if getattr(self, "_active_run_handle", None) is handle:
                self._active_run_handle = None

    async def _run_plan_mode(self, task: str, stream: bool = True) -> str:
        """运行计划模式流程，并统一处理完成、失败和取消。

        参数：
            task: 用户希望 Agent 完成的任务描述。
            stream: 是否将模型输出以增量事件形式返回。

        返回：
            处理后的文本或稳定标识。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        plan = await self._create_plan(task)
        result = self._prepare_plan_for_approval(plan)
        self.state.finish_run(result, success=True, run_id=self.state.run_id)
        self._save_session_messages()
        return result

    def _prepare_plan_for_approval(self, plan: Plan) -> str:
        """准备计划供用户审批：保存计划文件、更新状态、同步进度并触发审批界面。

        本函数是计划模式工作流的关键步骤，负责将生成的计划转换为可审批的状态。
        它执行以下操作：
            1. 将计划保存到 AgentState 中
            2. 将计划保存为 Markdown 文件到 .opennova/plan/ 目录
            3. 记录计划文件的哈希值，用于检测文件变化
            4. 将计划状态标记为等待用户审批
            5. 将计划步骤同步到 TodoWrite 工具，用于在 TUI 中显示任务进度
            6. 发送计划事件给 TUI，触发计划审批对话框

        参数：
            plan: 当前要保存、展示或执行的结构化计划，通常由 _create_plan() 生成。

        返回：
            固定返回 "Plan ready for approval" 字符串，表示计划已准备好供用户审批。

        说明：
            - 该函数会修改 AgentState 的状态（设置计划、标记等待审批等）
            - 该函数会访问本地文件系统保存计划文件
            - 该函数会发送事件通知 TUI 显示计划审批对话框
            - 调用此函数后，通常需要等待用户审批（通过 _confirm_plan()）
            - 用户审批后，可以通过 execute_approved_plan() 执行计划

        工作流程：
            _create_plan(task) → _prepare_plan_for_approval(plan) → 用户审批 → execute_approved_plan()
        """
        self.state.set_plan(plan)
        plan_file_path = self._save_plan_to_project(plan)
        self.state.set_plan_file_path(plan_file_path, self._hash_file(plan_file_path))
        self.state.mark_plan_awaiting_approval()
        self._sync_plan_progress(plan)

        self._emit("plan", plan, plan_file_path)
        return "Plan ready for approval"

    def _build_step_execution_task(self, plan: Plan, step: PlanStep) -> str:
        """根据当前输入和状态构造`build_step_execution_task`。

        参数：
            plan: 当前要保存、展示或执行的结构化计划。
            step: 当前要处理的计划步骤。

        返回：
            处理后的文本或稳定标识。
        """
        lines = [
            "Execute the approved development plan step.",
            f"Overall plan: {plan.task}",
            f"Current step ({step.id}): {step.description}",
            f"Plan file: {self.state.plan_file_path or '(not saved)'}",
            "",
            "Complete plan snapshot:",
            self._render_plan_snapshot(plan),
            "",
            "Follow the current approved plan exactly.",
            "Do not re-plan from scratch.",
            "If execution reveals the plan is stale or incorrect, update the plan status/notes first and then continue.",
        ]

        return "\n".join(lines)

    def _build_memory_messages(self, task: str) -> list[Message]:
        """根据当前输入和状态构造`build_memory_messages`。

        参数：
            task: 用户希望 Agent 完成的任务描述。

        返回：
            按调用约定排序的结果列表。
        """
        memory_parts = [self.project_memory.get_project_context()]
        relevant_decisions = self.project_memory.get_relevant_decisions(task, limit=3)

        if relevant_decisions:
            decision_lines = [
                f"- {decision.description}: {decision.reasoning}" for decision in relevant_decisions
            ]
            memory_parts.append("Relevant prior decisions:\n" + "\n".join(decision_lines))

        try:
            from opennova.memory.layered import LayeredMemoryManager
            from opennova.memory.project_guide import ProjectGuideManager

            project_path = getattr(self.project_memory, "project_path", Path(os.getcwd()))
            guide_manager = ProjectGuideManager(project_path=project_path)
            memory_budget = int(
                getattr(self, "config", {}).get("agent", {}).get("memory", {}).get(
                    "max_chars", 5000
                )
            )
            guide_text = guide_manager.load_for_context(max_chars=memory_budget)
            exclude_hashes = set()
            if guide_text:
                exclude_hashes.update(LayeredMemoryManager.paragraph_hashes(guide_text))
                memory_parts.append(
                    "Project guide (OPENNOVA.md) — follow these project-specific conventions when relevant:\n"
                    + guide_text
                )
            compressed_summary = self.context_manager.get_compressed_summary()
            if compressed_summary:
                exclude_hashes.update(
                    LayeredMemoryManager.paragraph_hashes(compressed_summary)
                )
            layered_text = LayeredMemoryManager(project_path=project_path).load_for_context(
                max_chars=memory_budget,
                exclude_hashes=exclude_hashes,
                scopes={"project", "user"},
            )
            if layered_text:
                memory_parts.append(
                    "Layered project memory (.opennova/memory) — additional maintained project notes:\n"
                    + layered_text
                )
        except Exception:
            pass

        memory_text = "\n\n".join(part for part in memory_parts if part)
        if not memory_text.strip():
            return []

        return [
            Message(
                role="system",
                content="Use this project memory when it is relevant to the current task:\n\n"
                + memory_text,
                name="opennova_project_memory",
            )
        ]

    def _record_run_session(self, task: str, success: bool, started_at: float) -> None:
        """记录运行会话，供状态展示、恢复或后续决策使用。

        参数：
            task: 用户希望 Agent 完成的任务描述。
            success: 本次操作使用的成功。
            started_at: 本次操作使用的`started_at`。
        """
        self.project_memory.record_session(
            task=task,
            success=success,
            duration_seconds=max(0.0, perf_counter() - started_at),
        )

    async def execute_approved_plan(self, stream: bool = True) -> str:
        """逐步执行已批准的计划：依次取出待执行步骤，为每一步构造独立任务并运行一次完整的 ReAct 循环，
        根据执行结果更新步骤状态，直到所有步骤完成或因失败/中断而终止。

        整体流程：
            1. 前置校验：确认当前有计划且审批状态为已批准/执行中/失败/中断。
            2. 准备阶段：重置中断/失败步骤为待执行，标记计划进入执行状态。
            3. 主循环：逐个执行步骤（详见下方循环内注释）。
            4. 收尾：根据所有步骤最终状态标记计划完成或失败。

        参数：
            stream: 是否将模型输出以增量事件形式返回。

        返回：
            最后一个步骤的结果文本，或 "Plan execution complete"。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        # ── 1. 前置校验 ─────────────────────────────────────────────
        # 确保当前有计划可执行，且审批状态允许进入执行流程。
        plan = self.state.current_plan
        if not plan:
            return "No plan available for execution"

        if self.state.plan_approval_status not in {
            PlanApprovalStatus.APPROVED,
            PlanApprovalStatus.EXECUTING,
            PlanApprovalStatus.FAILED,
            PlanApprovalStatus.INTERRUPTED,
        }:
            return "Plan approval required before execution"

        # ── 2. 准备阶段 ─────────────────────────────────────────────
        # 重置上一轮中断或失败的步骤为待执行，标记计划进入执行状态，
        # 同步待办项、发送 UI 更新、持久化计划快照。
        self._prepare_plan_for_execution(plan)
        self.state.mark_plan_executing()
        plan = self.state.current_plan or plan
        self._sync_plan_progress(plan)
        self._emit_plan_update(plan)
        self._persist_current_plan()

        # ── 3. 主循环：逐个执行计划步骤 ─────────────────────────────
        # 每次循环完成一个步骤：读取计划文件（支持用户手动编辑后热更新）→
        # 找到下一个待执行步骤 → 构造包含完整计划上下文的独立任务 →
        # 运行一次完整的 ReAct 循环（模型决定工具调用顺序）→ 根据结果更新状态。
        while True:
            # 3a. 从磁盘重新读取计划文件，如果用户手动修改过则采用最新版本。
            refreshed_plan = self._refresh_plan_from_file()
            if refreshed_plan is not None:
                plan = refreshed_plan

            # 3b. 取第一个状态为 PENDING 的步骤；全部完成则退出循环。
            step = plan.get_next_step()
            if not step:
                break

            # 3c. 标记当前步骤为"运行中"，同步到 state、UI 和磁盘。
            self.state.mark_step_running(step.id)
            plan = self.state.current_plan or plan
            step = next(item for item in plan.steps if item.id == step.id)
            step_plan_revision = self.state.plan_revision
            self._sync_plan_progress(plan, active_step_id=step.id)
            self._emit_plan_update(plan)
            self._persist_current_plan()
            self._emit("thought", f"Executing plan step {step.id}: {step.description}")

            # 3d. 为当前步骤构造独立任务（包含完整计划快照 + 步骤描述 + 指令），
            #     然后运行一次完整的 _run_act_mode（即一次 ReAct 循环）。
            #     preserve_plan_state=True 保证本轮结束后计划状态不被清空，
            #     route_workflow=False 跳过 Plan/Act 路由判断，直接执行。
            step_task = self._build_step_execution_task(plan, step)
            result = await self._run_act_mode(
                step_task,
                stream=stream,
                preserve_plan_state=True,
            )

            # 3e. 步骤执行完后，再次读取计划文件——执行期间用户可能手动编辑了计划，
            #     需要把最新的步骤状态合并进来（按 uid 或 id 匹配）。
            refreshed_after_execution = self._refresh_plan_from_file()
            if refreshed_after_execution is not None:
                plan = refreshed_after_execution
                refreshed_step = next(
                    (item for item in plan.steps if item.uid == step.uid or item.id == step.id),
                    None,
                )
                if refreshed_step is not None:
                    step = refreshed_step
                    step_plan_revision = self.state.plan_revision

            # 3f. 根据 ReAct 循环的返回结果更新步骤状态，分三种情况：
            #
            #   情况一：返回结果为空 → 标记失败，整个计划终止。
            if not result:
                self.state.mark_step_failed(
                    step.id,
                    "No result returned",
                    expected_plan_revision=step_plan_revision,
                )
                plan = self.state.current_plan or plan
                self.state.mark_plan_failed()
                self._sync_plan_progress(plan)
                self._emit_plan_update(plan)
                self._persist_current_plan()
                return self.state.last_result or "Plan execution complete"

            #   情况二：返回 "Task incomplete:" 或 "Task failed:" → 标记步骤失败，
            #           但 _should_continue_on_failure() 默认返回 False，即默认终止。
            if result.startswith("Task incomplete:") or result.startswith("Task failed:"):
                self.state.mark_step_failed(
                    step.id,
                    result,
                    expected_plan_revision=step_plan_revision,
                )
                plan = self.state.current_plan or plan
                self.state.mark_plan_failed()
                self._sync_plan_progress(plan)
                self._emit_plan_update(plan)
                self._persist_current_plan()
                if not self._should_continue_on_failure():
                    return self.state.last_result or result
                continue

            #   情况三：步骤成功 → 标记完成，记录结果，继续执行下一个步骤。
            self.state.mark_step_done(
                step.id,
                result,
                expected_plan_revision=step_plan_revision,
            )
            plan = self.state.current_plan or plan
            self._sync_plan_progress(plan)
            self._emit_plan_update(plan)
            self._persist_current_plan()

        # ── 4. 收尾 ─────────────────────────────────────────────────
        # 主循环结束后，根据计划中所有步骤的最终状态标记整个计划为"完成"或"失败"。
        final_result = self.state.last_result or "Plan execution complete"
        if plan.status == PlanStatus.DONE:
            self._sync_plan_progress(plan)
            self._emit_plan_update(plan)
            self.state.mark_plan_completed()
        elif plan.status == PlanStatus.FAILED:
            self._sync_plan_progress(plan)
            self.state.mark_plan_failed()
            self._emit_plan_update(plan)

        return final_result

    def _prepare_plan_for_execution(self, plan: Plan) -> None:
        """执行 `_prepare_plan_for_execution` 所定义的协调步骤，必要时更新Agent运行时维护的状态。

        参数：
            plan: 当前要保存、展示或执行的结构化计划。
        """
        if self.state.store is not None:
            self.state.requeue_interrupted_plan_steps()
            return
        for step in plan.steps:
            if step.status in {
                StepStatus.RUNNING,
                StepStatus.FAILED,
                StepStatus.INTERRUPTED,
            }:
                step.status = StepStatus.PENDING
                step.error = None

    def _is_plan_execution_approval(self, text: str) -> bool:
        """读取并返回 `_is_plan_execution_approval` 所表示的数据或流程，并遵守Agent运行时定义的边界与状态约束。

        参数：
            text: 需要解析、格式化或展示的文本。

        返回：
            表示条件是否成立。
        """
        if not self.state.current_plan:
            return False
        if self.state.plan_approval_status not in {
            PlanApprovalStatus.AWAITING_APPROVAL,
            PlanApprovalStatus.APPROVED,
        }:
            return False
        normalized = text.strip().lower()
        if not normalized:
            return False
        rejection_tokens = {"n", "no", "cancel", "取消", "不要", "别", "先不要", "不执行", "暂不"}
        if normalized in rejection_tokens or any(
            token in normalized for token in ("不要", "先不要", "别")
        ):
            return False
        approval_tokens = {
            "y",
            "yes",
            "approve",
            "approved",
            "execute",
            "run",
            "start",
            "go",
            "continue",
            "开始",
            "开始开发",
            "开始执行",
            "开始实现",
            "开始编码",
            "开始改",
            "开始做",
            "开始写代码",
            "开工",
            "开发",
            "执行",
            "执行计划",
            "继续",
            "继续开发",
            "继续执行",
            "同意",
            "批准",
        }
        if normalized in approval_tokens:
            return True
        return any(
            token in normalized
            for token in (
                "start coding",
                "execute plan",
                "implement the plan",
                "开始写代码",
                "开始开发",
                "开始实现",
                "开始执行",
                "按计划开发",
                "执行计划",
            )
        )

    def _emit_plan_update(self, plan: Plan) -> None:
        """发布`emit_plan_update`，通知已订阅的界面、SDK 或持久化组件。

        参数：
            plan: 当前要保存、展示或执行的结构化计划。
        """
        with suppress(Exception):
            self._emit("plan", plan, self.state.plan_file_path)

    async def _create_plan(self, task: str) -> Plan:
        """调用 Planner 生成结构化计划并优化合并。

        流程：
            1. 委托 self.planner.create_plan(task) 生成原始计划：
               - 先尝试按英文关键词匹配 COMMON_TEMPLATES（如含 "test"/"bug"/"refactor"）；
               - 匹配不到或匹配结果不是回退计划时，发送中文 PLANNING_PROMPT 给 LLM，
                 期望返回 JSON 格式的步骤列表，解析失败则回退为单步骤兜底计划。
            2. 委托 self.planner.optimize_plan(plan) 对超过 3 步的计划，
               将 tool_hint 相同的相邻步骤合并为一个步骤（用 "；然后" 连接描述）。

        参数：
            task: 用户希望 Agent 完成的任务描述。

        返回：
            经过生成和优化的 `Plan` 对象，包含任务描述、步骤列表和创建时间。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        plan = await self.planner.create_plan(task)
        return self.planner.optimize_plan(plan)

    def _save_plan_to_project(self, plan: Plan) -> Path:
        """保存计划转换到项目，并维持所在组件的一致性约束。

        参数：
            plan: 当前要保存、展示或执行的结构化计划。

        返回：
            `Path` 类型的处理结果。

        说明：
            该操作会访问本地文件系统，路径校验和原子写入约束由所在组件负责。
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plan_dir = Path(".opennova") / "plan"
        plan_dir.mkdir(parents=True, exist_ok=True)
        plan_path = plan_dir / f"plan_{timestamp}.md"
        content = self._render_saved_plan(plan, plan_path)
        temporary = plan_path.with_name(f".{plan_path.name}.tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, plan_path)
        return plan_path

    def _render_saved_plan(self, plan: Plan, plan_path: Path) -> str:
        """根据当前数据渲染`render_saved_plan`的界面或文本表示。

        参数：
            plan: 当前要保存、展示或执行的结构化计划。
            plan_path: 本次操作使用的计划路径。

        返回：
            处理后的文本或稳定标识。
        """
        summary = self._render_plan_snapshot(plan)
        lines = [
            f"# Saved Plan: {plan.task}",
            "",
            f"- Task: {self.state.current_task or plan.task}",
            f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
            f"- Saved path: {plan_path}",
            "",
            "## Summary",
            "",
            summary,
            "",
            "## Steps",
            "",
        ]

        for step in plan.steps:
            lines.append(f"### {step.id}")
            lines.append(f"<!-- opennova-step-uid: {step.uid} -->")
            lines.append(f"- Description: {step.description}")
            if step.tool_hint:
                lines.append(f"- Tool hint: `{step.tool_hint}`")
            lines.append(f"- Status: `{step.status.value}`")
            if step.result_summary:
                lines.append(f"- Result: {step.result_summary}")
            if step.error:
                lines.append(f"- Error: {step.error}")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def _render_plan_snapshot(self, plan: Plan) -> str:
        """根据当前数据渲染计划快照的界面或文本表示。

        参数：
            plan: 当前要保存、展示或执行的结构化计划。

        返回：
            处理后的文本或稳定标识。
        """
        lines = []
        for step in plan.steps:
            parts = [f"- [{step.status.value}] {step.id}: {step.description}"]
            if step.tool_hint:
                parts.append(f"(tool: {step.tool_hint})")
            if step.result_summary:
                parts.append(f"result={step.result_summary}")
            if step.error:
                parts.append(f"error={step.error}")
            lines.append(" ".join(parts))
        return "\n".join(lines) if lines else "- (no steps)"

    def _load_plan_from_markdown(self, content: str) -> Plan:
        """从 Markdown 文本解析并恢复 Plan 对象。

        支持两种格式：
            1. 规范格式：### heading + Description/Tool hint/Status/Result/Error 字段
            2. 旧版格式：数字编号 + **step_id** — description（回退兼容）

        参数：
            content: 计划文件的 Markdown 文本内容，通常由 .opennova/plan/*.md 读取

        返回：
            Plan: 解析后的计划对象，步骤状态已恢复，plan_revision 已同步
        """
        task = ""
        task_match = re.search(r"^# Saved Plan:\s*(.+)$", content, re.MULTILINE)
        if task_match:
            task = task_match.group(1).strip()
        task_line_match = re.search(r"^- Task:\s*(.+)$", content, re.MULTILINE)
        if task_line_match:
            task = task or task_line_match.group(1).strip()

        steps: list[PlanStep] = []
        canonical_lines = content.splitlines()
        current_step: PlanStep | None = None
        saw_canonical = False
        in_steps_section = False

        def append_current_step() -> None:
            nonlocal current_step
            if current_step is not None and current_step.description.strip():
                steps.append(current_step)
            current_step = None

        for raw_line in canonical_lines:
            line = raw_line.strip()
            section_match = re.match(r"^##\s+(.+)$", line)
            if section_match:
                if section_match.group(1).strip().lower() == "steps":
                    in_steps_section = True
                continue
            if not in_steps_section:
                continue
            heading_match = re.match(r"^###\s+(.+)$", line)
            if heading_match:
                heading = heading_match.group(1).strip()
                if self._is_canonical_plan_step_heading(heading):
                    saw_canonical = True
                    append_current_step()
                    current_step = PlanStep(id=heading, description="")
                continue
            if current_step is None:
                continue
            if line.startswith("<!-- opennova-step-uid:") and line.endswith("-->"):
                uid = line.removeprefix("<!-- opennova-step-uid:").removesuffix("-->").strip()
                if uid:
                    current_step.uid = uid
            elif line.startswith("- Description:"):
                current_step.description = line.split(":", 1)[1].strip()
            elif line.startswith("- Tool hint:"):
                current_step.tool_hint = line.split(":", 1)[1].strip().strip("`")
            elif line.startswith("- Status:"):
                status_value = line.split(":", 1)[1].strip().strip("`")
                current_step.status = current_step.status.__class__(status_value)
            elif line.startswith("- Result:"):
                current_step.result_summary = line.split(":", 1)[1].strip()
            elif line.startswith("- Error:"):
                current_step.error = line.split(":", 1)[1].strip()
        append_current_step()

        if not saw_canonical:
            steps = self._load_legacy_plan_steps(content)

        plan = Plan(task=task or "Saved plan", steps=steps).reindex_steps()
        plan._update_plan_status()
        return plan

    @staticmethod
    def _is_canonical_plan_step_heading(heading: str) -> bool:
        """读取并返回 `_is_canonical_plan_step_heading` 所表示的数据或流程，并遵守Agent运行时定义的边界与状态约束。

        参数：
            heading: 本次操作使用的标题。

        返回：
            表示条件是否成立。
        """
        normalized = heading.strip()
        if not re.match(r"^[A-Za-z0-9_.-]+$", normalized):
            return False
        lowered = normalized.lower()
        return lowered.startswith("step") or bool(re.search(r"\d", normalized))

    def _load_legacy_plan_steps(self, content: str) -> list[PlanStep]:
        """从配置、文件或持久化记录中加载`load_legacy_plan_steps`。

        参数：
            content: 需要处理、保存或分析的文本内容。

        返回：
            按调用约定排序的结果列表。
        """
        steps: list[PlanStep] = []
        current_step: PlanStep | None = None
        for raw_line in content.splitlines():
            line = raw_line.rstrip()
            step_match = re.match(r"^\d+\.\s+\*\*(.+?)\*\*\s+—\s+(.+)$", line.strip())
            if step_match:
                if current_step is not None:
                    steps.append(current_step)
                current_step = PlanStep(
                    id=step_match.group(1).strip(), description=step_match.group(2).strip()
                )
                continue
            if current_step is None:
                continue
            stripped = line.strip()
            if stripped.startswith("- Tool hint:"):
                current_step.tool_hint = stripped.split(":", 1)[1].strip().strip("`")
            elif stripped.startswith("- Status:"):
                status_value = stripped.split(":", 1)[1].strip().strip("`")
                current_step.status = current_step.status.__class__(status_value)
            elif stripped.startswith("- Result:"):
                current_step.result_summary = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("- Error:"):
                current_step.error = stripped.split(":", 1)[1].strip()
        if current_step is not None:
            steps.append(current_step)
        return steps

    def _persist_current_plan(self) -> None:
        """持久化当前计划，并按照当前组件的约定返回结果。

        说明：
            该操作会访问本地文件系统，路径校验和原子写入约束由所在组件负责。
        """
        plan = self.state.current_plan
        plan_path = self.state.plan_file_path
        if not plan or not plan_path:
            return
        path = Path(plan_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = self._render_saved_plan(plan, path)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
        state_store = getattr(self, "state_store", None)
        if state_store is not None:
            state_store.dispatch(
                RuntimeAction("plan_file_persisted", {"file_hash": self._hash_content(content)})
            )

    def _refresh_plan_from_file(self) -> Plan | None:
        """根据当前输入和Agent运行时的状态计算 `_refresh_plan_from_file`，并返回调用方需要的结果。

        返回：
            `Plan | None` 类型的处理结果。
        """
        plan_path = self.state.plan_file_path
        if not plan_path:
            return self.state.current_plan
        path = Path(plan_path)
        if not path.exists():
            return self.state.current_plan
        try:
            content = path.read_text(encoding="utf-8")
            file_hash = self._hash_content(content)
            state_store = getattr(self, "state_store", None)
            if state_store is not None and state_store.get_state().plan.file_hash == file_hash:
                return self.state.current_plan
            plan = self._load_plan_from_markdown(content)
        except Exception:
            return self.state.current_plan
        if state_store is not None:
            state_store.dispatch(
                RuntimeAction(
                    "plan_file_changed",
                    {"plan": plan, "file_hash": file_hash},
                    expected_plan_revision=self.state.plan_revision,
                )
            )
            refreshed = self.state.current_plan
            if refreshed is not None:
                self._emit_plan_update(refreshed)
            return refreshed
        self.state.current_plan = plan
        return self.state.current_plan

    @staticmethod
    def _hash_content(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @classmethod
    def _hash_file(cls, path: Path) -> str:
        return cls._hash_content(path.read_text(encoding="utf-8"))

    def _sync_plan_progress(self, plan: Plan, active_step_id: str | None = None) -> None:
        """同步计划进度，并按照当前组件的约定返回结果。

        参数：
            plan: 当前要保存、展示或执行的结构化计划。
            active_step_id: 可选的`active_step_id`。
        """
        from opennova.tools.todo_tools import TodoWriteTool

        state_store = getattr(self, "state_store", None)
        if state_store is not None:
            return

        todos = []
        for step in plan.steps:
            status = "pending"
            if step.status.value == "running" or step.id == active_step_id:
                status = "in_progress"
            elif step.status.value == "done":
                status = "done"
            elif step.status.value == "failed":
                status = "cancelled"
            content = step.description.strip() or step.id
            todos.append({"id": step.id, "content": content, "status": status})
        TodoWriteTool.replace_todos(todos)

    async def _confirm_plan(self, plan: Plan) -> bool:
        """根据当前输入和Agent运行时的状态计算 `_confirm_plan`，并返回调用方需要的结果。

        参数：
            plan: 当前要保存、展示或执行的结构化计划。

        返回：
            表示条件是否成立。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        event_bus = getattr(self, "events", None)
        plan_confirm = (
            event_bus.latest("plan_confirm")
            if event_bus is not None
            else getattr(self, "_callbacks", {}).get("plan_confirm")
        )
        if plan_confirm:
            return bool(plan_confirm(plan))
        return True

    def _should_continue_on_failure(self) -> bool:
        """根据当前输入和Agent运行时的状态计算 `_should_continue_on_failure`，并返回调用方需要的结果。

        返回：
            表示条件是否成立。
        """
        return False

    async def _run_act_mode(
        self,
        task: str,
        stream: bool = True,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        preserve_plan_state: bool = False,
        preserve_context: bool = False,
        route_workflow: bool = True,
    ) -> str:
        """执行 Act 模式的 Agent 任务：创建 ReActLoop 并运行推理-行动循环。

        本函数是单次 Agent 任务的"运行组织者"，负责组装运行环境、管理任务生命周期，
        但不直接决定具体的代码编写逻辑。真正的模型推理和工具调用由 ReActLoop.run() 完成。

        参数：
            task: 用户希望 Agent 完成的任务描述，例如 "帮我编写一段实现文件上传和下载的python代码"。
            stream: 是否将模型输出以增量事件形式返回。为 True 时，模型生成一小段内容就立即
                通过回调发送给 TUI 显示，而不是等整个回答完成后一次性展示。
            progress_callback: 每次运行进度变化时调用的回调函数，用于更新 TUI 中的进度显示。
            preserve_plan_state: 是否保留之前的计划状态。为 True 时，会保留之前生成的计划及其
                执行进度，适用于计划执行中断后继续的场景。
            preserve_context: 是否保留之前的对话上下文。为 True 时，不会清空之前的聊天记录，
                模型能够看到之前的对话内容。例如前一轮说"这个项目使用 FastAPI"，本轮再说
                "帮我编写文件上传代码"，模型会优先使用 FastAPI 实现。
            route_workflow: 是否进行 Plan/Act 工作流路由。为 True 时，会根据任务描述判断
                应该进入 Plan 模式（先生成计划再执行）还是 Act 模式（直接执行）。
                当 preserve_plan_state 为 True 时，此参数会被强制设为 False。

        返回：
            Agent 任务的执行结果文本。成功时返回模型的最终回答（如 "已创建 upload_download.py，
            实现了文件上传、下载和文件名安全检查……"），失败时返回以 "Task incomplete:" 或
            "Task failed:" 开头的错误描述。

        异常：
            KeyboardInterrupt: 用户按 Ctrl+C 取消任务时抛出。
            Exception: ReActLoop.run() 执行过程中发生任何异常时抛出，异常会被重新抛出
                由调用方（通常是 TUI）处理和显示。

        说明：
            - 执行过程中会更新当前实例维护的状态（AgentState、WorkingMemory、SessionManager 等）。
            - 这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
            - 每次调用都会新建一个 ReActLoop 实例，但会复用运行时已有的共享对象（LLM Provider、
              工具注册表、对话上下文、安全护栏等），因此对话历史和工具状态不会丢失。

        执行流程：
            1. 准备 MCP 等外部能力连接
            2. 准备取消令牌、文件缓存、会话和预算配置
            3. 创建新的 ReActLoop 并注入共享组件
            4. 处理对话上下文和项目记忆注入
            5. 注册 TUI 回调并运行 ReAct 循环
            6. 记录结果、更新状态、保存会话快照

        与 ReActLoop.run() 的关系：
            _run_act_mode() = 组装运行环境 + 管理任务生命周期
            ReActLoop.run() = 模型推理 + 工具调用循环
        """

        # 1. 确保 MCP 工具已经准备好
        # 如果配置了 MCP 服务，检查服务是否已连接；没有连接则尝试连接。
        # 没有配置 MCP 时直接返回，不影响普通内置工具。
        await self._ensure_mcp_ready()

        # 2. 准备本轮运行资源
        # 准备取消令牌：用户按 Ctrl+C 取消任务时，取消信号可以继续向 ReAct 循环和正在执行的工具传递。
        active_handle = getattr(self, "_active_run_handle", None)
        cancellation_token = (
            active_handle.token if active_handle is not None else CancellationToken()
        )
        # 读取配置，并准备以下资源：
        # - FileVersionCache：记录文件版本，辅助检测文件变化
        # - session_id：当前会话编号
        # - ArtifactStore：保存过大的工具结果或运行产物
        # - 预算配置：最大迭代次数、Token 预算、费用预算等
        runtime_config = getattr(self, "config", {})
        agent_config = runtime_config.get("agent", {})
        file_cache = getattr(self, "file_version_cache", None) or FileVersionCache()
        self.file_version_cache = file_cache
        session_manager = getattr(self, "session_manager", None)
        session_id = str(getattr(session_manager, "session_id", "") or "")
        artifact_store = getattr(self, "artifact_store", None) or ArtifactStore(
            Path.cwd(), session_id or "session"
        )
        self.artifact_store = artifact_store

        # 3. 创建一个新的 ReActLoop
        # 每次 _run_act_mode() 都会新建一个 ReActLoop，但会把运行时已有的共享对象传进去：
        # - llm：调用的大模型
        # - tool_registry：read_file、edit_file、execute_command 等工具
        # - state：当前任务、计划、运行次数和结果状态
        # - context_manager：保存对话历史
        # - working_memory：记录当前任务执行情况
        # - guardrails：检查工具调用是否允许
        # - cancellation_token：处理取消
        # - artifact_store：保存较大的运行结果
        # 所以 ReActLoop 虽然是新创建的，但对话上下文、工具和状态并不是全部重新创建。
        self.loop = ReActLoop(
            llm=self.llm,
            tool_registry=self.tool_registry,
            state=self.state,
            max_iterations=self.max_iterations,
            stream=stream,
            progress_callback=progress_callback,
            iteration_start_callback=lambda messages: self._emit("iteration_start", messages),
            interaction_callback=(
                self.events.latest("interaction")
                if getattr(self, "events", None) is not None
                else getattr(self, "_callbacks", {}).get("interaction")
            ),
            skill_registry=getattr(self, "skill_registry", None),
            context_manager=self.context_manager,
            working_memory=self.working_memory,
            guardrails=getattr(self, "guardrails", None),
            working_dir=os.getcwd(),
            hook_manager=getattr(self, "hook_manager", None),
            audit_logger=getattr(self, "security_audit_logger", None),
            cancellation_token=cancellation_token,
            file_cache=file_cache,
            artifact_store=artifact_store,
            parallel_tool_limit=int(
                agent_config.get("execution", {}).get("parallel_tool_limit", 4)
            ),
            per_turn_tool_result_chars=int(
                agent_config.get("execution", {}).get("per_turn_tool_result_chars", 160_000)
            ),
            session_id=session_id,
            deferred_tools_enabled=bool(
                agent_config.get("deferred_tools", {}).get("enabled", True)
            ),
            token_budget=int(agent_config.get("token_budget", 0)),
            cost_budget_usd=float(agent_config.get("cost_budget_usd", 0.0)),
            max_output_tokens=int(agent_config.get("max_output_tokens", 0)),
            input_cost_per_million=float(
                agent_config.get("input_cost_per_million", 0.0)
            ),
            output_cost_per_million=float(
                agent_config.get("output_cost_per_million", 0.0)
            ),
            fallback_providers=getattr(self, "fallback_providers", []),
            provider_retry_attempts=int(agent_config.get("provider_retry_attempts", 1)),
            provider_circuit_breaker=getattr(self, "provider_circuit_breaker", None),
        )
        started_at = perf_counter()

        # 4. 处理上下文和工作记忆
        # 本次调用中 preserve_context=True，所以不会清空之前的聊天记录。
        # 假设前一轮你说："这个项目使用 FastAPI"
        # 这一轮再说："帮我编写一段实现文件上传和下载的python代码"
        # 模型能够同时看到这两轮内容，从而优先使用 FastAPI。
        # 如果这是第一轮对话、上下文还是空的，则会调用 _build_memory_messages(task)，
        # 注入项目记忆、OPENNOVA.md 和 .opennova/memory 等内容。
        if not preserve_context:
            self.context_manager.clear()
        self.working_memory.set_task(task)
        self.working_memory.start_task()
        if not preserve_context:
            self.loop.set_context(self._build_memory_messages(task))
        elif not self.context_manager.messages:
            # 首轮需要直接注入项目记忆；这里不能调用 set_context，因为它会清空已保留的对话。
            for msg in self._build_memory_messages(task):
                self.context_manager.add_message(msg)

        # 5. 定义事件回调
        # 这些回调把 Agent 内部事件发送给 TUI：
        # - thought：模型思考信息
        # - action：准备调用什么工具
        # - result：工具执行结果
        # - stream：模型逐段输出的内容
        # - tool_event：更完整的工具生命周期事件
        # stream=True 时，模型返回一小段内容，TUI 就显示一小段，而不是等整个回答完成后一次性展示。
        # 内部仍会把这些片段合并成完整的 LLMResponse。
        def on_thought(thought: str) -> None:
            if self.show_thinking:
                self._emit("thought", thought)

        def on_action(tool_name: str, args: dict) -> None:
            self._emit("action", tool_name, args)

        def on_result(result: ToolResult) -> None:
            self._emit("result", result)

        def on_stream(chunk: StreamChunk) -> None:
            self._emit("stream", chunk)

        def on_tool_event(event: ToolEvent) -> None:
            self.tool_events.append(event.to_dict())
            self._emit("tool_event", event)

        try:
            # 6. 正式进入 ReAct 循环
            # 注意最后一个参数实际是：route_workflow=route_workflow and not preserve_plan_state
            # 普通任务中结果为：True and not False → True
            # 因此会进行 Plan/Act 工作流判断。
            # 如对于问题：帮我编写一段实现文件上传和下载的python代码
            # 它没有明确要求"先给计划，等我确认"，因此通常会被判断为 ACT，直接开始处理。
            # 如果判断为 PLAN，则只研究和生成计划，不能修改文件，之后由 _execute_task() 弹出审批窗口。
            # 进入 ACT 后，典型循环可能是：
            #     模型判断需要了解项目
            #         → 调用 list_directory
            #         → 获得目录结构
            #         → 调用 read_file 查看现有代码
            #         → 获得文件内容
            #         → 调用 create_file 或 edit_file 编写上传下载代码
            #         → 调用 execute_command 运行测试
            #         → 把工具结果再次交给模型
            #         → 模型给出最终回答
            # 这个工具顺序不是 _run_act_mode() 写死的，而是模型根据任务和工具结果动态决定的。
            result = await self.loop.run(
                task,
                on_thought=on_thought if self.show_thinking else None,
                on_action=on_action,
                on_result=on_result,
                on_stream=on_stream if stream else None,
                on_tool_event=on_tool_event,
                preserve_plan_state=preserve_plan_state,
                preserve_context=preserve_context,
                route_workflow=route_workflow and not preserve_plan_state,
            )
        except Exception:
            # 7. 如果 ReActLoop.run() 抛出异常，代码会：
            # - 将工作记忆标记为失败
            # - 结束本轮运行状态
            # - 记录失败会话
            # - 保存消息快照
            # - 继续抛出异常，由 TUI 显示错误
            self.working_memory.complete_task(success=False, error="Act mode execution failed")
            active_run_id = getattr(getattr(self, "loop", None), "active_run_id", None)
            self.state.finish_run(
                "Act mode execution failed",
                success=False,
                run_id=active_run_id,
            )
            self._record_run_session(task, success=False, started_at=started_at)
            self._save_session_messages()
            raise

        # 7. 处理成功，会话保存
        # - 将工作记忆标记为成功
        # - 结束本轮运行状态
        # - 记录成功会话
        # - 保存消息快照
        # 最终返回的 result，回到 _run_agent_task()，随后显示在 TUI 消息区域。
        # （如：已创建 upload_download.py，实现了文件上传、下载和文件名安全检查……）
        success = not (
            result.startswith("Task incomplete:")
            or result.startswith("Task failed:")
            or result == "Plan approval required before execution"
        )
        active_run_id = getattr(self.loop, "active_run_id", None)
        if active_run_id is not None and self.state.run_id != active_run_id:
            return result
        self.state.finish_run(result, success=success, run_id=active_run_id)
        self.working_memory.complete_task(success=success, error=None if success else result)
        self._record_run_session(task, success=success, started_at=started_at)
        self._save_session_messages()
        return result

    async def chat(self, message: str, stream: bool = True) -> str:
        """发送一次非流式模型请求，并把厂商响应规范化为 LLMResponse；具体 Provider 负责协议转换和异常归一化。

        参数：
            message: 用户提交或组件间传递的消息。
            stream: 是否将模型输出以增量事件形式返回。

        返回：
            处理后的文本或稳定标识。

        说明：
            该操作可能等待模型服务、MCP 服务或其他异步数据源返回。
        """
        messages = [
            Message(role="system", content="You are a helpful AI assistant."),
            Message(role="user", content=message),
        ]

        if stream:
            full_content = ""
            async for chunk in self.llm.stream_chat(messages, temperature=0.7):
                if chunk.content:
                    full_content += chunk.content
                    self._emit("stream", chunk)
            return full_content
        else:
            response = await self.llm.chat(messages, temperature=0.7)
            return response.content

    def clear_conversation(self) -> None:
        """清空对话并恢复到可继续使用的初始状态。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self._save_session_messages()
        self.context_manager.clear()
        self.context_manager.set_compressed_summary(None)
        self.state.reset("")
        self.session_transcript = []
        sm = getattr(self, "session_manager", None)
        if sm is not None:
            sm.clear_session()
            session_id = sm.start_session()
            self.artifact_store = ArtifactStore(self.project_path, session_id)
            self.file_version_cache.clear()
            state_store = getattr(self, "state_store", None)
            if state_store is not None:
                state_store.bind_session(session_id)

    def _save_session_messages(self) -> None:
        """保存会话消息，并维持所在组件的一致性约束。"""
        try:
            summary = self.context_manager.get_compressed_summary()
            self.session_manager.save_snapshot(
                self.context_manager.messages,
                compression_summary=summary,
                transcript_events=self.session_transcript,
                plan_state=self._serialize_plan_state(),
                runtime_state=(
                    self.state_store.serialize()
                    if getattr(self, "state_store", None) is not None
                    else None
                ),
            )
        except Exception as exc:
            _LOGGER.warning("Failed to persist OpenNova session state: %s", exc)
            with suppress(Exception):
                self._emit("diagnostic", "session_persistence_failed", str(exc))

    def record_session_transcript_event(self, kind: str, **payload: Any) -> None:
        """记录会话转录记录事件，供状态展示、恢复或后续决策使用。

        参数：
            kind: 本次操作使用的`kind`。
            **payload: 传递给底层实现的额外关键字参数。
        """
        self.session_transcript.append({"kind": kind, **payload})

    def flush_session(self) -> None:
        """执行 `flush_session` 所定义的协调步骤，必要时更新Agent运行时维护的状态。"""
        self._save_session_messages()
        with suppress(Exception):
            self.session_manager.flush_runtime_events()

    def resume_session(self, session_id: str) -> Any:
        """将当前写入器重新绑定到已有 session ID，加载对应消息和运行状态；后续保存继续写入原会话，不会隐式创建重复会话。

        参数：
            session_id: 目标会话的稳定标识。

        返回：
            `Any` 类型的处理结果。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        loaded = self.session_manager.load_session_with_summary(session_id)
        self.context_manager.clear()
        self.context_manager.set_compressed_summary(loaded.compression_summary)
        for msg in loaded.messages:
            self.context_manager.add_message(msg)
        # 恢复后继续绑定原会话写入，不能创建内容重复的新 session。
        self.session_manager.resume_session(session_id)
        project_path = getattr(
            self,
            "project_path",
            getattr(self.session_manager, "_project_root", Path.cwd()),
        )
        self.artifact_store = ArtifactStore(project_path, session_id)
        file_cache = getattr(self, "file_version_cache", None) or FileVersionCache()
        file_cache.clear()
        self.file_version_cache = file_cache
        self.session_transcript = [dict(event.payload) for event in loaded.transcript_events]
        state_store = getattr(self, "state_store", None)
        persistence_was_ready = getattr(self, "_state_persistence_ready", False)
        self._state_persistence_ready = False
        try:
            if state_store is not None:
                state_store.bind_session(session_id)
                if loaded.runtime_state:
                    state_store.restore(loaded.runtime_state)
                else:
                    state_store.restore(
                        {
                            "schema_version": 2,
                            "revision": 0,
                            "session_id": session_id,
                            "run": {"phase": "idle"},
                            "plan": {"lifecycle": "none", "revision": 0},
                            "todos": {},
                            "interaction": {},
                        }
                    )
                replay_result = state_store.replay(loaded.state_events)
                loaded.recovery_warnings.extend(replay_result.warnings)
                loaded.last_valid_revision = replay_result.last_valid_revision
                if not loaded.runtime_state and loaded.plan_state:
                    self._restore_plan_state(loaded.plan_state)
                self._refresh_plan_from_file()
                snapshot = state_store.get_state()
                if (
                    snapshot.run.phase == "running"
                    or snapshot.plan.lifecycle == PlanApprovalStatus.EXECUTING
                    or (
                        snapshot.plan.plan is not None
                        and any(
                            step.status == StepStatus.RUNNING for step in snapshot.plan.plan.steps
                        )
                    )
                ):
                    state_store.dispatch(RuntimeAction("session_interrupted_recovered"))
            else:
                self._restore_plan_state(loaded.plan_state)
        finally:
            self._state_persistence_ready = persistence_was_ready
        self._save_session_messages()
        return loaded

    def fork_session(self, session_id: str | None = None) -> str:
        """复制一个已持久化会话并分配新的 session ID，使新旧时间线可以独立继续写入。

        参数：
            session_id: 目标会话的稳定标识。

        返回：
            处理后的文本或稳定标识。
        """
        self.flush_session()
        source_id = session_id or self.session_manager.session_id
        if not source_id:
            raise ValueError("No source session is available to fork")
        fork_id = self.session_manager.fork_session(source_id)
        self.resume_session(fork_id)
        return fork_id

    def _serialize_plan_state(self) -> dict[str, Any]:
        """读取并返回 `_serialize_plan_state` 所表示的数据或流程，并遵守Agent运行时定义的边界与状态约束。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        state = getattr(self, "state", None)
        if state is None:
            return {}
        return {
            "current_plan": state.current_plan.to_dict()
            if getattr(state, "current_plan", None)
            else None,
            "plan_file_path": str(state.plan_file_path)
            if getattr(state, "plan_file_path", None)
            else None,
            "plan_approval_status": getattr(
                getattr(state, "plan_approval_status", None), "value", None
            ),
        }

    def _restore_plan_state(self, plan_state: dict[str, Any] | None) -> None:
        """从已有快照或持久化数据恢复计划状态。

        参数：
            plan_state: 可选的计划状态。
        """
        if not plan_state:
            return

        current_plan = plan_state.get("current_plan")
        plan_file_path = plan_state.get("plan_file_path")
        approval_status = plan_state.get("plan_approval_status")

        if plan_file_path and hasattr(self.state, "set_plan_file_path"):
            self.state.set_plan_file_path(plan_file_path)

        restored_plan: Plan | None = None
        if plan_file_path and Path(plan_file_path).exists():
            try:
                restored_plan = self._load_plan_from_markdown(
                    Path(plan_file_path).read_text(encoding="utf-8")
                )
            except Exception:
                restored_plan = None
        elif isinstance(current_plan, dict):
            restored_plan = Plan.from_dict(current_plan)

        if restored_plan is not None and hasattr(self.state, "set_plan"):
            self.state.set_plan(restored_plan)

        if approval_status:
            with suppress(ValueError):
                self.state.plan_approval_status = PlanApprovalStatus(approval_status)

    def get_sessions(self) -> list[Any]:
        """读取 `sessions` 对应的数据，不改变当前对象的业务状态。

        返回：
            按调用约定排序的结果列表。
        """
        return self.session_manager.list_sessions()

    def get_state(self) -> AgentState:
        """读取状态，不改变当前对象的业务状态。

        返回：
            `AgentState` 类型的处理结果。
        """
        return self.state

    def get_tools(self) -> list[str]:
        """读取工具，不改变当前对象的业务状态。

        返回：
            按调用约定排序的结果列表。
        """
        return self.tool_registry.list_names()

    async def init_project_guide_async(self, force: bool = False) -> ToolResult:
        """根据当前输入和Agent运行时的状态计算 `init_project_guide_async`，并返回调用方需要的结果。

        参数：
            force: 是否跳过可选保护并强制执行；硬性安全限制仍然生效。

        返回：
            `ToolResult` 类型的处理结果。

        说明：
            该操作可能等待模型服务、MCP 服务或其他异步数据源返回。
        """
        from opennova.memory.project_guide import ProjectGuideManager

        project_memory = getattr(self, "project_memory", None)
        project_path = getattr(project_memory, "project_path", Path(os.getcwd()))
        manager = ProjectGuideManager(project_path=project_path)

        if manager.exists() and not force:
            result = manager.create_or_skip(force=False)
            return ToolResult(
                success=True,
                output=result.message,
                metadata={
                    "status": result.status,
                    "file_path": str(result.path),
                    "overwritten": result.overwritten,
                    "force": force,
                    "source": "skip",
                },
            )

        brief = manager.build_generation_brief()
        messages = [
            Message(
                role="system",
                content=(
                    "You are generating an OPENNOVA.md project guide for an AI coding assistant. "
                    "Analyze the provided project facts and write a practical, high-signal guide in Markdown. "
                    "Do not output code fences around the whole document."
                ),
            ),
            Message(
                role="user",
                content=(
                    "Write OPENNOVA.md for this repository.\n\n"
                    "Required coverage (you decide structure/detail):\n"
                    "- project overview and goals\n"
                    "- tech stack and architecture conventions\n"
                    "- directory structure highlights\n"
                    "- common development commands\n"
                    "- coding standards and workflow preferences\n"
                    "- testing expectations\n"
                    "- environment variables and third-party services\n"
                    "- known issues / risks / forbidden operations\n"
                    "- practical collaboration guidance for the AI assistant\n\n"
                    "Requirements:\n"
                    "1. Be specific to the current repository, avoid generic filler.\n"
                    "2. If facts are unknown, explicitly mark as TODO rather than inventing.\n"
                    "3. Keep it concise but actionable.\n"
                    "4. Write in Chinese by default.\n\n"
                    f"Project facts:\n{brief}"
                ),
            ),
        ]

        try:
            response = await self.llm.chat(messages, temperature=0.2)
            content = ProjectGuideManager.normalize_generated_markdown(response.content)
            if not content.strip():
                raise ValueError("LLM returned empty content")
            result = manager.create_or_skip(force=force, content=content + "\n")
            source = "llm"
        except Exception:
            # 只有模型生成项目指南失败时才使用本地回退模板。
            result = manager.create_or_skip(force=force)
            source = "fallback_template"

        return ToolResult(
            success=True,
            output=result.message,
            metadata={
                "status": result.status,
                "file_path": str(result.path),
                "overwritten": result.overwritten,
                "force": force,
                "source": source,
            },
        )

    def get_model_info(self) -> dict[str, Any]:
        """返回当前 Provider 使用的模型名称、上下文窗口、最大输出和工具或推理能力。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        return self.llm.get_model_info()

    def get_skills(self) -> list[str]:
        """读取Skill，不改变当前对象的业务状态。

        返回：
            按调用约定排序的结果列表。
        """
        if self.skill_registry:
            return self.skill_registry.list_user_invocable_skills()
        return []

    def get_skill_argument_hint(self, skill_name: str, typed_args: str = "") -> str | None:
        """读取Skill参数提示，不改变当前对象的业务状态。

        参数：
            skill_name: 本次操作使用的`skill_name`。
            typed_args: 可选的`typed_args`。

        返回：
            `str | None` 类型的处理结果。
        """
        if not self.skill_registry:
            return None
        return self.skill_registry.get_skill_argument_hint(skill_name, typed_args)

    def notify_file_paths_touched(self, paths: list[str]) -> dict[str, list[str]]:
        """通知 `file_paths_touched` 对应的数据，并按照当前组件的约定返回结果。

        参数：
            paths: 本次操作使用的`paths`。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        if not self.skill_registry:
            return {"activated": [], "discovered": []}
        cwd = os.getcwd()
        discovered = self.skill_registry.discover_for_paths(paths, cwd)
        activated = self.skill_registry.activate_for_paths(paths, cwd)
        return {"activated": activated, "discovered": discovered}

    def invoke_skill(
        self, skill_name: str, skill_args: str = "", caller: str = "user"
    ) -> ToolResult:
        """根据当前输入和Agent运行时的状态计算 `invoke_skill`，并返回调用方需要的结果。

        参数：
            skill_name: 本次操作使用的`skill_name`。
            skill_args: 可选的`skill_args`。
            caller: 可选的`caller`。

        返回：
            `ToolResult` 类型的处理结果。
        """
        if not self.skill_registry:
            return ToolResult(success=False, output="", error="Skill registry is not available")

        normalized_name = str(skill_name).strip().lstrip("/")
        normalized_args = str(skill_args).strip()
        resolution = self.skill_registry.resolve_skill_name(normalized_name)
        if resolution.is_ambiguous:
            matches = ", ".join(resolution.matches)
            return ToolResult(
                success=False,
                output="",
                error=f"Ambiguous skill '{normalized_name}'. Use one of: {matches}",
            )
        if not resolution.resolved_name:
            return ToolResult(
                success=False, output="", error=f"Skill '{normalized_name}' is unavailable"
            )
        resolved_name = resolution.resolved_name

        if caller == "model":
            if not self.skill_registry.can_model_invoke(resolved_name):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Skill '{resolved_name}' cannot be invoked by the model",
                )
        else:
            if not self.skill_registry.can_user_invoke(resolved_name):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Skill '{resolved_name}' cannot be invoked directly by the user",
                )

        prompt = self.skill_registry.materialize_skill_prompt(resolved_name, normalized_args)
        if prompt is None:
            return ToolResult(
                success=False, output="", error=f"Skill '{resolved_name}' is unavailable"
            )

        return ToolResult(
            success=True,
            output=f"Invoked skill: {resolved_name}",
            metadata={
                "skill": normalized_name,
                "resolved_skill": resolved_name,
                "args": normalized_args,
                "skill_prompt": prompt.prompt,
                "allowed_tools": prompt.allowed_tools,
                "model": prompt.model,
                "argument_names": prompt.argument_names,
                "hooks": prompt.hooks,
                "activation_state": prompt.activation_state,
                "source_path": prompt.source_path,
                "skill_dir": prompt.skill_dir,
                "caller": caller,
            },
        )

    def get_mcp_servers(self) -> list[str]:
        """读取并规范化配置中的 MCP 服务列表，缺失配置时返回空列表。

        返回：
            按调用约定排序的结果列表。
        """
        if self.mcp_manager:
            return self.mcp_manager.get_server_names()
        return []

    def reload_skills(self) -> int:
        """根据当前输入和Agent运行时的状态计算 `reload_skills`，并返回调用方需要的结果。

        返回：
            `int` 类型的处理结果。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        from opennova.skills.examples import get_builtin_skill_dirs
        from opennova.skills.registry import SkillRegistry

        skills_config = self.config.get("skills", {})
        if not skills_config.get("enabled", True):
            if self.skill_registry:
                self.skill_registry.clear()
            return 0

        if not self.skill_registry:
            self.skill_registry = SkillRegistry()
        registry = self.skill_registry

        configured_dirs = [Path(path) for path in skills_config.get("dirs", [])]
        excluded = skills_config.get("exclude", [])

        registry.load_all(
            directories=[*get_builtin_skill_dirs(), *configured_dirs],
            sources=self.plugin_manager.get_skill_sources(),
            excluded=excluded,
        )
        return len(registry)
