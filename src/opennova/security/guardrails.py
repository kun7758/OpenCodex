"""安全控制子系统中的安全护栏模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from opennova.security.command_policy import CommandPolicy
from opennova.security.network_policy import NetworkPolicy
from opennova.security.permission_rules import PermissionRuleMatcher
from opennova.security.permissions import PermissionDecision, PermissionStore
from opennova.security.secrets import SecretScanner


class RiskLevel(StrEnum):
    """枚举`RiskLevel`允许出现的稳定取值，序列化和状态判断均使用这些值。"""

    SAFE = "safe"
    WARN = "warn"
    DANGER = "danger"
    BLOCK = "block"


class PermissionMode(StrEnum):
    """枚举权限模式允许出现的稳定取值，序列化和状态判断均使用这些值。"""

    REQUEST = "request"
    AUTO = "auto"
    FULL = "full"

    # 继续接受旧权限模式名称，避免现有配置文件和 API 调用立即失效。
    DEFAULT = "default"
    ASK = "ask"
    ALLOW_EDITS = "allowEdits"
    READ_ONLY = "readOnly"
    BYPASS = "bypass"

    @classmethod
    def normalize(cls, value: str | PermissionMode) -> PermissionMode:
        """处理规范化，并按照当前组件的约定返回结果。

        参数：
            value: 需要保存、转换或校验的值。

        返回：
            `PermissionMode` 类型的处理结果。
        """
        mode = value if isinstance(value, cls) else cls(value)
        aliases = {
            cls.DEFAULT: cls.AUTO,
            cls.ASK: cls.REQUEST,
            cls.ALLOW_EDITS: cls.AUTO,
            cls.BYPASS: cls.FULL,
        }
        return aliases.get(mode, mode)


@dataclass
class GuardResult:
    """保存安全检查结果所需的结构化数据，主要包含
    `allowed`、`risk_level`、`reason`、`requires_confirmation`、`suggestions`、`metadata`
    字段，便于在组件之间传递或持久化。
    """

    allowed: bool
    risk_level: RiskLevel
    reason: str
    requires_confirmation: bool = False
    suggestions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.allowed


DANGEROUS_COMMAND_PATTERNS = [
    (r"rm\s+-rf\s+/", "Delete root directory"),
    (r"rm\s+-rf\s+~", "Delete home directory"),
    (r"rm\s+-rf\s+\*", "Delete all files in current directory"),
    (r"chmod\s+(-R\s+)?777", "Set unsafe permissions"),
    (r"chown\s+.*:\s*/", "Change root ownership"),
    (r">\s*/etc/", "Write to system configuration"),
    (r">\s*/dev/", "Write to device files"),
    (r"dd\s+if=.*of=", "Disk write operation"),
    (r"mkfs", "Format filesystem"),
    (r"fdisk", "Disk partitioning"),
    (r":\(\)\s*\{\s*:\|\:&\s*\}\s*;:", "Fork bomb"),
    (r"curl\s+.*\|\s*(sh|bash|zsh)", "Execute remote script"),
    (r"wget\s+.*\|\s*(sh|bash|zsh)", "Execute remote script"),
    (r"eval\s+", "Eval command execution"),
    (r"exec\s+", "Exec command"),
    (r"sudo\s+", "Elevated privileges"),
    (r"su\s+", "Switch user"),
    (r"shutdown", "System shutdown"),
    (r"reboot", "System reboot"),
    (r"init\s+[06]", "System shutdown/reboot"),
]

PROTECTED_PATHS = [
    "/etc",
    "/usr",
    "/bin",
    "/sbin",
    "/boot",
    "/dev",
    "/proc",
    "/sys",
    "/root",
    "~/.ssh",
    "~/.gnupg",
    "~/.config/git",
]

SENSITIVE_FILE_PATTERNS = [
    r"\.env$",
    r"\.pem$",
    r"\.key$",
    r"\.p12$",
    r"\.pfx$",
    r"id_rsa",
    r"id_ed25519",
    r"\.gitconfig$",
    r"\.netrc$",
    r"\.pgpass$",
    r"credentials\.json$",
    r"secrets\.json$",
    r"secrets\.yaml$",
    r"\.htpasswd$",
]

NETWORK_COMMAND_PATTERNS = [
    (r"\bcurl\b", "Network command: curl"),
    (r"\bwget\b", "Network command: wget"),
    (r"\bpip(?:3)?\s+(install|download)\b", "Python package download/install"),
    (r"\buv\s+(sync|add|pip|tool\s+install)\b", "uv dependency/network operation"),
    (r"\bnpm\s+(install|i|update|add)\b", "npm package install/update"),
    (r"\byarn\s+(add|install)\b", "yarn package install"),
    (r"\bpnpm\s+(add|install)\b", "pnpm package install"),
    (r"\bgit\s+(clone|fetch|pull)\b", "git network operation"),
]

SHELL_FEATURE_PATTERNS = [
    r"\|",
    r">>",
    r"(?<!\d)>",
    r"<",
    r"\$\(",
    r"`[^`]+`",
    r"\*",
    r"\?",
]

APPROVAL_EXEMPT_TOOLS = {
    "ask_user_question",
    "enter_plan_mode",
    "exit_plan_mode",
}


class Guardrails:
    """工具执行前的安全决策中心。它组合权限模式、显式规则、路径沙箱、命令策略、网络策略和敏感信息检查，返回允许、阻止或需要用户确认的结果。"""

    def __init__(
        self,
        sandbox_mode: bool = True,
        allowed_paths: list[str] | None = None,
        blocked_commands: list[str] | None = None,
        auto_confirm_safe: bool = True,
        allow_network: bool = True,
        permission_mode: str | PermissionMode = PermissionMode.DEFAULT,
        always_allow_tools: list[str] | None = None,
        always_deny_tools: list[str] | None = None,
        always_ask_tools: list[str] | None = None,
        permission_store: PermissionStore | None = None,
        permission_rules: list[dict[str, Any]] | None = None,
        strict_shell_parsing: bool = False,
        network_policy: dict[str, Any] | None = None,
        secrets_policy: dict[str, Any] | None = None,
    ):
        """初始化安全护栏，保存后续操作需要的依赖、配置和初始状态。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self.sandbox_mode = sandbox_mode
        self.allowed_paths = allowed_paths or []
        self.blocked_commands = blocked_commands or []
        self.auto_confirm_safe = auto_confirm_safe
        self.allow_network = allow_network
        self.strict_shell_parsing = strict_shell_parsing
        self.permission_mode = PermissionMode(permission_mode)
        self.permission_store = permission_store
        self.always_allow_tools = set(always_allow_tools or [])
        self.always_deny_tools = set(always_deny_tools or [])
        self.always_ask_tools = set(always_ask_tools or [])
        self.network_policy = NetworkPolicy.from_config(network_policy)
        self.command_policy = CommandPolicy(
            allow_network=allow_network,
            strict_shell_parsing=strict_shell_parsing,
            network_policy=self.network_policy,
        )
        self.permission_rule_matcher = PermissionRuleMatcher.from_config(permission_rules)
        self.secrets_policy = secrets_policy or {}
        self.secret_scanner = SecretScanner.from_config(self.secrets_policy)
        if permission_store:
            self.always_allow_tools.update(permission_store.allowed_tools())
            self.always_deny_tools.update(permission_store.denied_tools())
            self.always_ask_tools.update(permission_store.ask_tools())

    @staticmethod
    def _tool_category(tool_name: str) -> str:
        if tool_name in {
            "read_file",
            "list_directory",
            "glob_files",
            "grep_code",
            "web_search",
            "web_fetch",
            "list_mcp_resources",
            "read_mcp_resource",
            "python_diagnostics",
            "python_symbols",
            "python_definition",
            "python_references",
        }:
            return "read"
        if tool_name in {
            "write_file",
            "create_file",
            "edit_file",
            "multi_edit_file",
            "delete_file",
            "enter_worktree",
            "exit_worktree",
        }:
            return "edit"
        if tool_name == "execute_command":
            return "command"
        return "other"

    @property
    def effective_permission_mode(self) -> PermissionMode:
        """计算并返回 `effective_permission_mode` 属性；读取该属性不会主动改变对象的业务状态。

        返回：
            `PermissionMode` 类型的处理结果。
        """
        return PermissionMode.normalize(self.permission_mode)

    def set_permission_mode(self, mode: str | PermissionMode) -> PermissionMode:
        """设置权限模式并保持相关派生状态同步。

        参数：
            mode: 本次运行采用的工作模式。

        返回：
            `PermissionMode` 类型的处理结果。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self.permission_mode = PermissionMode(mode)
        return self.effective_permission_mode

    def _check_permission_mode(self, tool_name: str) -> GuardResult | None:
        """检查`check_permission_mode`并返回明确的校验或策略结果。

        参数：
            tool_name: 目标工具在注册表中的名称。

        返回：
            `GuardResult | None` 类型的处理结果。
        """
        if tool_name in self.always_deny_tools:
            return GuardResult(False, RiskLevel.BLOCK, f"Tool is denied by policy: {tool_name}")
        if self.permission_store:
            decision = self.permission_store.decision_for(tool_name)
            if decision == PermissionDecision.ALWAYS_DENY:
                return GuardResult(False, RiskLevel.BLOCK, f"Tool is denied by policy: {tool_name}")

        category = self._tool_category(tool_name)
        if self.permission_mode == PermissionMode.READ_ONLY and category != "read":
            return GuardResult(False, RiskLevel.BLOCK, f"Read-only mode blocks tool: {tool_name}")
        return None

    def _apply_approval_policy(
        self,
        tool_name: str,
        safety_result: GuardResult,
        rule_result: GuardResult | None = None,
    ) -> GuardResult:
        """应用审批策略，并按照当前组件的约定返回结果。

        参数：
            tool_name: 目标工具在注册表中的名称。
            safety_result: 本次操作使用的`safety_result`。
            rule_result: 可选的规则结果。

        返回：
            `GuardResult` 类型的处理结果。
        """
        if not safety_result.allowed:
            return safety_result
        if rule_result and not rule_result.allowed:
            return rule_result

        if rule_result:
            safety_result.metadata.update(rule_result.metadata)

        configured_decision = (
            self.permission_store.decision_for(tool_name) if self.permission_store else None
        )
        explicitly_ask = (
            tool_name in self.always_ask_tools
            or configured_decision == PermissionDecision.ALWAYS_ASK
            or bool(rule_result and rule_result.requires_confirmation)
        )
        explicitly_allow = (
            tool_name in self.always_allow_tools
            or configured_decision == PermissionDecision.ALWAYS_ALLOW
            or bool(rule_result and rule_result.allowed and not rule_result.requires_confirmation)
        )

        mode = self.effective_permission_mode
        prior_confirmation = safety_result.requires_confirmation or explicitly_ask
        approval_source: str | None = None
        if tool_name in APPROVAL_EXEMPT_TOOLS:
            requires_confirmation = False
        elif mode == PermissionMode.REQUEST:
            requires_confirmation = True
            approval_source = "request_mode"
        elif mode == PermissionMode.FULL:
            requires_confirmation = False
        else:
            requires_confirmation = safety_result.risk_level == RiskLevel.DANGER or explicitly_ask
            if explicitly_ask:
                approval_source = "explicit_ask"
            elif safety_result.risk_level == RiskLevel.DANGER:
                approval_source = "danger"
            if explicitly_allow and safety_result.risk_level == RiskLevel.SAFE:
                requires_confirmation = False

        safety_result.requires_confirmation = requires_confirmation
        safety_result.metadata.update(
            {
                "permission_mode": mode.value,
                "approval_required": requires_confirmation,
                "approval_source": approval_source,
                "auto_approved": bool(
                    mode == PermissionMode.AUTO
                    and not requires_confirmation
                    and safety_result.risk_level in {RiskLevel.SAFE, RiskLevel.WARN}
                ),
                "approval_bypassed": bool(mode == PermissionMode.FULL and prior_confirmation),
            }
        )
        return safety_result

    @staticmethod
    def command_uses_shell_features(command: str) -> bool:
        """读取并返回 `command_uses_shell_features` 所表示的数据或流程，并遵守安全护栏定义的边界与状态约束。

        参数：
            command: 需要分析或执行的 Shell 命令。

        返回：
            表示条件是否成立。
        """
        return CommandPolicy.command_uses_shell_features(command)

    def check_command(self, command: str) -> GuardResult:
        """检查`check_command`并返回明确的校验或策略结果。

        参数：
            command: 需要分析或执行的 Shell 命令。

        返回：
            `GuardResult` 类型的处理结果。
        """
        command_stripped = command.strip()
        analysis = self.command_policy.analyze(command_stripped)
        metadata = {"command_analysis": analysis.to_dict()}

        for blocked in self.blocked_commands:
            if blocked.lower() in command_stripped.lower():
                return GuardResult(
                    allowed=False,
                    risk_level=RiskLevel.BLOCK,
                    reason=f"Command is in blocked list: {blocked}",
                    requires_confirmation=False,
                    metadata=metadata,
                )

        for pattern, description in DANGEROUS_COMMAND_PATTERNS:
            if re.search(pattern, command_stripped, re.IGNORECASE):
                return GuardResult(
                    allowed=False,
                    risk_level=RiskLevel.BLOCK,
                    reason=f"Potentially dangerous command detected: {description}",
                    requires_confirmation=True,
                    suggestions=[
                        "Review the command carefully before executing",
                        "Consider using a safer alternative",
                    ],
                    metadata=metadata,
                )

        if analysis.risk_level == "block":
            return GuardResult(
                allowed=False,
                risk_level=RiskLevel.BLOCK,
                reason=analysis.reason,
                requires_confirmation=False,
                metadata=metadata,
            )

        if analysis.risk_level == "danger":
            return GuardResult(
                allowed=True,
                risk_level=RiskLevel.DANGER,
                reason=analysis.reason,
                requires_confirmation=True,
                suggestions=[
                    "Review the target and irreversible effects before executing",
                    "Prefer a reversible or narrowly scoped alternative where possible",
                ],
                metadata=metadata,
            )

        if analysis.risk_level == "warn":
            return GuardResult(
                allowed=True,
                risk_level=RiskLevel.WARN,
                reason=analysis.reason,
                requires_confirmation=True,
                suggestions=[
                    "Review the command carefully before executing",
                    "Prefer narrowly scoped commands where possible",
                ],
                metadata=metadata,
            )

        if not self.allow_network:
            for pattern, description in NETWORK_COMMAND_PATTERNS:
                if re.search(pattern, command_stripped, re.IGNORECASE):
                    return GuardResult(
                        allowed=False,
                        risk_level=RiskLevel.BLOCK,
                        reason=f"Network access is disabled by policy: {description}",
                        requires_confirmation=False,
                        metadata=metadata,
                    )

        destructive_patterns = [
            (r"rm\s+", "File deletion"),
            (r"rmdir", "Directory removal"),
            (r"git\s+push\s+--force", "Force push to git"),
            (r"git\s+reset\s+--hard", "Hard reset git"),
            (r"drop\s+(table|database|schema)", "Database drop"),
            (r"delete\s+from", "Database deletion"),
            (r"truncate\s+", "Table truncation"),
        ]

        for pattern, description in destructive_patterns:
            if re.search(pattern, command_stripped, re.IGNORECASE):
                return GuardResult(
                    allowed=True,
                    risk_level=RiskLevel.DANGER,
                    reason=f"Destructive operation detected: {description}",
                    requires_confirmation=True,
                    suggestions=[
                        "Ensure you have backups",
                        "Double-check the target",
                    ],
                    metadata=metadata,
                )

        if self.command_uses_shell_features(command_stripped):
            return GuardResult(
                allowed=True,
                risk_level=RiskLevel.WARN,
                reason="Command uses shell syntax and requires shell fallback parsing",
                requires_confirmation=True,
                suggestions=[
                    "Prefer argument-based commands without shell operators",
                    "Confirm shell fallback execution if this is intentional",
                ],
                metadata=metadata,
            )

        return GuardResult(
            allowed=True,
            risk_level=RiskLevel.SAFE,
            reason="Command appears safe",
            requires_confirmation=not self.auto_confirm_safe,
            metadata=metadata,
        )

    def _check_parameter_rule(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        working_dir: str | None,
        command_analysis: dict[str, Any] | None = None,
    ) -> GuardResult | None:
        rule = self.permission_rule_matcher.match(
            tool_name,
            arguments,
            working_dir=working_dir,
            command_analysis=command_analysis,
        )
        if not rule:
            return None
        metadata: dict[str, Any] = {"rule_id": rule.id, "rule_reason": rule.reason}
        if command_analysis:
            metadata["command_analysis"] = command_analysis
        reason = rule.reason or f"Permission rule matched: {rule.id}"
        if rule.decision == "deny":
            return GuardResult(False, RiskLevel.BLOCK, reason, metadata=metadata)
        if rule.decision == "ask":
            return GuardResult(True, RiskLevel.WARN, reason, True, metadata=metadata)
        return GuardResult(True, RiskLevel.SAFE, reason, False, metadata=metadata)

    def _check_mcp_tool_context(
        self,
        tool_name: str,
        tool_context: dict[str, Any] | None,
    ) -> GuardResult | None:
        if not tool_context or tool_context.get("kind") != "mcp":
            return None
        mcp_tool = str(tool_context.get("tool") or tool_name)
        mcp_server = str(tool_context.get("server") or "")
        metadata = {
            "mcp_server": mcp_server,
            "mcp_tool": mcp_tool,
            "mcp_trusted": bool(tool_context.get("trusted", False)),
        }
        denied_tools = {str(item) for item in tool_context.get("denied_tools") or []}
        allowed_tools = {str(item) for item in tool_context.get("allowed_tools") or []}
        if mcp_tool in denied_tools or tool_name in denied_tools:
            return GuardResult(
                False,
                RiskLevel.BLOCK,
                f"MCP tool is denied by server policy: {mcp_server}.{mcp_tool}",
                metadata=metadata,
            )
        if allowed_tools and mcp_tool not in allowed_tools and tool_name not in allowed_tools:
            return GuardResult(
                False,
                RiskLevel.BLOCK,
                f"MCP tool is not allowed by server policy: {mcp_server}.{mcp_tool}",
                metadata=metadata,
            )
        if not bool(tool_context.get("trusted", False)) or bool(
            tool_context.get("require_confirmation", False)
        ):
            return GuardResult(
                True,
                RiskLevel.DANGER,
                f"MCP tool requires confirmation: {mcp_server}.{mcp_tool}",
                True,
                metadata=metadata,
            )
        return GuardResult(
            True, RiskLevel.SAFE, f"MCP tool trusted: {mcp_server}.{mcp_tool}", metadata=metadata
        )

    def _check_secret_content(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> GuardResult | None:
        if tool_name not in {"write_file", "create_file", "edit_file", "multi_edit_file"}:
            return None
        content = _extract_write_content(arguments)
        findings = self.secret_scanner.scan(content)
        if not findings:
            return None
        metadata = {
            "secret_findings_count": len(findings),
            "secret_findings": [finding.to_dict() for finding in findings],
        }
        if bool(self.secrets_policy.get("block_on_write", False)):
            return GuardResult(
                False,
                RiskLevel.BLOCK,
                "Write content appears to contain secrets",
                metadata=metadata,
            )
        if bool(self.secrets_policy.get("warn_on_write", True)):
            return GuardResult(
                True,
                RiskLevel.DANGER,
                "Write content appears to contain secrets",
                True,
                metadata=metadata,
            )
        return None

    def check_file_path(
        self,
        file_path: str,
        operation: str = "read",
        working_dir: str | None = None,
    ) -> GuardResult:
        """检查`check_file_path`并返回明确的校验或策略结果。

        参数：
            file_path: 目标文件的路径；访问范围仍受项目沙箱约束。
            operation: 可选的`operation`。
            working_dir: 所有相对路径和本地工具操作所基于的工作目录。

        返回：
            `GuardResult` 类型的处理结果。
        """
        try:
            path = Path(file_path).expanduser().resolve()
        except Exception as e:
            return GuardResult(
                allowed=False,
                risk_level=RiskLevel.BLOCK,
                reason=f"Invalid path: {e}",
            )

        for protected in PROTECTED_PATHS:
            protected_path = Path(protected).expanduser().resolve()
            try:
                path.relative_to(protected_path)
                return GuardResult(
                    allowed=False,
                    risk_level=RiskLevel.BLOCK,
                    reason=f"Access to protected system path: {protected}",
                    requires_confirmation=True,
                )
            except ValueError:
                pass

        for pattern in SENSITIVE_FILE_PATTERNS:
            if re.search(pattern, file_path, re.IGNORECASE):
                return GuardResult(
                    allowed=True,
                    risk_level=RiskLevel.WARN,
                    reason="Accessing potentially sensitive file",
                    requires_confirmation=True,
                    suggestions=["Verify this is intentional"],
                )

        if self.sandbox_mode and working_dir:
            work_path = Path(working_dir).resolve()
            try:
                path.relative_to(work_path)
            except ValueError:
                if not any(path.is_relative_to(Path(p).resolve()) for p in self.allowed_paths):
                    return GuardResult(
                        allowed=False,
                        risk_level=RiskLevel.DANGER,
                        reason=f"Path outside working directory: {file_path}",
                        requires_confirmation=True,
                        suggestions=[
                            f"Move to working directory: {working_dir}",
                            "Add path to allowed_paths in configuration",
                        ],
                    )

        if operation == "delete":
            return GuardResult(
                allowed=True,
                risk_level=RiskLevel.WARN,
                reason="Deleting a file is a destructive operation",
                requires_confirmation=True,
                suggestions=["Verify the target path before deleting it"],
            )
        return GuardResult(
            allowed=True,
            risk_level=RiskLevel.SAFE,
            reason="Path is safe to access",
            requires_confirmation=operation == "write",
        )

    def check_http_request(self, url: str, method: str = "GET") -> GuardResult:
        """检查`check_http_request`并返回明确的校验或策略结果。

        参数：
            url: 需要访问或校验的网络地址。
            method: 可选的`method`。

        返回：
            `GuardResult` 类型的处理结果。
        """
        if not self.allow_network:
            return GuardResult(
                allowed=False,
                risk_level=RiskLevel.BLOCK,
                reason="Network access is disabled by policy",
                requires_confirmation=False,
            )
        if not url.startswith(("http://", "https://")):
            return GuardResult(
                allowed=False,
                risk_level=RiskLevel.BLOCK,
                reason="HTTP request URL must use http or https",
            )
        analysis = self.network_policy.evaluate(url, method)
        metadata = {"network_analysis": analysis.to_dict()}
        if analysis.risk_level == "block":
            return GuardResult(
                allowed=False,
                risk_level=RiskLevel.BLOCK,
                reason=analysis.reason,
                metadata=metadata,
            )
        if analysis.risk_level == "danger":
            return GuardResult(
                allowed=True,
                risk_level=RiskLevel.DANGER,
                reason=analysis.reason,
                requires_confirmation=True,
                metadata=metadata,
            )
        if analysis.risk_level == "warn":
            return GuardResult(
                allowed=True,
                risk_level=RiskLevel.WARN,
                reason=analysis.reason,
                requires_confirmation=True,
                metadata=metadata,
            )
        suspicious_patterns = [
            r"login",
            r"signin",
            r"auth",
            r"password",
            r"secret",
            r"api[_-]?key",
            r"token",
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return GuardResult(
                    allowed=True,
                    risk_level=RiskLevel.WARN,
                    reason="URL contains potentially sensitive keywords",
                    requires_confirmation=True,
                    metadata=metadata,
                )

        return GuardResult(
            allowed=True,
            risk_level=RiskLevel.SAFE,
            reason="Request appears safe",
            requires_confirmation=False,
            metadata=metadata,
        )

    def check_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        working_dir: str | None = None,
        tool_context: dict[str, Any] | None = None,
    ) -> GuardResult:
        """综合工具声明、权限模式、参数规则、文件路径、Shell 命令、网络目标和敏感内容，生成本次工具调用的最终安全决策。

        参数：
            tool_name: 目标工具在注册表中的名称。
            arguments: 工具调用的结构化参数。
            working_dir: 所有相对路径和本地工具操作所基于的工作目录。
            tool_context: 可选的工具上下文。

        返回：
            `GuardResult` 类型的处理结果。
        """
        command_analysis = None
        if tool_name == "execute_command":
            command_analysis = self.command_policy.analyze(arguments.get("command", "")).to_dict()

        mcp_context_result = self._check_mcp_tool_context(tool_name, tool_context)
        if mcp_context_result is not None and not mcp_context_result.allowed:
            return mcp_context_result

        rule_result = self._check_parameter_rule(
            tool_name,
            arguments,
            working_dir,
            command_analysis=command_analysis,
        )
        if rule_result is not None and not rule_result.allowed:
            return rule_result

        permission_result = self._check_permission_mode(tool_name)
        if permission_result is not None and not permission_result.allowed:
            permission_result.metadata["permission_mode"] = self.effective_permission_mode.value
            return permission_result

        if tool_name == "execute_command":
            command_working_dir = arguments.get("working_dir")
            if command_working_dir:
                path_result = self.check_file_path(
                    str(command_working_dir),
                    "read",
                    working_dir,
                )
                if not path_result.allowed:
                    return path_result
            command = arguments.get("command", "")
            result = self.check_command(command)

        elif tool_name == "read_file":
            file_path = arguments.get("file_path", "")
            result = self.check_file_path(file_path, "read", working_dir)

        elif tool_name in ("write_file", "create_file", "edit_file", "multi_edit_file"):
            file_path = arguments.get("file_path", "")
            result = self.check_file_path(file_path, "write", working_dir)

        elif tool_name == "delete_file":
            file_path = arguments.get("file_path", "")
            result = self.check_file_path(file_path, "delete", working_dir)
        elif tool_name in ("list_directory", "glob_files", "grep_code"):
            directory = arguments.get("directory", ".")
            result = self.check_file_path(directory, "read", working_dir)

        elif tool_name in ("enter_worktree", "exit_worktree"):
            result = GuardResult(
                allowed=True,
                risk_level=RiskLevel.WARN,
                reason="Git worktree operation changes repository checkout state",
                requires_confirmation=True,
                suggestions=[
                    "Confirm the target path and branch name before proceeding",
                    "Review uncommitted changes before removing a worktree",
                ],
            )

        elif tool_name == "http_request":
            url = arguments.get("url", "")
            method = arguments.get("method", "GET")
            result = self.check_http_request(url, method)
        elif tool_name == "web_fetch":
            url = arguments.get("url", "")
            result = self.check_http_request(url, "GET")

        else:
            result = GuardResult(
                allowed=True,
                risk_level=RiskLevel.SAFE,
                reason="Tool not in high-risk category",
                requires_confirmation=False,
            )

        secret_result = self._check_secret_content(tool_name, arguments)
        if secret_result is not None:
            if not secret_result.allowed:
                return secret_result
            result.risk_level = secret_result.risk_level
            result.reason = secret_result.reason
            result.requires_confirmation = True
            result.metadata.update(secret_result.metadata)

        if mcp_context_result and mcp_context_result.allowed:
            result.metadata.update(mcp_context_result.metadata)
            if mcp_context_result.requires_confirmation and result.allowed:
                result.risk_level = mcp_context_result.risk_level
                result.reason = mcp_context_result.reason
                result.requires_confirmation = True
        return self._apply_approval_policy(tool_name, result, rule_result)


def _extract_write_content(arguments: dict[str, Any]) -> str:
    chunks: list[str] = []
    for key in ("content", "new_content", "replacement"):
        value = arguments.get(key)
        if isinstance(value, str):
            chunks.append(value)
    edits = arguments.get("edits")
    if isinstance(edits, list):
        for edit in edits:
            if isinstance(edit, dict):
                for key in ("new_text", "replacement", "content"):
                    value = edit.get(key)
                    if isinstance(value, str):
                        chunks.append(value)
    return "\n".join(chunks)
