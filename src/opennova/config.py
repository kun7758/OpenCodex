"""OpenNova中的配置模块，集中定义相关数据结构、边界适配和实现逻辑。"""

import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from opennova.security.secrets import redact_sensitive_data

DEFAULT_CONFIG: dict[str, Any] = {
    "default_provider": "deepseek",
    "providers": {
        "openai": {
            "api_key": "${OPENAI_API_KEY}",
            "base_url": "https://api.openai.com/v1",
            "default_model": "gpt-4o",
        },
        "anthropic": {
            "api_key": "${ANTHROPIC_API_KEY}",
            "default_model": "claude-sonnet-4",
        },
        "deepseek": {
            "api_key": "${DEEPSEEK_API_KEY}",
            "base_url": "https://api.deepseek.com/v1",
            "default_model": "deepseek-v4-pro",
        },
    },
    "agent": {
        "max_iterations": 20,
        "token_budget": 0,
        "cost_budget_usd": 0.0,
        "max_output_tokens": 0,
        "input_cost_per_million": 0.0,
        "output_cost_per_million": 0.0,
        "provider_retry_attempts": 1,
        "provider_failure_threshold": 3,
        "provider_cooldown_seconds": 30.0,
        "fallback_providers": [],
        "auto_confirm": False,
        "show_thinking": True,
        "execution": {
            "parallel_tool_limit": 4,
            "per_turn_tool_result_chars": 160000,
        },
        "deferred_tools": {"enabled": True},
        "memory": {"max_chars": 5000},
        "compression": {
            "enabled": True,
            "threshold": 0.55,
            "keep_last_pairs": 6,
            "max_tool_result_tokens": 8000,
        },
    },
    "session": {
        "persistence": {
            "debounce_ms": 250,
            "snapshot_event_threshold": 100,
            "snapshot_size_threshold": 1048576,
            "fsync_critical": True,
        }
    },
    "security": {
        "sandbox_mode": True,
        "command_timeout": 30,
        "allow_network": True,
        "auto_confirm_safe": True,
        "allowed_paths": [],
        "blocked_commands": [],
        "strict_shell_parsing": False,
        "permission_mode": "auto",
        "always_allow_tools": [],
        "always_deny_tools": [],
        "always_ask_tools": [],
        "permission_rules": [],
        "network": {
            "allowed_domains": [],
            "blocked_domains": [],
            "allow_localhost": False,
            "mutating_methods_require_confirmation": True,
        },
        "secrets": {
            "enabled": True,
            "redact_tool_outputs": True,
            "warn_on_write": True,
            "block_on_write": False,
            "max_scan_chars": 200000,
        },
        "process_sandbox": {
            "enabled": True,
            "backend": "auto",
            "enforce": False,
            "tmp_dir": None,
            "extra_read_roots": [],
            "extra_writable_roots": [],
        },
        "audit": {
            "enabled": True,
            "path": ".opennova/audit/security.jsonl",
            "max_arg_chars": 500,
        },
        "read_only": False,
        "max_file_size": 104857600,
    },
    "mcp": {
        "enabled": True,
        "servers": [],
    },
    "skills": {
        "enabled": True,
        "dirs": [],
        "exclude": [],
    },
}


@dataclass
class Config:
    """保存配置所需的结构化数据，主要包含 `data`、`config_path` 字段，便于在组件之间传递或持久化。"""

    data: dict[str, Any] = field(default_factory=lambda: deepcopy(DEFAULT_CONFIG))
    config_path: str | None = None

    def __post_init__(self) -> None:
        self.data = deepcopy(self.data)

    def get(self, key: str, default: Any = None) -> Any:
        """读取并返回 `get` 所表示的数据或流程，并遵守配置定义的边界与状态约束。

        参数：
            key: 本次操作使用的`key`。
            default: 可选的默认。

        返回：
            `Any` 类型的处理结果。
        """
        keys = key.split(".")
        value: Any = self.data

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

            if value is None:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """处理设置，并按照当前组件的约定返回结果。

        参数：
            key: 本次操作使用的`key`。
            value: 需要保存、转换或校验的值。
        """
        keys = key.split(".")
        data = self.data

        for k in keys[:-1]:
            child = data.get(k)
            if not isinstance(child, dict):
                child = {}
                data[k] = child
            data = child

        data[keys[-1]] = value

    def setdefault(self, key: str, default: Any = None) -> Any:
        """根据当前输入和配置的状态计算 `setdefault`，并返回调用方需要的结果。

        参数：
            key: 本次操作使用的`key`。
            default: 可选的默认。

        返回：
            `Any` 类型的处理结果。
        """
        return self.data.setdefault(key, deepcopy(default))

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def __contains__(self, key: object) -> bool:
        return key in self.data

    def to_dict(self) -> dict[str, Any]:
        """把配置转换为可序列化字典，供事件、会话或 API 边界使用。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        return deepcopy(self.data)

    def redacted_data(self) -> dict[str, Any]:
        """读取并返回 `redacted_data` 所表示的数据或流程，并遵守配置定义的边界与状态约束。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        redacted = redact_sensitive_data(self.data)
        return redacted if isinstance(redacted, dict) else {}

    def save(self, path: str | None = None) -> None:
        """处理保存，并按照当前组件的约定返回结果。

        参数：
            path: 需要读取、检查或写入的路径。

        说明：
            该操作会访问本地文件系统，路径校验和原子写入约束由所在组件负责。
        """
        save_path = path or self.config_path
        if not save_path:
            raise ValueError("No config path specified")

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(self.data, f, default_flow_style=False, sort_keys=False)

    def get_mcp_servers(self) -> list[dict[str, Any]]:
        """读取并规范化配置中的 MCP 服务列表，缺失配置时返回空列表。

        返回：
            按调用约定排序的结果列表。
        """
        mcp_config = self.get("mcp", {})
        if not isinstance(mcp_config, dict):
            return []
        if not mcp_config.get("enabled", True):
            return []
        servers = mcp_config.get("servers", [])
        if not isinstance(servers, list):
            return []
        return [deepcopy(server) for server in servers if isinstance(server, dict)]

    def get_skill_dirs(self) -> list[str]:
        """读取配置中额外扫描的 Skill 目录列表。

        返回：
            按调用约定排序的结果列表。
        """
        skills_config = self.get("skills", {})
        if not isinstance(skills_config, dict):
            return []
        if not skills_config.get("enabled", True):
            return []
        directories = skills_config.get("dirs", [])
        if not isinstance(directories, list):
            return []
        return [str(directory) for directory in directories]

    def get_excluded_skills(self) -> list[str]:
        """读取配置中明确禁止加载的 Skill 名称。

        返回：
            按调用约定排序的结果列表。
        """
        skills_config = self.get("skills", {})
        if not isinstance(skills_config, dict):
            return []
        excluded = skills_config.get("exclude", [])
        if not isinstance(excluded, list):
            return []
        return [str(name) for name in excluded]


def _expand_env_vars(value: Any) -> Any:
    """递归遍历配置值并展开 `${NAME}` 环境变量占位符，字典和列表保持原有结构。

    参数：
        value: 需要保存、转换或校验的值。

    返回：
        `Any` 类型的处理结果。

    说明：
        递归遍历整个配置字典，对每个字符串值检查是否是${...}格式，
        如果是就从os.environ中取对应的值替换，找不到则替换为空字符串。
    """
    if isinstance(value, str):
        if value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.environ.get(env_var, "")
        return value
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并基础配置和覆盖配置；双方都是字典时继续下钻，否则使用覆盖值。

    参数：
        base: 本次操作使用的基础抽象。
        override: 本次操作使用的`override`。

    返回：
        供后续逻辑或序列化使用的结构化字典。
    """
    result = deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result


def _config_mapping(value: Any, source: str) -> dict[str, Any]:
    """确认 YAML 根节点可以作为配置字典使用，并在格式错误时报告来源。

    参数：
        value: 需要保存、转换或校验的值。
        source: 数据、插件或 Hook 的来源。

    返回：
        供后续逻辑或序列化使用的结构化字典。
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Configuration in {source} must be a YAML mapping")
    return value


def find_config_file() -> Path | None:
    """查找配置文件，并按照当前组件的约定返回结果。

    返回：
        `Path | None` 类型的处理结果。
    """
    project_config = Path(".opennova/config.yaml")
    if project_config.exists():
        return project_config

    global_config = Path.home() / ".opennova" / "config.yaml"
    if global_config.exists():
        return global_config

    return None


def load_config(
    config_path: str | None = None,
    load_env: bool = True,
) -> Config:
    """加载并合并多层配置，返回最终的Config对象。

    配置加载顺序（后者覆盖前者）：
    1. 内置默认配置 DEFAULT_CONFIG
    2. 全局配置 ~/.opennova/config.yaml
    3. 项目配置 .opennova/config.yaml（或 config_path 指定的文件）
    4. 环境变量展开（${VAR_NAME} 格式的占位符会被替换为实际环境变量值）

    参数：
        config_path: 自定义配置文件路径。为None时自动查找项目目录下的.opennova/config.yaml；
                     非None时使用该路径替代项目配置文件。
        load_env: 是否加载环境变量。为True时会先加载.env文件到环境变量，
                  以便后续展开配置中的${VAR_NAME}占位符。

    返回：
        合并后的Config对象，包含最终配置数据和实际加载的配置文件路径。

    说明：
        该操作会访问本地文件系统读取配置文件，但不会创建Provider或会话。
    """
    # 第一步：加载环境变量，先加载系统环境变量，再加载.env文件覆盖
    if load_env:
        from dotenv import load_dotenv

        load_dotenv()  # 加载系统环境变量
        env_file = Path(".env")  # 相对路径，基于当前工作目录(CWD)，非脚本所在目录，取决于运行命令时的工作目录
        if env_file.exists():
            load_dotenv(env_file)  # 加载项目目录下的.env文件

    # 第二步：以内置默认配置为基础
    config_data = deepcopy(DEFAULT_CONFIG)
    loaded_path = None

    # 第三步：合并全局配置 ~/.opennova/config.yaml
    global_config = Path.home() / ".opennova" / "config.yaml"
    if global_config.exists():
        with open(global_config, encoding="utf-8") as f:
            global_data = _config_mapping(yaml.safe_load(f), str(global_config))
            config_data = _deep_merge(config_data, global_data)
            loaded_path = str(global_config)

    # 第四步：合并项目配置或自定义配置
    if config_path:
        # 使用用户指定的配置文件路径
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, encoding="utf-8") as f:
                file_data = _config_mapping(yaml.safe_load(f), str(config_file))
                config_data = _deep_merge(config_data, file_data)
                loaded_path = str(config_file)
    else:
        # 使用当前项目目录下的默认项目配置
        project_config = Path(".opennova/config.yaml")
        if project_config.exists():
            with open(project_config, encoding="utf-8") as f:
                project_data = _config_mapping(yaml.safe_load(f), str(project_config))
                config_data = _deep_merge(config_data, project_data)
                loaded_path = str(project_config)

    # 第五步：展开所有${ENV_VAR}格式的环境变量占位符
    config_data = _config_mapping(_expand_env_vars(config_data), "expanded configuration")

    return Config(data=config_data, config_path=loaded_path)


def get_default_config_path() -> Path:
    """读取默认配置路径，不改变当前对象的业务状态。

    返回：
        `Path` 类型的处理结果。
    """
    return Path.home() / ".opennova" / "config.yaml"


def create_default_config() -> Path:
    """创建默认配置并完成必要的初始化。

    返回：
        `Path` 类型的处理结果。

    说明：
        该操作会访问本地文件系统，路径校验和原子写入约束由所在组件负责。
    """
    config_path = get_default_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    if not config_path.exists():
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False)

    return config_path


def validate_config(config: Config) -> list[str]:
    """校验配置，发现问题时返回或抛出明确错误。

    参数：
        config: 控制当前组件行为的配置。

    返回：
        按调用约定排序的结果列表。
    """
    errors = []

    permission_mode = config.get("security.permission_mode", "auto")
    valid_permission_modes = {
        "request",
        "auto",
        "full",
        "default",
        "ask",
        "allowEdits",
        "readOnly",
        "bypass",
    }
    if permission_mode not in valid_permission_modes:
        errors.append(
            "security.permission_mode must be one of: request, auto, full, "
            "default, ask, allowEdits, readOnly, bypass"
        )

    default_provider = config.get("default_provider")
    if not default_provider:
        errors.append("No default_provider specified")
        return errors

    providers = config.get("providers", {})
    if default_provider not in providers:
        errors.append(f"Default provider '{default_provider}' not in providers")
        return errors

    provider_config = providers.get(default_provider, {})
    api_key = provider_config.get("api_key", "")

    if not api_key:
        env_var_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
        }
        env_var = env_var_map.get(default_provider)
        if env_var and not os.environ.get(env_var):
            errors.append(f"API key not configured for provider '{default_provider}'")

    return errors
