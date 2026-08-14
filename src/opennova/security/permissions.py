"""安全控制子系统中的权限模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class PermissionDecision(StrEnum):
    """枚举权限决策允许出现的稳定取值，序列化和状态判断均使用这些值。"""

    ALLOW_ONCE = "allow_once"
    ALWAYS_ALLOW = "always_allow"
    ALWAYS_DENY = "always_deny"
    ALWAYS_ASK = "always_ask"


@dataclass
class PermissionRule:
    """保存权限规则所需的结构化数据，主要包含 `tool_name`、`decision` 字段，便于在组件之间传递或持久化。"""

    tool_name: str
    decision: PermissionDecision


class PermissionStore:
    """负责权限存储的保存、读取和一致性维护，并隐藏具体持久化格式。"""

    def __init__(self, storage_path: str | Path | None = None):
        self.storage_path = Path(storage_path) if storage_path else None
        self.rules: dict[str, PermissionDecision] = {}
        self._load()

    def record(self, tool_name: str, decision: PermissionDecision | str) -> None:
        """处理记录，并按照当前组件的约定返回结果。

        参数：
            tool_name: 目标工具在注册表中的名称。
            decision: 用户或策略给出的决策结果。
        """
        decision_value = PermissionDecision(decision)
        if decision_value == PermissionDecision.ALLOW_ONCE:
            return
        self.rules[tool_name] = decision_value
        self.save()

    def decision_for(self, tool_name: str) -> PermissionDecision | None:
        """读取并返回 `decision_for` 所表示的数据或流程，并遵守权限存储定义的边界与状态约束。

        参数：
            tool_name: 目标工具在注册表中的名称。

        返回：
            `PermissionDecision | None` 类型的处理结果。
        """
        return self.rules.get(tool_name)

    def allowed_tools(self) -> list[str]:
        return [
            tool_name
            for tool_name, decision in self.rules.items()
            if decision == PermissionDecision.ALWAYS_ALLOW
        ]

    def denied_tools(self) -> list[str]:
        return [
            tool_name
            for tool_name, decision in self.rules.items()
            if decision == PermissionDecision.ALWAYS_DENY
        ]

    def ask_tools(self) -> list[str]:
        return [
            tool_name
            for tool_name, decision in self.rules.items()
            if decision == PermissionDecision.ALWAYS_ASK
        ]

    def save(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"rules": {tool: decision.value for tool, decision in self.rules.items()}}
        self.storage_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _load(self) -> None:
        if not self.storage_path or not self.storage_path.exists():
            return
        payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        for tool_name, decision in payload.get("rules", {}).items():
            self.rules[str(tool_name)] = PermissionDecision(decision)
