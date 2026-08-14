"""安全控制子系统中的`secrets`模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

SENSITIVE_CONFIG_KEY = re.compile(
    r"(^|[_-])(api[_-]?key|token|password|passwd|secret|authorization|private[_-]?key)$",
    re.IGNORECASE,
)
REDACTED_VALUE = "[REDACTED_SECRET]"


@dataclass
class SecretFinding:
    """数据对象 `SecretFinding` 主要保存 `kind`、`start`、`end` 字段，用于在组件之间传递或持久化这组状态。"""

    kind: str
    start: int
    end: int

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "start": self.start, "end": self.end}


class SecretScanner:
    """封装`SecretScanner`相关的状态和操作，使调用方通过稳定接口使用该能力。"""

    def __init__(self, enabled: bool = True, max_scan_chars: int = 200_000):
        self.enabled = enabled
        self.max_scan_chars = max_scan_chars
        self._patterns: list[tuple[str, re.Pattern[str]]] = [
            (
                "private-key",
                re.compile(
                    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"
                ),
            ),
            ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b")),
            ("openai-key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")),
            ("anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{16,}\b")),
            ("deepseek-key", re.compile(r"\bsk-deepseek-[A-Za-z0-9_-]{12,}\b")),
            (
                "secret-assignment",
                re.compile(
                    r"(?im)\b(api[_-]?key|token|password|secret)\b\s*[:=]\s*['\"]?([^\s'\"]{8,})['\"]?"
                ),
            ),
        ]

    @classmethod
    def from_config(cls, config: dict[str, object] | None) -> SecretScanner:
        data = config or {}
        raw_limit = data.get("max_scan_chars", 200_000)
        max_scan_chars = int(raw_limit) if isinstance(raw_limit, (str, int, float)) else 200_000
        return cls(
            enabled=bool(data.get("enabled", True)),
            max_scan_chars=max_scan_chars,
        )

    def scan(self, text: str) -> list[SecretFinding]:
        if not self.enabled or not text:
            return []
        haystack = text[: self.max_scan_chars]
        findings: list[SecretFinding] = []
        for kind, pattern in self._patterns:
            for match in pattern.finditer(haystack):
                findings.append(SecretFinding(kind=kind, start=match.start(), end=match.end()))
        return _dedupe_findings(findings)

    def redact(self, text: str) -> str:
        findings = self.scan(text)
        if not findings:
            return text
        redacted = []
        cursor = 0
        for finding in findings:
            redacted.append(text[cursor : finding.start])
            redacted.append("[REDACTED_SECRET]")
            cursor = finding.end
        redacted.append(text[cursor:])
        return "".join(redacted)


def redact_sensitive_data(value: Any, *, scanner: SecretScanner | None = None) -> Any:
    """读取并返回 `redact_sensitive_data` 所表示的数据或流程，并遵守当前模块定义的边界与状态约束。

    参数：
        value: 需要保存、转换或校验的值。
        scanner: 可选的`scanner`。

    返回：
        `Any` 类型的处理结果。
    """
    active_scanner = scanner or SecretScanner()

    def redact(item: Any, key: str | None = None) -> Any:
        if key and SENSITIVE_CONFIG_KEY.search(key):
            if item in (None, "", [], {}):
                return deepcopy(item)
            return REDACTED_VALUE
        if isinstance(item, dict):
            return {
                str(child_key): redact(child_value, str(child_key))
                for child_key, child_value in item.items()
            }
        if isinstance(item, list):
            return [redact(child) for child in item]
        if isinstance(item, tuple):
            return tuple(redact(child) for child in item)
        if isinstance(item, str):
            return active_scanner.redact(item)
        return deepcopy(item)

    return redact(value)


def _dedupe_findings(findings: list[SecretFinding]) -> list[SecretFinding]:
    ordered = sorted(findings, key=lambda item: (item.start, -(item.end - item.start)))
    selected: list[SecretFinding] = []
    last_end = -1
    for finding in ordered:
        if finding.start < last_end:
            continue
        selected.append(finding)
        last_end = finding.end
    return selected
