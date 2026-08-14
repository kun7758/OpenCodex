"""Agent 核心运行时中的模型策略模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

from opennova.providers.base import Usage
from opennova.providers.models import ModelProfile


@dataclass(frozen=True)
class BudgetSnapshot:
    """保存预算快照所需的结构化数据，主要包含
    `turns`、`prompt_tokens`、`completion_tokens`、`total_tokens`、`estimated_cost_usd`、`exhausted_reason`
    字段，便于在组件之间传递或持久化。
    """

    turns: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    exhausted_reason: str | None


class RunBudget:
    """封装运行预算相关的状态和操作，使调用方通过稳定接口使用该能力。"""

    def __init__(
        self,
        profile: ModelProfile,
        *,
        max_turns: int,
        token_budget: int = 0,
        cost_budget_usd: float = 0.0,
        max_output_tokens: int = 0,
        input_cost_per_million: float = 0.0,
        output_cost_per_million: float = 0.0,
    ) -> None:
        self.profile = profile
        self.max_turns = max(1, max_turns)
        self.token_budget = max(0, token_budget)
        self.cost_budget_usd = max(0.0, cost_budget_usd)
        configured_output = (
            max_output_tokens if max_output_tokens > 0 else profile.max_output_tokens
        )
        self.max_output_tokens = min(profile.max_output_tokens, configured_output)
        self.input_cost_per_million = max(0.0, input_cost_per_million)
        self.output_cost_per_million = max(0.0, output_cost_per_million)
        self.reset()

    def reset(self) -> None:
        self.turns = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.estimated_cost_usd = 0.0

    def record(self, usage: Usage | None) -> None:
        """处理记录，并按照当前组件的约定返回结果。

        参数：
            usage: 可选的用量。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self.turns += 1
        if usage is None:
            return
        self.prompt_tokens += max(0, usage.prompt_tokens)
        self.completion_tokens += max(0, usage.completion_tokens)
        self.total_tokens += max(0, usage.total_tokens)
        self.estimated_cost_usd += (
            usage.prompt_tokens * self.input_cost_per_million
            + usage.completion_tokens * self.output_cost_per_million
        ) / 1_000_000

    def output_limit(self) -> int:
        """读取并返回 `output_limit` 所表示的数据或流程，并遵守运行预算定义的边界与状态约束。

        返回：
            `int` 类型的处理结果。
        """
        if not self.token_budget:
            return self.max_output_tokens
        remaining = max(0, self.token_budget - self.total_tokens)
        return min(self.max_output_tokens, remaining)

    def exhausted_reason(self) -> str | None:
        if self.turns >= self.max_turns:
            return f"reached maximum model turns ({self.max_turns})"
        if self.token_budget and self.total_tokens >= self.token_budget:
            return f"reached token budget ({self.token_budget})"
        if self.cost_budget_usd and self.estimated_cost_usd >= self.cost_budget_usd:
            return f"reached cost budget (${self.cost_budget_usd:.4f})"
        return None

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            turns=self.turns,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            estimated_cost_usd=round(self.estimated_cost_usd, 8),
            exhausted_reason=self.exhausted_reason(),
        )


class ProviderCircuitBreaker:
    """封装模型服务熔断熔断器相关的状态和操作，使调用方通过稳定接口使用该能力。"""

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 30.0) -> None:
        self.failure_threshold = max(1, failure_threshold)
        self.cooldown_seconds = max(0.0, cooldown_seconds)
        self._failures: dict[str, int] = {}
        self._opened_at: dict[str, float] = {}

    @staticmethod
    def key(provider: object) -> str:
        return f"{getattr(provider, 'provider_name', 'unknown')}:{getattr(provider, 'model', '')}"

    def is_open(self, provider: object) -> bool:
        key = self.key(provider)
        opened_at = self._opened_at.get(key)
        if opened_at is None:
            return False
        if monotonic() - opened_at >= self.cooldown_seconds:
            self._failures.pop(key, None)
            self._opened_at.pop(key, None)
            return False
        return True

    def record_success(self, provider: object) -> None:
        key = self.key(provider)
        self._failures.pop(key, None)
        self._opened_at.pop(key, None)

    def record_failure(self, provider: object) -> None:
        key = self.key(provider)
        failures = self._failures.get(key, 0) + 1
        self._failures[key] = failures
        if failures >= self.failure_threshold:
            self._opened_at[key] = monotonic()
