"""记忆与上下文子系统中的上下文模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from opennova.providers.base import Message
from opennova.providers.models import context_window_for_model

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


DEFAULT_CONTEXT_WINDOW = 128000
RESERVED_OUTPUT_TOKENS = 4096


@dataclass
class ContextStats:
    """保存上下文统计信息所需的结构化数据，主要包含
    `total_messages`、`total_tokens`、`context_window`、`available_tokens`、`utilization_percent`
    字段，便于在组件之间传递或持久化。
    """

    total_messages: int
    total_tokens: int
    context_window: int
    available_tokens: int
    utilization_percent: float


@dataclass(frozen=True)
class ContextPresentationSnapshot:
    """数据对象 `ContextPresentationSnapshot` 主要保存
    `total_messages`、`total_tokens`、`context_window`、`available_tokens`、`utilization_percent`、`compression_count`、`has_compressed_summary`、`compression_threshold_percent`
    字段，用于在组件之间传递或持久化这组状态。
    """

    total_messages: int
    total_tokens: int
    context_window: int
    available_tokens: int
    utilization_percent: float
    compression_count: int
    has_compressed_summary: bool
    compression_threshold_percent: float


class MessageAddStatus(StrEnum):
    """枚举`MessageAddStatus`允许出现的稳定取值，序列化和状态判断均使用这些值。"""

    ADDED = "added"
    ADDED_AFTER_COMPRESSION = "added_after_compression"
    REJECTED = "rejected"


@dataclass(frozen=True)
class MessageAddResult:
    """数据对象 `MessageAddResult` 主要保存 `status`、`message_count`、`reason` 字段，用于在组件之间传递或持久化这组状态。"""

    status: MessageAddStatus
    message_count: int = 0
    reason: str | None = None

    @property
    def added(self) -> bool:
        return self.status is not MessageAddStatus.REJECTED

    def __bool__(self) -> bool:
        return self.added


class ContextCapacityError(RuntimeError):
    """表示`ContextCapacityError`失败；调用方可以捕获该异常并转换为稳定的用户提示或 SDK 事件。"""


class ContextManager:
    """管理发送给模型的上下文窗口，负责消息顺序、Token 统计、工具结果截断、上下文压缩以及最终模型消息的组装。"""

    def __init__(
        self,
        model: str = "gpt-4o",
        context_window: int | None = None,
        max_messages: int = 100,
        encoding_name: str = "cl100k_base",
        max_tool_result_tokens: int = 8000,
    ):
        """初始化上下文管理器。

        参数：
            model: 模型名称，用于自动推断上下文窗口大小。
            context_window: 上下文窗口的 Token 上限，为 None 时根据 model 自动查询。
            max_messages: 消息列表的最大条数，超出时会从头部淘汰旧消息。
            encoding_name: tiktoken 编码名称，用于 Token 计数。
            max_tool_result_tokens: 单条工具结果的最大 Token 数，超出时会被截断。
        """
        self.model = model  # 当前使用的模型名称
        self.context_window = context_window or self._get_context_window(model)  # 上下文窗口 Token 上限
        self.max_messages = max_messages  # 消息列表最大条数
        self.encoding_name = encoding_name  # tiktoken 编码名称
        self.max_tool_result_tokens = max_tool_result_tokens  # 单条工具结果的最大 Token 数

        self.messages: list[Message] = []  # 当前上下文中的消息列表
        self.system_prompt: str | None = None  # 系统提示词

        # 上下文压缩相关状态
        self._compressed_summary: str | None = None  # 压缩后的历史摘要文本
        self._compressor: Any = None  # 压缩器实例，由外部注入
        self._compressing: bool = False  # 是否正在压缩中，防止重入
        self._compression_count: int = 0  # 累计压缩次数
        self._compression_failures: int = 0  # 连续压缩失败次数
        self.compression_failure_limit: int = 3  # 压缩失败次数上限，达到后停止尝试
        self.compression_threshold: float = 0.55  # 触发压缩的 Token 利用率阈值
        self.keep_last_pairs: int = 6  # 压缩时保留的最近消息对数

        self._encoding = None  # tiktoken 编码器实例
        if TIKTOKEN_AVAILABLE:
            with suppress(Exception):
                self._encoding = tiktoken.get_encoding(encoding_name)

    def _get_context_window(self, model: str) -> int:
        """读取并返回 `_get_context_window` 所表示的数据或流程，并遵守上下文管理定义的边界与状态约束。

        参数：
            model: 本次操作使用的模型。

        返回：
            `int` 类型的处理结果。
        """
        return context_window_for_model(model, DEFAULT_CONTEXT_WINDOW)

    def _truncate_tool_result(self, content: str) -> str:
        """根据当前输入和上下文管理的状态计算 `_truncate_tool_result`，并返回调用方需要的结果。

        参数：
            content: 需要处理、保存或分析的文本内容。

        返回：
            处理后的文本或稳定标识。
        """
        if self._encoding is None:
            return content
        tokens = self._encoding.encode(content)
        limit = self.max_tool_result_tokens
        if len(tokens) <= limit:
            return content
        head_tokens = int(limit * 0.2)
        tail_tokens = limit - head_tokens
        head = self._encoding.decode(tokens[:head_tokens])
        tail = self._encoding.decode(tokens[-tail_tokens:])
        return (
            head + f"\n\n... [truncated: {len(tokens)} total tokens, {limit} limit] ...\n\n" + tail
        )

    def count_tokens(self, text: str) -> int:
        """统计 `tokens` 对应的数据，并按照当前组件的约定返回结果。

        参数：
            text: 需要解析、格式化或展示的文本。

        返回：
            `int` 类型的处理结果。
        """
        if self._encoding:
            return len(self._encoding.encode(text))

        return len(text) // 4

    def count_message_tokens(self, message: Message) -> int:
        """统计 `message_tokens` 对应的数据，并按照当前组件的约定返回结果。

        参数：
            message: 用户提交或组件间传递的消息。

        返回：
            `int` 类型的处理结果。
        """
        tokens = 4

        tokens += self.count_tokens(message.role)

        if message.content:
            tokens += self.count_tokens(message.content)

        if message.tool_calls:
            for tc in message.tool_calls:
                tokens += self.count_tokens(tc.name)
                tokens += self.count_tokens(str(tc.arguments))

        if message.name:
            tokens += self.count_tokens(message.name)

        return tokens

    def get_total_tokens(self) -> int:
        """读取 `total_tokens` 对应的数据，不改变当前对象的业务状态。

        返回：
            `int` 类型的处理结果。
        """
        total = 0

        if self.system_prompt:
            total += self.count_tokens(self.system_prompt) + 10

        for message in self.messages:
            total += self.count_message_tokens(message)

        return total

    def get_available_tokens(self) -> int:
        """读取 `available_tokens` 对应的数据，不改变当前对象的业务状态。

        返回：
            `int` 类型的处理结果。
        """
        total = self.get_total_tokens()
        return max(0, self.context_window - RESERVED_OUTPUT_TOKENS - total)

    def _get_effective_available_tokens(self) -> int:
        """读取并返回 `_get_effective_available_tokens` 所表示的数据或流程，并遵守上下文管理定义的边界与状态约束。

        返回：
            `int` 类型的处理结果。
        """
        if self.context_window <= RESERVED_OUTPUT_TOKENS:
            return max(0, self.context_window - self.get_total_tokens())
        return self.get_available_tokens()

    def get_stats(self) -> ContextStats:
        """读取统计信息，不改变当前对象的业务状态。

        返回：
            `ContextStats` 类型的处理结果。
        """
        total_tokens = self.get_total_tokens()
        available = self.context_window - RESERVED_OUTPUT_TOKENS - total_tokens

        return ContextStats(
            total_messages=len(self.messages),
            total_tokens=total_tokens,
            context_window=self.context_window,
            available_tokens=max(0, available),
            utilization_percent=(total_tokens / self.context_window) * 100,
        )

    def get_presentation_snapshot(self) -> ContextPresentationSnapshot:
        """读取 `presentation_snapshot` 对应的数据，不改变当前对象的业务状态。

        返回：
            `ContextPresentationSnapshot` 类型的处理结果。
        """
        stats = self.get_stats()
        return ContextPresentationSnapshot(
            total_messages=stats.total_messages,
            total_tokens=stats.total_tokens,
            context_window=stats.context_window,
            available_tokens=stats.available_tokens,
            utilization_percent=stats.utilization_percent,
            compression_count=self._compression_count,
            has_compressed_summary=bool(self._compressed_summary),
            compression_threshold_percent=self.compression_threshold * 100,
        )

    def _prepare_message(self, message: Message) -> Message:
        """根据当前输入和上下文管理的状态计算 `_prepare_message`，并返回调用方需要的结果。

        参数：
            message: 用户提交或组件间传递的消息。

        返回：
            `Message` 类型的处理结果。
        """
        if message.role != "tool":
            return message

        token_count = self.count_tokens(message.content)
        if token_count <= self.max_tool_result_tokens:
            return message

        return Message(
            role=message.role,
            content=self._truncate_tool_result(message.content),
            tool_call_id=message.tool_call_id,
            name=message.name,
            timestamp=message.timestamp,
        )

    def _append_messages(self, messages: list[Message]) -> MessageAddResult:
        """追加消息，并按照当前组件的约定返回结果。

        参数：
            messages: 按协议顺序排列的对话消息。

        返回：
            `MessageAddResult` 类型的处理结果。
        """
        prepared = [self._prepare_message(message) for message in messages]
        if not prepared:
            return MessageAddResult(MessageAddStatus.ADDED, message_count=0)

        required_tokens = sum(self.count_message_tokens(message) for message in prepared)
        if (
            len(self.messages) + len(prepared) > self.max_messages
            or required_tokens > self._get_effective_available_tokens()
        ):
            self._trim_old_messages(
                required_tokens=required_tokens,
                required_slots=len(prepared),
            )

        if (
            len(self.messages) + len(prepared) > self.max_messages
            or required_tokens > self._get_effective_available_tokens()
        ):
            return MessageAddResult(
                MessageAddStatus.REJECTED,
                reason=(
                    f"Message group requires {required_tokens} tokens and "
                    f"{len(prepared)} slots, but only "
                    f"{self._get_effective_available_tokens()} tokens and "
                    f"{max(0, self.max_messages - len(self.messages))} slots are available"
                ),
            )

        self.messages.extend(prepared)
        return MessageAddResult(MessageAddStatus.ADDED, message_count=len(prepared))

    def add_message(self, message: Message) -> MessageAddResult:
        """添加`add_message`，必要时执行去重或容量检查。

        参数：
            message: 用户提交或组件间传递的消息。

        返回：
            `MessageAddResult` 类型的处理结果。
        """
        return self._append_messages([message])

    def add_user_message(self, content: str) -> MessageAddResult:
        """添加`add_user_message`，必要时执行去重或容量检查。

        参数：
            content: 需要处理、保存或分析的文本内容。

        返回：
            `MessageAddResult` 类型的处理结果。
        """
        msg = Message(role="user", content=content)
        return self.add_message(msg)

    def add_assistant_message(
        self,
        content: str,
        tool_calls: list[Any] | None = None,
    ) -> MessageAddResult:
        """添加`add_assistant_message`，必要时执行去重或容量检查。

        参数：
            content: 需要处理、保存或分析的文本内容。
            tool_calls: 可选的`tool_calls`。

        返回：
            `MessageAddResult` 类型的处理结果。
        """
        msg = Message(role="assistant", content=content, tool_calls=tool_calls)
        return self.add_message(msg)

    def add_tool_message(
        self,
        content: str,
        tool_call_id: str,
        name: str | None = None,
    ) -> MessageAddResult:
        """添加`add_tool_message`，必要时执行去重或容量检查。

        参数：
            content: 需要处理、保存或分析的文本内容。
            tool_call_id: 模型工具调用与工具结果之间的关联标识。
            name: 待查询、注册或操作对象的名称。

        返回：
            `MessageAddResult` 类型的处理结果。
        """
        msg = Message(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            name=name,
        )
        return self.add_message(msg)

    def set_system_prompt(self, prompt: str) -> None:
        """设置系统提示词并保持相关派生状态同步。

        参数：
            prompt: 本次操作使用的提示词。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self.system_prompt = prompt

    # ── 上下文压缩 ────────────────────────────────────────────────

    def set_compressor(self, compressor: Any) -> None:
        """设置`set_compressor`并保持相关派生状态同步。

        参数：
            compressor: 本次操作使用的`compressor`。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self._compressor = compressor

    def set_compressed_summary(self, summary: str | None) -> None:
        """设置压缩摘要摘要并保持相关派生状态同步。

        参数：
            summary: 可选的摘要。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self._compressed_summary = summary
        if summary and self._compression_count == 0:
            self._compression_count = 1
        elif not summary:
            self._compression_count = 0

    def get_compressed_summary(self) -> str | None:
        """读取压缩摘要摘要，不改变当前对象的业务状态。

        返回：
            `str | None` 类型的处理结果。
        """
        return self._compressed_summary

    def _should_compress(self) -> bool:
        """校验 `_should_compress` 所表示的数据或流程，并遵守上下文管理定义的边界与状态约束。

        返回：
            表示条件是否成立。
        """
        tokens = self.get_total_tokens()
        return tokens > self.context_window * self.compression_threshold

    def _is_safe_to_compress(self) -> bool:
        """校验 `_is_safe_to_compress` 所表示的数据或流程，并遵守上下文管理定义的边界与状态约束。

        返回：
            表示条件是否成立。
        """
        if self._compressing:
            return False
        if self._compression_failures >= self.compression_failure_limit:
            return False
        min_messages = self.keep_last_pairs * 2 + 4
        return len(self.messages) >= min_messages

    def _find_safe_cut_point(self) -> int | None:
        """查找 `safe_cut_point` 对应的数据，并按照当前组件的约定返回结果。

        返回：
            `int | None` 类型的处理结果。
        """
        messages_to_keep = max(2, self.keep_last_pairs * 2)
        if len(self.messages) <= messages_to_keep:
            return None

        cut = len(self.messages) - messages_to_keep

        # 切分旧消息时不能留下失去对应 assistant 工具调用的孤立 tool 结果。
        while cut > 0 and self.messages[cut].role == "tool":
            cut -= 1

        return cut if cut > 0 else None

    async def compress(self) -> bool:
        """处理压缩，并按照当前组件的约定返回结果。

        返回：
            表示条件是否成立。

        说明：
            执行过程中会更新当前实例维护的状态。
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        if (
            self._compressor is None
            or self._compressing
            or self._compression_failures >= self.compression_failure_limit
        ):
            return False

        cut = self._find_safe_cut_point()
        if cut is None:
            return False

        old_messages = self.messages[:cut]
        if not old_messages:
            return False

        self._compressing = True
        try:
            try:
                summary = await self._compressor.compress(old_messages, self._compressed_summary)
            except Exception:
                self._compression_failures += 1
                return False

            if not summary:
                self._compression_failures += 1
                return False

            self._compressed_summary = summary
            self._compression_count += 1
            self._compression_failures = 0
            self.messages = self.messages[cut:]
            return True
        finally:
            self._compressing = False

    async def _maybe_compress(self) -> bool:
        """校验 `_maybe_compress` 所表示的数据或流程，并遵守上下文管理定义的边界与状态约束。

        返回：
            表示条件是否成立。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        if self._compressor is None:
            return False
        if not self._should_compress():
            return False
        if not self._is_safe_to_compress():
            return False
        return await self.compress()

    async def add_messages_and_compress(self, messages: list[Message]) -> MessageAddResult:
        """原子加入一组协议相关消息；容量不足时先尝试压缩旧上下文，仍放不下则明确拒绝整组，避免只插入部分消息。

        参数：
            messages: 按协议顺序排列的对话消息。

        返回：
            `MessageAddResult` 类型的处理结果。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        result = self._append_messages(messages)
        if result:
            await self._maybe_compress()
            return result

        if await self.compress():
            retried = self._append_messages(messages)
            if retried:
                await self._maybe_compress()
                return MessageAddResult(
                    MessageAddStatus.ADDED_AFTER_COMPRESSION,
                    message_count=retried.message_count,
                )
            return retried

        return result

    async def add_message_and_compress(self, message: Message) -> MessageAddResult:
        """添加`add_message_and_compress`，必要时执行去重或容量检查。

        参数：
            message: 用户提交或组件间传递的消息。

        返回：
            `MessageAddResult` 类型的处理结果。

        说明：
            这是异步操作，调用方应使用 `await`，并允许取消信号向下传播。
        """
        return await self.add_messages_and_compress([message])

    # ── 组装模型输入 ────────────────────────────────────────────────

    def get_messages_for_llm(self) -> list[Message]:
        """按 Provider 可消费的顺序组装系统提示词、压缩摘要和当前消息窗口，作为下一次模型请求的完整上下文。

        返回：
            按调用约定排序的结果列表。
        """
        result: list[Message] = []

        if self.system_prompt:
            result.append(Message(role="system", content=self.system_prompt))

        # 针对有会话压缩时，将压缩的次数、压缩后的内容，放在会话的最前面
        if self._compressed_summary:
            result.append(
                Message(
                    role="user",
                    content=(
                        "[Compressed conversation context"
                        f" ({self._compression_count} compression(s))]\n\n"
                        + self._compressed_summary
                    ),
                    is_compression_boundary=True,
                )
            )
            result.append(
                Message(
                    role="user",
                    content="[Continuing conversation after context compression]",
                    is_compression_boundary=True,
                )
            )

        result.extend(self.messages)
        return result

    def _oldest_protocol_group_end(self) -> int:
        """读取并返回 `_oldest_protocol_group_end` 所表示的数据或流程，并遵守上下文管理定义的边界与状态约束。

        返回：
            `int` 类型的处理结果。
        """
        if not self.messages:
            return 0
        first = self.messages[0]
        if first.role == "user":
            end = 1
            while end < len(self.messages) and self.messages[end].role != "user":
                end += 1
            return end
        if first.role == "assistant" and first.tool_calls:
            end = 1
            while end < len(self.messages) and self.messages[end].role == "tool":
                end += 1
            return end
        if first.role == "tool":
            end = 1
            while end < len(self.messages) and self.messages[end].role == "tool":
                end += 1
            return end
        return 1

    def _trim_old_messages(
        self,
        keep_last: int = 4,
        *,
        required_tokens: int = 0,
        required_slots: int = 0,
    ) -> None:
        """执行 `_trim_old_messages` 所定义的协调步骤，必要时更新上下文管理维护的状态。

        参数：
            keep_last: 可选的`keep_last`。
            required_tokens: 可选的`required_tokens`。
            required_slots: 可选的`required_slots`。
        """
        while self.messages:
            over_message_limit = len(self.messages) + required_slots > self.max_messages
            lacks_tokens = required_tokens > self._get_effective_available_tokens()
            over_soft_limit = (
                len(self.messages) > keep_last
                and self.get_total_tokens() > self.context_window * 0.7
            )
            if not (over_message_limit or lacks_tokens or over_soft_limit):
                break
            group_end = self._oldest_protocol_group_end()
            if group_end <= 0:
                break
            if (
                len(self.messages) - group_end < keep_last
                and not over_message_limit
                and not lacks_tokens
            ):
                break
            del self.messages[:group_end]

    def clear(self) -> None:
        """处理清理，并按照当前组件的约定返回结果。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self.messages.clear()
        self._compression_failures = 0

    def get_last_n_messages(self, n: int) -> list[Message]:
        """读取 `last_n_messages` 对应的数据，不改变当前对象的业务状态。

        参数：
            n: 本次操作使用的`n`。

        返回：
            按调用约定排序的结果列表。
        """
        return self.messages[-n:] if n > 0 else []

    def get_conversation_history(self) -> list[dict[str, Any]]:
        """读取对话历史，不改变当前对象的业务状态。

        返回：
            按调用约定排序的结果列表。
        """
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
            }
            for msg in self.messages
        ]

    def __len__(self) -> int:
        return len(self.messages)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"ContextManager(messages={len(self.messages)}, "
            f"tokens={stats.total_tokens}/{self.context_window}, "
            f"utilization={stats.utilization_percent:.1f}%)"
        )
