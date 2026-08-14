"""记忆与上下文子系统中的`compressor`模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from __future__ import annotations

from opennova.providers.base import BaseLLMProvider, Message


class ContextCompressor:
    """封装`ContextCompressor`相关的状态和操作，使调用方通过稳定接口使用该能力。"""

    COMPRESSION_PROMPT = """\
You are a context compressor for an AI coding agent. Summarize the conversation below, preserving:

1. User's explicit requests and goals
2. Key decisions made and their rationale
3. Files read, modified, or created (with full paths)
4. Errors encountered and how they were resolved
5. Current task state (completed, in-progress, remaining)
6. Important code patterns, conventions, or constraints discovered

Output a single paragraph (max 1500 tokens). Use flowing prose, NOT bullet points.

{previous_summary_block}

<conversation>
{messages}
</conversation>

Summary:"""

    def __init__(self, llm_provider: BaseLLMProvider):
        self.llm = llm_provider

    def build_compression_prompt(
        self,
        messages_text: str,
        previous_summary: str | None = None,
    ) -> str:
        if previous_summary:
            previous_summary_block = f"<previous_summary>\n{previous_summary}\n</previous_summary>"
        else:
            previous_summary_block = ""
        return self.COMPRESSION_PROMPT.format(
            previous_summary_block=previous_summary_block,
            messages=messages_text,
        )

    def _format_messages(self, messages: list[Message]) -> str:
        """把消息整理为稳定、便于展示的文本格式。

        参数：
            messages: 按协议顺序排列的对话消息。

        返回：
            处理后的文本或稳定标识。
        """
        lines: list[str] = []
        for msg in messages:
            role = msg.role
            content = msg.content or ""
            if msg.tool_calls:
                tools = ", ".join(tc.name for tc in msg.tool_calls)
                content = f"[called tools: {tools}] {content}"
            if len(content) > 2000:
                content = content[:2000] + "..."
            lines.append(f"[{role}]: {content}")
        return "\n".join(lines)

    async def compress(
        self,
        messages: list[Message],
        previous_summary: str | None = None,
    ) -> str:
        """处理压缩，并按照当前组件的约定返回结果。

        参数：
            messages: 按协议顺序排列的对话消息。
            previous_summary: 可选的上一个摘要。

        返回：
            处理后的文本或稳定标识。

        说明：
            该操作可能等待模型服务、MCP 服务或其他异步数据源返回。
        """
        messages_text = self._format_messages(messages)
        prompt = self.build_compression_prompt(messages_text, previous_summary)

        response = await self.llm.chat(
            messages=[Message(role="user", content=prompt)],
            tools=None,
            max_tokens=2048,
            temperature=0.3,
        )
        return response.content.strip()
