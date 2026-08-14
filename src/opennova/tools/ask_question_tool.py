"""内置工具系统中的`ask_question_tool`模块，集中定义相关数据结构、边界适配和实现逻辑。"""

import json
from typing import Any

from opennova.tools.base import BaseTool, ToolResult


class AskUserQuestionTool(BaseTool):
    """实现`AskUserQuestionTool`。模型通过统一工具 Schema 调用它，执行结果使用 ToolResult 返回，并服从运行时安全策略。"""

    name = "ask_user_question"
    description = (
        "Ask the user 1-4 questions to gather preferences, clarify requirements, "
        "or get decisions on implementation choices. "
        "Use this when you need user input during execution.\n"
        "Provide questions as an array. Each question can have:\n"
        "- 2-4 options with labels and optional descriptions (choice mode)\n"
        "- No options (free-text mode, user can skip to let you decide)\n"
        "- multiSelect: true for multiple answers\n"
        "- A recommended first option labelled with '(Recommended)'\n"
        "Users always get an 'Other' option for custom input."
    )

    def get_parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "description": "1-4 questions to ask the user",
                    "minItems": 1,
                    "maxItems": 4,
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The complete question to ask",
                            },
                            "header": {
                                "type": "string",
                                "description": "Short label shown as a chip/tag (max 12 chars)",
                            },
                            "options": {
                                "type": "array",
                                "description": "2-4 options for choice mode. Omit for free-text.",
                                "minItems": 2,
                                "maxItems": 4,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "label": {
                                            "type": "string",
                                            "description": "Display text (concise, 1-5 words)",
                                        },
                                        "description": {
                                            "type": "string",
                                            "description": "Explanation of what this option means",
                                        },
                                        "preview": {
                                            "type": "string",
                                            "description": "Optional preview content (markdown or HTML)",
                                        },
                                    },
                                    "required": ["label", "description"],
                                },
                            },
                            "multiSelect": {
                                "type": "boolean",
                                "description": "Allow multiple answers to be selected",
                                "default": False,
                            },
                        },
                        "required": ["question"],
                    },
                },
                "answers": {
                    "type": "object",
                    "description": "User answers (filled by the UI, not by the model)",
                },
                "annotations": {
                    "type": "object",
                    "description": "User notes/annotations (filled by the UI)",
                },
            },
            "required": ["questions"],
        }

    def execute(self, questions: list[dict[str, Any]] | None = None, **kwargs: Any) -> ToolResult:
        """执行`AskUserQuestionTool`对应的实际操作，校验输入并返回统一结果。

        参数：
            questions: 可选的`questions`。
            **kwargs: 传递给底层实现的额外关键字参数。

        返回：
            `ToolResult` 类型的处理结果。
        """
        try:
            # 兼容旧 API：允许直接传入扁平的 question/options 参数。
            if not questions and kwargs.get("question"):
                questions = [{
                    "question": kwargs["question"],
                    "header": kwargs.get("header"),
                    "options": kwargs.get("options", []),
                    "multiSelect": kwargs.get("multi_select", False),
                }]

            # 兼容模型把 questions 错误编码为 JSON 字符串的情况。
            if isinstance(questions, str):
                try:
                    questions = json.loads(questions)
                except (json.JSONDecodeError, TypeError):
                    questions = []

            questions = questions or []
            if not isinstance(questions, list):
                questions = []

            # 构造展示文本，并为交互处理器生成规范化问题列表。
            output_lines: list[str] = []
            normalized_questions: list[dict[str, Any]] = []

            for qi, q in enumerate(questions):
                if not isinstance(q, dict):
                    continue

                question_text = q.get("question", "")
                header = q.get("header")
                raw_options = q.get("options", []) or []
                multi_select = q.get("multiSelect", False)

                # 兼容模型把 options 编码为 JSON 字符串。
                if isinstance(raw_options, str):
                    try:
                        raw_options = json.loads(raw_options)
                    except (json.JSONDecodeError, TypeError):
                        raw_options = []
                if not isinstance(raw_options, list):
                    raw_options = []

                is_free_text = len(raw_options) < 2

                if qi > 0:
                    output_lines.append("")
                if len(questions) > 1:
                    output_lines.append(f"[Q{qi + 1}] {question_text}")
                else:
                    output_lines.append(f"Question: {question_text}")
                if header:
                    prefix = "     " if len(questions) > 1 else ""
                    output_lines.append(f"{prefix}[{header}]")

                normalized_options = []
                if is_free_text:
                    output_lines.append("     (Free-text answer, Enter to skip)")
                else:
                    if multi_select:
                        output_lines.append("     (Select multiple options, comma-separated)")
                    else:
                        output_lines.append("     (Select one option)")
                    for i, option in enumerate(raw_options, 1):
                        if isinstance(option, str):
                            label = option
                            description = ""
                            preview = None
                        else:
                            label = option.get("label", f"Option {i}")
                            description = option.get("description", "")
                            preview = option.get("preview")

                        normalized_option: dict[str, Any] = {
                            "index": i,
                            "label": label,
                            "description": description,
                        }
                        if preview:
                            normalized_option["preview"] = preview
                        normalized_options.append(normalized_option)

                        output_lines.append(f"  [{i}] {label}")
                        if description:
                            output_lines.append(f"      {description}")
                        if preview:
                            output_lines.append(
                                f"      Preview: {preview[:100]}{'...' if len(preview) > 100 else ''}"
                            )

                normalized_questions.append({
                    "question": question_text,
                    "header": header,
                    "options": normalized_options,
                    "multiSelect": multi_select,
                    "free_text": is_free_text,
                    "allow_custom_answer": True,
                })

            # 为交互处理器构造 prompt_payload。
            prompt_payload: dict[str, Any] = {
                "questions": normalized_questions,
            }
            # 同时暴露首个问题的顶层字段，保持旧回调兼容。
            if normalized_questions:
                first = normalized_questions[0]
                prompt_payload["question"] = first["question"]
                prompt_payload["header"] = first.get("header")
                prompt_payload["options"] = first.get("options", [])
                prompt_payload["multi_select"] = first.get("multiSelect", False)
                prompt_payload["free_text"] = first.get("free_text", False)
                prompt_payload["allow_custom_answer"] = first.get(
                    "allow_custom_answer", True
                )

            return ToolResult(
                success=True,
                output="\n".join(output_lines),
                metadata={
                    "interaction_required": True,
                    "interaction_type": "ask_user_question",
                    "questions": normalized_questions,
                    "prompt_payload": prompt_payload,
                },
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
