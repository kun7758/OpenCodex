"""Skill 扩展子系统中的参数模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from __future__ import annotations

import re
import shlex


def parse_arguments(args: str) -> list[str]:
    """解析参数并转换为内部使用的规范结构。

    参数：
        args: 调用方传入的位置参数或 Skill 参数文本。

    返回：
        按调用约定排序的结果列表。
    """
    if not args or not args.strip():
        return []

    try:
        return shlex.split(args, posix=True)
    except ValueError:
        return args.split()


def parse_argument_names(argument_names: str | list[str] | None) -> list[str]:
    """解析参数名称并转换为内部使用的规范结构。

    参数：
        argument_names: 可选的参数名称。

    返回：
        按调用约定排序的结果列表。
    """
    if not argument_names:
        return []

    if isinstance(argument_names, str):
        values = re.split(r"[\s,]+", argument_names.strip())
    else:
        values = [str(item).strip() for item in argument_names]

    return [value for value in values if value and not value.isdigit()]


def generate_progressive_argument_hint(
    argument_names: list[str],
    typed_args: list[str],
) -> str | None:
    """生成 `progressive_argument_hint` 对应的数据，并按照当前组件的约定返回结果。

    参数：
        argument_names: 本次操作使用的参数名称。
        typed_args: 本次操作使用的`typed_args`。

    返回：
        `str | None` 类型的处理结果。
    """
    remaining = argument_names[len(typed_args) :]
    if not remaining:
        return None
    return " ".join(f"[{name}]" for name in remaining)


def substitute_arguments(
    content: str,
    args: str | None,
    append_if_no_placeholder: bool = True,
    argument_names: list[str] | None = None,
) -> str:
    """更新 `substitute_arguments` 所表示的数据或流程，并遵守当前模块定义的边界与状态约束。

    参数：
        content: 需要处理、保存或分析的文本内容。
        args: 调用方传入的位置参数或 Skill 参数文本。
        append_if_no_placeholder: 可选的`append_if_no_placeholder`。
        argument_names: 可选的参数名称。

    返回：
        处理后的文本或稳定标识。
    """
    if args is None:
        return content

    parsed_args = parse_arguments(args)
    argument_names = argument_names or []
    original_content = content

    for index, name in enumerate(argument_names):
        content = re.sub(rf"\${re.escape(name)}(?![\[\w])", parsed_args[index] if index < len(parsed_args) else "", content)

    content = re.sub(
        r"\$ARGUMENTS\[(\d+)\]",
        lambda match: parsed_args[int(match.group(1))] if int(match.group(1)) < len(parsed_args) else "",
        content,
    )
    content = re.sub(
        r"\$(\d+)(?!\w)",
        lambda match: parsed_args[int(match.group(1))] if int(match.group(1)) < len(parsed_args) else "",
        content,
    )
    content = content.replace("$ARGUMENTS", args)
    content = content.replace("$ARGS", args)

    if content == original_content and append_if_no_placeholder and args:
        content = f"{content}\n\nARGUMENTS: {args}"

    return content
