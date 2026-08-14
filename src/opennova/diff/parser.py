"""差异与补丁子系统中的解析器模块，集中定义相关数据结构、边界适配和实现逻辑。"""

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ChangeType(StrEnum):
    """枚举变更类型允许出现的稳定取值，序列化和状态判断均使用这些值。"""

    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"


@dataclass
class FileChange:
    """保存文件变更所需的结构化数据，主要包含
    `file_path`、`change_type`、`original_content`、`new_content`、`diff`、`metadata`
    字段，便于在组件之间传递或持久化。
    """

    file_path: str
    change_type: ChangeType
    original_content: str | None = None
    new_content: str | None = None
    diff: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_diff(self) -> bool:
        """判断差异条件是否成立。

        返回：
            表示条件是否成立。
        """
        return self.diff is not None and self.diff.strip() != ""

    def get_lines_changed(self) -> tuple[int, int]:
        """读取 `lines_changed` 对应的数据，不改变当前对象的业务状态。

        返回：
            `tuple[int, int]` 类型的处理结果。
        """
        if not self.diff:
            return 0, 0

        added = 0
        removed = 0

        for line in self.diff.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                added += 1
            elif line.startswith("-") and not line.startswith("---"):
                removed += 1

        return added, removed


class DiffParser:
    """封装差异解析器相关的状态和操作，使调用方通过稳定接口使用该能力。"""

    XML_PATTERN = re.compile(
        r"<file_change>\s*"
        r"<path>(.*?)</path>\s*"
        r"<type>(.*?)</type>\s*"
        r"(?:<diff>(.*?)</diff>)?\s*"
        r"(?:<content>(.*?)</content>)?\s*"
        r"</file_change>",
        re.DOTALL,
    )

    MARKDOWN_DIFF_PATTERN = re.compile(
        r"```diff\s*\n(.*?)\n```",
        re.DOTALL,
    )

    FILE_PATH_PATTERN = re.compile(
        r"^(?:---|\+\+\+)\s+[ab]/(.+)$",
        re.MULTILINE,
    )

    def parse(self, llm_output: str) -> list[FileChange]:
        """处理解析，并按照当前组件的约定返回结果。

        参数：
            llm_output: 本次操作使用的`llm_output`。

        返回：
            按调用约定排序的结果列表。
        """
        changes = []

        changes.extend(self._parse_xml_format(llm_output))

        if not changes:
            changes.extend(self._parse_markdown_format(llm_output))

        if not changes:
            changes.extend(self._parse_json_format(llm_output))

        return changes

    def _parse_xml_format(self, text: str) -> list[FileChange]:
        """解析`parse_xml_format`并转换为内部使用的规范结构。

        参数：
            text: 需要解析、格式化或展示的文本。

        返回：
            按调用约定排序的结果列表。
        """
        changes = []

        for match in self.XML_PATTERN.finditer(text):
            path = match.group(1).strip()
            change_type_str = match.group(2).strip().lower()
            diff = match.group(3)
            content = match.group(4)

            try:
                change_type = ChangeType(change_type_str)
            except ValueError:
                change_type = ChangeType.MODIFY

            file_change = FileChange(
                file_path=path,
                change_type=change_type,
                diff=diff.strip() if diff else None,
                new_content=content.strip() if content else None,
            )

            changes.append(file_change)

        return changes

    def _parse_markdown_format(self, text: str) -> list[FileChange]:
        """解析Markdown格式化并转换为内部使用的规范结构。

        参数：
            text: 需要解析、格式化或展示的文本。

        返回：
            按调用约定排序的结果列表。
        """
        changes = []

        for match in self.MARKDOWN_DIFF_PATTERN.finditer(text):
            diff_text = match.group(1)

            file_path = self._extract_file_path(diff_text)
            if not file_path:
                continue

            file_change = FileChange(
                file_path=file_path,
                change_type=ChangeType.MODIFY,
                diff=diff_text,
            )

            changes.append(file_change)

        return changes

    def _parse_json_format(self, text: str) -> list[FileChange]:
        """解析JSON格式化并转换为内部使用的规范结构。

        参数：
            text: 需要解析、格式化或展示的文本。

        返回：
            按调用约定排序的结果列表。
        """
        changes = []

        json_pattern = re.compile(r"\{[^{}]*" r'"file_path"[^{}]*\}",?\s*', re.DOTALL)

        for match in json_pattern.finditer(text):
            try:
                json_str = match.group(0).rstrip(",")
                data = json.loads(json_str)

                file_path = data.get("file_path", "")
                change_type_str = data.get("type", "modify").lower()
                diff = data.get("diff")
                content = data.get("content")

                try:
                    change_type = ChangeType(change_type_str)
                except ValueError:
                    change_type = ChangeType.MODIFY

                file_change = FileChange(
                    file_path=file_path,
                    change_type=change_type,
                    diff=diff,
                    new_content=content,
                )

                changes.append(file_change)

            except json.JSONDecodeError:
                continue

        return changes

    def _extract_file_path(self, diff_text: str) -> str | None:
        """提取文件路径，并按照当前组件的约定返回结果。

        参数：
            diff_text: 本次操作使用的`diff_text`。

        返回：
            `str | None` 类型的处理结果。
        """
        match = self.FILE_PATH_PATTERN.search(diff_text)
        if match:
            return match.group(1)
        return None

    def parse_single_change(self, text: str) -> FileChange | None:
        """解析单个变更并转换为内部使用的规范结构。

        参数：
            text: 需要解析、格式化或展示的文本。

        返回：
            `FileChange | None` 类型的处理结果。
        """
        changes = self.parse(text)
        return changes[0] if changes else None

    @staticmethod
    def build_xml_change(
        file_path: str,
        change_type: ChangeType,
        diff: str | None = None,
        content: str | None = None,
    ) -> str:
        """根据当前输入和状态构造`build_xml_change`。

        参数：
            file_path: 目标文件的路径；访问范围仍受项目沙箱约束。
            change_type: 本次操作使用的变更类型。
            diff: 可选的差异。
            content: 需要处理、保存或分析的文本内容。

        返回：
            处理后的文本或稳定标识。
        """
        xml = f"""<file_change>
<path>{file_path}</path>
<type>{change_type.value}</type>
"""
        if diff:
            xml += f"<diff>\n{diff}\n</diff>\n"
        if content:
            xml += f"<content>\n{content}\n</content>\n"

        xml += "</file_change>"
        return xml


def parse_llm_file_changes(llm_output: str) -> list[FileChange]:
    """解析`parse_llm_file_changes`并转换为内部使用的规范结构。

    参数：
        llm_output: 本次操作使用的`llm_output`。

    返回：
        按调用约定排序的结果列表。
    """
    parser = DiffParser()
    return parser.parse(llm_output)
