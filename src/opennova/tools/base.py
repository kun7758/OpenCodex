"""内置工具系统中的基础抽象模块，集中定义相关数据结构、边界适配和实现逻辑。"""

import types
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Union, get_args, get_origin

from opennova.providers.base import ToolSchema


@dataclass
class ToolResult:
    """所有工具共用的返回结构。success 表示执行状态，output 面向模型和用户，error 保存失败原因，metadata 携带机器可读的扩展信息。"""

    success: bool
    output: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_string(self) -> str:
        """把工具结果转换为适合写入模型上下文或终端展示的文本。

        返回：
            处理后的文本或稳定标识。
        """
        if self.success:
            return self.output
        return f"Error: {self.error}\n{self.output}"


@dataclass
class ToolParameter:
    """描述一个工具参数的类型、说明、默认值、必填状态和嵌套结构，可进一步转换为 JSON Schema。"""

    type: str
    description: str = ""
    default: Any = None
    required: bool = True
    enum: list[Any] | None = None
    items: dict[str, Any] | None = None
    properties: dict[str, Any] | None = None

    def to_json_schema(self) -> dict[str, Any]:
        """把当前参数描述转换为 JSON Schema 属性；默认值、枚举、数组元素和对象属性仅在实际存在时写入。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        schema: dict[str, Any] = {"type": self.type, "description": self.description}

        if self.default is not None:
            schema["default"] = self.default

        if self.enum:
            schema["enum"] = self.enum

        if self.items:
            schema["items"] = self.items

        if self.properties:
            schema["properties"] = self.properties

        return schema


class BaseTool(ABC):
    """全部工具的抽象基类。子类实现实际操作，并通过类型注解生成工具参数 Schema；只读性、破坏性、权限、并发和外部访问属性也在这里提供统一约定。"""

    name: str = ""
    description: str = ""
    aliases: list[str] = []
    search_hint: str = ""
    max_result_chars: int = 100_000
    progress_metadata: dict[str, Any] = {}
    output_schema: dict[str, Any] | None = None

    def __init__(self, config: dict[str, Any] | None = None):
        """初始化基础抽象工具，保存后续操作需要的依赖、配置和初始状态。

        参数：
            config: 控制当前组件行为的配置。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self.config = config or {}

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """执行基础抽象工具对应的实际操作，校验输入并返回统一结果。

        参数：
            **kwargs: 传递给底层实现的额外关键字参数。

        返回：
            `ToolResult` 类型的处理结果。
        """
        pass

    def get_parameters_schema(self) -> dict[str, Any]:
        """通过反射工具 `execute()` 方法的签名和类型注解生成 JSON Schema；没有默认值的参数会进入 required 列表。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        import inspect
        from typing import get_type_hints

        sig = inspect.signature(self.execute)
        hints = get_type_hints(self.execute) if hasattr(self, "__class__") else {}

        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name == "kwargs":
                continue

            param_type = hints.get(param_name, str)
            prop = self._python_type_to_schema(param_type)

            if param.default is inspect.Parameter.empty:
                required.append(param_name)
            else:
                prop["default"] = param.default

            properties[param_name] = prop

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def get_schema(self) -> ToolSchema:
        """组合工具名称、说明和参数 Schema，生成可以直接交给模型 Provider 的 ToolSchema。

        返回：
            `ToolSchema` 类型的处理结果。
        """
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.get_parameters_schema(),
        )

    def describe(self, **kwargs: Any) -> str:
        """返回当前工具面向模型和界面的说明；子类可以根据调用参数提供更具体的动态说明。

        参数：
            **kwargs: 传递给底层实现的额外关键字参数。

        返回：
            处理后的文本或稳定标识。
        """
        return self.description

    @staticmethod
    def _python_type_to_json(python_type: type) -> str:
        """把 Python 类型注解转换为 JSON Schema 的基础类型名称。

        参数：
            python_type: 需要转换为 JSON Schema 的 Python 类型注解。

        返回：
            处理后的文本或稳定标识。
        """
        schema = BaseTool._python_type_to_schema(python_type)
        schema_type = schema.get("type")
        return schema_type if isinstance(schema_type, str) else "string"

    @staticmethod
    def _python_type_to_schema(python_type: Any) -> dict[str, Any]:
        """递归把 Python 类型注解转换为 JSON Schema 片段，支持容器、枚举、Literal 和可空联合类型。

        参数：
            python_type: 需要转换为 JSON Schema 的 Python 类型注解。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
            Any: "object",
        }

        if python_type is None or python_type is type(None):
            return {"type": "null"}

        if isinstance(python_type, type) and issubclass(python_type, Enum):
            values = [member.value for member in python_type]
            return {"type": BaseTool._json_type_for_values(values), "enum": values}

        origin = get_origin(python_type)
        if origin is not None:
            if origin in (list, tuple, set):
                args = get_args(python_type)
                item_schema = (
                    BaseTool._python_type_to_schema(args[0]) if args else {"type": "string"}
                )
                return {"type": "array", "items": item_schema}
            if origin is dict:
                return {"type": "object"}
            if origin is Literal:
                values = list(get_args(python_type))
                return {"type": BaseTool._json_type_for_values(values), "enum": values}
            if origin in (Union, types.UnionType):
                non_none_args = [arg for arg in get_args(python_type) if arg is not type(None)]
                if len(non_none_args) == 1:
                    return BaseTool._python_type_to_schema(non_none_args[0])
                return {
                    "anyOf": [BaseTool._python_type_to_schema(arg) for arg in non_none_args],
                }

        return {"type": type_mapping.get(python_type, "string")}

    @staticmethod
    def _json_type_for_values(values: list[Any]) -> str:
        """检查一组枚举值并推断它们共同使用的 JSON 标量类型。

        参数：
            values: 用于推断共同 JSON 标量类型的一组值。

        返回：
            处理后的文本或稳定标识。
        """
        if values and all(isinstance(value, bool) for value in values):
            return "boolean"
        if values and all(
            isinstance(value, int) and not isinstance(value, bool) for value in values
        ):
            return "integer"
        if values and all(
            isinstance(value, (int, float)) and not isinstance(value, bool) for value in values
        ):
            return "number"
        return "string"

    def is_read_only(self, **kwargs: Any) -> bool:
        """声明本次工具调用是否只读取状态；默认返回假，具体只读工具需要显式覆盖。

        参数：
            **kwargs: 传递给底层实现的额外关键字参数。

        返回：
            表示条件是否成立。
        """
        return False

    def is_enabled(self) -> bool:
        """声明工具当前是否应暴露给模型；默认启用。

        返回：
            表示条件是否成立。
        """
        return True

    def is_destructive(self, **kwargs: Any) -> bool:
        """声明本次工具调用是否可能删除或覆盖数据；默认返回假。

        参数：
            **kwargs: 传递给底层实现的额外关键字参数。

        返回：
            表示条件是否成立。
        """
        return False

    def requires_permission(self, **kwargs: Any) -> bool:
        """判断工具是否需要用户批准；默认沿用破坏性判断，子类可以收紧策略。

        参数：
            **kwargs: 传递给底层实现的额外关键字参数。

        返回：
            表示条件是否成立。
        """
        return self.is_destructive(**kwargs)

    def is_concurrency_safe(self, **kwargs: Any) -> bool:
        """判断工具是否可以与同一模型回合中的其他调用并发执行；默认只有只读调用可并发。

        参数：
            **kwargs: 传递给底层实现的额外关键字参数。

        返回：
            表示条件是否成立。
        """
        return self.is_read_only(**kwargs)

    def interrupt_behavior(self) -> str:
        """声明取消信号到达时应立即取消还是等待不可中断操作结束。

        返回：
            处理后的文本或稳定标识。
        """
        return "cancel"

    def inputs_equivalent(self, a: dict[str, Any], b: dict[str, Any]) -> bool:
        """判断两组工具参数是否表示同一次语义调用；默认按字典内容直接比较。

        参数：
            a: 用于比较的第一组输入。
            b: 用于比较的第二组输入。

        返回：
            表示条件是否成立。
        """
        return a == b

    def is_open_world(self, **kwargs: Any) -> bool:
        """声明工具是否会访问本地项目之外的开放资源，例如公网或外部服务。

        参数：
            **kwargs: 传递给底层实现的额外关键字参数。

        返回：
            表示条件是否成立。
        """
        return False

    def __repr__(self) -> str:
        return f"Tool({self.name})"


class ToolRegistry:
    """当前 AgentRuntime 私有的工具注册表。它保存工具实例并向模型提供 Schema，实例级隔离可以避免不同会话或子 Agent 共享可变工具状态。"""

    _global_registry: "ToolRegistry | None" = None

    def __init__(self, tools: list[BaseTool] | None = None):
        self._tools: dict[str, BaseTool] = {}
        for tool in tools or []:
            self.register(tool)

    @classmethod
    def global_registry(cls) -> "ToolRegistry":
        """返回仅供旧接口使用的显式全局工具注册表；正常 AgentRuntime 应持有独立注册表。

        返回：
            `'ToolRegistry'` 类型的处理结果。
        """
        if cls._global_registry is None:
            cls._global_registry = cls()
        return cls._global_registry

    def register(self, tool: BaseTool) -> None:
        """处理注册，并按照当前组件的约定返回结果。

        参数：
            tool: 要注册、检查或调用的工具实例。
        """
        if not tool.name:
            raise ValueError("Tool must have a name attribute")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        """读取并返回 `get` 所表示的数据或流程，并遵守工具注册表定义的边界与状态约束。

        参数：
            name: 待查询、注册或操作对象的名称。

        返回：
            `BaseTool` 类型的处理结果。
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found. Available: {list(self._tools.keys())}")
        return self._tools[name]

    def list_tools(self) -> list[ToolSchema]:
        """列出工具，并按当前组件约定返回稳定顺序。

        返回：
            按调用约定排序的结果列表。
        """
        return [tool.get_schema() for tool in self._tools.values()]

    def list_names(self) -> list[str]:
        """列出名称，并按当前组件约定返回稳定顺序。

        返回：
            按调用约定排序的结果列表。
        """
        return list(self._tools.keys())

    def has_tool(self, name: str) -> bool:
        """判断工具条件是否成立。

        参数：
            name: 待查询、注册或操作对象的名称。

        返回：
            表示条件是否成立。
        """
        return name in self._tools

    def unregister(self, name: str) -> bool:
        """处理注销，并按照当前组件的约定返回结果。

        参数：
            name: 待查询、注册或操作对象的名称。

        返回：
            表示条件是否成立。
        """
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def clear(self) -> None:
        """处理清理，并按照当前组件的约定返回结果。"""
        self._tools.clear()

    @classmethod
    def reset(cls) -> None:
        """处理重置，并按照当前组件的约定返回结果。"""
        cls._global_registry = None

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"ToolRegistry({len(self._tools)} tools: {list(self._tools.keys())})"


def register_builtin_tools(registry: ToolRegistry | None = None) -> ToolRegistry:
    """注册`register_builtin_tools`，使后续运行能够发现并调用它。

    参数：
        registry: 可选的注册表。

    返回：
        `ToolRegistry` 类型的处理结果。
    """
    if registry is None:
        registry = ToolRegistry()

    from opennova.tools.diagnostics_tools import (
        PythonDefinitionTool,
        PythonDiagnosticsTool,
        PythonReferencesTool,
        PythonSymbolsTool,
    )
    from opennova.tools.file_tools import (
        CreateFileTool,
        DeleteFileTool,
        EditFileTool,
        ListDirectoryTool,
        MultiEditFileTool,
        ReadFileTool,
        WriteFileTool,
    )
    from opennova.tools.mcp_resource_tools import ListMCPResourcesTool, ReadMCPResourceTool
    from opennova.tools.project_guide_tool import InitProjectGuideTool
    from opennova.tools.search_tools import GlobFilesTool, GrepCodeTool
    from opennova.tools.shell_tools import ExecuteCommandTool
    from opennova.tools.worktree_tools import EnterWorktreeTool, ExitWorktreeTool

    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(CreateFileTool())
    registry.register(EditFileTool())
    registry.register(MultiEditFileTool())
    registry.register(DeleteFileTool())
    registry.register(ListDirectoryTool())
    registry.register(ExecuteCommandTool())
    registry.register(GlobFilesTool())
    registry.register(GrepCodeTool())
    registry.register(PythonDiagnosticsTool())
    registry.register(PythonSymbolsTool())
    registry.register(PythonDefinitionTool())
    registry.register(PythonReferencesTool())
    registry.register(InitProjectGuideTool())
    registry.register(ListMCPResourcesTool())
    registry.register(ReadMCPResourceTool())
    registry.register(EnterWorktreeTool())
    registry.register(ExitWorktreeTool())

    return registry
