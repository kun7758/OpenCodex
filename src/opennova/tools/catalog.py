"""内置工具系统中的目录模块，集中定义相关数据结构、边界适配和实现逻辑。"""

BUILTIN_TOOL_NAMES: tuple[str, ...] = (
    "read_file",
    "write_file",
    "create_file",
    "edit_file",
    "multi_edit_file",
    "delete_file",
    "list_directory",
    "execute_command",
    "glob_files",
    "grep_code",
    "tool_search",
    "python_diagnostics",
    "python_symbols",
    "python_definition",
    "python_references",
    "task_create",
    "task_list",
    "task_get",
    "task_update",
    "task_stop",
    "task_output",
    "todo_write",
    "agent",
    "send_message",
    "ask_user_question",
    "skill",
    "enter_plan_mode",
    "exit_plan_mode",
    "web_search",
    "web_fetch",
    "init_project_guide",
    "list_mcp_resources",
    "read_mcp_resource",
    "git_commit",
    "git_status",
    "git_diff",
    "git_log",
    "git_branch",
    "enter_worktree",
    "exit_worktree",
)


def builtin_tool_names() -> list[str]:
    """返回当前版本声明的全部内置工具名称，供无副作用检查命令使用。

    返回：
        按调用约定排序的结果列表。
    """
    return list(BUILTIN_TOOL_NAMES)
