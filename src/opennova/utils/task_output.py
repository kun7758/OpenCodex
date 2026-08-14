"""通用辅助模块中的任务输出模块，集中定义相关数据结构、边界适配和实现逻辑。"""

import os
import tempfile
from pathlib import Path


def get_task_output_dir() -> Path:
    """读取任务输出目录，不改变当前对象的业务状态。

    返回：
        `Path` 类型的处理结果。

    说明：
        该操作会访问本地文件系统，路径校验和原子写入约束由所在组件负责。
    """
    # 优先使用 XDG_DATA_HOME；未设置时使用 `~/.local/share`。
    data_home = os.environ.get("XDG_DATA_HOME")
    if data_home:
        base = Path(data_home) / "opennova"
    else:
        base = Path.home() / ".local" / "share" / "opennova"

    output_dir = base / "task_outputs"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        probe = output_dir / ".write-test"
        probe.write_text("", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return output_dir
    except Exception:
        fallback_dir = Path(tempfile.gettempdir()) / "opennova" / "task_outputs"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        return fallback_dir


def get_task_output_path(task_id: str) -> str:
    """读取任务输出路径，不改变当前对象的业务状态。

    参数：
        task_id: 目标任务的稳定标识。

    返回：
        处理后的文本或稳定标识。
    """
    output_dir = get_task_output_dir()
    return str(output_dir / f"{task_id}.txt")


def write_task_output(task_id: str, content: str, offset: int = 0) -> int:
    """写入任务输出，并按照当前组件的约定返回结果。

    参数：
        task_id: 目标任务的稳定标识。
        content: 需要处理、保存或分析的文本内容。
        offset: 可选的`offset`。

    返回：
        `int` 类型的处理结果。

    说明：
        该操作会访问本地文件系统，路径校验和原子写入约束由所在组件负责。
    """
    output_path = get_task_output_path(task_id)

    mode = "a" if offset == 0 else "r+"

    try:
        with open(output_path, mode, encoding="utf-8") as f:
            if offset > 0:
                f.seek(offset)
            f.write(content)
            new_offset = f.tell()
        return new_offset
    except Exception:
        return offset


def read_task_output(task_id: str, max_length: int = 10000, offset: int = 0) -> tuple[str, int]:
    """读取任务输出，并按照当前组件的约定返回结果。

    参数：
        task_id: 目标任务的稳定标识。
        max_length: 允许返回的最大文本长度。
        offset: 可选的`offset`。

    返回：
        `tuple[str, int]` 类型的处理结果。

    说明：
        该操作会访问本地文件系统，路径校验和原子写入约束由所在组件负责。
    """
    output_path = get_task_output_path(task_id)

    if not os.path.exists(output_path):
        return "", offset

    try:
        with open(output_path, encoding="utf-8") as f:
            f.seek(offset)
            content = f.read(max_length)
            new_offset = f.tell()
        return content, new_offset
    except Exception:
        return "", offset


def delete_task_output(task_id: str) -> bool:
    """移除删除任务输出指向的数据，并清理相关索引或资源。

    参数：
        task_id: 目标任务的稳定标识。

    返回：
        表示条件是否成立。
    """
    output_path = get_task_output_path(task_id)
    try:
        if os.path.exists(output_path):
            os.remove(output_path)
            return True
    except Exception:
        pass
    return False


def get_task_output_size(task_id: str) -> int:
    """读取 `task_output_size` 对应的数据，不改变当前对象的业务状态。

    参数：
        task_id: 目标任务的稳定标识。

    返回：
        `int` 类型的处理结果。
    """
    output_path = get_task_output_path(task_id)
    try:
        return os.path.getsize(output_path)
    except Exception:
        return 0
