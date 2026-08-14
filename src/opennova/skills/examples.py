"""Skill 扩展子系统中的`examples`模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from pathlib import Path


def get_builtin_skill_dirs() -> list[Path]:
    """读取 `builtin_skill_dirs` 对应的数据，不改变当前对象的业务状态。

    返回：
        按调用约定排序的结果列表。
    """
    bundled_dir = Path(__file__).parent / "bundled"
    return [bundled_dir] if bundled_dir.exists() else []
