"""通用辅助模块中的`encoding`模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from __future__ import annotations

import os
from collections.abc import Mapping


def utf8_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """读取并返回 `utf8_environment` 所表示的数据或流程，并遵守当前模块定义的边界与状态约束。

    参数：
        base: 可选的基础抽象。

    返回：
        供后续逻辑或序列化使用的结构化字典。
    """
    env = dict(os.environ if base is None else base)
    env.setdefault("LC_ALL", "en_US.UTF-8")
    env.setdefault("LANG", "en_US.UTF-8")
    env["PYTHONUTF8"] = "1"
    return env
