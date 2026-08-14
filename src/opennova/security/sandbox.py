"""安全控制子系统中的沙箱模块，集中定义相关数据结构、边界适配和实现逻辑。"""

import contextlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SandboxConfig:
    """保存沙箱配置所需的结构化数据，主要包含
    `working_dir`、`allowed_paths`、`denied_paths`、`max_file_size`、`allow_network`、`read_only`、`temp_dir`
    字段，便于在组件之间传递或持久化。
    """

    working_dir: str
    allowed_paths: list[str] | None = None
    denied_paths: list[str] | None = None
    max_file_size: int = 100 * 1024 * 1024  # 默认最大文件大小为 100 MB。
    allow_network: bool = False
    read_only: bool = False
    temp_dir: str | None = None


class Sandbox:
    """封装沙箱相关的状态和操作，使调用方通过稳定接口使用该能力。"""

    DEFAULT_DENIED_PATHS = [
        "/etc",
        "/usr",
        "/bin",
        "/sbin",
        "/boot",
        "/dev",
        "/proc",
        "/sys",
        "/root",
    ]

    def __init__(self, config: SandboxConfig):
        """初始化沙箱，保存后续操作需要的依赖、配置和初始状态。

        参数：
            config: 控制当前组件行为的配置。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self.config = config
        self.working_dir = Path(config.working_dir).resolve()
        self.allowed_paths = [
            Path(p).resolve() for p in (config.allowed_paths or [])
        ]
        self.denied_paths = [
            Path(p).resolve() for p in (config.denied_paths or self.DEFAULT_DENIED_PATHS)
        ]

        self.modifications: list[dict[str, Any]] = []
        self._original_files: dict[str, str] = {}

    def is_path_allowed(self, path: str | Path) -> tuple[bool, str]:
        """判断路径允许项条件是否成立。

        参数：
            path: 需要读取、检查或写入的路径。

        返回：
            `tuple[bool, str]` 类型的处理结果。
        """
        try:
            target_path = Path(path).resolve()
        except Exception as e:
            return False, f"Invalid path: {e}"

        for denied in self.denied_paths:
            try:
                target_path.relative_to(denied)
                return False, f"Access to protected path denied: {denied}"
            except ValueError:
                pass

        try:
            target_path.relative_to(self.working_dir)
            return True, "Within working directory"
        except ValueError:
            pass

        for allowed in self.allowed_paths:
            try:
                target_path.relative_to(allowed)
                return True, f"Within allowed path: {allowed}"
            except ValueError:
                pass

        return False, f"Path outside allowed directories: {path}"

    def safe_read(self, file_path: str | Path) -> tuple[bool, str | bytes]:
        """根据当前输入和沙箱的状态计算 `safe_read`，并返回调用方需要的结果。

        参数：
            file_path: 目标文件的路径；访问范围仍受项目沙箱约束。

        返回：
            `tuple[bool, str | bytes]` 类型的处理结果。
        """
        is_allowed, reason = self.is_path_allowed(file_path)
        if not is_allowed:
            return False, reason

        path = Path(file_path)

        if not path.exists():
            return False, f"File not found: {file_path}"

        if not path.is_file():
            return False, f"Not a file: {file_path}"

        if path.stat().st_size > self.config.max_file_size:
            return False, f"File too large: {path.stat().st_size} bytes"

        try:
            content = path.read_bytes()
            return True, content
        except Exception as e:
            return False, f"Failed to read file: {e}"

    def safe_write(
        self,
        file_path: str | Path,
        content: bytes | str,
        backup: bool = True,
    ) -> tuple[bool, str]:
        """根据当前输入和沙箱的状态计算 `safe_write`，并返回调用方需要的结果。

        参数：
            file_path: 目标文件的路径；访问范围仍受项目沙箱约束。
            content: 需要处理、保存或分析的文本内容。
            backup: 可选的`backup`。

        返回：
            `tuple[bool, str]` 类型的处理结果。

        说明：
            该操作会访问本地文件系统，路径校验和原子写入约束由所在组件负责。
        """
        if self.config.read_only:
            return False, "Sandbox is in read-only mode"

        is_allowed, reason = self.is_path_allowed(file_path)
        if not is_allowed:
            return False, reason

        path = Path(file_path)

        if isinstance(content, str):
            content = content.encode("utf-8")

        if len(content) > self.config.max_file_size:
            return False, f"Content too large: {len(content)} bytes"

        if backup and path.exists():
            with contextlib.suppress(Exception):
                self._original_files[str(path)] = path.read_text(encoding="utf-8")

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

            self.modifications.append(
                {
                    "type": "write",
                    "path": str(path),
                    "size": len(content),
                }
            )

            return True, f"Successfully wrote to {file_path}"

        except Exception as e:
            return False, f"Failed to write file: {e}"

    def safe_delete(
        self,
        file_path: str | Path,
        backup: bool = True,
    ) -> tuple[bool, str]:
        """根据当前输入和沙箱的状态计算 `safe_delete`，并返回调用方需要的结果。

        参数：
            file_path: 目标文件的路径；访问范围仍受项目沙箱约束。
            backup: 可选的`backup`。

        返回：
            `tuple[bool, str]` 类型的处理结果。

        说明：
            该操作会访问本地文件系统，路径校验和原子写入约束由所在组件负责。
        """
        if self.config.read_only:
            return False, "Sandbox is in read-only mode"

        is_allowed, reason = self.is_path_allowed(file_path)
        if not is_allowed:
            return False, reason

        path = Path(file_path)

        if not path.exists():
            return False, f"File not found: {file_path}"

        if backup and path.is_file():
            with contextlib.suppress(Exception):
                self._original_files[str(path)] = path.read_text(encoding="utf-8")

        try:
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)

            self.modifications.append(
                {
                    "type": "delete",
                    "path": str(path),
                }
            )

            return True, f"Successfully deleted {file_path}"

        except Exception as e:
            return False, f"Failed to delete: {e}"

    def create_temp_file(
        self,
        content: bytes | str = "",
        suffix: str = "",
        prefix: str = "sandbox_",
    ) -> tuple[bool, str | Path]:
        """创建`create_temp_file`并完成必要的初始化。

        参数：
            content: 需要处理、保存或分析的文本内容。
            suffix: 可选的`suffix`。
            prefix: 可选的`prefix`。

        返回：
            `tuple[bool, str | Path]` 类型的处理结果。

        说明：
            该操作会访问本地文件系统，路径校验和原子写入约束由所在组件负责。
        """
        temp_dir = Path(self.config.temp_dir or tempfile.gettempdir())

        try:
            temp_dir.mkdir(parents=True, exist_ok=True)

            fd, temp_path = tempfile.mkstemp(
                suffix=suffix,
                prefix=prefix,
                dir=str(temp_dir),
            )

            if content:
                if isinstance(content, str):
                    content = content.encode("utf-8")
                os.write(fd, content)

            os.close(fd)

            return True, Path(temp_path)

        except Exception as e:
            return False, f"Failed to create temp file: {e}"

    def rollback(self) -> list[str]:
        """处理回滚，并按照当前组件的约定返回结果。

        返回：
            按调用约定排序的结果列表。

        说明：
            该操作会访问本地文件系统，路径校验和原子写入约束由所在组件负责。
        """
        results = []

        for original_path, content in self._original_files.items():
            try:
                Path(original_path).write_text(content, encoding="utf-8")
                results.append(f"Restored: {original_path}")
            except Exception as e:
                results.append(f"Failed to restore {original_path}: {e}")

        for mod in reversed(self.modifications):
            if mod["type"] == "write":
                modified_path = Path(mod["path"])
                if str(modified_path) not in self._original_files and modified_path.exists():
                    try:
                        modified_path.unlink()
                        results.append(f"Removed new file: {modified_path}")
                    except Exception:
                        pass

        self.modifications.clear()
        self._original_files.clear()

        return results

    def get_modifications(self) -> list[dict[str, Any]]:
        """读取 `modifications` 对应的数据，不改变当前对象的业务状态。

        返回：
            按调用约定排序的结果列表。
        """
        return self.modifications.copy()

    def __repr__(self) -> str:
        return (
            f"Sandbox(working_dir={self.working_dir}, "
            f"modifications={len(self.modifications)}, "
            f"read_only={self.config.read_only})"
        )
