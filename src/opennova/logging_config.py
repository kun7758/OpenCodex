"""OpenNova 日志配置模块，提供统一的日志初始化和格式化功能。"""

import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# 默认日志格式
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
CONSOLE_FORMAT = "%(asctime)s | %(levelname)-5s | %(message)s"

# 默认配置
DEFAULT_LOGGING_CONFIG: dict[str, Any] = {
    "enabled": True,
    "level": "DEBUG",
    "console_level": "INFO",
    "file_level": "DEBUG",
    "log_dir": "~/.opennova/logs",
    "max_file_size_mb": 10,
    "backup_count": 5,
    "format": DEFAULT_FORMAT,
}


def get_log_directory(log_dir: str | None = None) -> Path:
    """获取日志目录路径，不存在时自动创建。

    参数：
        log_dir: 可选的日志目录路径，为 None 时使用默认路径。

    返回：
        日志目录的 Path 对象。
    """
    log_path = Path(log_dir) if log_dir else Path.home() / ".opennova" / "logs"
    log_path = log_path.expanduser().resolve()
    log_path.mkdir(parents=True, exist_ok=True)
    return log_path


def generate_log_filename() -> str:
    """生成日志文件名，格式为 opennova_YYYYMMDD_HHMMSS.log。

    返回：
        日志文件名字符串。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"opennova_{timestamp}.log"


def setup_logging(config: dict[str, Any] | None = None) -> logging.Logger:
    """初始化 OpenNova 日志系统。

    参数：
        config: 日志配置字典，为 None 时使用默认配置。

    返回：
        根日志记录器。
    """
    # 合并配置
    log_config = DEFAULT_LOGGING_CONFIG.copy()
    if config:
        log_config.update(config)

    # 检查是否启用
    if not log_config.get("enabled", True):
        logging.disable(logging.CRITICAL)
        return logging.getLogger("opennova")

    # 获取根日志记录器
    root_logger = logging.getLogger("opennova")
    root_logger.setLevel(logging.DEBUG)

    # 清除现有处理器，避免重复
    root_logger.handlers.clear()

    # 日志格式
    log_format = log_config.get("format", DEFAULT_FORMAT)
    formatter = logging.Formatter(log_format)

    # 控制台处理器
    console_level = getattr(logging, log_config.get("console_level", "INFO").upper(), logging.INFO)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter(CONSOLE_FORMAT)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # 文件处理器
    file_level = getattr(logging, log_config.get("file_level", "DEBUG").upper(), logging.DEBUG)
    log_dir = get_log_directory(log_config.get("log_dir"))
    log_file = log_dir / generate_log_filename()

    # 将 MB 转换为字节
    max_file_size_mb = log_config.get("max_file_size_mb", 10)
    max_bytes = max_file_size_mb * 1024 * 1024
    backup_count = log_config.get("backup_count", 5)

    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 记录初始化信息
    root_logger.info("Logging initialized: file=%s, level=%s", log_file, log_config.get("level"))
    root_logger.debug("Log config: %s", log_config)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的日志记录器。

    参数：
        name: 日志记录器名称，通常使用 __name__。

    返回：
        日志记录器实例。
    """
    return logging.getLogger(f"opennova.{name}")
