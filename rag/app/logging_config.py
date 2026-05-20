"""应用日志：同时输出控制台与可选日志文件；通过 LOG_LEVEL / LOG_FILE 配置。"""
import logging
import sys
from pathlib import Path
from typing import Final

LOGGER_NAME: Final[str] = "rag"


def get_logger(modname: str = __name__) -> logging.Logger:
    """命名空间与包一致：例如 ``app.main`` → ``rag.main``，便于检索。"""
    if modname.startswith("app."):
        canon = LOGGER_NAME + modname[3:]
    elif modname.startswith(LOGGER_NAME):
        canon = modname
    else:
        canon = f"{LOGGER_NAME}.{modname}"
    return logging.getLogger(canon)


def configure_logging(
    level_str: str,
    log_file: Path | None,
    *,
    encoding: str = "utf-8",
) -> None:
    """配置 ``rag`` 根日志器（子 logger 逐级冒泡）；重复调用会先清空既有 handler。"""
    level = getattr(logging, level_str.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger(LOGGER_NAME)
    root.setLevel(level)
    root.handlers.clear()
    root.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding=encoding)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
