"""Compatibility logger: prefer loguru when available, otherwise provide
a thin stdlib-logging-compatible facade exposing a `logger` object.

Use by importing: `from src.logging_fallback import logger`
"""
from __future__ import annotations

try:
    from loguru import logger  # type: ignore
except Exception:
    import logging
    from typing import Any

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _std = logging.getLogger("acord_extractor")

    class _ProxyLogger:
        def debug(self, *args: Any, **kwargs: Any) -> None:
            _std.debug(*args, **kwargs)

        def info(self, *args: Any, **kwargs: Any) -> None:
            _std.info(*args, **kwargs)

        def warning(self, *args: Any, **kwargs: Any) -> None:
            _std.warning(*args, **kwargs)

        def error(self, *args: Any, **kwargs: Any) -> None:
            _std.error(*args, **kwargs)

        def exception(self, *args: Any, **kwargs: Any) -> None:
            _std.exception(*args, **kwargs)

        def critical(self, *args: Any, **kwargs: Any) -> None:
            _std.critical(*args, **kwargs)

        # loguru-specific convenience methods
        def success(self, *args: Any, **kwargs: Any) -> None:
            _std.info(*args, **kwargs)

        def trace(self, *args: Any, **kwargs: Any) -> None:
            _std.debug(*args, **kwargs)

    logger = _ProxyLogger()
