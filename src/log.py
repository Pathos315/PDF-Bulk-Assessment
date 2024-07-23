from __future__ import annotations

import json
import logging
import logging.config

from pathlib import Path
from typing import Callable, TypeVar


T = TypeVar("T")

logger = logging.getLogger("sciscraper")

logging_configs = Path("logging_config.json").resolve()
with open(logging_configs) as f_in:
    log_config = json.load(f_in)
    logging.config.dictConfig(log_config)


def log_debug(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to log debug information for a function."""

    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        logger.debug("function=%s, result=%s", func.__name__, result)
        return result

    return wrapper
