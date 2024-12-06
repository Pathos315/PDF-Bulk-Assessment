"""change_dir.py governs the file creation
processes for the sciscraper program.
"""

from __future__ import annotations

from contextlib import contextmanager
from os import chdir, getcwd, makedirs, path
from typing import TYPE_CHECKING

from src.config import FilePath

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def change_dir(destination: FilePath) -> Iterator[None]:
    """Sets a destination for exported files."""
    cwd = getcwd()
    try:
        dest = path.realpath(destination)
        makedirs(dest, exist_ok=True)
        chdir(dest)
        yield
    finally:
        chdir(cwd)
