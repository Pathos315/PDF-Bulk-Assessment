"""A robust behavioral science paper analyzer and downloader.

By John Fallot <john.fallot@gmail.com>
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from src.argsbuilder import build_parser
from src.factories import SCISCRAPERS, read_factory
from src.log import logger
from src.profilers import get_profiler, get_time

if TYPE_CHECKING:
    from collections.abc import Sequence


@get_time
def main(argv: Sequence[str] | None = None) -> None:
    """
    Entry point for the `sciscraper` application. Parses command line arguments and executes the appropriate actions.

    The function uses the `argparse` module to parse command line arguments passed in the `argv` parameter.
    The parsed arguments are used to determine the actions to be taken, which may include running the `sciscrape`
    function with the specified file and export options, running a benchmark on the `sciscrape` function, or
    conducting a memory profile of the `sciscrape` function.

    Parameters
    ---------
    argv:
    A sequence of strings representing the command line arguments. If not provided, the default value is
    `None`, in which case the `argv` list is constructed from `sys.argv`.

    Returns
    -------
    None
    """
    args = build_parser(argv)

    sciscrape = read_factory() if args.mode is None else SCISCRAPERS[args.mode]
    logger.debug(repr(args.file))
    get_profiler(args, sciscrape)


if __name__ == "__main__":
    main()
