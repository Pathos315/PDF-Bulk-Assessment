"""A robust behavioral science paper analyzer and downloader.

By John Fallot <john.fallot@gmail.com>

Simple usage example: TODO ->

...
"""

__version__ = "1.00"

__copyright__ = """
Copyright (c) 2021- John Fallot.  All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
from typing import Optional, Sequence
from time import perf_counter
from dotenv import load_dotenv

from sciscrape.log import logger
from sciscrape.factories import read_factory
from sciscrape.config import config


def main(argv: Optional[Sequence[str]] = None) -> None:
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

    load_dotenv()

    parser = argparse.ArgumentParser(
        prog=config.prog,
        usage="%(prog)s [options] filepath",
        description=config.description,
    )
    parser.add_argument(
        "-f",
        "--file",
        metavar="FILE",
        type=str,
        default=config.source_file,
        help="Specify the target file: default: %(default)s)",
    )
    parser.add_argument(
        "-d",
        "--debug",
        default=True,
        action="store_true",
        help="Specify debug logging output: default: %(default)s)",
    )
    parser.add_argument(
        "-e",
        "--export",
        default=True,
        action="store_true",
        help="Specify if exporting dataframe\
            to .csv: default: %(default)s)",
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        default=False,
        action="store_true",
        help="Specify if benchmarking is to be conducted: default: %(default)s)",
    )
    parser.add_argument(
        "-m",
        "--memory",
        default=False,
        action="store_true",
        help="Specify if memory profiling is to be conducted: default: %(default)s)",
    )
    args = parser.parse_args(argv)

    start = perf_counter()
    sciscrape = read_factory()
    logger.debug(repr(sciscrape))
    logger.debug(repr(args.file))

    sciscrape(args.file, args.export, args.debug)

    elapsed = perf_counter() - start
    logger.info(f"Extraction finished in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()