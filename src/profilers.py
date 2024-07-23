from __future__ import annotations

import time
from time import perf_counter, sleep
from functools import wraps
import dis
import pstats
import subprocess
import sys
from cProfile import Profile
from functools import partial
from typing import TYPE_CHECKING, Callable, Any

import memory_profiler
import psutil

from src.config import config
from src.log import logger

if TYPE_CHECKING:
    from argparse import Namespace

    from src.fetch import SciScraper


def _kill(proc_pid: int) -> None:
    """Kill the current benchmark process with SIGKILL, pre-emptively checking whether PID has been reused.

    Args:
        proc_pid (int): The process ID of the process to be killed.

    Returns:
        None
    """
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


def run_benchmark(args: Namespace, sciscrape: SciScraper) -> None:
    """
    Run a benchmark on the given SciScraper instance by profiling its execution and printing the results.

    The benchmark is run by calling the `sciscrape` function with the given `args` and profiling its execution
    using the `Profile` context manager from the `profile` module. The resulting profile statistics are then
    sorted by time and printed, and the stats are also dumped to a file specified by `config.profiling_path`.
    Finally, the `snakeviz` module is used to visualize the profile stats in a web browser.

    Args:
        args (Namespace): An object containing the arguments for the `sciscrape` function.
        sciscrape (SciScraper): An instance of the `SciScraper` class to be benchmarked.

    Returns:
        None
    """
    with Profile() as pr:
        sciscrape(args.file)
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
    stats.dump_stats(config.profiling_path)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "snakeviz",
            config.profiling_path,
        ],
        shell=True,
    )
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        _kill(proc.pid)


@memory_profiler.profile(precision=4)  # type: ignore[misc]
def run_memory_profiler(args: Namespace, sciscrape: SciScraper) -> None:
    """Benchmark the line by line memory usage of the `sciscraper` program."""
    sciscrape(args.file)


def run_bytecode_profiler(sciscrape: SciScraper) -> None:
    """
    Reproduces the bytecode of the entire `sciscraper` program.

    Args:
        sciscrape (SciScraper): The sciscraper function to be profiled.

    Returns:
        None
    """
    dis.dis(sciscrape.__call__)


def get_profiler(args: Namespace, sciscrape: SciScraper) -> None:  # type: ignore[misc]
    """
    Get the profiler based on the given arguments and sciscraper function.

    Args:
        args (Namespace): The arguments passed to the script.
        sciscrape (SciScraper): The sciscraper function to be profiled.

    Returns:
        None
    """
    profiler_dict = {
        "benchmark": partial(run_benchmark, args=args),
        "memory": partial(run_memory_profiler, args=args),
        "bytecode": run_bytecode_profiler,
    }
    profiler_dict.get(args.profilers, sciscrape(args.file))


def get_time(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:

        # Note that timing your code once isn't the most reliable option
        # for timing your code. Look into the timeit module for more accurate
        # timing.
        start_time: float = perf_counter()
        result: Any = func(*args, **kwargs)
        end_time: float = perf_counter()

        logger.debug(
            f'"{func.__name__}()" took {end_time - start_time:.2f} seconds to execute'
        )
        return result

    return wrapper
