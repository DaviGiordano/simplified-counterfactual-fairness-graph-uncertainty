import sys
import time
from contextlib import contextmanager
from pathlib import Path


def find_root(marker="requirements.txt"):
    p = Path(__file__).resolve().parent
    for ancestor in (p, *p.parents):
        if (ancestor / marker).exists():
            return ancestor
    raise FileNotFoundError(f"no {marker} above {p!s}")


@contextmanager
def record_time(key: str, times_dict: dict):
    """Context manager to record execution time of code blocks."""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        times_dict[key] = end_time - start_time
