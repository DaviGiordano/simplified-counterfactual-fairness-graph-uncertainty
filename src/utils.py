import sys
from pathlib import Path


def find_root(marker="requirements.txt"):
    p = Path(__file__).resolve().parent
    for ancestor in (p, *p.parents):
        if (ancestor / marker).exists():
            return ancestor
    raise FileNotFoundError(f"no {marker} above {p!s}")
