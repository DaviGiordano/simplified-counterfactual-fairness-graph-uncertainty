#!/usr/bin/env python3
import argparse
import pickle
from typing import Any, List, Tuple


def walk_for_entropy(obj: Any, path: str, results: List[Tuple[str, str, str, str]], max_values: int = 3) -> None:
    """Recursively walk an object and collect entries whose keys look like entropy-related.

    We collect tuples of (parent_path, key, value_type, value_preview).
    """
    try:
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_str = str(key)
                if any(tok in key_str.lower() for tok in ("entropy", "subgraph")):
                    preview = repr(value)
                    if isinstance(value, (list, tuple)):
                        preview = repr(value[:max_values])
                    elif isinstance(value, dict):
                        preview = repr(list(value.keys())[:max_values])
                    results.append((path, key_str, type(value).__name__, preview))
                walk_for_entropy(value, f"{path}.{key_str}", results, max_values)
        elif isinstance(obj, (list, tuple)):
            for idx, item in enumerate(obj):
                walk_for_entropy(item, f"{path}[{idx}]", results, max_values)
        # For other atomic types we do nothing
    except Exception:
        # Be permissive in traversal; skip problematic branches
        return


def main() -> None:

if __name__ == "__main__":
    main()


