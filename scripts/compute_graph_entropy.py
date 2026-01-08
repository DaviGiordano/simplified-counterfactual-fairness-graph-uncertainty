#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import networkx as nx

from src.causality.causal_world import CausalWorld
from src.graph.inspection import compute_edgewise_entropy, get_subdag_from_root


def load_causal_worlds(pkl_path: Path) -> list[CausalWorld]:
    # Numpy BitGenerator compatibility shim for cross-version pickles
    try:
        import numpy.random._pickle as _np_pickle  # type: ignore
        from numpy.random import MT19937 as _MT19937  # type: ignore

        bitgens = getattr(_np_pickle, "BitGenerators", None)
        if isinstance(bitgens, dict):
            bitgens.setdefault("MT19937", _MT19937)
            bitgens.setdefault("numpy.random._mt19937.MT19937", _MT19937)
            bitgens.setdefault("numpy.random.bit_generator.MT19937", _MT19937)
            bitgens.setdefault("numpy.random._bit_generator.MT19937", _MT19937)

        # Wrap __bit_generator_ctor to accept class objects by name
        _orig_ctor = getattr(_np_pickle, "__bit_generator_ctor", None)
        if callable(_orig_ctor):

            def _shim_ctor(bit_generator_name, *args, **kwargs):  # type: ignore
                if not isinstance(bit_generator_name, str) and hasattr(
                    bit_generator_name, "__name__"
                ):
                    bit_generator_name = bit_generator_name.__name__
                return _orig_ctor(bit_generator_name, *args, **kwargs)

            _np_pickle.__bit_generator_ctor = _shim_ctor  # type: ignore
    except Exception:
        pass

    with open(pkl_path, "rb") as f:
        worlds: list[CausalWorld] = pickle.load(f)
    return worlds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute graph entropy from saved causal worlds."
    )
    parser.add_argument("pickle_path", type=Path, help="Path to causal_worlds.pkl")
    parser.add_argument(
        "--sensitive", default="sex", help="Sensitive/root node name (default: sex)"
    )
    args = parser.parse_args()

    causal_worlds = load_causal_worlds(args.pickle_path)
    dags: list[nx.DiGraph] = [cw.dag for cw in causal_worlds]
    subdags: list[nx.DiGraph] = [get_subdag_from_root(d, args.sensitive) for d in dags]

    H_edges = compute_edgewise_entropy(dags)
    H_sub = compute_edgewise_entropy(subdags)

    print(f"Loaded causal worlds: {len(causal_worlds)} from {args.pickle_path}")
    print(f"Sensitive/root node: {args.sensitive}")
    print(f"Edgewise entropy across DAGs: {H_edges}")
    print(f"Edgewise entropy across subgraphs rooted at {args.sensitive}: {H_sub}")


if __name__ == "__main__":
    main()
