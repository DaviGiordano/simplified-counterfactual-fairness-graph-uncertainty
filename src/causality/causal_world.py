from dataclasses import dataclass

import networkx as nx
import pandas as pd
import pydot

from src.graph.inspection import (
    compute_edgewise_entropy,
    count_subdags_from_root,
    get_edge_frequencies,
    get_subdag_from_root,
    get_unique_cpdags_count,
    get_unique_dags_count,
)


@dataclass(frozen=True)
class CausalWorld:
    """Wrapper for causal world with one dag, its encoded version, its original cpdag and the original data_index"""

    dag: nx.DiGraph
    enc_dag: nx.DiGraph
    cpdag: pydot.Dot
    data_index: pd.Index


def inspect_graph_uncertainty(cws: list[CausalWorld], sensitive_feat):
    """Returns a string with an analysis of graph uncertainty"""
    dags = [cw.dag for cw in cws]
    cpdags = [cw.cpdag for cw in cws]
    dag_cnt = count_subdags_from_root(dags, sensitive_feat)
    subdags = [get_subdag_from_root(dag, sensitive_feat) for dag in dags]

    result = []
    result.append(f"Number of unique cpdags: {get_unique_cpdags_count(cpdags)}\n")
    result.append(f"Number of causal worlds: {len(cws)}\n")
    result.append(
        f"Edgewise entropy across causal worlds: {compute_edgewise_entropy(dags)}\n"
    )
    result.append(
        f"Edgewise entropy across subgraphs with root {sensitive_feat}: {compute_edgewise_entropy(subdags)}\n"
    )
    result.append(str(get_unique_dags_count(dags)))
    result.append(f"\n{get_edge_frequencies(dags)}")
    result.append(f"\nDistinct subdags starting from {sensitive_feat}:")
    for k, v in dict(dag_cnt).items():
        result.append(f"{v} time(s): {k}")

    return "\n".join(result)
