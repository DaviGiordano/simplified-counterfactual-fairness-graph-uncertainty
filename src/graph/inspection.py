import math
from collections import Counter
from typing import Union

import networkx as nx
import pydot


def get_edge_frequencies(dags: list[nx.DiGraph]) -> str:
    """From a list of dags, count the frequency of each edge then return as string.

    Returns:
        str: Formatted string containing edge frequencies
    """
    edge_counts = {}
    total_dags = len(dags)
    result = []

    # Count edges across all DAGs
    for dag in dags:
        for edge in dag.edges():
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    # Build result string
    result.append(f"Edge frequencies across {total_dags} DAGs:")
    for edge, count in sorted(edge_counts.items()):
        percentage = (count / total_dags) * 100
        result.append(f"{count} times ({percentage:.1f}%): {edge} ")

    return "\n".join(result)


def get_unique_dags_count(dags: list[nx.DiGraph]) -> str:
    """Count unique DAGs in a list and return result as string.

    Args:
        dags: A list of NetworkX DiGraph objects representing DAGs

    Returns:
        str: Formatted string containing unique DAGs count
    """
    # Convert each DAG to a hashable representation (frozen set of edges)
    unique_dags = set()
    dag_to_count = {}  # Dictionary to keep track of each unique DAG's frequency

    for dag in dags:
        # Convert edges to a frozenset of tuples which is hashable
        edge_set = frozenset(dag.edges())
        unique_dags.add(edge_set)
        dag_to_count[edge_set] = dag_to_count.get(edge_set, 0) + 1

    # Build result string
    total_dags = len(dags)
    unique_count = len(unique_dags)
    result = [f"Found {unique_count} unique DAGs out of {total_dags} total DAGs"]

    # Add details for each unique DAG
    for i, unique_dag in enumerate(sorted(unique_dags), 1):
        count = dag_to_count[unique_dag]
        percentage = (count / total_dags) * 100
        result.append(
            f"DAG {i} ({count} times, {percentage:.1f}%):  {list(unique_dag)} "
        )

    return "\n".join(result)


def get_unique_cpdags_count(cpdags: list[pydot.Dot]) -> int:
    """Count unique CPDAGs from a list of pydot graphs.

    Since CPDAGs can contain both directed and undirected edges, we need a
    special representation that preserves this information when checking
    for uniqueness.

    Args:
        cpdags: A list of pydot.Dot objects representing CPDAGs

    Returns:
        int: Number of unique CPDAGs in the list
    """
    unique_cpdags = set()

    for cpdag in cpdags:
        # Create sets for directed and undirected edges
        directed_edges = set()
        undirected_edges = set()

        for edge in cpdag.get_edges():
            source = edge.get_source().strip('"')
            dest = edge.get_destination().strip('"')

            # Check if the edge is undirected
            attrs = edge.get_attributes()
            is_undirected = attrs.get("dir") == "none" or (
                attrs.get("arrowhead") == "none" and attrs.get("arrowtail") == "none"
            )

            if is_undirected:
                # Store undirected edges as frozen sets to make order irrelevant
                undirected_edges.add(frozenset([source, dest]))
            else:
                directed_edges.add((source, dest))

        # Create a hashable representation containing both edge types
        hashable_cpdag = (frozenset(directed_edges), frozenset(undirected_edges))
        unique_cpdags.add(hashable_cpdag)

    return len(unique_cpdags)


def get_subdag_from_root(dag: nx.DiGraph, root) -> nx.DiGraph:
    """Given a dag and one of its nodes, return the subdag where the node is root"""
    reachable = {root} | nx.descendants(dag, root)
    subdag = dag.subgraph(reachable).copy()
    return subdag


def count_subdags_from_root(dags: list[nx.DiGraph], root) -> Counter:
    """Get counter of subdags from each dag in dags, starting on root"""

    subdags = []
    for dag in dags:
        reachable = {root} | nx.descendants(dag, root)
        subdag = dag.subgraph(reachable).copy()
        edges = tuple(sorted(subdag.edges()))

        subdags.append(edges)

    return Counter(subdags)


def compute_edgewise_entropy(dags: list[nx.DiGraph], normalize: bool = True) -> float:
    """
    Given a list of DAGs, compute the Shannon entropy of the graph based on edge inclusion frequencies.
    If normalize=True, returns H_norm in [0,1], else returns raw entropy (nats).

    Returns:
        float: normalized entropy H_norm = H / (n_edges * ln2)
    """
    B = len(dags)
    edge_counts = Counter(edge for dag in dags for edge in dag.edges())

    # raw entropy (nats)
    H = 0.0
    for count in edge_counts.values():
        p = count / B
        if 0 < p < 1:
            H += -p * math.log(p) - (1 - p) * math.log(1 - p)

    if normalize:
        n_edges = len(edge_counts)
        return H / (n_edges * math.log(2)) if n_edges > 0 else 0.0
    else:
        return H
