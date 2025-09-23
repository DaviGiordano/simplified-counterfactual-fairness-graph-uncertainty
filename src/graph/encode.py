from typing import Dict, Tuple

import networkx as nx
from dowhy.gcm import InvertibleStructuralCausalModel
from sklearn.compose import ColumnTransformer


def encode_graph(
    raw_graph: nx.DiGraph,
    column_transformer: ColumnTransformer,
) -> nx.DiGraph:
    """Encodes raw graph using column transformer."""
    all_encoded = column_transformer.get_feature_names_out()
    transformers = {name: feats for name, _, feats in column_transformer.transformers_}
    cat = transformers.get("onehot", [])
    num = transformers.get("scale", [])
    unsupported = set(transformers) - {"onehot", "scale"}
    if unsupported:
        raise ValueError(f"Unsupported transformers: {sorted(unsupported)}")

    mapping_initial = {
        **{f: [c for c in all_encoded if c.startswith(f"onehot__{f}_")] for f in cat},
        **{f: [f"scale__{f}"] for f in num},
    }

    raw_feats = set(raw_graph.nodes)
    mapping = {feat: mapping_initial.get(feat, [feat]) for feat in raw_feats}

    encoded_graph = nx.DiGraph()

    # Add nodes and edges
    for encoded_nodes in mapping.values():
        encoded_graph.add_nodes_from(encoded_nodes)

    for src, tgt in raw_graph.edges():
        for se in mapping[src]:
            for te in mapping[tgt]:
                encoded_graph.add_edge(se, te)

    return encoded_graph


def encode_graphs(
    raw_graphs: list[nx.DiGraph],
    column_transformer: ColumnTransformer,
):
    """Helper to encode a list of raw graphs with a column transformer"""
    encoded_graphs = []

    for raw_graph in raw_graphs:
        encoded_graph = encode_graph(
            raw_graph,
            column_transformer,
        )
        encoded_graphs.append(encoded_graph)

    return encoded_graphs
