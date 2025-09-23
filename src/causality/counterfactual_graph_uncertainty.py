import logging
import os
from pathlib import Path

import networkx as nx
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.causal_discovery.causal_discovery import CausalDiscovery
from src.causality.causal_world import CausalWorld
from src.causality.counterfactual_estimator import CounterfactualEstimator
from src.dataset.counterfactuals_wrapper import MultiWorldCounterfactuals
from src.graph.encode import encode_graph

logger = logging.getLogger(__name__)


def generate_counterfactuals_from_worlds(
    train_data: pd.DataFrame,
    causal_worlds: list[CausalWorld],
    observed_data: pd.DataFrame,
    sensitive_name: str,
    num_workers=1,
) -> MultiWorldCounterfactuals:
    """Generates counterfactuals from observed data using the previously defined encoded graphs."""

    logger.info(f"Generating counterfactuals from worlds with {num_workers} workers..")
    counterfactuals = {}

    for world_num, causal_world in enumerate(causal_worlds):
        cf_estimator = CounterfactualEstimator(causal_world.enc_dag)
        cf_estimator.fit_causal_model(train_data.loc[causal_world.data_index])
        counterfactuals[world_num] = cf_estimator.generate_counterfactuals(
            observed_data,
            sensitive_name,
        )

    df_counterfactuals = pd.concat(counterfactuals, axis=0)
    df_counterfactuals.index.set_names(["causal_world", "individual"], inplace=True)
    return MultiWorldCounterfactuals(df_counterfactuals)
