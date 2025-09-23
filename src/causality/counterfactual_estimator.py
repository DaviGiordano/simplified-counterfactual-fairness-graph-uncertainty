import logging
from typing import Callable, Dict, Optional

import dowhy.gcm as gcm
import networkx as nx
import pandas as pd
from dowhy.gcm import (
    AdditiveNoiseModel,
    EmpiricalDistribution,
    InvertibleStructuralCausalModel,
)
from dowhy.graph import is_root_node
from scipy.stats import norm

logger = logging.getLogger(__name__)


class CounterfactualEstimator:
    def __init__(
        self,
        encoded_graph: nx.DiGraph,
        mechanism_factories: Optional[Dict[str, Callable]] = None,
    ):
        self.encoded_graph = encoded_graph
        self.mechanism_factories = mechanism_factories or {}
        self.causal_model = InvertibleStructuralCausalModel(encoded_graph)  # type: ignore

    def fit_causal_model(self, train_data: pd.DataFrame) -> None:
        """Sets mechanisms and fits the causal model"""
        self._set_causal_mechanisms(self.causal_model, self.mechanism_factories)
        self._fit_causal_mechanisms(self.causal_model, train_data)

    def evaluate_causal_model(self, test_data: pd.DataFrame) -> str:
        """Evaluates causal model against test data"""
        nodes = set(self.causal_model.graph.nodes)  # type: ignore
        test_cols = set(test_data.columns)
        missing = nodes - test_cols
        if missing:
            raise ValueError(f"Missing columns in test data: {sorted(missing)}")

        subset = test_data.loc[:, sorted(nodes)]
        if subset.empty:
            raise ValueError("test_encoded_data empty")

        return str(
            gcm.evaluate_causal_model(
                self.causal_model,
                subset,
            )
        )

    def generate_counterfactuals(
        self,
        observed_data: pd.DataFrame,
        sensitive_name: str,
    ) -> pd.DataFrame:
        """Generate counterfactual samples for a protected attribute."""
        logger.info(f"Generating counterfactuals for '{sensitive_name}'..")

        cf_int = {sensitive_name: lambda x: 1 - x}

        df_cf = gcm.counterfactual_samples(
            self.causal_model,
            cf_int,
            observed_data=observed_data,
        )

        cols = observed_data.columns
        idx = observed_data.index

        df_cf.set_index(idx, inplace=True)
        df_cf = df_cf[cols]

        return df_cf

    @staticmethod
    def _set_causal_mechanisms(
        causal_model: InvertibleStructuralCausalModel,
        mechanism_factories: Dict[str, Callable],
    ) -> None:
        """Sets causal mechanisms for nodes"""
        logger.info("Setting causal mechanisms..")
        for node in causal_model.graph.nodes:  # type: ignore
            if is_root_node(causal_model.graph, node):  # type: ignore
                mech = mechanism_factories.get("root", EmpiricalDistribution)()
                causal_model.set_causal_mechanism(node, mech)
            else:
                factory = mechanism_factories.get(
                    "non_root",
                    lambda: AdditiveNoiseModel(
                        prediction_model=gcm.ml.create_linear_regressor(),
                        noise_model=gcm.ScipyDistribution(norm),
                    ),
                )
                mech = factory()
                causal_model.set_causal_mechanism(node, mech)

    @staticmethod
    def _fit_causal_mechanisms(
        causal_model: InvertibleStructuralCausalModel,
        train_data: pd.DataFrame,
    ) -> None:
        """Fits causal mechanisms"""
        logger.info("Fitting causal mechanisms..")

        nodes = set(causal_model.graph.nodes)  # type: ignore
        all_train_cols = set(train_data.columns)

        missing = nodes - all_train_cols
        if missing:
            raise ValueError(f"Missing columns in train data: {sorted(missing)}")

        subset = train_data.loc[:, sorted(nodes)]
        if subset.empty:
            raise ValueError("train_encoded_data empty")

        gcm.fit(causal_model, subset)
