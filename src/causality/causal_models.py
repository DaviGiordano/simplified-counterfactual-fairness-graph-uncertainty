"""
Enhanced causal model implementations for linear, LGBM, diffusion, and causalflow.
Includes counterfactual quality evaluation.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import dowhy.gcm as gcm
import networkx as nx
import pandas as pd
from dowhy.gcm import InvertibleStructuralCausalModel
from scipy.stats import norm

logger = logging.getLogger(__name__)


class CausalModelFitter:
    """Base class for causal model fitting with counterfactual quality evaluation."""

    def __init__(self, graph: nx.DiGraph, model_type: str, **kwargs):
        self.graph = graph
        self.model_type = model_type
        self.kwargs = kwargs
        self.model = None

    def fit(self, train_data: pd.DataFrame) -> Dict[str, Any]:
        """Fit the causal model and return performance metrics."""
        raise NotImplementedError

    def generate_counterfactuals(
        self, test_data: pd.DataFrame, sensitive_name: str
    ) -> pd.DataFrame:
        """Generate counterfactuals for the sensitive attribute."""
        raise NotImplementedError

    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate the fitted model."""
        raise NotImplementedError

    def evaluate_counterfactual_quality(
        self, cf_data: pd.DataFrame, test_data: pd.DataFrame, sensitive_name: str
    ) -> Dict[str, float]:
        """Evaluate counterfactual quality metrics."""
        logger.info("Starting counterfactual quality evaluation...")
        logger.info(
            f"CF data shape: {cf_data.shape}, Test data shape: {test_data.shape}"
        )

        from src.metrics.counterfactual_metrics import evaluate_counterfactual_world

        logger.info("Calling evaluate_counterfactual_world...")
        result = evaluate_counterfactual_world(
            df_observed=test_data,
            df_counterfactuals=cf_data,
            sensitive_feature=test_data[sensitive_name],
        )
        logger.info("Counterfactual quality evaluation completed.")

        return result


class LinearCausalModel(CausalModelFitter):
    """Linear causal model using DoWhy GCM with linear regression."""

    def __init__(self, graph: nx.DiGraph, **kwargs):
        super().__init__(graph, "linear", **kwargs)
        self.scm = InvertibleStructuralCausalModel(graph)

    def fit(self, train_data: pd.DataFrame) -> Dict[str, Any]:
        """Fit linear causal mechanisms."""
        logger.info("Fitting linear causal mechanisms...")

        # Set linear mechanisms for non-root nodes
        for node in self.graph.nodes:
            if self.graph.in_degree(node) == 0:
                # Root nodes use empirical distribution
                mechanism = gcm.EmpiricalDistribution()
            else:
                # Non-root nodes use linear regression
                mechanism = gcm.AdditiveNoiseModel(
                    prediction_model=gcm.ml.create_linear_regressor(),
                    noise_model=gcm.ScipyDistribution(norm),
                )
            self.scm.set_causal_mechanism(node, mechanism)

        # Fit the model
        gcm.fit(self.scm, train_data)

        # Evaluate performance with minimal evaluation for speed
        # Use a subset of data for faster evaluation
        eval_data = train_data.sample(n=min(1000, len(train_data)), random_state=42)
        evaluation = gcm.evaluate_causal_model(
            self.scm,
            eval_data,
            evaluate_causal_mechanisms=True,  # Keep this for basic metrics
            compare_mechanism_baselines=False,  # Skip baseline comparison
            evaluate_invertibility_assumptions=False,  # Skip invertibility tests
            evaluate_overall_kl_divergence=True,  # Keep KL divergence
            evaluate_causal_structure=False,  # Skip structure evaluation
        )

        return {
            "model_type": "linear",
            "evaluation": str(evaluation),
            "mean_mse": _extract_mean_mse(evaluation),
            "overall_kl": _extract_overall_kl(evaluation),
        }

    def generate_counterfactuals(
        self, test_data: pd.DataFrame, sensitive_name: str
    ) -> pd.DataFrame:
        """Generate counterfactuals using linear model."""
        cf_intervention = {sensitive_name: lambda x: 1 - x}
        df_cf = gcm.counterfactual_samples(
            self.scm, cf_intervention, observed_data=test_data
        )
        df_cf.index = test_data.index.copy()
        return df_cf[test_data.columns]

    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate linear model performance."""
        evaluation = gcm.evaluate_causal_model(self.scm, test_data)
        return {
            "mean_mse": _extract_mean_mse(evaluation),
            "overall_kl": _extract_overall_kl(evaluation),
        }


class LGBMCausalModel(CausalModelFitter):
    """LGBM-based causal model using DoWhy GCM."""

    def __init__(self, graph: nx.DiGraph, **kwargs):
        super().__init__(graph, "lgbm", **kwargs)
        self.scm = InvertibleStructuralCausalModel(graph)

    def fit(self, train_data: pd.DataFrame) -> Dict[str, Any]:
        """Fit LGBM causal mechanisms."""
        logger.info("Fitting LGBM causal mechanisms...")
        logger.info(f"Setting mechanisms for {len(self.graph.nodes)} nodes...")

        # Set LGBM mechanisms for non-root nodes
        for i, node in enumerate(self.graph.nodes):
            logger.info(
                f"Setting mechanism for node {i+1}/{len(self.graph.nodes)}: {node}"
            )
            if self.graph.in_degree(node) == 0:
                mechanism = gcm.EmpiricalDistribution()
                logger.info(f"  -> Root node, using EmpiricalDistribution")
            else:
                mechanism = gcm.AdditiveNoiseModel(
                    prediction_model=gcm.ml.create_hist_gradient_boost_regressor()
                )
                logger.info(f"  -> Non-root node, using LGBM AdditiveNoiseModel")
            self.scm.set_causal_mechanism(node, mechanism)

        logger.info("All mechanisms set. Starting GCM fitting...")
        gcm.fit(self.scm, train_data)
        logger.info("GCM fitting completed. Starting minimal evaluation...")

        # Use minimal evaluation for faster execution
        # Use a subset of data for faster evaluation
        eval_data = train_data.sample(n=min(1000, len(train_data)), random_state=42)
        evaluation = gcm.evaluate_causal_model(
            self.scm,
            eval_data,
            evaluate_causal_mechanisms=True,  # Keep this for basic metrics
            compare_mechanism_baselines=False,  # Skip baseline comparison
            evaluate_invertibility_assumptions=False,  # Skip invertibility tests
            evaluate_overall_kl_divergence=True,  # Keep KL divergence
            evaluate_causal_structure=False,  # Skip structure evaluation
        )
        logger.info("Minimal evaluation completed.")

        return {
            "model_type": "lgbm",
            "evaluation": str(evaluation),
            "mean_mse": _extract_mean_mse(evaluation),
            "overall_kl": _extract_overall_kl(evaluation),
        }

    def generate_counterfactuals(
        self, test_data: pd.DataFrame, sensitive_name: str
    ) -> pd.DataFrame:
        """Generate counterfactuals using LGBM model."""
        logger.info(
            f"Generating counterfactuals for sensitive attribute: {sensitive_name}"
        )
        logger.info(f"Test data shape: {test_data.shape}")

        cf_intervention = {sensitive_name: lambda x: 1 - x}
        logger.info("Starting counterfactual sampling...")
        df_cf = gcm.counterfactual_samples(
            self.scm, cf_intervention, observed_data=test_data
        )
        df_cf.index = test_data.index.copy()

        logger.info("Counterfactual sampling completed.")

        return df_cf[test_data.columns]

    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate LGBM model performance."""
        logger.info("Starting model evaluation on test data...")
        evaluation = gcm.evaluate_causal_model(self.scm, test_data)
        logger.info("Model evaluation completed.")
        return {
            "mean_mse": _extract_mean_mse(evaluation),
            "overall_kl": _extract_overall_kl(evaluation),
        }


class DiffusionCausalModel(CausalModelFitter):
    """Diffusion-based causal model."""

    def __init__(self, graph: nx.DiGraph, **kwargs):
        super().__init__(graph, "diffusion", **kwargs)
        self.scm = InvertibleStructuralCausalModel(graph)

    def fit(self, train_data: pd.DataFrame) -> Dict[str, Any]:
        """Fit diffusion causal mechanisms."""
        logger.info("Fitting diffusion causal mechanisms...")

        # For now, use linear mechanisms as placeholder for diffusion
        # TODO: Implement actual diffusion mechanisms when external library is available
        for node in self.graph.nodes:
            if self.graph.in_degree(node) == 0:
                mechanism = gcm.EmpiricalDistribution()
            else:
                # Placeholder: use linear regression until diffusion is implemented
                mechanism = gcm.AdditiveNoiseModel(
                    prediction_model=gcm.ml.create_linear_regressor(),
                    noise_model=gcm.ScipyDistribution(norm),
                )
            self.scm.set_causal_mechanism(node, mechanism)

        gcm.fit(self.scm, train_data)
        # Use minimal evaluation for faster execution
        # Use a subset of data for faster evaluation
        eval_data = train_data.sample(n=min(1000, len(train_data)), random_state=42)
        evaluation = gcm.evaluate_causal_model(
            self.scm,
            eval_data,
            evaluate_causal_mechanisms=True,  # Keep this for basic metrics
            compare_mechanism_baselines=False,  # Skip baseline comparison
            evaluate_invertibility_assumptions=False,  # Skip invertibility tests
            evaluate_overall_kl_divergence=True,  # Keep KL divergence
            evaluate_causal_structure=False,  # Skip structure evaluation
        )

        return {
            "model_type": "diffusion",
            "evaluation": str(evaluation),
            "mean_mse": _extract_mean_mse(evaluation),
            "overall_kl": _extract_overall_kl(evaluation),
        }

    def generate_counterfactuals(
        self, test_data: pd.DataFrame, sensitive_name: str
    ) -> pd.DataFrame:
        """Generate counterfactuals using diffusion model."""
        cf_intervention = {sensitive_name: lambda x: 1 - x}
        df_cf = gcm.counterfactual_samples(
            self.scm, cf_intervention, observed_data=test_data
        )
        df_cf.index = test_data.index.copy()

        return df_cf[test_data.columns]


class CausalFlowModel(CausalModelFitter):
    """CausalFlow (CausalMAF) based causal model."""

    def __init__(self, graph: nx.DiGraph, **kwargs):
        super().__init__(graph, "causalflow", **kwargs)
        self.flow_model = None

    def fit(self, train_data: pd.DataFrame) -> Dict[str, Any]:
        """Fit CausalFlow model."""
        logger.info("Fitting CausalFlow model...")

        # For now, use linear mechanisms as placeholder for CausalFlow
        # TODO: Implement actual CausalFlow when external library is available
        self.scm = InvertibleStructuralCausalModel(self.graph)

        for node in self.graph.nodes:
            if self.graph.in_degree(node) == 0:
                mechanism = gcm.EmpiricalDistribution()
            else:
                # Placeholder: use linear regression until CausalFlow is implemented
                mechanism = gcm.AdditiveNoiseModel(
                    prediction_model=gcm.ml.create_linear_regressor(),
                    noise_model=gcm.ScipyDistribution(norm),
                )
            self.scm.set_causal_mechanism(node, mechanism)

        gcm.fit(self.scm, train_data)
        # Use minimal evaluation for faster execution
        # Use a subset of data for faster evaluation
        eval_data = train_data.sample(n=min(1000, len(train_data)), random_state=42)
        evaluation = gcm.evaluate_causal_model(
            self.scm,
            eval_data,
            evaluate_causal_mechanisms=True,  # Keep this for basic metrics
            compare_mechanism_baselines=False,  # Skip baseline comparison
            evaluate_invertibility_assumptions=False,  # Skip invertibility tests
            evaluate_overall_kl_divergence=True,  # Keep KL divergence
            evaluate_causal_structure=False,  # Skip structure evaluation
        )

        return {
            "model_type": "causalflow",
            "evaluation": str(evaluation),
            "mean_mse": _extract_mean_mse(evaluation),
            "overall_kl": _extract_overall_kl(evaluation),
        }

    def generate_counterfactuals(
        self, test_data: pd.DataFrame, sensitive_name: str
    ) -> pd.DataFrame:
        """Generate counterfactuals using CausalFlow model."""
        cf_intervention = {sensitive_name: lambda x: 1 - x}
        df_cf = gcm.counterfactual_samples(
            self.scm, cf_intervention, observed_data=test_data
        )
        df_cf.index = test_data.index.copy()

        return df_cf[test_data.columns]

    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate CausalFlow model performance."""
        evaluation = gcm.evaluate_causal_model(self.scm, test_data)
        return {
            "mean_mse": _extract_mean_mse(evaluation),
            "overall_kl": _extract_overall_kl(evaluation),
        }


def create_causal_model(
    model_type: str, graph: nx.DiGraph, **kwargs
) -> CausalModelFitter:
    """Factory function to create causal models."""
    if model_type == "linear":
        return LinearCausalModel(graph, **kwargs)
    elif model_type == "lgbm":
        return LGBMCausalModel(graph, **kwargs)
    elif model_type == "diffusion":
        return DiffusionCausalModel(graph, **kwargs)
    elif model_type == "causalflow":
        return CausalFlowModel(graph, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Helper methods for extracting metrics from DoWhy evaluations
def _extract_mean_mse(evaluation) -> float:
    """Extract mean MSE from DoWhy evaluation results."""
    try:
        if hasattr(evaluation, "mechanism_performances"):
            mse_values = [
                perf.mse
                for perf in evaluation.mechanism_performances.values()
                if perf.mse is not None
            ]
            return sum(mse_values) / len(mse_values) if mse_values else 0.0
    except:
        pass
    return 0.0


def _extract_overall_kl(evaluation) -> float:
    """Extract overall KL divergence from DoWhy evaluation results."""
    try:
        if hasattr(evaluation, "overall_kl_divergence"):
            return evaluation.overall_kl_divergence
    except:
        pass
    return 0.0


# Add the helper methods to the base class
# CausalModelFitter._extract_mean_mse = staticmethod(_extract_mean_mse)
# CausalModelFitter._extract_overall_kl = staticmethod(_extract_overall_kl)
