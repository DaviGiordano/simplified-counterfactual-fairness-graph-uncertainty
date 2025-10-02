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

        # Import the external diffusion model
        from external.DiffusionBasedCausalModels.model.diffusion import (
            CausalDiffusionModel,
        )

        # Set diffusion mechanisms for non-root nodes with default parameters
        for node in self.graph.nodes:
            if self.graph.in_degree(node) == 0:
                mechanism = gcm.EmpiricalDistribution()
            else:
                # Use actual diffusion mechanism with default parameters
                mechanism = CausalDiffusionModel(
                    hidden_dim=64,  # Default from library
                    use_positional_encoding=False,  # Default from library
                    t_dim=8,  # Default from library
                    lr=1e-4,  # Default from library
                    weight_decay=0.001,  # Default from library
                    batch_size=64,  # Default from library
                    num_epochs=10,  # Default from library
                    use_gpu_if_available=True,  # Default from library
                    verbose=False,  # Default from library
                    w=0,  # Default from library
                    lambda_loss=0,  # Default from library
                    T=100,  # Default from library
                    betas=None,  # Default from library
                    clip=False,  # Default from library
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

    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate diffusion model performance."""
        evaluation = gcm.evaluate_causal_model(self.scm, test_data)
        return {
            "mean_mse": _extract_mean_mse(evaluation),
            "overall_kl": _extract_overall_kl(evaluation),
        }


class CausalFlowModel(CausalModelFitter):
    """CausalFlow (CausalMAF) based causal model."""

    def __init__(self, graph: nx.DiGraph, **kwargs):
        super().__init__(graph, "causalflow", **kwargs)
        self.flow_model = None

    def fit(self, train_data: pd.DataFrame) -> Dict[str, Any]:
        """Fit CausalFlow model."""
        logger.info("Fitting CausalFlow model...")

        # Import the CausalMAFModel
        from src.causality.causal_flow import CausalMAFModel

        # Create CausalMAF model with default parameters
        # Note: We need to determine the binary_root from the data
        # For now, we'll assume it's the first binary column or use a default
        binary_root = self._find_binary_root(train_data)

        self.flow_model = CausalMAFModel(
            graph=self.graph,
            columns=train_data.columns.tolist(),
            binary_root=binary_root,
            hidden_features=(128, 128),  # Default from reference
        )

        # Fit the model with default parameters
        losses = self.flow_model.fit(train_data)

        # For evaluation, we'll use a simple approach since CausalFlow doesn't use DoWhy's evaluation
        # We'll generate some samples and compute basic metrics
        eval_data = train_data.sample(n=min(1000, len(train_data)), random_state=42)

        # Compute test NLL
        test_nll = self.flow_model.nll(eval_data)

        # Generate samples for evaluation
        from src.causality.causal_flow import evaluate_flow

        _, summary = evaluate_flow(self.flow_model, eval_data, n_gen=len(eval_data))
        summary["test_nll"] = test_nll

        return {
            "model_type": "causalflow",
            "evaluation": f"CausalFlow evaluation - NLL: {test_nll:.4f}",
            "mean_mse": summary.get("mean_mse", 0.0),
            "overall_kl": summary.get("overall_kl", 0.0),
            "test_nll": test_nll,
        }

    def _find_binary_root(self, data: pd.DataFrame) -> str:
        """Find the binary root column in the data."""
        # Look for columns that contain only 0 and 1 values
        for col in data.columns:
            unique_vals = set(data[col].dropna().astype(float))
            if unique_vals.issubset({0.0, 1.0}):
                return col

        # If no binary column found, return the first column as fallback
        logger.warning("No binary column found, using first column as binary_root")
        return data.columns[0]

    def generate_counterfactuals(
        self, test_data: pd.DataFrame, sensitive_name: str
    ) -> pd.DataFrame:
        """Generate counterfactuals using CausalFlow model."""
        if self.flow_model is None:
            raise ValueError("Model must be fitted before generating counterfactuals")

        # Use the CausalMAF model's counterfactual generation
        return self.flow_model.generate_counterfactuals(test_data)

    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate CausalFlow model performance."""
        if self.flow_model is None:
            raise ValueError("Model must be fitted before evaluation")

        # Compute test NLL
        test_nll = self.flow_model.nll(test_data)

        # Generate samples for evaluation
        from src.causality.causal_flow import evaluate_flow

        _, summary = evaluate_flow(self.flow_model, test_data, n_gen=len(test_data))

        return {
            "mean_mse": summary.get("mean_mse", 0.0),
            "overall_kl": summary.get("overall_kl", 0.0),
            "test_nll": test_nll,
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
