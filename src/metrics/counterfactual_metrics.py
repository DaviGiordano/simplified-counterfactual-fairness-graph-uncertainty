import logging

import numpy as np
import pandas as pd
from dowhy.gcm.divergence import (
    estimate_kl_divergence_continuous_clf,
    estimate_kl_divergence_continuous_knn,
)
from prdc import compute_prdc
from sklearn.neighbors import LocalOutlierFactor

from src.log.mlflow import log_to_mlflow

logger = logging.getLogger(__name__)


# @log_to_mlflow
def evaluate_counterfactual_world(
    df_observed: pd.DataFrame,
    df_counterfactuals: pd.DataFrame,
    sensitive_feature: pd.Series,
) -> dict[str, float]:
    """Evaluates counterfactuals against observed data. Use to evaluate a single counterfactual world"""
    if set(df_observed.columns) != set(df_counterfactuals.columns):
        raise ValueError("Data columns mismatch")
    if not df_observed.index.equals(df_counterfactuals.index):
        raise ValueError("Data index mismatch")

    logger.info("Evaluating counterfactual quality..")
    diff = compute_diff(df_observed, df_counterfactuals)

    # Overall distance metrics
    metrics = {f"overall.{k}": v for k, v in compute_distance_metrics(diff).items()}

    # Metrics per subgroup
    unique_values = sorted(sensitive_feature.unique())
    if len(unique_values) != 2:
        raise ValueError("Expected exactly two unique values in sensitive feature")

    for orig_value in unique_values:
        cf_value = [v for v in unique_values if v != orig_value][0]

        # Original A
        observed_mask = sensitive_feature == orig_value
        # Original B, Counterfactual A
        counterfactual_mask = sensitive_feature != orig_value

        transition_name = f"{orig_value}_to_{cf_value}"
        same_subgroup_name = f"orig_{orig_value}_cf_{orig_value}"

        diff_subset = diff.loc[observed_mask]
        df_observed_subset = df_observed.loc[observed_mask]
        df_counterfactuals_subset = df_counterfactuals.loc[counterfactual_mask]

        # Add distance metrics for this transition
        distance_metrics = compute_distance_metrics(diff_subset)
        metrics.update(
            {f"{transition_name}.{k}": v for k, v in distance_metrics.items()}
        )
        metrics[f"{transition_name}.sample_size"] = len(df_observed_subset)

        # Add divergence metrics for this subgroup.
        divergence_metrics = compute_divergence_metrics(
            df_observed_subset, df_counterfactuals_subset
        )
        metrics.update(
            {f"{same_subgroup_name}.{k}": v for k, v in divergence_metrics.items()}
        )

        # Add prdc metrics for this subgroup.
        prdc_metrics = compute_prdc_metrics(
            df_observed_subset, df_counterfactuals_subset
        )

        metrics.update(
            {f"{same_subgroup_name}.{k}": v for k, v in prdc_metrics.items()}
        )
        # Add LOF metrics for this subgroup
        lof_metrics = compute_lof_metrics(df_observed_subset, df_counterfactuals_subset)
        metrics.update({f"{same_subgroup_name}.{k}": v for k, v in lof_metrics.items()})

    return metrics


def compute_distance_metrics(diff: pd.DataFrame) -> dict[str, float]:
    """Computes distance metrics for counterfactual evaluation."""
    return {
        "l1_mean": l1_mean(diff),
        "l1_std": l1_std(diff),
        "l2_mean": l2_mean(diff),
        "l2_std": l2_std(diff),
    }


def compute_divergence_metrics(
    df_observed: pd.DataFrame, df_counterfactuals: pd.DataFrame
) -> dict[str, float]:
    """
    Computes divergence metrics between observed and counterfactual data.
    Note: both observed and counterfactual should contain the same value of the sensitive feature.
    """
    return {
        "kl_div_clf": estimate_kl_divergence_continuous_clf(
            np.array(df_observed),
            np.array(df_counterfactuals),
        ),
    }


def compute_lof_metrics(
    df_observed: pd.DataFrame, df_counterfactuals: pd.DataFrame, n_neighbors: int = 20
) -> dict[str, float]:
    """
    Computes Local Outlier Factor metrics for observed and counterfactual data.

    Parameters:
    -----------
    df_observed: DataFrame containing the observed data
    df_counterfactuals: DataFrame containing the counterfactual data
    n_neighbors: Number of neighbors to use for LOF calculation

    Returns:
    --------
    Dictionary containing LOF metrics:
    - observed_lof_mean: Average LOF score for observed data
    - counterfactual_lof_mean: Average LOF score for counterfactuals
    - lof_diff: Difference between counterfactual and observed LOF means
    """
    # Convert dataframes to numpy arrays
    observed_array = np.array(df_observed)
    counterfactual_array = np.array(df_counterfactuals)

    # Fit LOF on observed data
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    lof.fit(observed_array)

    # Calculate negative outlier scores (higher means more outlier-like)
    observed_scores = -lof.score_samples(observed_array)
    counterfactual_scores = -lof.score_samples(counterfactual_array)

    # Calculate metrics
    observed_lof_mean = float(observed_scores.mean())
    counterfactual_lof_mean = float(counterfactual_scores.mean())
    lof_diff = counterfactual_lof_mean - observed_lof_mean

    return {
        "observed_lof_mean": observed_lof_mean,
        "counterfactual_lof_mean": counterfactual_lof_mean,
        "lof_diff": lof_diff,
    }


def compute_prdc_metrics(
    df_observed: pd.DataFrame,
    df_counterfactuals: pd.DataFrame,
    nearest_k: int = 5,
) -> dict[str, float]:
    return compute_prdc(
        real_features=df_observed,
        fake_features=df_counterfactuals,
        nearest_k=nearest_k,
    )


def compute_diff(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Computes difference between two dataframes"""
    if set(df1.columns) != set(df2.columns):
        raise ValueError("Data columns mismatch")
    if not df1.index.equals(df2.index):
        raise ValueError("Data index mismatch")
    return df1 - df2


def l1_mean(diff: pd.DataFrame) -> float:
    return float(diff.abs().mean(axis=1).mean())


def l1_std(diff: pd.DataFrame) -> float:
    return float(diff.abs().mean(axis=1).std())


def l2_mean(diff: pd.DataFrame) -> float:
    return float(np.sqrt(diff.pow(2).sum(axis=1)).mean())


def l2_std(diff: pd.DataFrame) -> float:
    return float(np.sqrt(diff.pow(2).sum(axis=1)).std())
