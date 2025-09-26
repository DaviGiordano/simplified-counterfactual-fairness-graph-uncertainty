import logging
from typing import Optional

import numpy as np
import pandas as pd
from pymdma.tabular.measures.synthesis_val import Coverage, Density, StatisticalSimScore
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

from src.log.mlflow import log_to_mlflow
from src.utils import record_time

logger = logging.getLogger(__name__)


def outlier_conformal_isoforest(
    df_test: pd.DataFrame,
    df_cf: pd.DataFrame,
    binary_root: str,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> pd.Series:
    """Alpha: desired false positive rate among real data"""
    if binary_root not in df_test.columns or binary_root not in df_cf.columns:
        raise ValueError(f"'{binary_root}' must exist in both dataframes.")

    features = [c for c in df_test.columns if c != binary_root and c in df_cf.columns]
    if not features:
        raise ValueError("No overlapping features (excluding binary_root).")

    flags = pd.Series(0, index=df_cf.index, dtype=int)

    for root_val in [0, 1]:
        X_real = df_test.loc[df_test[binary_root] == root_val, features]
        X_cf = df_cf.loc[df_cf[binary_root] == root_val, features]

        if X_cf.empty:
            continue
        # Not enough data to calibrate; mark as non-outliers (conservative).
        if len(X_real) < 5:
            flags.loc[X_cf.index] = 0
            continue

        X_train, X_cal = train_test_split(
            X_real,
            test_size=min(0.3, max(1, int(0.3 * len(X_real))) / len(X_real)),
            random_state=random_state,
            shuffle=True,
        )
        # if X_cal.empty:  # fallback if too small
        # X_train, X_cal = X_real.iloc[:-1], X_real.iloc[-1:].copy()

        clf = IsolationForest(
            n_estimators=300,
            max_samples="auto",
            bootstrap=True,
            contamination="auto",  # not used, we are recalibrating
            random_state=random_state,
        )
        clf.fit(X_train)

        # Nonconformity = -score_samples (higher = more outlier)
        s_cal = -clf.score_samples(X_cal)
        # Conformal threshold at (1 - alpha)-quantile of calibration scores
        T = np.quantile(s_cal, 1 - alpha)

        # Score counterfactuals and flag
        s_cf = -clf.score_samples(X_cf)
        flags.loc[X_cf.index] = (s_cf >= T).astype(int)

    return flags


def evaluate_cf_quality(
    df_real: pd.DataFrame,
    df_cf: pd.DataFrame,
    binary_root: str,
    k: int = 5,
    metric: str = "euclidean",
) -> dict:
    """
    Compare real vs. counterfactual data within each subgroup of `binary_root` using pyMDMA:
      - Density (fidelity-like)
      - Coverage (diversity-like)
      - StatisticalSimScore (distributional similarity)
    Returns a DataFrame with one row per subgroup (0/1).
    """
    if binary_root not in df_real.columns or binary_root not in df_cf.columns:
        raise ValueError(f"'{binary_root}' must exist in both dataframes.")

    features = [c for c in df_real.columns if c != binary_root and c in df_cf.columns]
    if not features:
        raise ValueError("No overlapping features (excluding binary_root).")

    results = {}
    for root_val in [0, 1]:
        Xr = df_real.loc[df_real[binary_root] == root_val, features]
        Xf = df_cf.loc[df_cf[binary_root] == root_val, features]

        Xr = Xr.to_numpy()
        Xf = Xf.to_numpy()

        ctx = {}
        dens_res = (
            Density(k=k, metric=metric).compute(Xr, Xf, context=ctx).dataset_level.value
        )
        cov_res = (
            Coverage(k=k, metric=metric)
            .compute(Xr, Xf, context=ctx)
            .dataset_level.value
        )
        sim_res = float(
            np.asarray(
                list(StatisticalSimScore().compute(Xr, Xf).dataset_level.value.values())
            ).mean()
        )

        results.update(
            {
                f"n_real_{root_val}": len(Xr),
                f"n_cf_{root_val}": len(Xf),
                f"density_{root_val}": dens_res,
                f"coverage_{root_val}": cov_res,
                f"stat_sim_{root_val}": sim_res,
            }
        )

    return results


def evaluate_counterfactuals(df_cf, df_test, binary_root):
    times = {}
    with record_time("time_evaluate_outliers", times):
        df_cf["outlier_flag"] = outlier_conformal_isoforest(
            df_test,
            df_cf,
            binary_root,
            alpha=0.05,
            random_state=42,
        )

    mask0 = df_cf[binary_root] == 0
    mask1 = ~mask0
    cf_outliers = {
        "overall_outlier_percent": float(df_cf["outlier_flag"].mean()),
        "sbg0_outlier_percent": (
            float(df_cf.loc[mask0, "outlier_flag"].mean()) if mask0.any() else 0.0
        ),
        "sbg1_outlier_percent": (
            float(df_cf.loc[mask1, "outlier_flag"].mean()) if mask1.any() else 0.0
        ),
    }

    with record_time("time_evaluate_cf_quality", times):
        cf_quality = evaluate_cf_quality(df_test, df_cf, binary_root)

    cf_metrics = cf_outliers | cf_quality
    return cf_metrics, times


# @log_to_mlflow
def evaluate_counterfactual_world(
    df_observed: pd.DataFrame,
    df_counterfactuals: pd.DataFrame,
    sensitive_feature: pd.Series,
) -> dict[str, float]:
    """Evaluates counterfactuals against observed data using Isolation Forest and pyMDMA metrics."""
    if set(df_observed.columns) != set(df_counterfactuals.columns):
        raise ValueError("Data columns mismatch")
    if not df_observed.index.equals(df_counterfactuals.index):
        raise ValueError("Data index mismatch")

    logger.info("Evaluating counterfactual quality with Isolation Forest and pyMDMA..")

    # Get the sensitive feature name from the dataframe columns
    # Assuming the sensitive feature is binary and matches one of the dataframe columns
    unique_values = sorted(sensitive_feature.unique())
    if len(unique_values) != 2:
        raise ValueError("Expected exactly two unique values in sensitive feature")

    # Find the sensitive feature column name
    sensitive_col = None

    # First, try to find exact match (for non-encoded data)
    for col in df_observed.columns:
        if set(df_observed[col].unique()) == set(unique_values):
            sensitive_col = col
            break

    # If no exact match, look for one-hot encoded columns
    if sensitive_col is None:
        # Look for columns that might be one-hot encoded sensitive features
        for col in df_observed.columns:
            col_unique = set(df_observed[col].unique())
            # Check if this looks like a binary one-hot encoded column
            if col_unique.issubset({0.0, 1.0}) and len(col_unique) == 2:
                # This could be a one-hot encoded sensitive feature
                # We'll use this column and convert the sensitive feature to binary
                sensitive_col = col
                break

    if sensitive_col is None:
        raise ValueError("Could not find sensitive feature column in dataframe")

    # Create copies of dataframes with the sensitive feature
    df_obs_with_sensitive = df_observed.copy()
    df_cf_with_sensitive = df_counterfactuals.copy()

    # Check if we need to convert sensitive feature to binary values
    if set(df_observed[sensitive_col].unique()).issubset({0.0, 1.0}):
        # The dataframe column is binary, so we need to convert the sensitive feature
        # Map the original values to binary (assuming first value maps to 0, second to 1)
        value_map = {unique_values[0]: 0.0, unique_values[1]: 1.0}
        binary_sensitive = sensitive_feature.map(value_map)
        df_obs_with_sensitive[sensitive_col] = binary_sensitive
        df_cf_with_sensitive[sensitive_col] = binary_sensitive
    else:
        # The dataframe column has the same values as the sensitive feature
        df_obs_with_sensitive[sensitive_col] = sensitive_feature
        df_cf_with_sensitive[sensitive_col] = sensitive_feature

    # Evaluate using the new approach
    cf_metrics, times = evaluate_counterfactuals(
        df_cf_with_sensitive, df_obs_with_sensitive, sensitive_col
    )

    # Add timing information to metrics
    cf_metrics.update(times)

    return cf_metrics


def compute_distance_metrics(diff: pd.DataFrame) -> dict[str, float]:
    """Computes distance metrics for counterfactual evaluation."""
    return {
        "l1_mean": l1_mean(diff),
        "l1_std": l1_std(diff),
        "l2_mean": l2_mean(diff),
        "l2_std": l2_std(diff),
    }


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
