import numpy as np
import pandas as pd


def build_summary_statistics(
    df: dict[int, pd.DataFrame],
    quantile_alpha=0.05,
) -> dict[str, pd.DataFrame]:
    """
    Build a dataset with summary statistics for all individuals, organized by metric type.

    Args:
        counterfactuals_by_individual: Dictionary mapping individual indices to their counterfactuals

    Returns:
        dict: Dictionary containing separate DataFrames for means, variances, and confidence intervals
    """
    if not df:
        raise ValueError("Counterfactuals by individual is empty")

    # Initialize dictionaries for each metric
    means_data = {}
    var_data = {}
    ci_lower_data = {}
    ci_upper_data = {}

    # Process each individual's data
    for idx, cf_data in sorted(df.items()):
        # Get basic stats
        means = cf_data.mean(numeric_only=True)
        variances = cf_data.var(numeric_only=True)
        ci_lower = cf_data.quantile(quantile_alpha / 2)
        ci_upper = cf_data.quantile(1 - quantile_alpha / 2)

        # Store stats in respective dictionaries
        means_data[idx] = means
        var_data[idx] = variances
        ci_lower_data[idx] = ci_lower
        ci_upper_data[idx] = ci_upper

    # Convert dictionaries to DataFrames
    means_df = pd.DataFrame.from_dict(means_data, orient="index")
    var_df = pd.DataFrame.from_dict(var_data, orient="index")
    ci_lower_df = pd.DataFrame.from_dict(ci_lower_data, orient="index")
    ci_upper_df = pd.DataFrame.from_dict(ci_upper_data, orient="index")

    return {
        "mean": means_df,
        "var": var_df,
        "ci_lower": ci_lower_df,
        "ci_upper": ci_upper_df,
    }
