import pandas as pd


def load_score_variance_data(base_path, dataset_tag, knowledge_levels, classifiers):
    """
    Load and combine score variance data across different knowledge levels and classifiers.

    Args:
        base_path (Path): Base path to the project
        dataset_tag (str): Dataset identifier
        knowledge_levels (list): List of knowledge levels
        classifiers (list): List of classifier names

    Returns:
        pd.DataFrame: Combined dataframe with score variance data
    """
    df_score_var = pd.DataFrame()
    for know in knowledge_levels:
        for clf in classifiers:
            fpath = (
                base_path
                / "output"
                / dataset_tag
                / know
                / clf
                / "score_var_across_cws_by_individual.csv"
            )
            df_curr = pd.read_csv(fpath, index_col=0).reset_index()
            df_curr["classifier"] = clf
            df_curr["knowledge"] = know
            df_score_var = pd.concat([df_score_var, df_curr], axis=0).reset_index(
                drop=True
            )

    # Removing results that used sensitive feature to classify
    df_score_var = df_score_var[
        df_score_var["classifier"].isin(
            [
                "GB_no_sensitive",
                "RF_no_sensitive",
                "LR_no_sensitive",
                "FAIRGBM",
                "FAIRGBM_equal_opportunity",
                "FAIRGBM_predictive_equality",
            ]
        )
    ]
    df_score_var["classifier"] = df_score_var["classifier"].replace(
        {
            "GB_no_sensitive": "GB",
            "RF_no_sensitive": "RF",
            "LR_no_sensitive": "LR",
            "FAIRGBM": "FAIRGBM",
            "FAIRGBM_equal_opportunity": "FAIRGBM_equal_opportunity",
            "FAIRGBM_predictive_equality": "FAIRGBM_predictive_equality",
        }
    )
    df_score_var["knowledge"] = df_score_var["knowledge"].map(
        {"low": "Low", "med": "High"}
    )
    return df_score_var


from pathlib import Path

import pandas as pd


def load_and_tidy_fairness_metrics(
    base_path: Path, dataset_tag: str, knows: list[str], clfs: list[str]
) -> pd.DataFrame:
    """
    Load and tidy fairness metrics from multiple classifiers and knowledge sources.

    Parameters:
    - base_path: Path to the base directory containing dataset folders.
    - dataset_tag: Subdirectory name for the specific dataset.
    - knows: List of knowledge source directory names.
    - clfs: List of classifier directory names.

    Returns:
    - Tidy DataFrame with columns [Classifier, Knowledge, intervention, metric, Rate].
    """
    df_list = []
    for know in knows:
        for clf in clfs:
            fpath = base_path / "output" / dataset_tag / know / clf / "cf_metrics.csv"
            df = pd.read_csv(fpath, index_col=0).reset_index()
            df["Classifier"] = clf
            df["Knowledge"] = know
            df_list.append(df)

    df_fairness = pd.concat(df_list, ignore_index=True)

    # Identify rate columns and melt into tidy format
    rate_cols = [c for c in df_fairness.columns if c.endswith("_rate")]
    tidy = df_fairness.melt(
        id_vars=["Classifier", "Knowledge"],
        value_vars=rate_cols,
        var_name="intervention_metric",
        value_name="Rate",
    )

    # Split intervention and metric
    tidy[["intervention", "metric"]] = tidy["intervention_metric"].str.rsplit(
        ".", n=1, expand=True
    )
    tidy = tidy.drop(columns=["intervention_metric"])
    tidy.replace(
        {
            "Knowledge": {
                "low": "Low",
                "med": "Medium",
                "high": "High",
                "forbidx1": "Forbid $X_1$",
                "forbidx2": "Forbid $X_2$",
            },
            "metric": {
                "negative_to_positive_switch_rate": "PSR",
                "positive_to_negative_switch_rate": "NSR",
            },
        },
        inplace=True,
    )
    # Removing results that used sensitive feature to classify
    tidy = tidy[
        tidy["Classifier"].isin(
            [
                "GB_no_sensitive",
                "RF_no_sensitive",
                "LR_no_sensitive",
                "FAIRGBM",
                "FAIRGBM_equal_opportunity",
                "FAIRGBM_predictive_equality",
            ]
        )
    ]
    tidy["Classifier"] = tidy["Classifier"].replace(
        {
            "GB_no_sensitive": "GB",
            "RF_no_sensitive": "RF",
            "LR_no_sensitive": "LR",
            "FAIRGBM": "FAIRGBM",
            "FAIRGBM_equal_opportunity": "FAIRGBM_equal_opportunity",
            "FAIRGBM_predictive_equality": "FAIRGBM_predictive_equality",
        }
    )
    tidy["Knowledge"] = tidy["Knowledge"].map({"Low": "Low", "Medium": "High"})

    return tidy
