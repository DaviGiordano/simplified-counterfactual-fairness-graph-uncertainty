import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    count,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
    true_negative_rate,
    true_positive_rate,
)
from flatten_dict import flatten
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from src.log import log_to_mlflow
from src.metrics.custom_metrics import (
    negative_switch_count,
    negative_switch_rate,
    positive_switch_count,
    positive_switch_rate,
)


# @log_to_mlflow
def evaluate_performance(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict:
    return {
        "overall.accuracy": accuracy_score(
            y_true=y_true,
            y_pred=y_pred,
        ),
        "overall.precision": precision_score(
            y_true=y_true,
            y_pred=y_pred,
        ),
        "overall.recall": recall_score(
            y_true=y_true,
            y_pred=y_pred,
        ),
        "overall.f1": f1_score(
            y_true=y_true,
            y_pred=y_pred,
        ),
        "overall.matthews_corrcoef": matthews_corrcoef(
            y_true=y_true,
            y_pred=y_pred,
        ),
        "overall.true_positive_rate": float(
            true_positive_rate(
                y_true=y_true,
                y_pred=y_pred,
            )
        ),
        "overall.true_negative_rate": float(
            true_negative_rate(
                y_true=y_true,
                y_pred=y_pred,
            )
        ),
        "overall.false_positive_rate": float(
            false_positive_rate(
                y_true=y_true,
                y_pred=y_pred,
            )
        ),
        "overall.false_negative_rate": float(
            false_negative_rate(
                y_true=y_true,
                y_pred=y_pred,
            )
        ),
        "overall.selection_rate": float(
            selection_rate(
                y_true=y_true,
                y_pred=y_pred,
            )
        ),
    }


# @log_to_mlflow
def evaluate_group_fairness(
    y_true: pd.Series,
    y_pred: pd.Series,
    sensitive_features: pd.Series,
) -> dict:
    """
    Evaluates group fairness metrics.
    Uses MetricFrame from fairlearn to get breakdown by subgroup.
    """
    overall_fairness = {
        "dem_parity_diff": float(
            demographic_parity_difference(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features,
            )
        ),
        "dem_parity_ratio": float(
            demographic_parity_ratio(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features,
            )
        ),
        "eq_odds_diff": equalized_odds_difference(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
        ),
        "eq_odds_ratio": equalized_odds_ratio(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
        ),
    }

    subgroup_metrics_fns = {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
        "matthews_corrcoef": matthews_corrcoef,
        "true_positive_rate": true_positive_rate,
        "true_negative_rate": true_negative_rate,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "selection_rate": selection_rate,
        "count": count,
    }

    metric_frame = MetricFrame(
        metrics=subgroup_metrics_fns,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )

    # each metric will be broken down by subgroup: 'subgroup'.'metric_name'
    subgroup_fairness = flatten(
        pd.DataFrame(metric_frame.by_group).to_dict(orient="index"),
        reducer="dot",
    )

    return overall_fairness | subgroup_fairness


@log_to_mlflow
def evaluate_stability(y_pred, y_cf):
    """
    Evaluates overall counterfactual fairness metrics.
    Aims to measure general stability.
    """
    stability_kwargs = {
        "y_pred": y_pred,
        "y_cf": y_cf,
    }
    stability_metrics = {
        "overall.positive_switch_count": positive_switch_count(**stability_kwargs),
        "overall.negative_switch_count": negative_switch_count(**stability_kwargs),
        "overall.positive_switch_rate": positive_switch_rate(**stability_kwargs),
        "overall.negative_switch_rate": negative_switch_rate(**stability_kwargs),
        "overall.counterfactual_matthews_corrcoef": matthews_corrcoef(
            y_true=y_pred, y_pred=y_cf
        ),
    }

    return stability_metrics


# @log_to_mlflow
def compute_cf_metrics(
    y_pred: pd.Series,
    y_cf: pd.Series,
    original_sensitive_feature: pd.Series,
) -> dict[str, float]:
    """
    Evaluates subgroup counterfactual fairness metrics.
    Calculates metrics for each transition from original to counterfactual sensitive feature value.
    """
    unique_values = sorted(original_sensitive_feature.unique())
    if len(unique_values) != 2:
        raise ValueError("Expected exactly two unique values in sensitive feature")

    cf_metrics = {}

    for orig_value in unique_values:
        cf_value = [v for v in unique_values if v != orig_value][0]
        mask = original_sensitive_feature == orig_value
        transition_name = f"{orig_value}_to_{cf_value}"

        cf_metrics[transition_name] = {
            "negative_to_positive_switch_count": positive_switch_count(
                y_pred[mask], y_cf[mask]
            ),
            "positive_to_negative_switch_count": negative_switch_count(
                y_pred[mask], y_cf[mask]
            ),
            "negative_to_positive_switch_rate": positive_switch_rate(
                y_pred[mask], y_cf[mask]
            ),
            "positive_to_negative_switch_rate": negative_switch_rate(
                y_pred[mask], y_cf[mask]
            ),
            "counterfactual_matthews_corrcoef": matthews_corrcoef(
                y_true=y_pred[mask], y_pred=y_cf[mask]
            ),
            "sample_size": len(y_pred[mask]),
        }

    flat_metrics = {}
    for transition, metrics in cf_metrics.items():
        for metric_name, value in metrics.items():
            flat_metrics[f"{transition}.{metric_name}"] = value

    return flat_metrics


def evaluate_extended_counterfactual_fairness(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_cf: pd.Series,
    sensitive_feature: pd.Series,
):
    pass
