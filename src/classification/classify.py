import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.dataset.dataset_wrappers import EncodedDatasetWrapper
from src.metrics.model_metrics import evaluate_group_fairness, evaluate_performance

# Import FAIRGBM simple wrapper
try:
    from src.classification.fairgbm_simple import fit_fairgbm_simple

    FAIRGBM_AVAILABLE = True
except ImportError:
    FAIRGBM_AVAILABLE = False

# Import FairLearn wrapper
try:
    from src.classification.fairlearn_wrapper import fit_fairlearn_classifier

    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


def fit_evaluate_classifier(
    model_tag: str,
    enc_dataset: EncodedDatasetWrapper,
    output_dir: Path,
    **kwargs,  # Additional arguments for FAIRGBM
) -> tuple[ClassifierMixin, pd.Series]:
    MODEL_MAPPING = {
        "LR": LogisticRegression,
        "RF": RandomForestClassifier,
        "GB": GradientBoostingClassifier,
    }

    # Handle FAIRGBM models
    if model_tag.startswith("FAIRGBM"):
        if not FAIRGBM_AVAILABLE:
            raise ImportError("FAIRGBM not available. Install required packages.")
        return fit_fairgbm_simple(model_tag, enc_dataset, output_dir, **kwargs)

    # Handle FairLearn models
    if model_tag.startswith("FAIRLEARN"):
        if not FAIRLEARN_AVAILABLE:
            raise ImportError("FairLearn not available. Install required packages.")
        return fit_fairlearn_classifier(model_tag, enc_dataset, output_dir, **kwargs)

    # Handle standard models
    if model_tag not in MODEL_MAPPING:
        raise ValueError(f"Unknown model tag: {model_tag}")

    logger.info(f"Fit evaluate classifier {model_tag}")
    classifier = MODEL_MAPPING[model_tag]()
    df_train = enc_dataset.X_enc_train.copy()
    df_test = enc_dataset.X_enc_test.copy()

    logger.info(f"Dropping {enc_dataset.enc_sensitive_name} for model tag {model_tag}")
    df_train.drop(columns=enc_dataset.enc_sensitive_name, inplace=True)
    df_test.drop(columns=enc_dataset.enc_sensitive_name, inplace=True)

    classifier.fit(
        df_train,
        enc_dataset.y_train,
    )
    y_pred = pd.Series(
        classifier.predict(df_test),
        index=enc_dataset.X_enc_test.index,
    )  # type: ignore

    model_performance = evaluate_performance(
        enc_dataset.y_test,
        y_pred,
    )
    group_fairness = evaluate_group_fairness(
        enc_dataset.y_test,
        y_pred,
        enc_dataset.sensitive_test,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Model performance:\n{json.dumps(model_performance, indent=4)}")
    with open(output_dir / "model_performance.json", "w") as f:
        json.dump(model_performance, f, indent=4)

    logger.info(f"Group fairness:\n{json.dumps(group_fairness, indent=4)}")
    with open(output_dir / "group_fairness.json", "w") as f:
        json.dump(group_fairness, f, indent=4)

    return classifier, y_pred


def fit_evaluate_classifier_with_tuning(
    model_tag: str,
    enc_dataset: EncodedDatasetWrapper,
    output_dir: Path,
    tune_hyperparameters: bool = False,
    tuning_config_file: Optional[Path] = None,
    **kwargs,  # Additional arguments for FAIRGBM
) -> tuple[ClassifierMixin, pd.Series, Optional[Dict[str, Any]]]:
    """
    Fit and evaluate classifier with optional hyperparameter tuning.

    Args:
        model_tag: Model tag (LR, RF, GB, FAIRGBM)
        enc_dataset: Encoded dataset wrapper
        output_dir: Output directory
        tune_hyperparameters: Whether to perform hyperparameter tuning
        tuning_config_file: Path to tuning configuration file
        **kwargs: Additional arguments

    Returns:
        Tuple of (classifier, predictions, tuning_results)
    """
    if tune_hyperparameters:
        return _fit_with_hyperparameter_tuning(
            model_tag, enc_dataset, output_dir, tuning_config_file, **kwargs
        )
    else:
        classifier, y_pred = fit_evaluate_classifier(
            model_tag, enc_dataset, output_dir, **kwargs
        )
        return classifier, y_pred, None


def _fit_with_hyperparameter_tuning(
    model_tag: str,
    enc_dataset: EncodedDatasetWrapper,
    output_dir: Path,
    tuning_config_file: Optional[Path] = None,
    **kwargs,
) -> tuple[ClassifierMixin, pd.Series, Dict[str, Any]]:
    """
    Fit classifier with hyperparameter tuning.

    Args:
        model_tag: Model tag
        enc_dataset: Encoded dataset wrapper
        output_dir: Output directory
        tuning_config_file: Path to tuning configuration file
        **kwargs: Additional arguments

    Returns:
        Tuple of (classifier, predictions, tuning_results)
    """
    from src.classification.hyperparameter_tuning import HyperparameterTuner
    from src.utils import find_root

    # Set default tuning config file if not provided
    if tuning_config_file is None:
        base_path = find_root()
        tuning_config_file = base_path / "config/tuning/tuning_config.yaml"

    # Create tuner
    classifier_config_dir = find_root() / "config/classifiers"
    tuner = HyperparameterTuner(
        classifier_config_dir=classifier_config_dir,
        tuning_config_file=tuning_config_file,
        output_dir=output_dir / "tuning",
    )

    # Prepare data
    X_train = enc_dataset.X_enc_train.copy()
    X_test = enc_dataset.X_enc_test.copy()

    # Always drop the encoded sensitive feature from inputs
    logger.info(f"Dropping {enc_dataset.enc_sensitive_name} for model {model_tag}")
    X_train = X_train.drop(columns=enc_dataset.enc_sensitive_name)
    X_test = X_test.drop(columns=enc_dataset.enc_sensitive_name)

    # For FairGBM, pass sensitive as numeric constraint_group
    if model_tag.startswith("FAIRGBM"):
        sensitive_train = _convert_sensitive_to_numeric(enc_dataset.sensitive_train)
        sensitive_test = _convert_sensitive_to_numeric(enc_dataset.sensitive_test)
    else:
        sensitive_train = None
        sensitive_test = None

    # Perform hyperparameter tuning
    tuning_results = tuner.tune_classifier(
        classifier_name=model_tag,
        X_train=X_train,
        y_train=enc_dataset.y_train,
        sensitive_train=sensitive_train,
        X_test=X_test,
        y_test=enc_dataset.y_test,
        sensitive_test=sensitive_test,
    )

    # Get the best model
    classifier = tuning_results["final_model"]

    # Make predictions
    y_pred = pd.Series(
        classifier.predict(X_test),
        index=enc_dataset.X_enc_test.index,
    )

    # Evaluate and save metrics
    model_performance = evaluate_performance(enc_dataset.y_test, y_pred)
    group_fairness = evaluate_group_fairness(
        enc_dataset.y_test, y_pred, enc_dataset.sensitive_test
    )

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Model performance:\n{json.dumps(model_performance, indent=4)}")
    with open(output_dir / "model_performance.json", "w") as f:
        json.dump(model_performance, f, indent=4)

    logger.info(f"Group fairness:\n{json.dumps(group_fairness, indent=4)}")
    with open(output_dir / "group_fairness.json", "w") as f:
        json.dump(group_fairness, f, indent=4)

    # Save tuning results
    with open(output_dir / "tuning_results.json", "w") as f:
        json.dump(
            {
                "best_params": tuning_results["best_params"],
                "best_score": tuning_results["best_score"],
                "test_results": tuning_results["test_results"],
            },
            f,
            indent=4,
        )

    return classifier, y_pred, tuning_results


def _convert_sensitive_to_numeric(sensitive_series: pd.Series) -> pd.Series:
    """
    Convert sensitive attributes to numeric format.

    Args:
        sensitive_series: Sensitive attribute series

    Returns:
        Numeric sensitive attribute series
    """
    unique_values = sorted(sensitive_series.unique())
    if len(unique_values) != 2:
        raise ValueError(
            f"Expected binary sensitive attribute, got {len(unique_values)} unique values: {unique_values}"
        )

    # Map to 0 and 1
    value_map = {unique_values[0]: 0, unique_values[1]: 1}
    return sensitive_series.map(value_map).astype(int)
