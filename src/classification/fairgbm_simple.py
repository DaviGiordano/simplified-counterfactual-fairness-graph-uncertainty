import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.base import ClassifierMixin

from src.dataset.dataset_wrappers import EncodedDatasetWrapper
from src.metrics.model_metrics import evaluate_group_fairness, evaluate_performance

logger = logging.getLogger(__name__)

try:
    from fairgbm import FairGBMClassifier

    FAIRGBM_AVAILABLE = True
except ImportError:
    FAIRGBM_AVAILABLE = False
    logger.warning("FAIRGBM not available. Install with: pip install fairgbm")


def get_config_path(model_tag: str) -> Path:
    """Map model tag to appropriate config file."""
    if "equal_opportunity" in model_tag:
        return Path("config/fairgbm/equal_opportunity.yaml")
    elif "predictive_equality" in model_tag:
        return Path("config/fairgbm/predictive_equality.yaml")
    else:
        return Path("config/fairgbm/equalized_odds.yaml")


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def fit_fairgbm_simple(
    model_tag: str,
    enc_dataset: EncodedDatasetWrapper,
    output_dir: Path,
    **kwargs,  # Additional arguments for future use
) -> Tuple[ClassifierMixin, pd.Series]:
    """Fit and evaluate FAIRGBM classifier with simple default parameters."""

    if not FAIRGBM_AVAILABLE:
        raise ImportError("FAIRGBM not available. Install required packages.")

    logger.info(f"Fitting FAIRGBM classifier (simple): {model_tag}")

    # Prepare data
    X_train = enc_dataset.X_enc_train.copy()
    X_test = enc_dataset.X_enc_test.copy()

    # Handle sensitive attribute inclusion/exclusion
    if "no_sensitive" in model_tag:
        logger.info(
            f"Dropping {enc_dataset.enc_sensitive_name} for model tag {model_tag}"
        )
        X_train = X_train.drop(columns=enc_dataset.enc_sensitive_name)
        X_test = X_test.drop(columns=enc_dataset.enc_sensitive_name)
        # For FAIRGBM without sensitive features, we can't use fairness constraints
        logger.warning(
            "FAIRGBM without sensitive features will not enforce fairness constraints"
        )

    # Convert sensitive attributes to numeric format for FAIRGBM
    # FAIRGBM requires numeric sensitive attributes (0, 1)
    s_train_numeric = enc_dataset.sensitive_train.copy()
    s_test_numeric = enc_dataset.sensitive_test.copy()

    # Convert string values to numeric (assuming binary sensitive attribute)
    unique_values = sorted(enc_dataset.sensitive_train.unique())
    if len(unique_values) != 2:
        raise ValueError(
            f"Expected binary sensitive attribute, got {len(unique_values)} unique values: {unique_values}"
        )

    # Map to 0 and 1
    value_map = {unique_values[0]: 0, unique_values[1]: 1}
    s_train_numeric = s_train_numeric.map(value_map).astype(int)
    s_test_numeric = s_test_numeric.map(value_map).astype(int)

    logger.info(f"Converted sensitive attributes: {value_map}")

    # Load configuration from YAML file
    config_path = get_config_path(model_tag)
    config = load_config(config_path)

    # Extract constraint type from config file
    fairgbm_config = config.get("FairGBM", {})
    kwargs_config = fairgbm_config.get("kwargs", {})
    constraint_type_options = kwargs_config.get("constraint_type", ["FNR,FPR"])

    # Use the first constraint type (config files should have only one)
    constraint_type = (
        constraint_type_options[0]
        if isinstance(constraint_type_options, list)
        else constraint_type_options
    )

    logger.info(f"Using constraint type from config: {constraint_type}")

    # Helper function to extract single value from config (handle lists and search spaces)
    def get_single_value(key, default):
        value = kwargs_config.get(key, default)
        if isinstance(value, list):
            return value[0]  # Take first value from list
        elif isinstance(value, dict):
            # This is a hyperparameter search space, use default
            return default
        return value

    # Initialize FAIRGBM with parameters from config file
    # Use reasonable defaults for parameters not specified in config
    classifier = FairGBMClassifier(
        n_estimators=get_single_value("n_estimators", 100),
        constraint_type=constraint_type,
        multiplier_learning_rate=get_single_value("multiplier_learning_rate", 0.1),
        learning_rate=get_single_value("learning_rate", 0.1),
        num_leaves=get_single_value("num_leaves", 31),
        max_depth=get_single_value("max_depth", -1),
        min_child_samples=get_single_value("min_child_samples", 20),
        reg_alpha=get_single_value("reg_alpha", 0.0),
        reg_lambda=get_single_value("reg_lambda", 0.0),
        boosting_type=get_single_value("boosting_type", "gbdt"),
        random_state=42,
        verbose=-1,  # Suppress output
    )

    # Fit the model
    if "no_sensitive" in model_tag:
        # Train without fairness constraints
        classifier.fit(X_train, enc_dataset.y_train)
    else:
        # Train with fairness constraints
        classifier.fit(X_train, enc_dataset.y_train, constraint_group=s_train_numeric)

    # Make predictions on test set
    y_pred = pd.Series(
        classifier.predict(X_test),
        index=enc_dataset.X_enc_test.index,
    )

    # Evaluate performance and fairness
    model_performance = evaluate_performance(
        enc_dataset.y_test,
        y_pred,
    )
    group_fairness = evaluate_group_fairness(
        enc_dataset.y_test,
        y_pred,
        enc_dataset.sensitive_test,
    )

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model performance
    logger.info(f"Model performance:\n{model_performance}")
    with open(output_dir / "model_performance.json", "w") as f:
        import json

        json.dump(model_performance, f, indent=4)

    # Save group fairness
    logger.info(f"Group fairness:\n{group_fairness}")
    with open(output_dir / "group_fairness.json", "w") as f:
        json.dump(group_fairness, f, indent=4)

    # Save the trained model
    with open(output_dir / "fairgbm_model.pkl", "wb") as f:
        pickle.dump(classifier, f)

    return classifier, y_pred
