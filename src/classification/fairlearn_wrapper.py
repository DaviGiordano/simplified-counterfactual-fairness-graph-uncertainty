import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.dataset.dataset_wrappers import EncodedDatasetWrapper
from src.metrics.model_metrics import evaluate_group_fairness, evaluate_performance

logger = logging.getLogger(__name__)

try:
    from fairlearn.reductions import (
        DemographicParity,
        EqualizedOdds,
        ExponentiatedGradient,
        FalsePositiveRateParity,
        TruePositiveRateParity,
    )

    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False
    logger.warning("FairLearn not available. Install with: pip install fairlearn")


class ExponentiatedGradientWrapper:
    """Wrapper for ExponentiatedGradient to add missing methods for compatibility."""

    def __init__(self, exp_grad):
        self.exp_grad = exp_grad
        # Add compatibility attributes
        if hasattr(exp_grad, "predictors_") and len(exp_grad.predictors_) > 0:
            self.classes_ = exp_grad.predictors_[0].classes_
            if hasattr(exp_grad.predictors_[0], "feature_names_in_"):
                self.feature_names_in_ = exp_grad.predictors_[0].feature_names_in_

    def predict(self, X):
        return self.exp_grad.predict(X)

    def predict_proba(self, X):
        if hasattr(self.exp_grad, "predictors_") and len(self.exp_grad.predictors_) > 0:
            return self.exp_grad.predictors_[0].predict_proba(X)
        else:
            raise AttributeError("No predictors available for probability predictions")

    def __getattr__(self, name):
        # Delegate all other attributes to the wrapped object
        return getattr(self.exp_grad, name)


try:
    from hpt.tuner import ObjectiveFunction, OptunaTuner

    HPT_AVAILABLE = True
except ImportError:
    HPT_AVAILABLE = False
    logger.warning(
        "Hyperparameter tuning not available. Install with: pip install hyperparameter-tuning"
    )


class FairLearnWrapper:
    """Wrapper for FairLearn ExponentiatedGradient classifier with hyperparameter tuning."""

    def __init__(
        self,
        base_estimator_type: str = "LogisticRegression",
        constraint_type: str = "DemographicParity",
        config_path: Optional[Path] = None,
        n_trials: int = 20,
        n_jobs: int = 1,
        seed: int = 42,
        perf_metric: str = "accuracy",
        fair_metric: str = "demographic_parity_difference",
        alpha: float = 0.5,
    ):
        if not FAIRLEARN_AVAILABLE:
            raise ImportError("FairLearn not available. Install required packages.")

        self.base_estimator_type = base_estimator_type
        self.constraint_type = constraint_type
        self.config_path = config_path
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.seed = seed
        self.perf_metric = perf_metric
        self.fair_metric = fair_metric
        self.alpha = alpha
        self.tuner = None
        self.best_model = None

    def _get_base_estimator(self, **kwargs) -> ClassifierMixin:
        """Get base estimator instance based on type."""
        estimator_map = {
            "LogisticRegression": LogisticRegression,
            "RandomForestClassifier": RandomForestClassifier,
            "GradientBoostingClassifier": GradientBoostingClassifier,
        }

        if self.base_estimator_type not in estimator_map:
            raise ValueError(f"Unknown base estimator type: {self.base_estimator_type}")

        return estimator_map[self.base_estimator_type](**kwargs)

    def _get_constraint(self, **kwargs):
        """Get fairness constraint instance based on type."""
        constraint_map = {
            "DemographicParity": DemographicParity,
            "EqualizedOdds": EqualizedOdds,
            "EqualOpportunity": TruePositiveRateParity,  # Equal opportunity = TPR parity
            "PredictiveEquality": FalsePositiveRateParity,  # Predictive equality = FPR parity
        }

        if self.constraint_type not in constraint_map:
            raise ValueError(f"Unknown constraint type: {self.constraint_type}")

        return constraint_map[self.constraint_type](**kwargs)

    def fit_with_tuning(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        s_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        s_val: pd.Series,
    ) -> Tuple[ClassifierMixin, dict]:
        """Fit FairLearn with hyperparameter tuning."""

        if not HPT_AVAILABLE:
            raise ImportError(
                "Hyperparameter tuning not available. Install hyperparameter-tuning package."
            )

        logger.info("Starting FairLearn hyperparameter tuning...")

        # Create objective function
        obj_func = ObjectiveFunction(
            X_train=X_train,
            y_train=y_train,
            s_train=s_train,
            X_val=X_val,
            y_val=y_val,
            s_val=s_val,
            hyperparameter_space=str(self.config_path),
            eval_metric=self.perf_metric,
            other_eval_metric=self.fair_metric,
            threshold=0.50,
            alpha=self.alpha,
        )

        # Create tuner
        self.tuner = OptunaTuner(
            objective_function=obj_func,
            direction="maximize",
            seed=self.seed,
        )

        # Run optimization
        self.tuner.optimize(
            n_trials=self.n_trials, n_jobs=self.n_jobs, show_progress_bar=True
        )

        # Get best model
        self.best_model = obj_func.reconstruct_model(obj_func.best_trial)

        # Log best trial information
        logger.info(f"Best trial completed successfully")

        # Access validation results for performance metrics
        if (
            hasattr(obj_func.best_trial, "validation_results")
            and obj_func.best_trial.validation_results
        ):
            val_results = obj_func.best_trial.validation_results
            if hasattr(val_results, self.perf_metric):
                logger.info(
                    f"Best {self.perf_metric}: {getattr(val_results, self.perf_metric):.4f}"
                )
            if hasattr(val_results, self.fair_metric):
                logger.info(
                    f"Best {self.fair_metric}: {getattr(val_results, self.fair_metric):.4f}"
                )

        return self.best_model, obj_func.best_trial.hyperparameters

    def fit_simple(
        self, X_train: pd.DataFrame, y_train: pd.Series, s_train: pd.Series, **kwargs
    ) -> ClassifierMixin:
        """Fit FairLearn with default parameters (no hyperparameter tuning)."""

        logger.info("Fitting FairLearn with default parameters...")

        # Create base estimator
        base_estimator = self._get_base_estimator(random_state=self.seed)

        # Create constraint
        constraint = self._get_constraint()

        # Create ExponentiatedGradient
        exp_grad = ExponentiatedGradient(
            estimator=base_estimator,
            constraints=constraint,
            eps=kwargs.get("eps", 0.01),
            nu=kwargs.get("nu", 0.01),
            eta0=kwargs.get("eta0", 2.0),
            max_iter=kwargs.get("max_iter", 50),
        )

        # Fit the model
        exp_grad.fit(X_train, y_train, sensitive_features=s_train)

        # Wrap the ExponentiatedGradient for compatibility
        wrapped_model = ExponentiatedGradientWrapper(exp_grad)

        self.best_model = wrapped_model
        return wrapped_model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.best_model is None:
            raise ValueError("Model not fitted yet.")
        return self.best_model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions."""
        if self.best_model is None:
            raise ValueError("Model not fitted yet.")

        # ExponentiatedGradient doesn't have predict_proba, but its predictors do
        if (
            hasattr(self.best_model, "predictors_")
            and len(self.best_model.predictors_) > 0
        ):
            # Use the first predictor for probability predictions
            # In practice, you might want to average across all predictors
            return self.best_model.predictors_[0].predict_proba(X)
        else:
            # Fallback: if no predictors available, raise an error
            raise AttributeError("No predictors available for probability predictions")

    @property
    def classes_(self):
        """Get class labels."""
        if self.best_model is None:
            raise ValueError("Model not fitted yet.")

        # ExponentiatedGradient doesn't have classes_, but its predictors do
        if (
            hasattr(self.best_model, "predictors_")
            and len(self.best_model.predictors_) > 0
        ):
            return self.best_model.predictors_[0].classes_
        else:
            # Fallback: try to get classes_ directly
            return self.best_model.classes_

    @property
    def feature_names_in_(self):
        """Get feature names."""
        if self.best_model is None:
            raise ValueError("Model not fitted yet.")
        # FairLearn ExponentiatedGradient should have feature_names_in_
        if hasattr(self.best_model, "feature_names_in_"):
            return self.best_model.feature_names_in_
        else:
            # Fallback: return column names from training data
            return None


def get_fairlearn_config_path(
    constraint_type: str, base_estimator_type: str = "LogisticRegression"
) -> Path:
    """Map constraint type and base estimator to appropriate config file."""

    # Map base estimator types to directory names
    estimator_to_dir = {
        "LogisticRegression": "lr",
        "RandomForestClassifier": "rf",
        "GradientBoostingClassifier": "gb",
    }

    # Map constraint types to filenames
    constraint_to_file = {
        "DemographicParity": "demographic_parity.yaml",
        "EqualizedOdds": "equalized_odds.yaml",
        "EqualOpportunity": "equal_opportunity.yaml",
        "PredictiveEquality": "predictive_equality.yaml",
    }

    estimator_dir = estimator_to_dir.get(base_estimator_type, "lr")
    filename = constraint_to_file.get(constraint_type, "demographic_parity.yaml")

    return Path(f"config/fairlearn/{estimator_dir}/{filename}")


def parse_fairlearn_model_tag(model_tag: str) -> Tuple[str, str]:
    """Parse model tag to extract base estimator type and constraint type."""

    # Default values
    base_estimator_type = "LogisticRegression"
    constraint_type = "DemographicParity"

    # Extract base estimator type
    if "_LR" in model_tag:
        base_estimator_type = "LogisticRegression"
    elif "_RF" in model_tag:
        base_estimator_type = "RandomForestClassifier"
    elif "_GB" in model_tag:
        base_estimator_type = "GradientBoostingClassifier"

    # Extract constraint type
    if "demographic_parity" in model_tag:
        constraint_type = "DemographicParity"
    elif "equalized_odds" in model_tag:
        constraint_type = "EqualizedOdds"
    elif "equal_opportunity" in model_tag:
        constraint_type = "EqualOpportunity"
    elif "predictive_equality" in model_tag:
        constraint_type = "PredictiveEquality"

    return base_estimator_type, constraint_type


def prepare_fairlearn_data(
    enc_dataset: EncodedDatasetWrapper, model_tag: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare training and test data for FairLearn."""

    X_train = enc_dataset.X_enc_train.copy()
    X_test = enc_dataset.X_enc_test.copy()

    # Handle sensitive attribute inclusion/exclusion
    if "no_sensitive" in model_tag:
        logger.info(
            f"Dropping {enc_dataset.enc_sensitive_name} for model tag {model_tag}"
        )
        X_train = X_train.drop(columns=enc_dataset.enc_sensitive_name)
        X_test = X_test.drop(columns=enc_dataset.enc_sensitive_name)

    return X_train, X_test


def prepare_sensitive_attributes(
    enc_dataset: EncodedDatasetWrapper,
) -> Tuple[pd.Series, pd.Series]:
    """Prepare sensitive attributes for FairLearn (convert to numeric 0/1)."""

    s_train = enc_dataset.sensitive_train.copy()
    s_test = enc_dataset.sensitive_test.copy()

    # Convert string values to numeric (assuming binary sensitive attribute)
    unique_values = sorted(enc_dataset.sensitive_train.unique())
    if len(unique_values) != 2:
        raise ValueError(
            f"Expected binary sensitive attribute, got {len(unique_values)} unique values: {unique_values}"
        )

    # Map to 0 and 1
    value_map = {unique_values[0]: 0, unique_values[1]: 1}
    s_train_numeric = s_train.map(value_map).astype(int)
    s_test_numeric = s_test.map(value_map).astype(int)

    logger.info(f"Converted sensitive attributes: {value_map}")

    return s_train_numeric, s_test_numeric


def create_validation_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    s_train: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Create validation split from training data."""

    from sklearn.model_selection import train_test_split

    (
        X_train_split,
        X_val_split,
        y_train_split,
        y_val_split,
        s_train_split,
        s_val_split,
    ) = train_test_split(
        X_train,
        y_train,
        s_train,
        test_size=test_size,
        random_state=random_state,
        stratify=y_train,
    )

    # Reset indices to ensure consistency
    X_train_split = X_train_split.reset_index(drop=True)
    X_val_split = X_val_split.reset_index(drop=True)
    y_train_split = y_train_split.reset_index(drop=True)
    y_val_split = y_val_split.reset_index(drop=True)
    s_train_split = s_train_split.reset_index(drop=True)
    s_val_split = s_val_split.reset_index(drop=True)

    return (
        X_train_split,
        X_val_split,
        y_train_split,
        y_val_split,
        s_train_split,
        s_val_split,
    )


def fit_base_estimator(
    base_estimator_type: str,
    enc_dataset: EncodedDatasetWrapper,
    output_dir: Path,
    seed: int = 42,
) -> Tuple[ClassifierMixin, pd.Series]:
    """Fit base estimator without fairness constraints (for no_sensitive variants)."""

    logger.info(
        f"Fitting base estimator {base_estimator_type} without fairness constraints"
    )

    # Prepare data
    X_train = enc_dataset.X_enc_train.copy()
    X_test = enc_dataset.X_enc_test.copy()

    # Drop sensitive attributes
    if hasattr(enc_dataset, "enc_sensitive_name"):
        X_train = X_train.drop(columns=enc_dataset.enc_sensitive_name)
        X_test = X_test.drop(columns=enc_dataset.enc_sensitive_name)

    # Create base estimator
    estimator_map = {
        "LogisticRegression": LogisticRegression,
        "RandomForestClassifier": RandomForestClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
    }

    if base_estimator_type not in estimator_map:
        raise ValueError(f"Unknown base estimator type: {base_estimator_type}")

    classifier = estimator_map[base_estimator_type](random_state=seed)

    # Fit and predict
    classifier.fit(X_train, enc_dataset.y_train)
    y_pred = pd.Series(
        classifier.predict(X_test),
        index=enc_dataset.X_enc_test.index,
    )

    # Evaluate and save results
    model_performance = evaluate_performance(enc_dataset.y_test, y_pred)
    group_fairness = evaluate_group_fairness(
        enc_dataset.y_test, y_pred, enc_dataset.sensitive_test
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    import json

    logger.info(f"Model performance:\n{model_performance}")
    with open(output_dir / "model_performance.json", "w") as f:
        json.dump(model_performance, f, indent=4)

    logger.info(f"Group fairness:\n{group_fairness}")
    with open(output_dir / "group_fairness.json", "w") as f:
        json.dump(group_fairness, f, indent=4)

    return classifier, y_pred


def save_fairlearn_results(
    classifier: ClassifierMixin,
    y_pred: pd.Series,
    best_params: dict,
    output_dir: Path,
    tuning_results: Optional[dict] = None,
) -> None:
    """Save FairLearn results to output directory."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate performance and fairness
    # Note: We need the original test data for evaluation
    # This will be handled in the main fit function

    # Save hyperparameter tuning results
    if tuning_results is not None:
        import json

        with open(output_dir / "best_hyperparameters.json", "w") as f:
            json.dump(best_params, f, indent=4)

        if hasattr(tuning_results, "results"):
            tuning_results.results.to_csv(
                output_dir / "hyperparameter_tuning_results.csv"
            )

    # Save the trained model
    with open(output_dir / "fairlearn_model.pkl", "wb") as f:
        pickle.dump(classifier, f)


def fit_fairlearn_classifier(
    model_tag: str,
    enc_dataset: EncodedDatasetWrapper,
    output_dir: Path,
    config_path: Optional[Path] = None,
    n_trials: int = 20,
    n_jobs: int = 1,
    seed: int = 42,
) -> Tuple[ClassifierMixin, pd.Series]:
    """Fit and evaluate FairLearn ExponentiatedGradient classifier."""

    if not FAIRLEARN_AVAILABLE:
        raise ImportError("FairLearn not available. Install required packages.")

    logger.info(f"Fitting FairLearn classifier: {model_tag}")

    # Parse model tag
    base_estimator_type, constraint_type = parse_fairlearn_model_tag(model_tag)

    # Handle no_sensitive variants
    if "no_sensitive" in model_tag:
        logger.warning(
            "FairLearn without sensitive features will not enforce fairness constraints"
        )
        return fit_base_estimator(base_estimator_type, enc_dataset, output_dir, seed)

    # Set default config path if not provided
    if config_path is None:
        config_path = get_fairlearn_config_path(constraint_type, base_estimator_type)

    # Prepare data
    X_train, X_test = prepare_fairlearn_data(enc_dataset, model_tag)
    s_train, s_test = prepare_sensitive_attributes(enc_dataset)

    # Create validation split
    (
        X_train_split,
        X_val_split,
        y_train_split,
        y_val_split,
        s_train_split,
        s_val_split,
    ) = create_validation_split(
        X_train, enc_dataset.y_train, s_train, random_state=seed
    )

    # Initialize FairLearn wrapper
    fairlearn_wrapper = FairLearnWrapper(
        base_estimator_type=base_estimator_type,
        constraint_type=constraint_type,
        config_path=config_path,
        n_trials=n_trials,
        n_jobs=n_jobs,
        seed=seed,
    )

    # For now, use simple fitting (we'll add hyperparameter tuning later)
    # TODO: Add hyperparameter tuning support
    classifier = fairlearn_wrapper.fit_simple(
        X_train_split, y_train_split, s_train_split
    )

    # Make predictions on test set
    y_pred = pd.Series(
        classifier.predict(X_test),
        index=enc_dataset.X_enc_test.index,
    )

    # Evaluate performance and fairness
    model_performance = evaluate_performance(enc_dataset.y_test, y_pred)
    group_fairness = evaluate_group_fairness(
        enc_dataset.y_test, y_pred, enc_dataset.sensitive_test
    )

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    import json

    logger.info(f"Model performance:\n{model_performance}")
    with open(output_dir / "model_performance.json", "w") as f:
        json.dump(model_performance, f, indent=4)

    logger.info(f"Group fairness:\n{group_fairness}")
    with open(output_dir / "group_fairness.json", "w") as f:
        json.dump(group_fairness, f, indent=4)

    # Save default parameters
    default_params = {
        "base_estimator_type": base_estimator_type,
        "constraint_type": constraint_type,
        "eps": 0.01,
        "nu": 0.01,
        "eta0": 2.0,
        "max_iter": 50,
        "note": "Default parameters used (no hyperparameter tuning)",
    }
    with open(output_dir / "best_hyperparameters.json", "w") as f:
        json.dump(default_params, f, indent=4)

    # Save the trained model
    with open(output_dir / "fairlearn_model.pkl", "wb") as f:
        pickle.dump(classifier, f)

    return classifier, y_pred
