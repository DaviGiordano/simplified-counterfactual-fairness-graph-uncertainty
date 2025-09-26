import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

from src.dataset.dataset_wrappers import EncodedDatasetWrapper
from src.metrics.model_metrics import evaluate_group_fairness, evaluate_performance

logger = logging.getLogger(__name__)

try:
    from fairgbm import FairGBMClassifier
    from hpt.tuner import ObjectiveFunction, OptunaTuner

    FAIRGBM_AVAILABLE = True
except ImportError:
    FAIRGBM_AVAILABLE = False
    logger.warning(
        "FAIRGBM not available. Install with: pip install fairgbm hyperparameter-tuning"
    )


class FairGBMWrapper:
    """Wrapper for FAIRGBM classifier with hyperparameter tuning."""

    def __init__(
        self,
        config_path: Path,
        n_trials: int = 20,
        n_jobs: int = 1,
        seed: int = 42,
        perf_metric: str = "accuracy",
        fair_metric: str = "equalized_odds_ratio",
        alpha: float = 0.5,
    ):
        if not FAIRGBM_AVAILABLE:
            raise ImportError("FAIRGBM not available. Install required packages.")

        self.config_path = config_path
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.seed = seed
        self.perf_metric = perf_metric
        self.fair_metric = fair_metric
        self.alpha = alpha
        self.tuner = None
        self.best_model = None

    def fit_with_tuning(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        s_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        s_val: pd.Series,
    ) -> Tuple[ClassifierMixin, dict]:
        """Fit FAIRGBM with hyperparameter tuning."""

        logger.info("Starting FAIRGBM hyperparameter tuning...")

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

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.best_model is None:
            raise ValueError("Model not fitted yet.")
        return self.best_model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions."""
        if self.best_model is None:
            raise ValueError("Model not fitted yet.")
        return self.best_model.predict_proba(X)

    @property
    def classes_(self):
        """Get class labels."""
        if self.best_model is None:
            raise ValueError("Model not fitted yet.")
        return self.best_model.classes_

    @property
    def feature_names_in_(self):
        """Get feature names."""
        if self.best_model is None:
            raise ValueError("Model not fitted yet.")
        # FAIRGBM uses feature_name_ instead of feature_names_in_
        if hasattr(self.best_model, "feature_names_in_"):
            return self.best_model.feature_names_in_
        elif hasattr(self.best_model, "feature_name_"):
            return self.best_model.feature_name_
        else:
            # Fallback: return column names from training data
            return (
                self.best_model._Booster.feature_name()
                if hasattr(self.best_model, "_Booster")
                else None
            )


def get_config_path(model_tag: str) -> Path:
    """Map model tag to appropriate config file."""
    if "equal_opportunity" in model_tag:
        return Path("config/fairgbm/equal_opportunity.yaml")
    elif "predictive_equality" in model_tag:
        return Path("config/fairgbm/predictive_equality.yaml")
    else:
        return Path("config/fairgbm/equalized_odds.yaml")


def fit_fairgbm_classifier(
    model_tag: str,
    enc_dataset: EncodedDatasetWrapper,
    output_dir: Path,
    config_path: Optional[Path] = None,
    n_trials: int = 20,
    n_jobs: int = 1,
    seed: int = 42,
) -> Tuple[ClassifierMixin, pd.Series]:
    """Fit and evaluate FAIRGBM classifier."""

    if not FAIRGBM_AVAILABLE:
        raise ImportError("FAIRGBM not available. Install required packages.")

    logger.info(f"Fitting FAIRGBM classifier: {model_tag}")

    # Set default config path if not provided
    if config_path is None:
        config_path = get_config_path(model_tag)

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

    # Create validation split from training data
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
        enc_dataset.y_train,
        s_train_numeric,
        test_size=0.2,
        random_state=seed,
        stratify=enc_dataset.y_train,
    )

    # Reset indices to ensure consistency for hpt library
    X_train_split = X_train_split.reset_index(drop=True)
    X_val_split = X_val_split.reset_index(drop=True)
    y_train_split = y_train_split.reset_index(drop=True)
    y_val_split = y_val_split.reset_index(drop=True)
    s_train_split = s_train_split.reset_index(drop=True)
    s_val_split = s_val_split.reset_index(drop=True)

    # Initialize FAIRGBM wrapper
    fairgbm_wrapper = FairGBMWrapper(
        config_path=config_path,
        n_trials=n_trials,
        n_jobs=n_jobs,
        seed=seed,
    )

    # Fit with hyperparameter tuning
    if "no_sensitive" in model_tag:
        # For no_sensitive case, we can't use fairness constraints in hyperparameter tuning
        # Use a simple approach without hyperparameter tuning for fairness
        logger.warning(
            "No_sensitive FAIRGBM models will use default parameters (no hyperparameter tuning for fairness)"
        )

        # Create a simple FairGBMClassifier with default parameters
        from fairgbm import FairGBMClassifier

        classifier = FairGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            reg_alpha=0.0,
            reg_lambda=0.0,
            boosting_type="gbdt",
            random_state=seed,
            verbose=-1,
        )
        classifier.fit(X_train_split, y_train_split)
        best_params = {}
    else:
        # Fit with hyperparameter tuning for fairness-aware models
        classifier, best_params = fairgbm_wrapper.fit_with_tuning(
            X_train_split,
            y_train_split,
            s_train_split,
            X_val_split,
            y_val_split,
            s_val_split,
        )

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

    # Save hyperparameter tuning results
    if fairgbm_wrapper.tuner is not None and best_params:
        tuning_results = fairgbm_wrapper.tuner.results
        tuning_results.to_csv(output_dir / "hyperparameter_tuning_results.csv")

        # Save best parameters
        with open(output_dir / "best_hyperparameters.json", "w") as f:
            json.dump(best_params, f, indent=4)
    elif "no_sensitive" in model_tag:
        # Save default parameters for no_sensitive models
        default_params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "boosting_type": "gbdt",
            "note": "Default parameters used for no_sensitive model (no hyperparameter tuning)",
        }
        with open(output_dir / "best_hyperparameters.json", "w") as f:
            json.dump(default_params, f, indent=4)

    # Save the trained model
    with open(output_dir / "fairgbm_model.pkl", "wb") as f:
        pickle.dump(classifier, f)

    return classifier, y_pred
