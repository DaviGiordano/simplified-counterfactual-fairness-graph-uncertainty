"""
Hyperparameter tuning engine using Optuna for Bayesian optimization.
Supports single-objective (F1-score) and multi-objective (F1-score + fairness) optimization.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from src.classification.config_loader import ClassifierConfigLoader, TuningConfigLoader
from src.classification.objective_evaluators import ObjectiveEvaluatorFactory
from src.log.mlflow import log_to_mlflow
from src.metrics.model_metrics import evaluate_group_fairness, evaluate_performance

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Main hyperparameter tuning orchestrator."""

    def __init__(
        self, classifier_config_dir: Path, tuning_config_file: Path, output_dir: Path
    ):
        """
        Initialize hyperparameter tuner.

        Args:
            classifier_config_dir: Directory containing classifier configurations
            tuning_config_file: Path to tuning configuration file
            output_dir: Output directory for results
        """
        self.classifier_config_dir = Path(classifier_config_dir)
        self.tuning_config_file = Path(tuning_config_file)
        self.output_dir = Path(output_dir)

        # Load configurations
        self.classifier_loader = ClassifierConfigLoader(self.classifier_config_dir)
        self.tuning_loader = TuningConfigLoader(self.tuning_config_file)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def tune_classifier(
        self,
        classifier_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sensitive_train: pd.Series = None,
        X_test: pd.DataFrame = None,
        y_test: pd.Series = None,
        sensitive_test: pd.Series = None,
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters for a classifier.

        Args:
            classifier_name: Name of the classifier (LR, RF, GB, FAIRGBM)
            X_train: Training features
            y_train: Training targets
            sensitive_train: Training sensitive attributes
            X_test: Test features (optional, for final evaluation)
            y_test: Test targets (optional, for final evaluation)
            sensitive_test: Test sensitive attributes (optional, for final evaluation)

        Returns:
            Dictionary containing tuning results
        """
        logger.info(f"Starting hyperparameter tuning for {classifier_name}")

        # Load configurations
        classifier_config = self.classifier_loader.load_classifier_config(
            classifier_name
        )
        tuning_config = self.tuning_loader.get_tuning_config()

        # Get optimization objective
        optimization_objective = self.classifier_loader.get_optimization_objective(
            classifier_name
        )

        # Create study
        study = self._create_study(
            classifier_name, optimization_objective, tuning_config
        )

        # Create objective function
        objective_func = self._create_objective_function(
            classifier_name, X_train, y_train, sensitive_train, optimization_objective
        )

        # Run optimization
        logger.info(f"Running {tuning_config['n_trials']} trials for {classifier_name}")
        study.optimize(
            objective_func,
            n_trials=tuning_config["n_trials"],
            timeout=tuning_config.get("timeout", None),
        )

        # Get best results
        best_params = study.best_params
        best_score = study.best_value

        logger.info(f"Best score for {classifier_name}: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")

        # Train final model with best parameters
        final_model = self._train_final_model(
            classifier_name, best_params, X_train, y_train, sensitive_train
        )

        # Evaluate on test set if provided
        test_results = {}
        if X_test is not None and y_test is not None:
            test_results = self._evaluate_final_model(
                final_model, X_test, y_test, sensitive_test
            )

        # Save results
        results = {
            "classifier_name": classifier_name,
            "best_params": best_params,
            "best_score": best_score,
            "study": study,
            "final_model": final_model,
            "test_results": test_results,
            "optimization_objective": optimization_objective,
        }

        self._save_results(results)

        return results

    def _create_study(
        self,
        classifier_name: str,
        optimization_objective: Union[str, Dict[str, Any]],
        tuning_config: Dict[str, Any],
    ) -> optuna.Study:
        """
        Create Optuna study for optimization.

        Args:
            classifier_name: Name of the classifier
            optimization_objective: Optimization objective
            tuning_config: Tuning configuration

        Returns:
            Optuna study
        """
        # Create study name
        study_name = f"{classifier_name}_hyperparameter_tuning"

        # Create sampler
        sampler = TPESampler(seed=tuning_config["random_state"])

        # Create pruner
        pruner = (
            MedianPruner()
            if tuning_config.get("pruning", {}).get("enabled", True)
            else None
        )

        # Create study
        study = optuna.create_study(
            direction="maximize",  # We always maximize (F1-score or combined score)
            study_name=study_name,
            sampler=sampler,
            pruner=pruner,
        )

        return study

    def _create_objective_function(
        self,
        classifier_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sensitive_train: pd.Series,
        optimization_objective: Union[str, Dict[str, Any]],
    ):
        """
        Create objective function for optimization.

        Args:
            classifier_name: Name of the classifier
            X_train: Training features
            y_train: Training targets
            sensitive_train: Training sensitive attributes
            optimization_objective: Optimization objective

        Returns:
            Objective function
        """

        def objective(trial):
            try:
                # Sample hyperparameters
                params = self._sample_hyperparameters(trial, classifier_name)

                # Create classifier
                classifier = self._create_classifier(classifier_name, params)

                # Create evaluator
                evaluator = ObjectiveEvaluatorFactory.create_evaluator(
                    classifier_name, optimization_objective
                )

                # Evaluate
                score = evaluator.evaluate(
                    classifier, X_train, y_train, sensitive_train
                )

                return score

            except Exception as e:
                logger.error(f"Error in trial: {e}")
                return 0.0  # Return worst possible score

        return objective

    def _sample_hyperparameters(
        self, trial: optuna.Trial, classifier_name: str
    ) -> Dict[str, Any]:
        """
        Sample hyperparameters for a trial.

        Args:
            trial: Optuna trial
            classifier_name: Name of the classifier

        Returns:
            Dictionary of sampled hyperparameters
        """
        search_space = self.classifier_loader.get_search_space(classifier_name)
        default_params = self.classifier_loader.get_default_params(classifier_name)

        params = default_params.copy()

        for param_name, param_config in search_space.items():
            if isinstance(param_config, list):
                # Categorical parameter
                params[param_name] = trial.suggest_categorical(param_name, param_config)
            else:
                # Numeric parameter
                param_type = param_config["type"]
                param_range = param_config["range"]
                use_log = param_config.get("log", False)

                if param_type == "int":
                    params[param_name] = trial.suggest_int(
                        param_name,
                        int(param_range[0]),
                        int(param_range[1]),
                        log=use_log,
                    )
                elif param_type == "float":
                    params[param_name] = trial.suggest_float(
                        param_name,
                        float(param_range[0]),
                        float(param_range[1]),
                        log=use_log,
                    )

        return params

    def _create_classifier(
        self, classifier_name: str, params: Dict[str, Any]
    ) -> BaseEstimator:
        """
        Create classifier instance with given parameters.

        Args:
            classifier_name: Name of the classifier
            params: Hyperparameters

        Returns:
            Classifier instance
        """
        classpath = self.classifier_loader.get_classpath(classifier_name)

        # Import the class
        module_name, class_name = classpath.rsplit(".", 1)
        module = __import__(module_name, fromlist=[class_name])
        classifier_class = getattr(module, class_name)

        # Create instance
        classifier = classifier_class(**params)

        return classifier

    def _train_final_model(
        self,
        classifier_name: str,
        best_params: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sensitive_train: pd.Series,
    ) -> BaseEstimator:
        """
        Train final model with best parameters.

        Args:
            classifier_name: Name of the classifier
            best_params: Best hyperparameters
            X_train: Training features
            y_train: Training targets
            sensitive_train: Training sensitive attributes

        Returns:
            Trained classifier
        """
        classifier = self._create_classifier(classifier_name, best_params)

        # Handle FairGBM special case
        if classifier_name == "FAIRGBM" and sensitive_train is not None:
            classifier.fit(X_train, y_train, constraint_group=sensitive_train)
        else:
            classifier.fit(X_train, y_train)

        return classifier

    def _evaluate_final_model(
        self,
        model: BaseEstimator,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        sensitive_test: pd.Series = None,
    ) -> Dict[str, float]:
        """
        Evaluate final model on test set.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            sensitive_test: Test sensitive attributes

        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = model.predict(X_test)
        # Compute comprehensive metrics using shared utilities
        model_performance = evaluate_performance(
            y_true=y_test,
            y_pred=pd.Series(y_pred, index=y_test.index),
        )

        group_fairness: Dict[str, float] = {}
        if sensitive_test is not None:
            group_fairness = evaluate_group_fairness(
                y_true=y_test,
                y_pred=pd.Series(y_pred, index=y_test.index),
                sensitive_features=sensitive_test,
            )

        return {
            "model_performance": model_performance,
            "group_fairness": group_fairness,
        }

    def _save_results(self, results: Dict[str, Any]) -> None:
        """
        Save tuning results.

        Args:
            results: Tuning results
        """
        classifier_name = results["classifier_name"]
        output_file = self.output_dir / f"{classifier_name}_tuning_results.pkl"

        # Remove study and model from results for saving (they're large)
        save_results = results.copy()
        save_results.pop("study", None)
        save_results.pop("final_model", None)

        with open(output_file, "wb") as f:
            pickle.dump(save_results, f)

        logger.info(f"Saved tuning results to {output_file}")

        # Save model separately
        model_file = self.output_dir / f"{classifier_name}_best_model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(results["final_model"], f)

        logger.info(f"Saved best model to {model_file}")
