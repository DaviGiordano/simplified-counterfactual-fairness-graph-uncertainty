"""
Objective evaluators for hyperparameter tuning.
Handles single-objective (F1-score) and multi-objective (F1-score + fairness) evaluation.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

logger = logging.getLogger(__name__)


class SingleObjectiveEvaluator:
    """Evaluator for single-objective optimization (F1-score for LR, RF, GB)."""

    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        """
        Initialize single-objective evaluator.

        Args:
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.cv = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=random_state
        )

    def evaluate(
        self,
        classifier: Any,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_attributes: Optional[pd.Series] = None,
    ) -> float:
        """
        Evaluate classifier using cross-validation.

        Args:
            classifier: Trained classifier
            X: Feature matrix
            y: Target variable
            sensitive_attributes: Sensitive attributes (not used for single-objective)

        Returns:
            F1-score (maximize)
        """
        try:
            # Perform cross-validation
            cv_scores = cross_val_score(
                classifier,
                X,
                y,
                cv=self.cv,
                scoring="f1",
                n_jobs=1,  # Avoid nested parallelism
            )

            mean_f1 = float(cv_scores.mean())
            logger.debug(f"CV F1-scores: {cv_scores}, Mean: {mean_f1:.4f}")

            return mean_f1

        except Exception as e:
            logger.error(f"Error in single-objective evaluation: {e}")
            return 0.0  # Return worst possible score


class MultiObjectiveEvaluator:
    """Evaluator for multi-objective optimization (F1-score + fairness for FairGBM)."""

    def __init__(
        self,
        cv_folds: int = 5,
        random_state: int = 42,
        weights: List[float] = [0.7, 0.3],
    ):
        """
        Initialize multi-objective evaluator.

        Args:
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
            weights: Weights for [f1_score, fairness] objectives
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.weights = weights
        self.cv = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=random_state
        )

    def evaluate(
        self,
        classifier: Any,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_attributes: pd.Series,
    ) -> float:
        """
        Evaluate classifier using cross-validation with multi-objective optimization.

        Args:
            classifier: Trained classifier
            X: Feature matrix
            y: Target variable
            sensitive_attributes: Sensitive attributes for fairness evaluation

        Returns:
            Combined objective score (maximize)
        """
        try:
            f1_scores = []
            fairness_scores = []

            for train_idx, val_idx in self.cv.split(X, y):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                s_train_fold, s_val_fold = (
                    sensitive_attributes.iloc[train_idx],
                    sensitive_attributes.iloc[val_idx],
                )

                # Train classifier on fold
                fold_classifier = self._clone_classifier(classifier)

                # Handle FairGBM special case
                if (
                    hasattr(fold_classifier, "fit")
                    and "constraint_group" in fold_classifier.fit.__code__.co_varnames
                ):
                    fold_classifier.fit(
                        X_train_fold, y_train_fold, constraint_group=s_train_fold
                    )
                else:
                    fold_classifier.fit(X_train_fold, y_train_fold)

                # Make predictions
                y_pred_fold = fold_classifier.predict(X_val_fold)

                # Calculate F1-score
                f1_fold = f1_score(y_val_fold, y_pred_fold)
                f1_scores.append(f1_fold)

                # Calculate fairness metric (equalized odds difference)
                fairness_fold = abs(
                    equalized_odds_difference(
                        y_val_fold, y_pred_fold, sensitive_features=s_val_fold
                    )
                )
                fairness_scores.append(fairness_fold)

            # Calculate mean scores
            mean_f1 = float(np.mean(f1_scores))
            mean_fairness = float(np.mean(fairness_scores))

            # Combine objectives (F1-score to maximize, fairness difference to minimize)
            # Convert fairness to a maximization problem by using (1 - fairness)
            combined_score = float(
                self.weights[0] * mean_f1 + self.weights[1] * (1 - mean_fairness)
            )

            logger.debug(f"CV F1-scores: {f1_scores}, Mean: {mean_f1:.4f}")
            logger.debug(
                f"CV Fairness scores: {fairness_scores}, Mean: {mean_fairness:.4f}"
            )
            logger.debug(f"Combined score: {combined_score:.4f}")

            return combined_score

        except Exception as e:
            logger.error(f"Error in multi-objective evaluation: {e}")
            return 0.0  # Return worst possible score

    def _clone_classifier(self, classifier: Any) -> Any:
        """
        Clone a classifier to avoid modifying the original.

        Args:
            classifier: Original classifier

        Returns:
            Cloned classifier
        """
        try:
            # Try to use sklearn's clone if available
            from sklearn.base import clone

            return clone(classifier)
        except:
            # Fallback: create new instance with same parameters
            import copy

            return copy.deepcopy(classifier)


class ObjectiveEvaluatorFactory:
    """Factory for creating appropriate objective evaluators."""

    @staticmethod
    def create_evaluator(
        classifier_name: str,
        optimization_objective: Union[str, Dict[str, Any]],
        cv_folds: int = 5,
        random_state: int = 42,
        weights: List[float] = [0.7, 0.3],
    ) -> Union[SingleObjectiveEvaluator, MultiObjectiveEvaluator]:
        """
        Create appropriate evaluator based on classifier and objective.

        Args:
            classifier_name: Name of the classifier
            optimization_objective: Optimization objective configuration
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
            weights: Weights for multi-objective optimization

        Returns:
            Appropriate evaluator instance
        """
        if (
            optimization_objective == "f1_score"
            or optimization_objective == "single_objective"
        ):
            return SingleObjectiveEvaluator(cv_folds, random_state)
        elif optimization_objective == "multi_objective":
            return MultiObjectiveEvaluator(cv_folds, random_state, weights)
        else:
            raise ValueError(
                f"Unknown optimization objective: {optimization_objective}"
            )
