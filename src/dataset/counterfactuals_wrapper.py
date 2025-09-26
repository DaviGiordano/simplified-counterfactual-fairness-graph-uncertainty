import logging
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.dataset.dataset_wrappers import EncodedDatasetWrapper
from src.metrics.counterfactual_metrics import evaluate_counterfactual_world
from src.metrics.model_metrics import compute_cf_metrics

logger = logging.getLogger(__name__)


class MultiWorldCounterfactuals:

    def __init__(self, counterfactuals: pd.DataFrame) -> None:
        self.counterfactuals = counterfactuals

        self.scores: Optional[pd.Series] = None
        self.counterfactual_predictions: Optional[pd.Series] = None
        self.cf_metrics: Optional[pd.DataFrame] = None
        self.counterfactuals_quality: Optional[pd.DataFrame] = None

    def score_counterfactuals(self, classifier):
        """Computes and stores the scores of the counterfactuals, for a given classifier"""
        # Get feature names with fallback for different classifier types
        if hasattr(classifier, "feature_names_in_"):
            feature_names = classifier.feature_names_in_
        elif hasattr(classifier, "feature_name_"):
            feature_names = classifier.feature_name_
        elif hasattr(classifier, "_Booster") and hasattr(
            classifier._Booster, "feature_name"
        ):
            feature_names = classifier._Booster.feature_name()
        else:
            # Fallback: use all columns from counterfactuals
            feature_names = self.counterfactuals.columns.tolist()

        logger.info(
            f"Predicting counterfactual scores with features: {list(feature_names)}"
        )

        idx_positive_class = np.where(classifier.classes_ == 1)[0][0]  # type: ignore
        self.scores = pd.Series(
            classifier.predict_proba(self.counterfactuals[list(feature_names)])[
                :, idx_positive_class
            ],
            index=self.counterfactuals.index,
        )

    def predict_counterfactuals(self, classifier: pd.Series):
        """
        Computes and stores counterfactual predictions, given a classifier
        """
        # Get feature names with fallback for different classifier types
        if hasattr(classifier, "feature_names_in_"):
            feature_names = classifier.feature_names_in_
        elif hasattr(classifier, "feature_name_"):
            feature_names = classifier.feature_name_
        elif hasattr(classifier, "_Booster") and hasattr(
            classifier._Booster, "feature_name"
        ):
            feature_names = classifier._Booster.feature_name()
        else:
            # Fallback: use all columns from counterfactuals
            feature_names = self.counterfactuals.columns.tolist()

        logger.info(
            f"Predicting counterfactual outputs with features {list(feature_names)}"
        )

        self.counterfactual_predictions = pd.Series(
            classifier.predict(self.counterfactuals[list(feature_names)]),
            index=self.counterfactuals.index,
        )

    def evaluate_counterfactual_fairness(
        self,
        classifier,
        original_predictions: pd.Series,
        original_sensitive_feature: pd.Series,
    ):
        """
        Evaluates counterfactual fairness, given a classifier and original predictions.
        If the counterfactual predictions were not computed yet, runs `predict_counterfactuals`
        """
        if self.counterfactual_predictions is None:
            self.predict_counterfactuals(classifier)

        cf_metrics_list = []
        for cw_id, cf_preds in self.counterfactual_predictions.groupby(  # type: ignore
            level="causal_world"
        ):
            cf_preds = cf_preds.droplevel("causal_world")
            curr_metrics = compute_cf_metrics(
                original_predictions,
                cf_preds,
                original_sensitive_feature,
            )
            curr_metrics["causal_world"] = int(cw_id)  # type: ignore
            cf_metrics_list.append(curr_metrics)

        cf_metrics_df = pd.DataFrame.from_records(cf_metrics_list)
        cf_metrics_df.set_index("causal_world", inplace=True)
        self.cf_metrics = cf_metrics_df

    def evaluate_counterfactuals_quality(
        self,
        original_features: pd.DataFrame,
        original_sensitive: pd.Series,
        num_workers=1,
    ):
        logger.info(f"Evaluating counterfactuals.")
        quality_list = []
        for cw_id, curr_counterfactuals in self.counterfactuals.groupby(  # type: ignore
            level="causal_world"
        ):
            curr_counterfactuals = curr_counterfactuals.droplevel("causal_world")
            curr_metrics = evaluate_counterfactual_world(
                original_features,
                curr_counterfactuals,
                original_sensitive,
            )
            curr_metrics["causal_world"] = int(cw_id)  # type: ignore
            quality_list.append(curr_metrics)
        quality_df = pd.DataFrame.from_records(quality_list)
        quality_df.set_index("causal_world", inplace=True)
        self.counterfactuals_quality = quality_df

    def get_causal_world_ids(self) -> List[str]:
        """Returns a list of causal worlds in the counterfactuals dataset."""
        assert (
            "causal_world" in self.counterfactuals.index.names
        ), "causal_world must be a named index"
        return self.counterfactuals.index.unique("causal_world").tolist()

    def get_individual_ids(self) -> List[str]:
        """Returns a list of causal worlds in the counterfactuals dataset."""
        assert (
            "individual" in self.counterfactuals.index.names
        ), "individual must be a named index"
        return self.counterfactuals.index.unique("individual").tolist()

    def get_values_variance_by_individual(
        self,
    ) -> pd.Series:
        """
        Get mean, variance, upper and lower cis of some given data by the index 'individual'
        Use to summarize the counterfactuals or the score.
        """
        variance = self.counterfactuals.groupby(level="individual").var()
        return (
            variance.iloc[0]
            .sort_values(ascending=False)
            .rename("variance_by_individual")
        )

    # def get_causal_world_counterfactuals(
    #     self, causal_world_id
    # ) -> Union[pd.DataFrame, pd.Series]:
    #     """Returns the causal world of a given id"""
    #     if "causal_world" not in self.counterfactuals.index.names:
    #         raise ValueError("causal_world must be a named index")
    #     return self.counterfactuals.xs(causal_world_id, level="causal_world")

    # def get_individual_counterfactuals(
    #     self, individual_id
    # ) -> Union[pd.DataFrame, pd.Series]:
    #     """Returns the causal world of a given id"""
    #     if "individual" not in self.counterfactuals.index.names:
    #         raise ValueError("individual must be a named index")
    #     return self.counterfactuals.xs(individual_id, level="individual")

    # def get_causal_world_scores(
    #     self, causal_world_id
    # ) -> Union[pd.DataFrame, pd.Series]:
    #     """Returns the causal world of a given id"""
    #     if self.scores is None:
    #         raise ValueError(
    #             "Scores have not yet been computed. Call `score_counterfactuals`"
    #         )
    #     if "causal_world" not in self.scores.index.names:
    #         raise ValueError("causal_world must be a named index")
    #     return self.scores.xs(causal_world_id, level="causal_world")

    # def get_individual_scores(self, individual_id) -> Union[pd.DataFrame, pd.Series]:
    #     """Returns the causal world of a given id"""
    #     if self.scores is None:
    #         raise ValueError(
    #             "Scores have not yet been computed. Call `score_counterfactuals`"
    #         )
    #     if "individual" not in self.scores.index.names:
    #         raise ValueError("individual must be a named index")
    #     return self.counterfactuals.xs(individual_id, level="individual")

    def get_v(self) -> pd.Series:
        """Returns the variance of scores across causal worlds for each individual"""
        if self.scores is None:
            raise ValueError(
                "Scores have not been computed. Call `score_counterfactuals`"
            )
        return self.scores.groupby(level="individual").var()
