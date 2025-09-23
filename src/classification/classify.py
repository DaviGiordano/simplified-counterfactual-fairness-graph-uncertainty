import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.dataset.dataset_wrappers import EncodedDatasetWrapper
from src.metrics.model_metrics import evaluate_group_fairness, evaluate_performance

logger = logging.getLogger(__name__)


def fit_evaluate_classifier(
    model_tag: str,
    enc_dataset: EncodedDatasetWrapper,
    output_dir: Path,
) -> tuple[ClassifierMixin, pd.Series]:
    MODEL_MAPPING = {
        "LR": LogisticRegression,
        "RF": RandomForestClassifier,
        "GB": GradientBoostingClassifier,
        "LR_no_sensitive": LogisticRegression,
        "RF_no_sensitive": RandomForestClassifier,
        "GB_no_sensitive": GradientBoostingClassifier,
    }

    logger.info(f"Fit evaluate classifier {model_tag}")
    classifier = MODEL_MAPPING[model_tag]()
    df_train = enc_dataset.X_enc_train.copy()
    df_test = enc_dataset.X_enc_test.copy()

    if "no_sensitive" in model_tag:
        logger.info(
            f"Dropping {enc_dataset.enc_sensitive_name} for model tag {model_tag}"
        )
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
