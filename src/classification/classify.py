import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.dataset.dataset_wrappers import EncodedDatasetWrapper
from src.metrics.model_metrics import evaluate_group_fairness, evaluate_performance

# Import FAIRGBM wrapper
try:
    from src.classification.fairgbm_wrapper import fit_fairgbm_classifier

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
        return fit_fairgbm_classifier(model_tag, enc_dataset, output_dir, **kwargs)

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
