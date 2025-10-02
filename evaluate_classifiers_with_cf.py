#!/usr/bin/env python3
"""
Script to fit and evaluate classification models (LR, RF, GB, FairGBM) with counterfactual fairness analysis.
Evaluates models without sensitive features (except FairGBM which uses sensitive features for regularization).
Evaluates counterfactual fairness using PSR and NSR metrics across all causal worlds.
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from fairgbm import FairGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.dataset.counterfactuals_wrapper import MultiWorldCounterfactuals
from src.dataset.load import load_dataset_col_trf
from src.log.logging_config import setup_logging
from src.metrics.model_metrics import evaluate_group_fairness, evaluate_performance
from src.utils import find_root

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a classifier with counterfactual fairness analysis across multiple causal models"
    )
    parser.add_argument(
        "--dataset", type=str, default="adult", help="Dataset to use (default: adult)"
    )
    parser.add_argument(
        "--classifier",
        type=str,
        choices=["LR", "RF", "GB", "FAIRGBM"],
        required=True,
        help="Classifier type",
    )
    parser.add_argument(
        "--knowledge", type=str, default="med", help="Knowledge level (default: med)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: output/{dataset}/{knowledge}/classification)",
    )

    return parser.parse_args()


def fit_evaluate_classifier(
    classifier_name: str,
    enc_dataset,
    output_dir: Path,
) -> Tuple[Any, pd.Series, Dict, Dict]:
    """
    Fit a classifier without hyperparameter tuning.

    Args:
        classifier_name: Name of the classifier ('LR', 'RF', 'GB', 'FAIRGBM')
        enc_dataset: Encoded dataset wrapper
        output_dir: Output directory for saving results

    Returns:
        Tuple of (classifier, predictions)
    """
    logger.info(f"Fitting {classifier_name} classifier")

    # Prepare data - drop sensitive features for all models
    X_train = enc_dataset.X_enc_train.copy()
    X_test = enc_dataset.X_enc_test.copy()

    logger.info(
        f"Dropping {enc_dataset.enc_sensitive_name} for model {classifier_name}"
    )
    X_train = X_train.drop(columns=enc_dataset.enc_sensitive_name)
    X_test = X_test.drop(columns=enc_dataset.enc_sensitive_name)

    # Fit classifier based on type
    if classifier_name == "FAIRGBM":
        # Convert sensitive attributes to numeric format for FAIRGBM
        s_train_numeric = enc_dataset.sensitive_train.copy()

        # Convert string values to numeric (assuming binary sensitive attribute)
        unique_values = sorted(enc_dataset.sensitive_train.unique())
        if len(unique_values) != 2:
            raise ValueError(
                f"Expected binary sensitive attribute, got {len(unique_values)} unique values: {unique_values}"
            )

        # Map to 0 and 1
        value_map = {"Female": 0, "Male": 1}
        s_train_numeric = s_train_numeric.map(value_map).astype(int)

        logger.info(f"Converted sensitive attributes: {value_map}")

        # Fit FairGBM with default parameters
        classifier = FairGBMClassifier()
        classifier.fit(
            X_train,
            enc_dataset.y_train,
            constraint_group=s_train_numeric,
        )
    else:
        # Standard sklearn classifiers
        MODEL_MAPPING = {
            "LR": LogisticRegression,
            "RF": RandomForestClassifier,
            "GB": GradientBoostingClassifier,
        }

        if classifier_name not in MODEL_MAPPING:
            raise ValueError(f"Unknown classifier: {classifier_name}")

        # Fit classifier with default parameters
        classifier = MODEL_MAPPING[classifier_name]()
        classifier.fit(X_train, enc_dataset.y_train)

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

    return classifier, y_pred, model_performance, group_fairness


def load_counterfactual_worlds(
    base_path: Path, dataset: str, knowledge: str, causal_models: List[str]
) -> Dict[str, List[pd.DataFrame]]:
    """
    Load counterfactual worlds for all causal models.

    Args:
        base_path: Base path to the project
        dataset: Dataset name
        knowledge: Knowledge level
        causal_models: List of causal model names

    Returns:
        Dictionary mapping causal model names to lists of counterfactual DataFrames
    """
    counterfactuals = {}

    for causal_model in causal_models:
        model_dir = base_path / f"output/{dataset}/{knowledge}/{causal_model}"
        if not model_dir.exists():
            logger.warning(
                f"Directory {model_dir} does not exist, skipping {causal_model}"
            )
            continue

        # Find all counterfactual world files
        cf_files = sorted(model_dir.glob("counterfactuals_world_*.csv"))
        if not cf_files:
            logger.warning(f"No counterfactual files found in {model_dir}")
            continue

        logger.info(f"Loading {len(cf_files)} counterfactual worlds for {causal_model}")
        counterfactuals[causal_model] = []

        for cf_file in cf_files:
            try:
                cf_df = pd.read_csv(cf_file, index_col=0)
                counterfactuals[causal_model].append(cf_df)
            except Exception as e:
                logger.error(f"Error loading {cf_file}: {e}")

    return counterfactuals


def evaluate_counterfactual_fairness(
    classifier: Any,
    y_pred: pd.Series,
    counterfactuals: Dict[str, List[pd.DataFrame]],
    sensitive_test: pd.Series,
    causal_models: List[str],
    enc_dataset,
) -> pd.DataFrame:
    """
    Evaluate counterfactual fairness using PSR and NSR metrics.

    Args:
        classifier: Trained classifier
        y_pred: Original predictions
        counterfactuals: Dictionary of counterfactual worlds by causal model
        sensitive_test: Sensitive attributes for test set
        causal_models: List of causal model names

    Returns:
        DataFrame with counterfactual fairness metrics
    """
    cf_results = []

    for causal_model in causal_models:
        if causal_model not in counterfactuals:
            logger.warning(f"No counterfactuals found for {causal_model}")
            continue

        logger.info(f"Evaluating counterfactual fairness for {causal_model}")

        for world_idx, cf_world in enumerate(counterfactuals[causal_model]):
            try:
                # Make predictions on counterfactual data
                # Drop sensitive features if they exist in counterfactual data
                cf_features = cf_world.copy()

                # Remove the sensitive feature column that was dropped during training
                sensitive_col = "onehot__sex_Male"
                if sensitive_col in cf_features.columns:
                    cf_features = cf_features.drop(columns=[sensitive_col])

                # Debug logging
                logger.debug(
                    f"Counterfactual features shape after dropping sensitive: {cf_features.shape}"
                )
                logger.debug(f"Test data shape: {enc_dataset.X_enc_test.shape}")
                logger.debug(f"Counterfactual columns: {list(cf_features.columns)}")
                logger.debug(
                    f"Test data columns: {list(enc_dataset.X_enc_test.columns)}"
                )

                # Ensure feature alignment with the features used during training
                # The classifier was trained without sensitive features, so we need to match that
                # Create the expected columns by removing the sensitive feature from test data columns
                expected_columns = [
                    col
                    for col in enc_dataset.X_enc_test.columns
                    if col != sensitive_col
                ]
                cf_features = cf_features.reindex(
                    columns=expected_columns, fill_value=0
                )

                logger.debug(
                    f"Final counterfactual features shape: {cf_features.shape}"
                )

                # Make predictions
                cf_pred = pd.Series(
                    classifier.predict(cf_features), index=cf_world.index
                )

                # Calculate PSR and NSR metrics
                from src.metrics.model_metrics import compute_cf_metrics

                cf_metrics = compute_cf_metrics(y_pred, cf_pred, sensitive_test)

                # Add metadata
                result = {
                    "causal_model": causal_model,
                    "causal_world": world_idx,
                    **cf_metrics,
                }
                cf_results.append(result)

            except Exception as e:
                logger.error(
                    f"Error evaluating world {world_idx} for {causal_model}: {e}"
                )

    return pd.DataFrame(cf_results)


def save_results(
    classifier_name: str,
    model_performance: Dict,
    group_fairness: Dict,
    cf_results: pd.DataFrame,
    output_dir: Path,
):
    """Save evaluation results to files."""
    classifier_dir = output_dir / classifier_name
    classifier_dir.mkdir(parents=True, exist_ok=True)

    # Save classifier evaluation metrics
    classifier_eval = {
        "model_performance": model_performance,
        "group_fairness": group_fairness,
    }

    with open(classifier_dir / "classifier_evaluation.json", "w") as f:
        json.dump(classifier_eval, f, indent=4)

    # Save counterfactual fairness results
    cf_results.to_csv(classifier_dir / "cf_evaluation.csv", index=False)

    logger.info(f"Saved results for {classifier_name} to {classifier_dir}")


def main():
    """Main function to evaluate a classifier with counterfactual fairness."""
    args = parse_arguments()
    base_path = find_root()

    # Set up output directory
    if args.output_dir is None:
        output_dir = (
            base_path / f"output/{args.dataset}/{args.knowledge}/classification"
        )
    else:
        output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    setup_logging(output_dir / "evaluate_classifiers.log")
    logger.info(f"Starting classifier evaluation with counterfactual fairness")
    logger.info(f"Dataset: {args.dataset}, Knowledge: {args.knowledge}")
    logger.info(f"Output directory: {output_dir}")

    # Load dataset
    logger.info("Loading dataset...")
    enc_dataset, col_trf = load_dataset_col_trf(args.dataset)

    # Define causal models
    causal_models = ["linear", "lgbm", "causalflow", "diffusion"]

    # Load counterfactual worlds
    logger.info("Loading counterfactual worlds...")
    counterfactuals = load_counterfactual_worlds(
        base_path, args.dataset, args.knowledge, causal_models
    )

    # Evaluate each classifier
    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluating {args.classifier}")
    logger.info(f"{'='*50}")

    try:
        # Set up model output directory
        model_output_dir = (
            base_path / f"output/{args.dataset}/model_metrics/{args.classifier}"
        )

        # Fit and evaluate classifier
        classifier, y_pred, model_performance, group_fairness = fit_evaluate_classifier(
            args.classifier,
            enc_dataset,
            model_output_dir,
        )

        # Evaluate counterfactual fairness
        cf_results = evaluate_counterfactual_fairness(
            classifier,
            y_pred,
            counterfactuals,
            enc_dataset.sensitive_test,
            causal_models,
            enc_dataset,
        )

        # Save results
        save_results(
            args.classifier,
            model_performance,
            group_fairness,
            cf_results,
            output_dir,
        )

        logger.info(f"Completed evaluation for {args.classifier}")

    except Exception as e:
        logger.error(f"Error evaluating {args.classifier}: {e}")

    logger.info("Classifier evaluation completed!")


if __name__ == "__main__":
    main()
