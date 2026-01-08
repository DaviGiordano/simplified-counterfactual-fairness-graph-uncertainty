#!/usr/bin/env python3
"""
Script to evaluate previously tuned classifiers with counterfactual fairness analysis.
Loads pre-trained models from tuning output directory and evaluates them with CF metrics.
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.base import ClassifierMixin

from src.dataset.counterfactuals_wrapper import MultiWorldCounterfactuals
from src.dataset.load import load_dataset_col_trf
from src.log.logging_config import setup_logging
from src.metrics.model_metrics import evaluate_group_fairness, evaluate_performance
from src.utils import find_root

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate previously tuned classifiers with counterfactual fairness analysis"
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
        "--tuning-root",
        type=Path,
        help="Root directory of tuning outputs (default: output/{dataset}/{knowledge}/tuning_comprehensive)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: output/{dataset}/{knowledge}/tuned_classification_cf)",
    )

    return parser.parse_args()


def load_tuned_classifier(
    classifier_name: str, tuning_root: Path, enc_dataset
) -> Tuple[Any, pd.Series, Dict, Dict]:
    """
    Load a previously tuned classifier and its evaluation metrics.

    Args:
        classifier_name: Name of the classifier ('LR', 'RF', 'GB', 'FAIRGBM')
        tuning_root: Root directory containing tuning outputs
        enc_dataset: Encoded dataset wrapper

    Returns:
        Tuple of (classifier, predictions, model_performance, group_fairness)
    """
    logger.info(f"Loading tuned {classifier_name} classifier")

    classifier_dir = tuning_root / classifier_name
    if not classifier_dir.exists():
        raise ValueError(f"Tuning directory not found: {classifier_dir}")

    # Load the best model
    model_file = classifier_dir / f"tuning/{classifier_name}_best_model.pkl"
    if not model_file.exists():
        raise ValueError(f"Best model file not found: {model_file}")

    with open(model_file, "rb") as f:
        classifier = pickle.load(f)

    # Load performance metrics
    perf_file = classifier_dir / "model_performance.json"
    if not perf_file.exists():
        raise ValueError(f"Model performance file not found: {perf_file}")

    with open(perf_file, "r") as f:
        model_performance = json.load(f)

    # Load group fairness metrics
    gf_file = classifier_dir / "group_fairness.json"
    if not gf_file.exists():
        raise ValueError(f"Group fairness file not found: {gf_file}")

    with open(gf_file, "r") as f:
        group_fairness = json.load(f)

    # Make predictions on test set
    X_test = enc_dataset.X_enc_test.copy()
    # Always drop sensitive feature to match training (including FAIRGBM)
    logger.info(
        f"Dropping {enc_dataset.enc_sensitive_name} for model {classifier_name}"
    )
    X_test = X_test.drop(columns=enc_dataset.enc_sensitive_name)

    y_pred = pd.Series(
        classifier.predict(X_test),
        index=enc_dataset.X_enc_test.index,
    )

    logger.info(f"Successfully loaded {classifier_name} classifier")
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
    classifier_name: str,
) -> pd.DataFrame:
    """
    Evaluate counterfactual fairness using PSR and NSR metrics.

    Args:
        classifier: Trained classifier
        y_pred: Original predictions
        counterfactuals: Dictionary of counterfactual worlds by causal model
        sensitive_test: Sensitive attributes for test set
        causal_models: List of causal model names
        enc_dataset: Encoded dataset wrapper
        classifier_name: Name of the classifier

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
                    "classifier": classifier_name,
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
    """Main function to evaluate a tuned classifier with counterfactual fairness."""
    args = parse_arguments()
    base_path = find_root()

    # Set up tuning root directory
    if args.tuning_root is None:
        tuning_root = (
            base_path / f"output/{args.dataset}/{args.knowledge}/tuning_comprehensive"
        )
    else:
        tuning_root = args.tuning_root

    # Set up output directory
    if args.output_dir is None:
        output_dir = (
            base_path
            / f"output/{args.dataset}/{args.knowledge}/tuned_classification_cf"
        )
    else:
        output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    setup_logging(output_dir / "evaluate_tuned_classifiers.log")
    logger.info(f"Starting tuned classifier evaluation with counterfactual fairness")
    logger.info(f"Dataset: {args.dataset}, Knowledge: {args.knowledge}")
    logger.info(f"Tuning root: {tuning_root}")
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

    # Evaluate the classifier
    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluating tuned {args.classifier}")
    logger.info(f"{'='*50}")

    try:
        # Load tuned classifier
        classifier, y_pred, model_performance, group_fairness = load_tuned_classifier(
            args.classifier, tuning_root, enc_dataset
        )

        # Evaluate counterfactual fairness
        cf_results = evaluate_counterfactual_fairness(
            classifier,
            y_pred,
            counterfactuals,
            enc_dataset.sensitive_test,
            causal_models,
            enc_dataset,
            args.classifier,
        )

        # Save results
        save_results(
            args.classifier,
            model_performance,
            group_fairness,
            cf_results,
            output_dir,
        )

        logger.info(f"Completed evaluation for tuned {args.classifier}")

    except Exception as e:
        logger.error(f"Error evaluating tuned {args.classifier}: {e}")
        raise

    logger.info("Tuned classifier evaluation completed!")


if __name__ == "__main__":
    main()
