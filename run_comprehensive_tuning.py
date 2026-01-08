#!/usr/bin/env python3
"""
Comprehensive hyperparameter tuning script for all classifiers.
Runs 20 trials for each classifier (LR, RF, GB, FairGBM) and saves all outputs.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

from src.classification.classify import fit_evaluate_classifier_with_tuning
from src.dataset.load import load_dataset_col_trf
from src.log.logging_config import setup_logging
from src.utils import find_root

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive hyperparameter tuning for all classifiers"
    )
    parser.add_argument(
        "--dataset", type=str, default="adult", help="Dataset to use (default: adult)"
    )
    parser.add_argument(
        "--knowledge", type=str, default="med", help="Knowledge level (default: med)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: output/{dataset}/{knowledge}/tuning_comprehensive)",
    )
    parser.add_argument(
        "--tuning-config",
        type=Path,
        help="Path to tuning configuration file (default: config/tuning/tuning_config.yaml)",
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=["LR", "RF", "GB", "FAIRGBM"],
        help="Classifiers to tune (default: all)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip classifiers that already have tuning results",
    )

    return parser.parse_args()


def run_comprehensive_tuning(
    dataset: str,
    knowledge: str,
    output_dir: Path,
    tuning_config_file: Path,
    classifiers: List[str],
    skip_existing: bool = False,
) -> Dict[str, Dict]:
    """
    Run comprehensive hyperparameter tuning for all specified classifiers.

    Args:
        dataset: Dataset name
        knowledge: Knowledge level
        output_dir: Output directory
        tuning_config_file: Path to tuning configuration file
        classifiers: List of classifiers to tune
        skip_existing: Whether to skip existing results

    Returns:
        Dictionary containing results for each classifier
    """
    logger.info(f"Starting comprehensive hyperparameter tuning")
    logger.info(f"Dataset: {dataset}, Knowledge: {knowledge}")
    logger.info(f"Classifiers: {classifiers}")
    logger.info(f"Output directory: {output_dir}")

    # Load dataset
    logger.info("Loading dataset...")
    enc_dataset, col_trf = load_dataset_col_trf(dataset)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Results storage
    all_results = {}

    # Run tuning for each classifier
    for classifier in classifiers:
        logger.info(f"\n{'='*60}")
        logger.info(f"Tuning {classifier} classifier")
        logger.info(f"{'='*60}")

        # Check if results already exist
        classifier_output_dir = output_dir / classifier
        if skip_existing and classifier_output_dir.exists():
            results_file = classifier_output_dir / "tuning_results.json"
            if results_file.exists():
                logger.info(f"Skipping {classifier} - results already exist")
                try:
                    with open(results_file, "r") as f:
                        all_results[classifier] = json.load(f)
                    continue
                except Exception as e:
                    logger.warning(
                        f"Could not load existing results for {classifier}: {e}"
                    )

        try:
            start_time = time.time()

            # Run hyperparameter tuning
            model, y_pred, tuning_results = fit_evaluate_classifier_with_tuning(
                classifier,
                enc_dataset,
                classifier_output_dir,
                tune_hyperparameters=True,
                tuning_config_file=tuning_config_file,
            )

            end_time = time.time()
            tuning_time = end_time - start_time

            # Store results
            results_summary = {
                "classifier": classifier,
                "best_params": tuning_results["best_params"],
                "best_score": tuning_results["best_score"],
                "test_results": tuning_results["test_results"],
                "optimization_objective": tuning_results["optimization_objective"],
                "tuning_time_seconds": tuning_time,
                "tuning_time_minutes": tuning_time / 60,
            }

            all_results[classifier] = results_summary

            # Save individual classifier results
            with open(classifier_output_dir / "tuning_summary.json", "w") as f:
                json.dump(results_summary, f, indent=4)

            logger.info(
                f"✓ {classifier} tuning completed in {tuning_time/60:.2f} minutes"
            )
            logger.info(f"  Best score: {tuning_results['best_score']:.4f}")
            logger.info(f"  Best params: {tuning_results['best_params']}")

        except Exception as e:
            logger.error(f"✗ Error tuning {classifier}: {e}")
            all_results[classifier] = {"error": str(e)}

    return all_results


def save_comprehensive_results(
    all_results: Dict[str, Dict], output_dir: Path, dataset: str, knowledge: str
) -> None:
    """
    Save comprehensive tuning results.

    Args:
        all_results: Results from all classifiers
        output_dir: Output directory
        dataset: Dataset name
        knowledge: Knowledge level
    """
    # Save comprehensive results
    comprehensive_file = output_dir / "comprehensive_tuning_results.json"
    with open(comprehensive_file, "w") as f:
        json.dump(all_results, f, indent=4)

    # Create summary report
    summary_file = output_dir / "tuning_summary_report.md"
    with open(summary_file, "w") as f:
        f.write(f"# Hyperparameter Tuning Summary Report\n\n")
        f.write(f"**Dataset:** {dataset}\n")
        f.write(f"**Knowledge Level:** {knowledge}\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Results Summary\n\n")
        f.write("| Classifier | Best Score | Tuning Time (min) | Status |\n")
        f.write("|------------|------------|-------------------|--------|\n")

        for classifier, results in all_results.items():
            if "error" in results:
                f.write(f"| {classifier} | N/A | N/A | ❌ Error |\n")
            else:
                score = results.get("best_score", 0)
                time_min = results.get("tuning_time_minutes", 0)
                f.write(
                    f"| {classifier} | {score:.4f} | {time_min:.2f} | ✅ Success |\n"
                )

        f.write("\n## Detailed Results\n\n")

        for classifier, results in all_results.items():
            f.write(f"### {classifier}\n\n")

            if "error" in results:
                f.write(f"**Error:** {results['error']}\n\n")
            else:
                f.write(f"**Best Score:** {results.get('best_score', 'N/A'):.4f}\n")
                f.write(
                    f"**Tuning Time:** {results.get('tuning_time_minutes', 0):.2f} minutes\n"
                )
                f.write(
                    f"**Optimization Objective:** {results.get('optimization_objective', 'N/A')}\n\n"
                )

                f.write("**Best Parameters:**\n")
                best_params = results.get("best_params", {})
                for param, value in best_params.items():
                    f.write(f"- {param}: {value}\n")

                f.write("\n**Test Results:**\n")
                test_results = results.get("test_results", {})
                if isinstance(test_results, dict):
                    # Handle nested dictionaries (e.g., model_performance and group_fairness)
                    for metric, value in test_results.items():
                        if isinstance(value, dict):
                            f.write(f"\n- {metric}:\n")
                            for sub_metric, sub_value in value.items():
                                try:
                                    f.write(
                                        f"  - {sub_metric}: {float(sub_value):.4f}\n"
                                    )
                                except (TypeError, ValueError):
                                    f.write(f"  - {sub_metric}: {sub_value}\n")
                        else:
                            try:
                                f.write(f"- {metric}: {float(value):.4f}\n")
                            except (TypeError, ValueError):
                                f.write(f"- {metric}: {value}\n")
                else:
                    f.write(f"- {test_results}\n")

                f.write("\n")

    logger.info(f"Saved comprehensive results to {comprehensive_file}")
    logger.info(f"Saved summary report to {summary_file}")


def main():
    """Main function to run comprehensive hyperparameter tuning."""
    args = parse_arguments()
    base_path = find_root()

    # Set up output directory
    if args.output_dir is None:
        output_dir = (
            base_path / f"output/{args.dataset}/{args.knowledge}/tuning_comprehensive"
        )
    else:
        output_dir = args.output_dir

    # Set up tuning config file
    if args.tuning_config is None:
        tuning_config_file = base_path / "config/tuning/tuning_config.yaml"
    else:
        tuning_config_file = args.tuning_config

    # Set up logging
    setup_logging(output_dir / "comprehensive_tuning.log")

    logger.info("=" * 80)
    logger.info("COMPREHENSIVE HYPERPARAMETER TUNING")
    logger.info("=" * 80)

    # Run comprehensive tuning
    all_results = run_comprehensive_tuning(
        dataset=args.dataset,
        knowledge=args.knowledge,
        output_dir=output_dir,
        tuning_config_file=tuning_config_file,
        classifiers=args.classifiers,
        skip_existing=args.skip_existing,
    )

    # Save comprehensive results
    save_comprehensive_results(all_results, output_dir, args.dataset, args.knowledge)

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("TUNING COMPLETED")
    logger.info("=" * 80)

    successful = sum(1 for r in all_results.values() if "error" not in r)
    total = len(all_results)

    logger.info(f"Successfully tuned: {successful}/{total} classifiers")

    for classifier, results in all_results.items():
        if "error" in results:
            logger.error(f"❌ {classifier}: {results['error']}")
        else:
            score = results.get("best_score", 0)
            time_min = results.get("tuning_time_minutes", 0)
            logger.info(f"✅ {classifier}: Score={score:.4f}, Time={time_min:.2f}min")

    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("Comprehensive hyperparameter tuning completed!")


if __name__ == "__main__":
    main()
