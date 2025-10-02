#!/usr/bin/env python3
"""
Script to fit causal models (linear, LGBM, diffusion, causalflow) using existing causal worlds.
Assumes causal_worlds.pkl exists from previous causal discovery runs.
Focused on adult dataset with medium knowledge.
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.causality.causal_models import create_causal_model
from src.causality.causal_world import CausalWorld
from src.dataset.load import load_dataset_col_trf
from src.log.logging_config import setup_logging
from src.utils import find_root, record_time

logger = logging.getLogger(__name__)


def main():
    """Main function to fit causal models using existing causal worlds for adult dataset."""
    # Setup

    BASE_PATH = find_root()
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(args.output_dir / f"fit_{args.model_type}_models.log")
    logger = logging.getLogger(__name__)

    logger.info(f"Starting causal model fitting: {args.model_type}")
    logger.info(f"Output directory: {args.output_dir}")

    # Fixed parameters for adult dataset with medium knowledge
    DATASET = "adult"
    KNOWLEDGE = "med"

    # Load existing causal worlds
    causal_worlds = load_causal_worlds(DATASET, KNOWLEDGE, BASE_PATH)

    # Load adult dataset
    enc_dataset, col_trf = load_dataset_col_trf(DATASET)

    # Prepare data
    train_data = enc_dataset.X_enc_train
    test_data = enc_dataset.X_enc_test
    sensitive_name = enc_dataset.enc_sensitive_name

    logger.info(f"Dataset: {DATASET}, Knowledge: {KNOWLEDGE}")
    logger.info(f"Train data shape: {train_data.shape}")
    logger.info(f"Test data shape: {test_data.shape}")
    logger.info(f"Sensitive attribute: {sensitive_name}")

    # Load configuration
    config = load_model_config(args.config, args.model_type)

    # Fit models for each causal world
    results = []

    for world_idx, causal_world in enumerate(causal_worlds):
        logger.info(f"Processing causal world {world_idx + 1}/{len(causal_worlds)}")

        try:
            # Create and fit causal model
            model = create_causal_model(
                args.model_type,
                causal_world.enc_dag,
                binary_root=sensitive_name,
                **config,
            )

            # Fit model
            with record_time("time_fit", {}):
                fit_results = model.fit(train_data.loc[causal_world.data_index])
                fit_results.pop("evaluation")

            # Generate counterfactuals
            with record_time("time_generate_cf", {}):
                cf_data = model.generate_counterfactuals(test_data, sensitive_name)

            # Evaluate counterfactual quality
            with record_time("time_evaluate_cf_quality", {}):
                cf_quality = model.evaluate_counterfactual_quality(
                    cf_data, test_data, sensitive_name
                )

            # Store results
            result = {
                "world_idx": world_idx,
                "dataset": DATASET,
                "knowledge": KNOWLEDGE,
                "model_type": args.model_type,
                **fit_results,
                **cf_quality,
                "num_counterfactuals": len(cf_data),
                "num_train_samples": len(causal_world.data_index),
            }
            results.append(result)

            # Save counterfactuals for this world to model-specific directory
            save_counterfactuals(cf_data, args.output_dir, world_idx)

            # Save metrics to CSV file
            save_metrics_to_csv(result, args.output_dir, args.model_type)

            logger.info(
                f"Completed world {world_idx + 1}: MSE={fit_results.get('mean_mse', 'N/A'):.4f}, "
                f"KL={fit_results.get('overall_kl', 'N/A'):.4f}"
            )

        except Exception as e:
            logger.error(f"Error processing world {world_idx}: {str(e)}")
            continue

    # Save aggregated results
    save_results(results, args.output_dir)

    # Generate summary report
    # generate_summary_report(results, args.output_dir, args.model_type)

    logger.info(
        f"Completed fitting {args.model_type} models for {len(results)} causal worlds"
    )
    logger.info(f"Results saved to: {args.output_dir}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fit causal models using existing causal worlds (adult dataset, medium knowledge)"
    )

    parser.add_argument(
        "--model-type",
        choices=["linear", "lgbm", "diffusion", "causalflow"],
        required=True,
        help="Causal model type to fit",
    )

    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory for results"
    )

    parser.add_argument(
        "--config", type=Path, default=None, help="Configuration file path"
    )

    parser.add_argument(
        "--num-workers", type=int, default=1, help="Number of parallel workers"
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def load_causal_worlds(
    dataset: str, knowledge: str, base_path: Path
) -> List[CausalWorld]:
    """Load existing causal worlds from pickle file."""
    causal_worlds_path = base_path / f"output/{dataset}/{knowledge}/causal_worlds.pkl"

    if not causal_worlds_path.exists():
        raise FileNotFoundError(f"Causal worlds not found: {causal_worlds_path}")

    with open(causal_worlds_path, "rb") as f:
        causal_worlds = pickle.load(f)

    logger = logging.getLogger(__name__)
    logger.info(f"Loaded {len(causal_worlds)} causal worlds from {causal_worlds_path}")
    return causal_worlds


def load_model_config(config_path: Optional[Path], model_type: str) -> Dict[str, Any]:
    """Load model configuration from file or use defaults."""
    if config_path and config_path.exists():
        import yaml

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config.get(model_type, {})

    # Default configurations
    defaults = {
        "linear": {"random_state": 42},
        "lgbm": {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "random_state": 42,
        },
        "diffusion": {
            "num_epochs": 100,
            "verbose": True,
            "use_gpu_if_available": False,
            "learning_rate": 0.001,
            "batch_size": 64,
        },
        "causalflow": {
            "hidden_features": (128, 128),
            "epochs": 200,
            "lr": 1e-3,
            "batch_size": 512,
            "val_split": 0.2,
            "patience": 50,
            "warmup_steps": 1000,
            "seed": 42,
        },
    }

    return defaults.get(model_type, {})


def save_counterfactuals(cf_data: pd.DataFrame, output_dir: Path, world_idx: int):
    """Save counterfactuals for a specific causal world."""
    cf_path = output_dir / f"counterfactuals_world_{world_idx:03d}.csv"
    cf_data.to_csv(cf_path, index=True)
    logger = logging.getLogger(__name__)
    logger.info(f"Saved counterfactuals for world {world_idx} to {cf_path}")


def save_metrics_to_csv(metrics: Dict[str, Any], output_dir: Path, model_type: str):
    """Save metrics to a CSV file for easy analysis."""
    metrics_file = output_dir / f"{model_type}_metrics.csv"

    # Prepare metrics data
    metrics_data = {
        "timestamp": [pd.Timestamp.now()],
        "model_type": [model_type],
        "world_idx": [metrics.get("world_idx", "N/A")],
        "dataset": [metrics.get("dataset", "N/A")],
        "knowledge": [metrics.get("knowledge", "N/A")],
        "num_counterfactuals": [metrics.get("num_counterfactuals", "N/A")],
        "num_train_samples": [metrics.get("num_train_samples", "N/A")],
    }

    # Add causal model evaluation metrics
    metrics_data.update(
        {
            "mean_mse": [metrics.get("mean_mse", "N/A")],
            "overall_kl": [metrics.get("overall_kl", "N/A")],
        }
    )

    # Add counterfactual quality metrics
    cf_metrics = [
        "overall_outlier_percent",
        "sbg0_outlier_percent",
        "sbg1_outlier_percent",
        "n_real_0",
        "n_cf_0",
        "density_0",
        "coverage_0",
        "stat_sim_0",
        "n_real_1",
        "n_cf_1",
        "density_1",
        "coverage_1",
        "stat_sim_1",
    ]

    for metric in cf_metrics:
        metrics_data[metric] = [metrics.get(metric, "N/A")]

    # Compute averages
    density_0 = metrics.get("density_0", None)
    density_1 = metrics.get("density_1", None)
    coverage_0 = metrics.get("coverage_0", None)
    coverage_1 = metrics.get("coverage_1", None)

    avg_density = "N/A"
    avg_coverage = "N/A"

    if density_0 is not None and density_1 is not None:
        try:
            avg_density = (float(density_0) + float(density_1)) / 2
        except (ValueError, TypeError):
            avg_density = "N/A"

    if coverage_0 is not None and coverage_1 is not None:
        try:
            avg_coverage = (float(coverage_0) + float(coverage_1)) / 2
        except (ValueError, TypeError):
            avg_coverage = "N/A"

    metrics_data.update(
        {
            "avg_density": [avg_density],
            "avg_coverage": [avg_coverage],
        }
    )

    # Add timing information
    timing_metrics = ["time_fit", "time_generate_cf", "time_evaluate_cf_quality"]
    for metric in timing_metrics:
        metrics_data[metric] = [metrics.get(metric, "N/A")]

    # Create DataFrame and save to CSV
    df = pd.DataFrame(metrics_data)

    # Check if file exists to determine if we need headers
    file_exists = metrics_file.exists()

    # Append to CSV (with headers only if file doesn't exist)
    df.to_csv(metrics_file, mode="a", header=not file_exists, index=False)

    logger = logging.getLogger(__name__)
    logger.info(f"Saved metrics to {metrics_file}")


def save_results(results: List[Dict], output_dir: Path):
    """Save aggregated results."""
    if not results:
        logger = logging.getLogger(__name__)
        logger.warning("No results to save")
        return

    results_df = pd.DataFrame(results)
    results_path = output_dir / "causal_model_results.csv"
    results_df.to_csv(results_path, index=False)

    logger = logging.getLogger(__name__)
    logger.info(f"Saved results to {results_path}")


def generate_summary_report(results: List[Dict], output_dir: Path, model_type: str):
    """Generate a comprehensive summary report."""
    if not results:
        return

    results_df = pd.DataFrame(results)

    # Calculate summary statistics
    summary = {
        "model_type": model_type,
        "dataset": "adult",
        "knowledge": "med",
        "total_worlds": len(results),
        "successful_fits": len(results_df),
        "model_performance": {
            "mean_mse": (
                results_df["mean_mse"].mean()
                if "mean_mse" in results_df.columns
                else None
            ),
            "std_mse": (
                results_df["mean_mse"].std()
                if "mean_mse" in results_df.columns
                else None
            ),
            "mean_kl": (
                results_df["overall_kl"].mean()
                if "overall_kl" in results_df.columns
                else None
            ),
            "std_kl": (
                results_df["overall_kl"].std()
                if "overall_kl" in results_df.columns
                else None
            ),
            "mean_test_nll": (
                results_df["test_nll"].mean()
                if "test_nll" in results_df.columns
                else None
            ),
        },
        "counterfactual_quality": {
            "mean_proximity": (
                results_df["proximity"].mean()
                if "proximity" in results_df.columns
                else None
            ),
            "mean_sparsity": (
                results_df["sparsity"].mean()
                if "sparsity" in results_df.columns
                else None
            ),
            "mean_diversity": (
                results_df["diversity"].mean()
                if "diversity" in results_df.columns
                else None
            ),
            "mean_validity": (
                results_df["validity"].mean()
                if "validity" in results_df.columns
                else None
            ),
        },
        "data_statistics": {
            "mean_train_samples": results_df["num_train_samples"].mean(),
            "total_counterfactuals": results_df["num_counterfactuals"].sum(),
        },
    }

    # Save summary
    import json

    summary_path = output_dir / "summary_report.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Generate text report
    report_path = output_dir / "summary_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Causal Model Fitting Summary Report\n")
        f.write(f"=====================================\n\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Dataset: adult\n")
        f.write(f"Knowledge Level: med\n")
        f.write(f"Total Causal Worlds: {summary['total_worlds']}\n")
        f.write(f"Successful Fits: {summary['successful_fits']}\n\n")

        f.write(f"Model Performance:\n")
        f.write(f"  Mean MSE: {summary['model_performance']['mean_mse']:.4f}\n")
        f.write(f"  Std MSE: {summary['model_performance']['std_mse']:.4f}\n")
        f.write(
            f"  Mean KL Divergence: {summary['model_performance']['mean_kl']:.4f}\n"
        )
        f.write(
            f"  Std KL Divergence: {summary['model_performance']['std_kl']:.4f}\n\n"
        )

        f.write(f"Counterfactual Quality:\n")
        f.write(
            f"  Mean Proximity: {summary['counterfactual_quality']['mean_proximity']:.4f}\n"
        )
        f.write(
            f"  Mean Sparsity: {summary['counterfactual_quality']['mean_sparsity']:.4f}\n"
        )
        f.write(
            f"  Mean Diversity: {summary['counterfactual_quality']['mean_diversity']:.4f}\n"
        )
        f.write(
            f"  Mean Validity: {summary['counterfactual_quality']['mean_validity']:.4f}\n\n"
        )

        f.write(f"Data Statistics:\n")
        f.write(
            f"  Mean Train Samples per World: {summary['data_statistics']['mean_train_samples']:.0f}\n"
        )
        f.write(
            f"  Total Counterfactuals Generated: {summary['data_statistics']['total_counterfactuals']}\n"
        )

    logger = logging.getLogger(__name__)
    logger.info(f"Generated summary report: {report_path}")


if __name__ == "__main__":
    main()
