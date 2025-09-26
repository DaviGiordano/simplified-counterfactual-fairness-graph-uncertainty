#!/usr/bin/env python3
"""
Comprehensive plotting script for all available classification models.
Based on the analysis patterns from notebooks/analyse_adult_results.ipynb
"""

import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add parent directory to path
parent_dir = str(Path().absolute().parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from notebooks.notebook_utils import (
    load_and_tidy_fairness_metrics,
    load_score_variance_data,
)
from src.plot.bar_charts import (
    _maybe_mkdir,
    load_model_metrics,
    plot_counterfactual_metrics,
    plot_counterfactual_quality,
    plot_group_fairness,
    plot_model_performance,
    plot_score_variance,
    set_pub_theme,
    tol_palette,
)
from src.utils import find_root


def setup_plotting_environment():
    """Set up the plotting environment with consistent styling."""
    # Set seaborn theme
    sns.set_theme(
        style="ticks",
        rc={
            "text.usetex": False,
            "font.family": "serif",
            "axes.grid": True,
            "lines.linewidth": 0.8,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.edgecolor": "black",
        },
        palette="deep",
    )
    sns.set_palette("Set2")


def get_available_models(
    base_path: Path, dataset_tag: str, knowledge_level: str
) -> list[str]:
    """Get list of available classification models."""
    models_dir = base_path / "output" / dataset_tag / knowledge_level
    if not models_dir.exists():
        return []

    # Get all directories that contain cf_metrics.csv
    available_models = []
    for item in models_dir.iterdir():
        if item.is_dir() and (item / "cf_metrics.csv").exists():
            available_models.append(item.name)

    return sorted(available_models)


def plot_graph_uncertainty(
    base_path: Path,
    dataset_tag: str,
    knowledge_level: str,
    output_dir: Path,
    sensitive_feat: str = "sex",
):
    """Plot graph uncertainty analysis."""
    print("📊 Plotting graph uncertainty...")

    try:
        from src.causality.causal_world import inspect_graph_uncertainty

        fpath = (
            base_path / "output" / dataset_tag / knowledge_level / "causal_worlds.pkl"
        )
        if fpath.exists():
            with open(fpath, "rb") as f:
                cws = pickle.load(f)

            uncertainty_info = inspect_graph_uncertainty(cws, sensitive_feat)
            print(uncertainty_info)

            # Save uncertainty info to file
            uncertainty_file = output_dir / "graph_uncertainty_analysis.txt"
            with open(uncertainty_file, "w") as f:
                f.write(uncertainty_info)

            print(f"✅ Graph uncertainty analysis saved to {uncertainty_file}")
        else:
            print(f"⚠️  Causal worlds file not found: {fpath}")

    except Exception as e:
        print(f"❌ Error plotting graph uncertainty: {e}")


def plot_score_variance_analysis(
    base_path: Path,
    dataset_tag: str,
    knowledge_level: str,
    available_models: list[str],
    output_dir: Path,
):
    """Plot score variance analysis across causal worlds."""
    print("📊 Plotting score variance analysis...")

    try:
        # Load score variance data
        df_score_var = load_score_variance_data(
            base_path, dataset_tag, [knowledge_level], available_models
        )

        if not df_score_var.empty:
            # Create the plot
            g = plot_score_variance(
                df_score_var, str(output_dir / "score_variance.pdf")
            )
            plt.close()  # Close the plot to prevent display
            print(
                f"✅ Score variance plot saved to {output_dir / 'score_variance.pdf'}"
            )
        else:
            print("⚠️  No score variance data available")

    except Exception as e:
        print(f"❌ Error plotting score variance: {e}")


def plot_counterfactual_metrics_analysis(
    base_path: Path,
    dataset_tag: str,
    knowledge_level: str,
    available_models: list[str],
    output_dir: Path,
):
    """Plot counterfactual metrics analysis."""
    print("📊 Plotting counterfactual metrics analysis...")

    try:
        # Load and tidy fairness metrics
        df = load_and_tidy_fairness_metrics(
            base_path, dataset_tag, [knowledge_level], available_models
        )

        if not df.empty:
            interventions = ["Female_to_Male", "Male_to_Female"]
            g = plot_counterfactual_metrics(
                df, interventions, str(output_dir / "counterfactual_metrics.pdf")
            )
            plt.close()  # Close the plot to prevent display
            print(
                f"✅ Counterfactual metrics plot saved to {output_dir / 'counterfactual_metrics.pdf'}"
            )

            # Generate summary statistics
            summary_df = (
                df.groupby(["Knowledge", "intervention", "metric", "Classifier"])[
                    "Rate"
                ]
                .agg(
                    mean_rate="mean",
                    ci_low=lambda s: np.percentile(s, 2.5),
                    ci_high=lambda s: np.percentile(s, 97.5),
                )
                .reset_index()
            )

            # Save summary to CSV
            summary_file = output_dir / "counterfactual_metrics_summary.csv"
            summary_df.round(4).to_csv(summary_file, index=False)
            print(f"✅ Counterfactual metrics summary saved to {summary_file}")
        else:
            print("⚠️  No counterfactual metrics data available")

    except Exception as e:
        print(f"❌ Error plotting counterfactual metrics: {e}")


def plot_counterfactual_quality_analysis(
    base_path: Path, dataset_tag: str, knowledge_level: str, output_dir: Path
):
    """Plot counterfactual quality analysis."""
    print("📊 Plotting counterfactual quality analysis...")

    try:
        # Load counterfactual quality data
        fpath = (
            base_path
            / "output"
            / dataset_tag
            / knowledge_level
            / "mw_counterfactuals.pkl"
        )
        if fpath.exists():
            with open(fpath, "rb") as f:
                mw_cf = pickle.load(f)

            # Calculate averages across causal worlds
            avg_metrics = pd.DataFrame(
                {
                    "avg_coverage": (
                        mw_cf.counterfactuals_quality["coverage_0"]
                        + mw_cf.counterfactuals_quality["coverage_1"]
                    )
                    / 2,
                    "avg_density": (
                        mw_cf.counterfactuals_quality["density_0"]
                        + mw_cf.counterfactuals_quality["density_1"]
                    )
                    / 2,
                    "outlier_pct": mw_cf.counterfactuals_quality[
                        "overall_outlier_percent"
                    ],
                }
            )

            # Create quality plot with confidence intervals
            fig, ax = plot_counterfactual_quality(
                avg_metrics, save_path=str(output_dir / "counterfactual_quality_ci.pdf")
            )
            plt.close()  # Close the plot to prevent display
            print(
                f"✅ Counterfactual quality plot saved to {output_dir / 'counterfactual_quality_ci.pdf'}"
            )

            # Create scatter plot
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(
                avg_metrics["avg_density"],
                avg_metrics["avg_coverage"],
                c=avg_metrics["outlier_pct"],
                s=100,
                alpha=0.6,
                cmap="viridis",
            )

            plt.xlabel("Average Density", fontsize=12)
            plt.ylabel("Average Coverage", fontsize=12)
            plt.title(
                "Counterfactual Quality Metrics by Causal World\n(Color indicates outlier percentage)",
                fontsize=14,
                pad=20,
            )
            plt.grid(True, alpha=0.3)

            cbar = plt.colorbar(scatter)
            cbar.set_label("Outlier Percentage", fontsize=10)
            cbar.ax.tick_params(labelsize=8)

            plt.tight_layout()
            plt.savefig(
                output_dir / "quality_metrics_scatter.pdf", bbox_inches="tight", dpi=300
            )
            plt.close()
            print(
                f"✅ Quality metrics scatter plot saved to {output_dir / 'quality_metrics_scatter.pdf'}"
            )

            # Find best causal world
            avg_metrics["overall_avg"] = (
                avg_metrics["avg_coverage"] + avg_metrics["avg_density"]
            ) / 2
            best_world = avg_metrics["overall_avg"].idxmax()
            best_metrics = avg_metrics.loc[best_world]

            print(f"🏆 Best causal world: {best_world}")
            print(f"   Average coverage: {best_metrics['avg_coverage']:.3f}")
            print(f"   Average density: {best_metrics['avg_density']:.3f}")
            print(f"   Overall average: {best_metrics['overall_avg']:.3f}")
            print(f"   Outlier percentage: {best_metrics['outlier_pct']:.3f}")

        else:
            print(f"⚠️  Multi-world counterfactuals file not found: {fpath}")

    except Exception as e:
        print(f"❌ Error plotting counterfactual quality: {e}")


def plot_model_performance_analysis(
    base_path: Path, dataset_tag: str, available_models: list[str], output_dir: Path
):
    """Plot model performance and group fairness analysis."""
    print("📊 Plotting model performance and group fairness analysis...")

    try:
        # Load model metrics
        performance_df, fairness_df = load_model_metrics(
            base_path, dataset_tag, available_models
        )

        if not performance_df.empty:
            # Plot model performance
            fig, axes = plot_model_performance(
                performance_df,
                save_path=str(output_dir / "model_performance_metrics.pdf"),
            )
            plt.close()  # Close the plot to prevent display
            print(
                f"✅ Model performance plot saved to {output_dir / 'model_performance_metrics.pdf'}"
            )

            # Generate performance summary
            performance_summary = performance_df.pivot(
                index="metric", columns="classifier", values="value"
            )
            performance_file = output_dir / "model_performance_summary.csv"
            performance_summary.round(4).to_csv(performance_file)
            print(f"✅ Model performance summary saved to {performance_file}")
        else:
            print("⚠️  No model performance data available")

        if not fairness_df.empty:
            # Plot group fairness
            fig, ax = plot_group_fairness(
                fairness_df, save_path=str(output_dir / "group_fairness_metrics.pdf")
            )
            plt.close()  # Close the plot to prevent display
            print(
                f"✅ Group fairness plot saved to {output_dir / 'group_fairness_metrics.pdf'}"
            )

            # Generate fairness summary
            fairness_summary = fairness_df.pivot(
                index="metric", columns="classifier", values="value"
            )
            fairness_file = output_dir / "group_fairness_summary.csv"
            fairness_summary.round(4).to_csv(fairness_file)
            print(f"✅ Group fairness summary saved to {fairness_file}")
        else:
            print("⚠️  No group fairness data available")

    except Exception as e:
        print(f"❌ Error plotting model performance: {e}")


def create_model_comparison_table(
    base_path: Path, dataset_tag: str, available_models: list[str], output_dir: Path
):
    """Create a comprehensive model comparison table."""
    print("📊 Creating model comparison table...")

    try:
        comparison_data = []

        for model in available_models:
            # Load model performance
            perf_path = (
                base_path
                / "output"
                / dataset_tag
                / "model_metrics"
                / model
                / "model_performance.json"
            )
            fair_path = (
                base_path
                / "output"
                / dataset_tag
                / "model_metrics"
                / model
                / "group_fairness.json"
            )

            model_data = {"Model": model}

            if perf_path.exists():
                with open(perf_path, "r") as f:
                    perf_metrics = json.load(f)
                for metric, value in perf_metrics.items():
                    if metric.startswith("overall."):
                        model_data[metric.replace("overall.", "")] = round(value, 4)

            if fair_path.exists():
                with open(fair_path, "r") as f:
                    fair_metrics = json.load(f)
                for metric, value in fair_metrics.items():
                    if not any(group in metric for group in ["Female", "Male"]):
                        model_data[metric] = round(value, 4)

            comparison_data.append(model_data)

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_file = output_dir / "model_comparison_table.csv"
            comparison_df.to_csv(comparison_file, index=False)
            print(f"✅ Model comparison table saved to {comparison_file}")

            # Print summary
            print("\n📋 Model Comparison Summary:")
            print("=" * 80)
            print(comparison_df.to_string(index=False))
        else:
            print("⚠️  No model comparison data available")

    except Exception as e:
        print(f"❌ Error creating model comparison table: {e}")


def main():
    """Main function to generate all plots."""
    print("🚀 Starting comprehensive results plotting...")
    print("=" * 60)

    # Setup
    setup_plotting_environment()
    base_path = find_root()
    dataset_tag = "adult"
    knowledge_level = "med"
    sensitive_feat = "sex"

    # Create output directory
    output_dir = Path("adult_charts")
    output_dir.mkdir(exist_ok=True)

    # Get available models
    available_models = get_available_models(base_path, dataset_tag, knowledge_level)
    print(f"📋 Found {len(available_models)} available models:")
    for model in available_models:
        print(f"   - {model}")
    print()

    if not available_models:
        print("❌ No models found. Please run some experiments first.")
        return

    # Generate all plots
    plot_graph_uncertainty(
        base_path, dataset_tag, knowledge_level, output_dir, sensitive_feat
    )
    print()

    plot_score_variance_analysis(
        base_path, dataset_tag, knowledge_level, available_models, output_dir
    )
    print()

    plot_counterfactual_metrics_analysis(
        base_path, dataset_tag, knowledge_level, available_models, output_dir
    )
    print()

    plot_counterfactual_quality_analysis(
        base_path, dataset_tag, knowledge_level, output_dir
    )
    print()

    plot_model_performance_analysis(
        base_path, dataset_tag, available_models, output_dir
    )
    print()

    create_model_comparison_table(base_path, dataset_tag, available_models, output_dir)
    print()

    print("🎉 All plots generated successfully!")
    print(f"📁 Results saved to: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
