#!/usr/bin/env python3
"""
Plotting script for classification results comparison.
Compares different classifiers (LR, RF, GB, FAIRGBM) based on:
1. Model performance metrics
2. Group fairness metrics
3. Counterfactual fairness metrics (PSR and NSR)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def set_pub_theme(
    font_size: int = 9, line_width: float = 0.5, font_scale: float = 0.75
):
    """Seaborn/Matplotlib theme tuned for papers (serif fonts, subtle black)."""
    _new_black = "#373737"  # "never use pure black"
    sns.set_theme(
        style="ticks",
        font_scale=font_scale,
        rc={
            # Fonts/output
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
            "svg.fonttype": "none",
            "text.usetex": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Sizes
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "legend.fontsize": font_size,
            "legend.title_fontsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "axes.labelpad": 2,
            "axes.titlepad": 4,
            # Line widths
            "axes.linewidth": line_width,
            "lines.linewidth": line_width,
            "xtick.major.width": line_width,
            "ytick.major.width": line_width,
            "xtick.minor.width": line_width,
            "ytick.minor.width": line_width,
            "xtick.major.size": 2,
            "xtick.major.size": 2,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            "xtick.major.pad": 1,
            "xtick.minor.pad": 1,
            "xtick.minor.pad": 1,
            "xtick.minor.pad": 1,
            # Neutral—not black
            "text.color": _new_black,
            "axes.edgecolor": _new_black,
            "axes.labelcolor": _new_black,
            "xtick.color": _new_black,
            "ytick.color": _new_black,
            "patch.edgecolor": _new_black,
            "patch.force_edgecolor": False,
            "hatch.color": _new_black,
        },
    )


def tol_palette(name: str = "bright"):
    """Paul Tol qualitative palettes."""
    sets = {
        "bright": [
            "#4477AA",
            "#EE6677",
            "#228833",
            "#CCBB44",
            "#66CCEE",
            "#AA3377",
            "#BBBBBB",
            "#000000",
        ],
        "high-contrast": ["#004488", "#DDAA33", "#BB5566", "#000000"],
        "vibrant": [
            "#EE7733",
            "#0077BB",
            "#33BBEE",
            "#EE3377",
            "#CC3311",
            "#009988",
            "#BBBBBB",
            "#000000",
        ],
        "muted": [
            "#CC6677",
            "#332288",
            "#DDCC77",
            "#117733",
            "#88CCEE",
            "#882255",
            "#44AA99",
            "#999933",
            "#AA4499",
            "#DDDDDD",
            "#000000",
        ],
    }
    return sets.get(name, sets["bright"])


def load_classification_data(base_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load classification results from all classifiers."""
    classifiers = ["LR", "RF", "GB", "FAIRGBM"]
    performance_data = []
    cf_data = []

    for classifier in classifiers:
        # Load performance and group fairness metrics
        json_path = (
            base_path
            / "output"
            / "adult"
            / "med"
            / "classification"
            / classifier
            / "classifier_evaluation.json"
        )
        if json_path.exists():
            with open(json_path, "r") as f:
                data = json.load(f)

            # Extract performance metrics
            perf_metrics = data["model_performance"]
            group_metrics = data["group_fairness"]

            # Combine into single row
            row = {"classifier": classifier}
            row.update(perf_metrics)
            row.update(group_metrics)
            performance_data.append(row)

            print(f"✅ Loaded performance data from {classifier}")
        else:
            print(f"⚠️  File not found: {json_path}")

        # Load counterfactual fairness data
        csv_path = (
            base_path
            / "output"
            / "adult"
            / "med"
            / "classification"
            / classifier
            / "cf_evaluation.csv"
        )
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["classifier"] = classifier
            cf_data.append(df)
            print(f"✅ Loaded counterfactual data from {classifier}: {len(df)} rows")
        else:
            print(f"⚠️  File not found: {csv_path}")

    if not performance_data:
        raise ValueError("No classification performance data found!")
    if not cf_data:
        raise ValueError("No counterfactual fairness data found!")

    # Combine all data
    performance_df = pd.DataFrame(performance_data)
    cf_df = pd.concat(cf_data, ignore_index=True)

    print(f"📊 Total performance data: {len(performance_df)} rows")
    print(f"📊 Total counterfactual data: {len(cf_df)} rows")

    return performance_df, cf_df


def plot_performance_metrics(df: pd.DataFrame, output_dir: Path):
    """Create bar charts comparing model performance metrics."""
    set_pub_theme()

    # Define the metrics to plot
    performance_metrics = [
        "overall.accuracy",
        "overall.precision",
        "overall.recall",
        "overall.f1",
        "overall.matthews_corrcoef",
    ]

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Color palette
    colors = tol_palette("vibrant")
    classifier_colors = {
        classifier: colors[i] for i, classifier in enumerate(df["classifier"].unique())
    }

    # Create individual plots for each metric
    for metric in performance_metrics:
        if metric not in df.columns:
            print(f"⚠️  Metric {metric} not found in data")
            continue

        print(f"📊 Plotting {metric}...")

        # Create figure
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

        # Create bar plot
        bars = sns.barplot(
            data=df,
            x="classifier",
            y=metric,
            hue="classifier",
            palette=classifier_colors,
            errorbar=None,  # No error bars for single values
            legend=False,
        )

        # Add value labels on bars
        for cont in bars.containers:
            bars.bar_label(cont, fmt="%.3f", padding=3)

        # Customize the plot
        ax.set_xlabel("Classifier")
        ax.set_ylabel(f"{metric.replace('overall.', '').replace('_', ' ').title()}")
        ax.set_title(
            f"{metric.replace('overall.', '').replace('_', ' ').title()} by Classifier",
            pad=20,
        )

        # Clean styling
        sns.despine()
        plt.tight_layout()

        # Save plot
        plot_path = output_dir / f"{metric.replace('.', '_')}_comparison.png"
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()

        print(f"✅ Saved {plot_path}")


def plot_group_fairness_metrics(df: pd.DataFrame, output_dir: Path):
    """Create bar charts comparing group fairness metrics."""
    set_pub_theme()

    # Define the metrics to plot
    fairness_metrics = [
        "dem_parity_diff",
        "dem_parity_ratio",
        "eq_odds_diff",
        "eq_odds_ratio",
    ]

    # Color palette
    colors = tol_palette("vibrant")
    classifier_colors = {
        classifier: colors[i] for i, classifier in enumerate(df["classifier"].unique())
    }

    # Create individual plots for each metric
    for metric in fairness_metrics:
        if metric not in df.columns:
            print(f"⚠️  Metric {metric} not found in data")
            continue

        print(f"📊 Plotting {metric}...")

        # Create figure
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

        # Create bar plot
        bars = sns.barplot(
            data=df,
            x="classifier",
            y=metric,
            hue="classifier",
            palette=classifier_colors,
            errorbar=None,  # No error bars for single values
            legend=False,
        )

        # Add value labels on bars
        for cont in bars.containers:
            bars.bar_label(cont, fmt="%.3f", padding=3)

        # Customize the plot
        ax.set_xlabel("Classifier")
        ax.set_ylabel(f"{metric.replace('_', ' ').title()}")
        ax.set_title(f"{metric.replace('_', ' ').title()} by Classifier", pad=20)

        # Clean styling
        sns.despine()
        plt.tight_layout()

        # Save plot
        plot_path = output_dir / f"{metric}_comparison.png"
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()

        print(f"✅ Saved {plot_path}")


def plot_counterfactual_metrics(cf_df: pd.DataFrame, output_dir: Path):
    """Create bar charts comparing counterfactual fairness metrics (PSR and NSR)."""
    set_pub_theme()

    # Prepare data for plotting
    plot_data = []

    for _, row in cf_df.iterrows():
        # Female to Male PSR and NSR
        plot_data.append(
            {
                "classifier": row["classifier"],
                "causal_model": row["causal_model"],
                "intervention": "Female to Male",
                "metric": "PSR",
                "rate": row["Female_to_Male.negative_to_positive_switch_rate"],
            }
        )
        plot_data.append(
            {
                "classifier": row["classifier"],
                "causal_model": row["causal_model"],
                "intervention": "Female to Male",
                "metric": "NSR",
                "rate": row["Female_to_Male.positive_to_negative_switch_rate"],
            }
        )

        # Male to Female PSR and NSR
        plot_data.append(
            {
                "classifier": row["classifier"],
                "causal_model": row["causal_model"],
                "intervention": "Male to Female",
                "metric": "PSR",
                "rate": row["Male_to_Female.negative_to_positive_switch_rate"],
            }
        )
        plot_data.append(
            {
                "classifier": row["classifier"],
                "causal_model": row["causal_model"],
                "intervention": "Male to Female",
                "metric": "NSR",
                "rate": row["Male_to_Female.positive_to_negative_switch_rate"],
            }
        )

    plot_df = pd.DataFrame(plot_data)

    # Map causal models to their display names
    causal_model_mapping = {
        "linear": "Linear",
        "lgbm": "LGBM",
        "diffusion": "FCM",
        "causalflow": "CNF",
    }

    # Apply the mapping to the dataframe
    plot_df["causal_model_display"] = plot_df["causal_model"].map(causal_model_mapping)

    # Define interventions and metrics
    interventions = ["Female to Male", "Male to Female"]
    metrics = ["PSR", "NSR"]

    # Color palette for classifiers
    colors = tol_palette("vibrant")
    classifier_colors = {
        "LR": colors[0],
        "RF": colors[1],
        "GB": colors[2],
        "FAIRGBM": colors[3],
    }

    # Create the main counterfactual fairness plot
    print("📊 Creating counterfactual fairness comparison plot...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)

    for i, metric in enumerate(metrics):  # PSR (row 0), NSR (row 1)
        for j, intervention in enumerate(
            interventions
        ):  # Female to Male (col 0), Male to Female (col 1)
            ax = axes[i, j]

            # Filter data for this metric and intervention
            metric_data = plot_df[
                (plot_df["metric"] == metric)
                & (plot_df["intervention"] == intervention)
            ]

            if len(metric_data) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            # Create bar plot with causal models as x-axis and classifiers as hue
            bars = sns.barplot(
                data=metric_data,
                x="causal_model_display",
                y="rate",
                hue="classifier",
                palette=classifier_colors,
                ax=ax,
                errorbar="ci",  # 95% confidence interval
                capsize=0.05,
            )

            # Customize subplot
            ax.set_title(f"{metric} - {intervention}", fontsize=12, pad=10)
            ax.set_xlabel("Causal Model")
            ax.set_ylabel(f"{metric} ± 95% CI")

            # Set y-axis limits
            ax.set_ylim(0, 1)

            # Remove legend for individual subplots except top-left
            if not (i == 0 and j == 0):
                ax.legend().remove()
            else:
                # Customize legend for top-left subplot
                ax.legend(
                    title="Classifier", bbox_to_anchor=(1.05, 1), loc="upper left"
                )

            # Clean styling
            sns.despine(ax=ax)

    # Add overall title
    fig.suptitle(
        "Counterfactual Fairness Metrics by Classifier and Causal Model",
        fontsize=14,
        y=0.98,
    )

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / "counterfactual_fairness_comparison.png"
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"✅ Saved {plot_path}")


def create_summary_plots(performance_df: pd.DataFrame, output_dir: Path):
    """Create summary plots with key metrics."""
    set_pub_theme()

    # Color palette
    colors = tol_palette("vibrant")
    classifier_colors = {
        classifier: colors[i]
        for i, classifier in enumerate(performance_df["classifier"].unique())
    }

    # Performance summary
    print("📊 Creating performance summary plot...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=300)
    axes = axes.flatten()

    performance_metrics = [
        "overall.accuracy",
        "overall.precision",
        "overall.recall",
        "overall.f1",
        "overall.matthews_corrcoef",
    ]

    for i, metric in enumerate(performance_metrics):
        if i >= len(axes):
            break

        ax = axes[i]

        if metric not in performance_df.columns:
            ax.text(
                0.5,
                0.5,
                f"Metric {metric}\nnot available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        # Create bar plot
        bars = sns.barplot(
            data=performance_df,
            x="classifier",
            y=metric,
            hue="classifier",
            palette=classifier_colors,
            ax=ax,
            errorbar=None,
            legend=False,
        )

        # Add value labels
        for cont in bars.containers:
            bars.bar_label(cont, fmt="%.3f", padding=3, fontsize=8)

        # Customize subplot
        ax.set_title(
            f"{metric.replace('overall.', '').replace('_', ' ').title()}", fontsize=10
        )
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Clean styling
        sns.despine(ax=ax)

    # Group fairness summary
    fairness_metrics = ["dem_parity_diff", "eq_odds_diff"]

    for i, metric in enumerate(fairness_metrics):
        ax_idx = len(performance_metrics) + i
        if ax_idx >= len(axes):
            break

        ax = axes[ax_idx]

        if metric not in performance_df.columns:
            ax.text(
                0.5,
                0.5,
                f"Metric {metric}\nnot available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        # Create bar plot
        bars = sns.barplot(
            data=performance_df,
            x="classifier",
            y=metric,
            hue="classifier",
            palette=classifier_colors,
            ax=ax,
            errorbar=None,
            legend=False,
        )

        # Add value labels
        for cont in bars.containers:
            bars.bar_label(cont, fmt="%.3f", padding=3, fontsize=8)

        # Customize subplot
        ax.set_title(f"{metric.replace('_', ' ').title()}", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Clean styling
        sns.despine(ax=ax)

    # Remove empty subplots
    for i in range(len(performance_metrics) + len(fairness_metrics), len(axes)):
        fig.delaxes(axes[i])

    # Add overall title
    fig.suptitle(
        "Classification Performance and Group Fairness Summary", fontsize=14, y=0.98
    )

    plt.tight_layout()

    # Save summary plot
    summary_path = output_dir / "classification_summary.png"
    plt.savefig(summary_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"✅ Saved summary plot: {summary_path}")


def create_metrics_summary_table(performance_df: pd.DataFrame, output_dir: Path):
    """Create a summary table of all metrics."""
    print("📊 Creating metrics summary table...")

    # Define key metrics
    key_metrics = [
        "overall.accuracy",
        "overall.precision",
        "overall.recall",
        "overall.f1",
        "overall.matthews_corrcoef",
        "dem_parity_diff",
        "dem_parity_ratio",
        "eq_odds_diff",
        "eq_odds_ratio",
    ]

    # Create summary table
    summary_df = performance_df[["classifier"] + key_metrics].copy()

    # Round to 4 decimal places
    for metric in key_metrics:
        if metric in summary_df.columns:
            summary_df[metric] = summary_df[metric].round(4)

    # Save to CSV
    summary_path = output_dir / "classification_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"✅ Saved summary table: {summary_path}")

    # Print summary to console
    print("\n📋 Classification Metrics Summary:")
    print("=" * 80)
    print(summary_df.to_string(index=False))


def main():
    """Main function to generate all classification result plots."""
    print("🚀 Starting classification results plotting...")
    print("=" * 60)

    # Setup paths
    base_path = Path(__file__).parent
    output_dir = base_path / "adult_classification_results"

    try:
        # Load data
        performance_df, cf_df = load_classification_data(base_path)

        # Create plots
        plot_performance_metrics(performance_df, output_dir)
        plot_group_fairness_metrics(performance_df, output_dir)
        plot_counterfactual_metrics(cf_df, output_dir)
        create_summary_plots(performance_df, output_dir)

        # Create summary table
        create_metrics_summary_table(performance_df, output_dir)

        print("\n🎉 All classification result plots generated successfully!")
        print(f"📁 Results saved to: {output_dir.absolute()}")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
