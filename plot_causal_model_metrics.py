#!/usr/bin/env python3
"""
Plotting script for causal model metrics comparison.
Compares different fitting mechanisms (linear, diffusion, causalflow, lgbm)
based on causal model results.
"""

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
            "ytick.major.pad": 1,
            "xtick.minor.pad": 1,
            "ytick.minor.pad": 1,
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


def load_causal_model_data(base_path: Path) -> pd.DataFrame:
    """Load causal model results from all fitting mechanisms."""
    data = []

    # Define the fitting mechanisms and their paths
    mechanisms = ["linear", "diffusion", "causalflow", "lgbm"]

    for mechanism in mechanisms:
        csv_path = (
            base_path
            / "output"
            / "adult"
            / "med"
            / mechanism
            / "causal_model_results.csv"
        )

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            data.append(df)
            print(f"✅ Loaded data from {mechanism}: {len(df)} rows")
        else:
            print(f"⚠️  File not found: {csv_path}")

    if not data:
        raise ValueError("No causal model data found!")

    # Combine all data
    combined_df = pd.concat(data, ignore_index=True)
    print(f"📊 Total combined data: {len(combined_df)} rows")

    return combined_df


def plot_causal_model_metrics(df: pd.DataFrame, output_dir: Path):
    """Create bar charts comparing causal model metrics across fitting mechanisms."""
    set_pub_theme()

    # Mechanism name mapping
    mechanism_names = {
        "linear": "Linear",
        "diffusion": "DCM",
        "causalflow": "CNF",
        "lgbm": "LGBM",
    }

    # Apply mechanism name mapping
    df = df.copy()
    df["model_type"] = df["model_type"].replace(mechanism_names)

    # Compute average metrics
    df["avg_coverage"] = (df["coverage_0"] + df["coverage_1"]) / 2
    df["avg_density"] = (df["density_0"] + df["density_1"]) / 2

    # Define the metrics to plot
    metrics = [
        "mean_mse",
        "overall_kl",
        "overall_outlier_percent",
        "avg_density",
        "avg_coverage",
        "coverage_0",
        "coverage_1",
        "density_0",
        "density_1",
    ]

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get unique model types (fitting mechanisms)
    model_types = df["model_type"].unique()
    print(f"📋 Found model types: {model_types}")

    # Color palette
    colors = tol_palette("vibrant")
    model_colors = {model: colors[i] for i, model in enumerate(model_types)}

    # Create individual plots for each metric
    for metric in metrics:
        if metric not in df.columns:
            print(f"⚠️  Metric {metric} not found in data")
            continue

        print(f"📊 Plotting {metric}...")

        # Create figure following the style guidelines
        fig, ax = plt.subplots(figsize=(2.5, 3.5), dpi=300)

        # Let seaborn handle the aggregation and CI calculation with capsize=0.2
        bars = sns.barplot(
            data=df,
            x=metric,
            y="model_type",
            color="steelblue",
            orient="h",
            errorbar="ci",  # Let seaborn calculate confidence intervals
            capsize=0.2,  # Increased cap size for better spacing
        )

        # Add value labels on bars with more padding
        for cont in bars.containers:
            # Extra padding for outlier rate to prevent text overlap
            padding = 8 if metric == "overall_outlier_percent" else 5
            bars.bar_label(cont, fmt="%.3f", padding=padding)  # type: ignore[arg-type]

        # Customize the plot
        metric_display_name = metric.replace("_", " ").title()
        # Fix specific metric names
        if metric == "mean_mse":
            metric_display_name = "MSE"
        elif metric == "overall_outlier_percent":
            metric_display_name = "Outlier Rate"
        elif metric == "avg_coverage":
            metric_display_name = "Coverage"
        elif metric == "avg_density":
            metric_display_name = "Density"
        elif metric == "overall_kl":
            metric_display_name = "KL Divergence"

        ax.set_xlabel(f"Average {metric_display_name} ± 95% CI")
        ax.set_ylabel("Causal Model")
        ax.set_title(f"{metric_display_name} by Causal Model", pad=20)

        # Clean styling
        sns.despine()
        plt.tight_layout()

        # Save plot
        plot_path = output_dir / f"{metric}_comparison.png"
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()

        print(f"✅ Saved {plot_path}")

    # Create a combined summary plot
    print("📊 Creating summary comparison plot...")
    create_summary_plot(df, output_dir, model_colors)


def create_summary_plot(df: pd.DataFrame, output_dir: Path, model_colors: dict):
    """Create a summary plot with all metrics in subplots."""
    set_pub_theme()

    metrics = [
        "mean_mse",
        "overall_kl",
        "overall_outlier_percent",
        "avg_density",
        "avg_coverage",
    ]

    # Create 2x2 subplots with A4 aspect ratio (4x4)
    fig, axes = plt.subplots(2, 2, figsize=(4, 4), dpi=300)
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        if i >= len(axes):
            break

        ax = axes[i]

        if metric not in df.columns:
            ax.text(
                0.5,
                0.5,
                f"Metric {metric}\nnot available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        # Let seaborn handle the aggregation and CI calculation with capsize=0.2
        bars = sns.barplot(
            data=df,
            x=metric,
            y="model_type",
            color="steelblue",
            orient="h",
            errorbar="ci",  # Let seaborn calculate confidence intervals
            capsize=0.2,  # Increased cap size for better spacing
            ax=ax,
        )

        # Add value labels on bars with more padding
        for cont in bars.containers:
            # Extra padding for outlier rate to prevent text overlap
            padding = 8 if metric == "overall_outlier_percent" else 5
            bars.bar_label(cont, fmt="%.3f", padding=padding, fontsize=8)  # type: ignore[arg-type]

        # Customize subplot
        metric_display_name = metric.replace("_", " ").title()
        # Fix specific metric names
        if metric == "mean_mse":
            metric_display_name = "MSE"
        elif metric == "overall_outlier_percent":
            metric_display_name = "Outlier Rate"
        elif metric == "avg_coverage":
            metric_display_name = "Coverage"
        elif metric == "avg_density":
            metric_display_name = "Density"
        elif metric == "overall_kl":
            metric_display_name = "KL Divergence"

        ax.set_title(f"{metric_display_name} by Causal Model", fontsize=10)
        ax.set_xlabel(f"Average {metric_display_name} ± 95% CI")
        ax.set_ylabel("Causal Model")

        # Clean styling
        sns.despine(ax=ax)

    # Add overall title
    fig.suptitle("Causal Model Metrics Comparison", fontsize=14, y=0.98)

    plt.tight_layout()

    # Save summary plot
    summary_path = output_dir / "causal_model_metrics_summary.png"
    plt.savefig(summary_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"✅ Saved summary plot: {summary_path}")


def create_metrics_summary_table(df: pd.DataFrame, output_dir: Path):
    """Create a summary table of all metrics."""
    print("📊 Creating metrics summary table...")

    # Mechanism name mapping
    mechanism_names = {
        "linear": "Linear",
        "diffusion": "DCM",
        "causalflow": "CNF",
        "lgbm": "LGBM",
    }

    # Apply mechanism name mapping
    df = df.copy()
    df["model_type"] = df["model_type"].replace(mechanism_names)

    # Compute average metrics (same as in plot_causal_model_metrics)
    df["avg_coverage"] = (df["coverage_0"] + df["coverage_1"]) / 2
    df["avg_density"] = (df["density_0"] + df["density_1"]) / 2

    metrics = [
        "mean_mse",
        "overall_kl",
        "overall_outlier_percent",
        "avg_density",
        "avg_coverage",
    ]

    summary_data = []

    for model_type in df["model_type"].unique():
        model_data = df[df["model_type"] == model_type]

        row = {"Fitting_Mechanism": model_type}

        for metric in metrics:
            if metric in model_data.columns:
                values = model_data[metric]
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_ci_lower"] = float(
                    np.quantile(np.asarray(values), 0.025)
                )
                row[f"{metric}_ci_upper"] = float(
                    np.quantile(np.asarray(values), 0.975)
                )
                row[f"{metric}_count"] = len(values)
            else:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_ci_lower"] = np.nan
                row[f"{metric}_ci_upper"] = np.nan
                row[f"{metric}_count"] = 0

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)

    # Compute rank-based composite score (weighted average of ranks)
    # Lower is better for MSE, Outlier Rate, KL; higher is better for Density, Coverage
    summary_df["rank_mse"] = summary_df["mean_mse_mean"].rank(
        ascending=True, method="average"
    )
    summary_df["rank_outlier"] = summary_df["overall_outlier_percent_mean"].rank(
        ascending=True, method="average"
    )
    summary_df["rank_kl"] = summary_df["overall_kl_mean"].rank(
        ascending=True, method="average"
    )
    summary_df["rank_density"] = summary_df["avg_density_mean"].rank(
        ascending=False, method="average"
    )
    summary_df["rank_coverage"] = summary_df["avg_coverage_mean"].rank(
        ascending=False, method="average"
    )

    # Weighted average: weight 2 for MSE
    total_weight = 2 + 1 + 1 + 1 + 1
    summary_df["composite_score"] = (
        summary_df["rank_mse"]
        + summary_df["rank_outlier"]
        + summary_df["rank_kl"]
        + summary_df["rank_density"]
        + summary_df["rank_coverage"]
    ) / float(total_weight)

    # Save to CSV
    summary_path = output_dir / "causal_model_metrics_summary.csv"
    summary_df.round(4).to_csv(summary_path, index=False)

    print(f"✅ Saved summary table: {summary_path}")

    # Print summary to console
    print("\n📋 Causal Model Metrics Summary:")
    print("=" * 80)
    print(summary_df.to_string(index=False, float_format="%.4f"))

    # Save model ranking by composite score to TXT
    ranking_df = summary_df[["Fitting_Mechanism", "composite_score"]].copy()
    ranking_df = ranking_df.sort_values(by=["composite_score"], ascending=True).reset_index(drop=True)  # type: ignore[call-arg]
    ranking_df.index = ranking_df.index + 1  # start rank at 1

    ranking_lines = [
        "Model ranking by weighted average rank (lower is better)",
        "Weights: MSE x2; Outlier, KL, Density, Coverage x1",
        "=" * 70,
        "",
    ]
    for rank, row in ranking_df.iterrows():
        ranking_lines.append(
            f"{rank:2d}. {row['Fitting_Mechanism']}: {row['composite_score']:.4f}"
        )

    ranking_txt_path = output_dir / "model_ranking.txt"
    with open(ranking_txt_path, "w") as f:
        f.write("\n".join(ranking_lines))

    print(f"\n🏁 Saved model ranking: {ranking_txt_path}")


def save_metrics_with_ci_to_txt(df: pd.DataFrame, output_dir: Path):
    """Save mean values and 95% confidence intervals to a text file."""
    print("📊 Creating metrics with CI text file...")

    # Mechanism name mapping
    mechanism_names = {
        "linear": "Linear",
        "diffusion": "DCM",
        "causalflow": "CNF",
        "lgbm": "LGBM",
    }

    # Apply mechanism name mapping
    df = df.copy()
    df["model_type"] = df["model_type"].replace(mechanism_names)

    # Compute average metrics (same as in plot_causal_model_metrics)
    df["avg_coverage"] = (df["coverage_0"] + df["coverage_1"]) / 2
    df["avg_density"] = (df["density_0"] + df["density_1"]) / 2

    metrics = [
        "mean_mse",
        "overall_kl",
        "overall_outlier_percent",
        "avg_density",
        "avg_coverage",
    ]

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare text content
    lines = []
    lines.append("Causal Model Metrics: Mean Values and 95% Confidence Intervals")
    lines.append("=" * 70)
    lines.append("")

    for metric in metrics:
        if metric not in df.columns:
            continue

        lines.append(f"Metric: {metric.replace('_', ' ').title()}")
        lines.append("-" * 40)

        # Calculate statistics for each model type
        for model_type in sorted(df["model_type"].unique()):
            model_data = df[df["model_type"] == model_type][metric]
            if len(model_data) > 0:
                mean_val = model_data.mean()
                n = len(model_data)
                # 95% CI using quantiles
                ci_low = float(np.quantile(np.asarray(model_data), 0.025))
                ci_high = float(np.quantile(np.asarray(model_data), 0.975))

                lines.append(
                    f"  {model_type:12s}: {mean_val:.4f} [{ci_low:.4f}, {ci_high:.4f}] (n={n})"
                )
            else:
                lines.append(f"  {model_type:12s}: No data")

        lines.append("")

    # Save to text file
    txt_path = output_dir / "causal_model_metrics_with_ci.txt"
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))

    print(f"✅ Saved metrics with CI: {txt_path}")

    # Print to console as well
    print("\n📋 Causal Model Metrics with 95% Confidence Intervals:")
    print("=" * 70)
    for line in lines:
        print(line)


def main():
    """Main function to generate all causal model metric plots."""
    print("🚀 Starting causal model metrics plotting...")
    print("=" * 60)

    # Setup paths
    base_path = Path(__file__).parent
    output_dir = base_path / "adult_causal_model_metrics"

    try:
        # Load data
        df = load_causal_model_data(base_path)

        # Create plots
        plot_causal_model_metrics(df, output_dir)

        # Create summary table
        create_metrics_summary_table(df, output_dir)

        # Save metrics with CI to text file
        save_metrics_with_ci_to_txt(df, output_dir)

        print("\n🎉 All causal model metric plots generated successfully!")
        print(f"📁 Results saved to: {output_dir.absolute()}")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
