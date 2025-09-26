# rewritten_code1.py

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
            "ytick.major.size": 2,
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


# ---- small helper ----
def _maybe_mkdir(fpath):
    if fpath:
        Path(fpath).parent.mkdir(parents=True, exist_ok=True)


def plot_score_variance(df: pd.DataFrame, save_path: str | None = None):
    set_pub_theme()
    _maybe_mkdir(save_path)

    hue_order = ["RF", "GB", "LR", "FAIRGBM"]
    palette = tol_palette("vibrant")
    palette = [palette[1], palette[0], palette[2], palette[3]]

    g = sns.catplot(
        data=df,
        x="knowledge",
        y="score_variance_by_individual",
        hue="classifier",
        hue_order=hue_order,
        kind="bar",
        dodge=0.7,
        estimator="mean",
        errorbar="pi",
        capsize=0.1,
        height=4,  # increased size
        aspect=2.0,  # wider
        palette=palette,
    )

    g.set_axis_labels(
        "Domain Knowledge Level", "Average Variance of Score ± 95% CI", fontsize=14
    )
    g.legend.set_title("Classifier", prop={"size": 14})

    # ✅ Add title here
    g.fig.suptitle(
        "Score Average Variance and CI by Domain Knowledge Level", fontsize=14, y=1.05
    )
    g.tick_params(axis="both", labelsize=14)
    sns.despine(fig=g.figure)
    plt.tight_layout()

    sns.move_legend(g, "center right", bbox_to_anchor=(1.1, 0.5), ncol=1, frameon=True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()
    return g


def plot_counterfactual_metrics(
    data: pd.DataFrame, interventions: list[str], save_path: str | None = None
):
    set_pub_theme()
    _maybe_mkdir(save_path)

    hue_order = [
        "RF",
        "GB",
        "LR",
        "FAIRGBM",
    ]
    palette = tol_palette("vibrant")
    palette = [palette[1], palette[0], palette[2], palette[3]]

    g = sns.catplot(
        data=data,
        x="Knowledge",
        y="Rate",
        kind="bar",
        hue="Classifier",
        hue_order=hue_order,
        col="intervention",
        col_order=interventions,
        row="metric",
        row_order=["PSR", "NSR"],
        estimator="mean",
        errorbar="pi",
        capsize=0.05,
        height=4,
        aspect=1.5,
        palette=palette,
        sharey=False,
        sharex=False,
    )

    g.set_titles(template="{col_name}", size=14)
    for ax in g.axes.flatten():
        ax.set_title(
            f"Intervention: {ax.get_title().replace('_', ' ')}", pad=6, fontsize=14
        )
        ax.tick_params(axis="both", labelsize=14)

    for metric, ax in zip(g.row_names, g.axes[:, 0]):
        ax.set_ylabel(
            f"{metric} ± 95% CI", rotation=90, labelpad=8, va="center", fontsize=14
        )

    for ax in g.axes.flatten():
        ax.set_xlabel(ax.get_xlabel(), fontsize=14)

    g.legend.set_title("Classifier", prop={"size": 14})
    sns.move_legend(g, "center right", bbox_to_anchor=(1.1, 0.5), ncol=1, frameon=True)

    g.set(ylim=(0, 1))
    sns.despine(fig=g.figure)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()
    return g


def plot_counterfactual_quality(
    data: pd.DataFrame, save_path: str | None = None, metrics: list[str] | None = None
):
    """
    Plot counterfactual quality metrics with confidence intervals.

    Args:
        data: DataFrame with counterfactual quality metrics
        save_path: Optional path to save the plot
        metrics: List of metrics to plot (if None, plots all available metrics)
    """
    set_pub_theme()
    _maybe_mkdir(save_path)

    # Filter metrics if specified
    if metrics is None:
        # Get all quality metrics (exclude causal_world if it's an index)
        quality_metrics = [col for col in data.columns if col != "causal_world"]
    else:
        quality_metrics = metrics

    # Prepare data for plotting
    plot_data = []
    for metric in quality_metrics:
        if metric in data.columns:
            values = data[metric].dropna()
            if len(values) > 0:
                plot_data.append(
                    {
                        "metric": metric,
                        "mean": values.mean(),
                        "std": values.std(),
                        "ci_low": np.percentile(values, 2.5),
                        "ci_high": np.percentile(values, 97.5),
                        "count": len(values),
                    }
                )

    if not plot_data:
        print("No data available for plotting")
        return None

    df_plot = pd.DataFrame(plot_data)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bar plot with error bars
    x_pos = np.arange(len(df_plot))
    bars = ax.bar(
        x_pos,
        df_plot["mean"],
        yerr=[
            df_plot["mean"] - df_plot["ci_low"],
            df_plot["ci_high"] - df_plot["mean"],
        ],
        capsize=5,
        # capthick=1,
        alpha=0.7,
        color=tol_palette("vibrant")[0],
    )

    # Customize the plot
    ax.set_xlabel("Counterfactual Quality Metrics", fontsize=12)
    ax.set_ylabel("Metric Value ± 95% CI", fontsize=12)
    ax.set_title(
        "Counterfactual Quality Metrics with Confidence Intervals", fontsize=14
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_plot["metric"], rotation=45, ha="right")

    # Add value labels on bars
    for i, (bar, mean_val, count) in enumerate(
        zip(bars, df_plot["mean"], df_plot["count"])
    ):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{mean_val:.3f}\n(n={count})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Add grid
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()

    return fig, ax


def load_model_metrics(
    base_path: Path, dataset_tag: str, classifiers: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load model performance and group fairness metrics for all classifiers.

    Args:
        base_path: Base path to the project
        dataset_tag: Dataset identifier
        classifiers: List of classifier names

    Returns:
        Tuple of (performance_df, fairness_df)
    """
    performance_data = []
    fairness_data = []

    for clf in classifiers:
        # Load model performance
        perf_path = (
            base_path
            / "output"
            / dataset_tag
            / "model_metrics"
            / clf
            / "model_performance.json"
        )
        if perf_path.exists():
            with open(perf_path, "r") as f:
                perf_metrics = json.load(f)

            # Convert to DataFrame format
            for metric, value in perf_metrics.items():
                performance_data.append(
                    {
                        "classifier": clf,
                        "metric": metric.replace("overall.", ""),
                        "value": value,
                    }
                )

        # Load group fairness
        fair_path = (
            base_path
            / "output"
            / dataset_tag
            / "model_metrics"
            / clf
            / "group_fairness.json"
        )
        if fair_path.exists():
            with open(fair_path, "r") as f:
                fair_metrics = json.load(f)

            # Extract overall fairness metrics
            for metric, value in fair_metrics.items():
                if not any(group in metric for group in ["Female", "Male"]):
                    fairness_data.append(
                        {"classifier": clf, "metric": metric, "value": value}
                    )

    performance_df = pd.DataFrame(performance_data)
    fairness_df = pd.DataFrame(fairness_data)

    return performance_df, fairness_df


def plot_model_performance(performance_df: pd.DataFrame, save_path: str | None = None):
    """
    Plot model performance metrics across classifiers.

    Args:
        performance_df: DataFrame with model performance metrics
        save_path: Optional path to save the plot
    """
    set_pub_theme()
    _maybe_mkdir(save_path)

    # Select key metrics for plotting
    key_metrics = ["accuracy", "precision", "recall", "f1", "matthews_corrcoef"]
    plot_data = performance_df[performance_df["metric"].isin(key_metrics)].copy()

    if plot_data.empty:
        print("No performance data available for plotting")
        return None

    # Create the plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Color palette
    colors = tol_palette("vibrant")
    classifier_colors = {
        clf: colors[i] for i, clf in enumerate(plot_data["classifier"].unique())
    }

    for i, metric in enumerate(key_metrics):
        if i >= len(axes):
            break

        ax = axes[i]
        metric_data = plot_data[plot_data["metric"] == metric]

        if not metric_data.empty:
            # Create bar plot
            x_pos = np.arange(len(metric_data))
            bars = ax.bar(
                x_pos,
                metric_data["value"],
                color=[classifier_colors[clf] for clf in metric_data["classifier"]],
                alpha=0.7,
            )

            # Customize
            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12)
            ax.set_ylabel("Value", fontsize=10)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(metric_data["classifier"], rotation=45, ha="right")
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_axisbelow(True)

            # Add value labels on bars
            for bar, value in zip(bars, metric_data["value"]):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    # Remove empty subplots
    for i in range(len(key_metrics), len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle("Model Performance Metrics Across Classifiers", fontsize=16, y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()

    return fig, axes


def plot_group_fairness(fairness_df: pd.DataFrame, save_path: str | None = None):
    """
    Plot group fairness metrics across classifiers.

    Args:
        fairness_df: DataFrame with group fairness metrics
        save_path: Optional path to save the plot
    """
    set_pub_theme()
    _maybe_mkdir(save_path)

    if fairness_df.empty:
        print("No fairness data available for plotting")
        return None

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Prepare data for grouped bar plot
    metrics = fairness_df["metric"].unique()
    classifiers = fairness_df["classifier"].unique()

    x_pos = np.arange(len(metrics))
    width = 0.8 / len(classifiers)
    colors = tol_palette("vibrant")

    for i, clf in enumerate(classifiers):
        clf_data = fairness_df[fairness_df["classifier"] == clf]
        values = [
            (
                clf_data[clf_data["metric"] == metric]["value"].iloc[0]
                if not clf_data[clf_data["metric"] == metric].empty
                else 0
            )
            for metric in metrics
        ]

        bars = ax.bar(
            x_pos + i * width, values, width, label=clf, color=colors[i], alpha=0.7
        )

        # Add value labels
        for bar, value in zip(bars, values):
            if value != 0:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    # Customize the plot
    ax.set_xlabel("Fairness Metrics", fontsize=12)
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.set_title("Group Fairness Metrics Across Classifiers", fontsize=14)
    ax.set_xticks(x_pos + width * (len(classifiers) - 1) / 2)
    ax.set_xticklabels(
        [m.replace("_", " ").title() for m in metrics], rotation=45, ha="right"
    )
    ax.legend(title="Classifier", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()

    return fig, ax
