#!/usr/bin/env python3
"""
Plotting script for tuned classifier results.
Reads outputs produced by run_comprehensive_tuning.py and plots:
1) Model performance metrics
2) Group fairness metrics
Also writes a compact summary CSV.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def set_pub_theme(
    font_size: int = 14, line_width: float = 0.5, font_scale: float = 0.75
):
    """Seaborn/Matplotlib theme tuned for papers (serif fonts, subtle black)."""
    _new_black = "#373737"
    sns.set_theme(
        style="ticks",
        font_scale=font_scale,
        rc={
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
            "svg.fonttype": "none",
            "text.usetex": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "legend.fontsize": font_size,
            "legend.title_fontsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "axes.labelpad": 2,
            "axes.titlepad": 4,
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
            "xtick.minor.pad": 1,
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


def tol_palette(name: str = "vibrant"):
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
    }
    return sets.get(name, sets["vibrant"])


def find_available_classifiers(tuning_root: Path) -> list[str]:
    if not tuning_root.exists():
        raise ValueError(f"Tuning output root not found: {tuning_root}")
    return [p.name for p in tuning_root.iterdir() if p.is_dir()]


def load_tuned_results(tuning_root: Path, classifiers: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for clf in classifiers:
        clf_dir = tuning_root / clf
        perf_path = clf_dir / "model_performance.json"
        gf_path = clf_dir / "group_fairness.json"
        summary_path = clf_dir / "tuning_summary.json"

        if not perf_path.exists() or not gf_path.exists():
            print(f"⚠️ Skipping {clf}: missing metrics files")
            continue

        with open(perf_path, "r") as f:
            perf = json.load(f)
        with open(gf_path, "r") as f:
            gf = json.load(f)

        best_score = None
        try:
            if summary_path.exists():
                with open(summary_path, "r") as f:
                    summary = json.load(f)
                    best_score = summary.get("best_score")
        except Exception:
            pass

        row = {"classifier": clf}
        row.update(perf)
        row.update(gf)
        if best_score is not None:
            row["best_score"] = best_score
        rows.append(row)

    if not rows:
        raise ValueError(
            "No tuned metrics found. Have you run run_comprehensive_tuning.py?"
        )

    return pd.DataFrame(rows)


def plot_performance_metrics(df: pd.DataFrame, output_dir: Path):
    set_pub_theme()
    metrics = [
        "overall.accuracy",
        "overall.precision",
        "overall.recall",
        "overall.f1",
        "overall.matthews_corrcoef",
        "overall.true_positive_rate",
        "overall.true_negative_rate",
        "overall.false_positive_rate",
        "overall.false_negative_rate",
        "overall.selection_rate",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = tol_palette("vibrant")

    classifier_colors = {
        c: colors[i % len(colors)]
        for i, c in enumerate(sorted(df["classifier"].unique()))
    }
    for metric in metrics:
        if metric not in df.columns:
            print(f"⚠️ Metric not found: {metric}")
            continue

        fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
        bars = sns.barplot(
            data=df,
            x="classifier",
            y=metric,
            # hue="classifier",
            color="steelblue",
            # orient="h",
            # palette=classifier_colors,
            errorbar=None,
            legend=False,
            ax=ax,
        )
        for cont in bars.containers:
            ax.bar_label(cont, fmt="%.3f", padding=3)
        ax.set_xlabel("Classifier")
        ax.set_ylabel(metric.replace("overall.", "").replace("_", " ").title())
        ax.set_title(
            metric.replace("overall.", "").replace("_", " ").title() + " by Classifier",
            pad=12,
        )
        sns.despine()
        plt.tight_layout()
        out = output_dir / f"tuned_{metric.replace('.', '_')}_comparison.png"
        plt.savefig(out, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"✅ Saved {out}")


def plot_group_fairness_metrics(df: pd.DataFrame, output_dir: Path):
    set_pub_theme()
    fairness_metrics = [
        "dem_parity_diff",
        "dem_parity_ratio",
        "eq_odds_diff",
        "eq_odds_ratio",
        "Female.f1",
        "Female.true_positive_rate",
        "Female.true_negative_rate",
        "Female.false_positive_rate",
        "Female.false_negative_rate",
        "Female.selection_rate",
        "Male.f1",
        "Male.true_positive_rate",
        "Male.true_negative_rate",
        "Male.false_positive_rate",
        "Male.false_negative_rate",
        "Male.selection_rate",
    ]
    colors = tol_palette("vibrant")
    classifier_colors = {
        c: colors[i % len(colors)] for i, c in enumerate(df["classifier"].unique())
    }

    for metric in fairness_metrics:
        if metric not in df.columns:
            print(f"⚠️ Metric not found: {metric}")
            continue
        fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
        bars = sns.barplot(
            data=df,
            x="classifier",
            y=metric,
            # hue="classifier",
            color="steelblue",
            # orient="h",
            # palette=classifier_colors,
            errorbar=None,
            legend=False,
            ax=ax,
        )
        for cont in bars.containers:
            ax.bar_label(cont, fmt="%.3f", padding=3)
        ax.set_xlabel("Classifier")
        ax.set_ylabel(metric.replace("_", " ").title())

        # Custom titles for specific metrics
        if metric == "eq_odds_diff":
            title = "Equalized Odds Difference by Classifier"
        elif metric == "dem_parity_diff":
            title = "Demographic Parity Difference by Classifier"
        else:
            title = metric.replace("_", " ").title() + " by Classifier"

        ax.set_title(title, pad=12)
        sns.despine()
        plt.tight_layout()
        out = output_dir / f"tuned_{metric}_comparison.png"
        plt.savefig(out, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"✅ Saved {out}")


def plot_segmented_group_fairness_metrics(df: pd.DataFrame, output_dir: Path):
    set_pub_theme()

    # Define the base metrics (without gender prefix)
    base_metrics = [
        "f1",
        "true_positive_rate",
        "true_negative_rate",
        "false_positive_rate",
        "false_negative_rate",
        "selection_rate",
    ]

    # Create a reshaped DataFrame for plotting
    plot_data = []
    for _, row in df.iterrows():
        classifier = row["classifier"]
        for metric in base_metrics:
            female_col = f"Female.{metric}"
            male_col = f"Male.{metric}"

            if female_col in df.columns and male_col in df.columns:
                plot_data.append(
                    {
                        "classifier": classifier,
                        "sex": "Female",
                        "metric": metric,
                        "value": row[female_col],
                    }
                )
                plot_data.append(
                    {
                        "classifier": classifier,
                        "sex": "Male",
                        "metric": metric,
                        "value": row[male_col],
                    }
                )

    plot_df = pd.DataFrame(plot_data)

    if plot_df.empty:
        print("⚠️ No data available for segmented group fairness metrics")
        return

    # Define colors for sex groups
    sex_colors = {"Female": "#E74C3C", "Male": "#3498DB"}  # Red and Blue

    for metric in base_metrics:
        metric_data = plot_df[plot_df["metric"] == metric].copy()
        if metric_data.empty:
            print(f"⚠️ No data found for metric: {metric}")
            continue

        fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
        # Ensure we have a proper DataFrame
        if isinstance(metric_data, pd.DataFrame):
            bars = sns.barplot(
                data=metric_data,
                x="classifier",
                y="value",
                hue="sex",
                # orient="h",
                palette=sex_colors,
                errorbar=None,
                ax=ax,
            )

            # Add value labels on bars
            for cont in bars.containers:
                ax.bar_label(
                    cont, fmt="%.2f", padding=3, fontsize=13
                )  # Default fontsize is 10

            ax.set_xlabel(metric.replace("_", " ").title())
            ax.set_ylabel("Classifier")
            ax.set_title(
                f"{metric.replace('_', ' ').title()} by Classifier and Sex", pad=12
            )

            # Customize legend
            if metric == "f1" or metric == "false_negative_rate":
                ax.legend(title="Sex", loc="lower right")
            else:
                ax.legend().remove()
                # ax.legend(title="Sex", loc="center", bbox_to_anchor=(0.5, 1.15))

            sns.despine()
            plt.tight_layout()
            out = output_dir / f"tuned_{metric}_by_sex_comparison.png"
            plt.savefig(out, bbox_inches="tight", dpi=300)
            plt.close()
            print(f"✅ Saved {out}")
        else:
            print(f"⚠️ Invalid data type for metric: {metric}")


def create_metrics_summary_table(df: pd.DataFrame, output_dir: Path):
    print("📊 Creating metrics summary table...")
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
    cols = [c for c in ["classifier", *key_metrics, "best_score"] if c in df.columns]
    summary = df[cols].copy()
    for c in key_metrics:
        if c in summary.columns:
            summary[c] = summary[c].round(4)
    if "best_score" in summary.columns:
        summary["best_score"] = summary["best_score"].round(4)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "tuned_classification_metrics_summary.csv"
    summary.to_csv(out, index=False)
    print(f"✅ Saved summary table: {out}")


def main():
    parser = argparse.ArgumentParser(description="Plot tuned classification results")
    parser.add_argument("--dataset", type=str, default="adult", help="Dataset name")
    parser.add_argument("--knowledge", type=str, default="med", help="Knowledge level")
    parser.add_argument(
        "--tuning-root",
        type=Path,
        default=None,
        help="Root directory of tuned outputs (defaults to output/{dataset}/{knowledge}/tuning_comprehensive)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save plots (defaults to <tuning-root>/plots)",
    )
    args = parser.parse_args()

    base = Path(__file__).parent
    if args.tuning_root is None:
        tuning_root = (
            base / "output" / args.dataset / args.knowledge / "tuning_comprehensive"
        )
    else:
        tuning_root = args.tuning_root

    classifiers = find_available_classifiers(tuning_root)
    print(f"🔎 Found classifiers: {classifiers}")
    df = load_tuned_results(tuning_root, classifiers)

    output_dir = (
        args.output_dir if args.output_dir is not None else tuning_root / "plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # plot_performance_metrics(df, output_dir)
    # plot_group_fairness_metrics(df, output_dir)
    plot_segmented_group_fairness_metrics(df, output_dir)
    create_metrics_summary_table(df, output_dir)

    print("🎉 All tuned classification plots generated!")
    print(f"📁 Results saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
