#!/usr/bin/env python3
"""
Plotting script for tuned classifiers' counterfactual metrics.
Loads CF evaluation outputs from tuned models and plots PSR/NSR
in the style of plot_causal_model_metrics.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def set_pub_theme(
    font_size: int = 14, line_width: float = 0.5, font_scale: float = 0.75
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


def tol_palette(name: str = "vibrant"):
    sets = {
        "vibrant": [
            "#EE7733",
            "#0077BB",
            "#33BBEE",
            "#EE3377",
            "#CC3311",
            "#009988",
            "#BBBBBB",
            "#000000",
        ]
    }
    return sets[name]


def load_tuned_cf_data(base_path: Path) -> pd.DataFrame:
    """Load CF evaluation results for all tuned classifiers."""
    classifiers = ["LR", "RF", "GB", "FAIRGBM"]
    rows: list[pd.DataFrame] = []
    for clf in classifiers:
        cf_path = (
            base_path
            / "output"
            / "adult"
            / "med"
            / "tuned_classification_cf"
            / clf
            / "cf_evaluation.csv"
        )
        if not cf_path.exists():
            print(f"⚠️  Missing CF results for {clf}: {cf_path}")
            continue
        try:
            df = pd.read_csv(cf_path)
        except Exception as e:
            print(f"⚠️  Could not read {cf_path}: {e}. Skipping {clf}.")
            continue
        # Ensure required columns exist
        required = [
            "causal_model",
            "Female_to_Male.negative_to_positive_switch_rate",
            "Female_to_Male.positive_to_negative_switch_rate",
            "Male_to_Female.negative_to_positive_switch_rate",
            "Male_to_Female.positive_to_negative_switch_rate",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"⚠️  Missing expected columns for {clf}: {missing}")
        df["classifier"] = clf
        rows.append(df)

    if not rows:
        raise ValueError("No tuned CF evaluation data found.")

    all_df = pd.concat(rows, ignore_index=True)
    print(f"📊 Loaded tuned CF rows: {len(all_df)}")
    return all_df


def rename_causal_models(df: pd.DataFrame) -> pd.DataFrame:
    """Rename causal model names for better display."""
    model_mapping = {
        "linear": "Linear",
        "lgbm": "LGBM",
        "causalflow": "CNF",
        "diffusion": "DCM",
    }
    df["causal_model"] = df["causal_model"].map(lambda x: model_mapping.get(x, x))
    return df


def melt_cf_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape CF rate columns to long format with metric and intervention labels."""
    mapping = {
        "Female_to_Male.negative_to_positive_switch_rate": ("PSR", "Female→Male"),
        "Female_to_Male.positive_to_negative_switch_rate": ("NSR", "Female→Male"),
        "Male_to_Female.negative_to_positive_switch_rate": ("PSR", "Male→Female"),
        "Male_to_Female.positive_to_negative_switch_rate": ("NSR", "Male→Female"),
    }
    rate_cols = list(mapping.keys())
    present_cols = [c for c in rate_cols if c in df.columns]
    m = df.melt(
        id_vars=["classifier", "causal_model"],
        value_vars=present_cols,
        var_name="rate_col",
        value_name="rate",
    )
    m["metric"] = m["rate_col"].map(lambda c: mapping.get(c, ("Unknown", "Unknown"))[0])
    m["intervention"] = m["rate_col"].map(
        lambda c: mapping.get(c, ("Unknown", "Unknown"))[1]
    )
    m.drop(columns=["rate_col"], inplace=True)
    return m


def plot_cf_metrics(df: pd.DataFrame, output_dir: Path):
    """Create a single 2x2 combined chart of PSR/NSR for both interventions."""
    set_pub_theme()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Colors by classifier
    colors = tol_palette("vibrant")
    clf_colors = {
        c: colors[i % len(colors)]
        for i, c in enumerate(sorted(df["classifier"].unique()))
    }

    interventions = ["Female→Male", "Male→Female"]
    metrics = ["PSR", "NSR"]

    print("📊 Creating combined tuned CF comparison plot...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)

    for i, metric in enumerate(metrics):
        for j, intervention in enumerate(interventions):
            ax = axes[i, j]
            sub = df[
                (df["metric"] == metric) & (df["intervention"] == intervention)
            ].copy()
            if len(sub) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            if isinstance(sub, pd.DataFrame) and not sub.empty:
                bars = sns.barplot(
                    data=sub,
                    x="causal_model",
                    y="rate",
                    hue="classifier",
                    palette=clf_colors,
                    errorbar="ci",
                    capsize=0.05,
                    ax=ax,
                )

            ax.set_title(f"{metric} - {intervention}", fontsize=14, pad=10)
            ax.set_xlabel("Causal Model")
            ax.set_ylabel(f"{metric} ± 95% CI")
            ax.set_ylim(0, 1)

            ax.legend().remove()

            sns.despine(ax=ax)

    fig.suptitle(
        "Counterfactual Fairness Metrics by Classifier and Causal Model",
        fontsize=16,
        y=0.98,
    )

    # Add figure-level legend centered at the top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Classifier",
        bbox_to_anchor=(0.5, 0.85),
        loc="lower center",
        ncol=4,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.90))
    out = output_dir / "tuned_counterfactual_fairness_comparison.png"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✅ Saved {out}")


def plot_single_causal_model_family(
    df: pd.DataFrame, output_dir: Path, causal_model_family: str
):
    """Plot CF metrics for a single causal model family across all classifiers."""
    set_pub_theme()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter data for the specified causal model family
    family_data = df[df["causal_model"] == causal_model_family].copy()

    if family_data.empty:
        print(f"⚠️ No data found for causal model family: {causal_model_family}")
        return

    # Colors by classifier
    colors = tol_palette("vibrant")
    unique_classifiers = sorted(pd.Series(family_data["classifier"]).unique())
    clf_colors = {c: colors[i % len(colors)] for i, c in enumerate(unique_classifiers)}

    interventions = ["Female→Male", "Male→Female"]
    metrics = ["PSR", "NSR"]

    print(f"📊 Creating CF metrics plot for {causal_model_family}...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)

    for i, metric in enumerate(metrics):
        for j, intervention in enumerate(interventions):
            ax = axes[i, j]
            sub = family_data[
                (family_data["metric"] == metric)
                & (family_data["intervention"] == intervention)
            ].copy()

            if len(sub) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            if isinstance(sub, pd.DataFrame) and not sub.empty:
                bars = sns.barplot(
                    data=sub,
                    x="classifier",
                    y="rate",
                    hue="classifier",
                    palette=clf_colors,
                    errorbar="ci",
                    capsize=0.05,
                    ax=ax,
                    legend=False,
                )

                # Add value labels on bars
                for cont in bars.containers:
                    ax.bar_label(cont, fmt="%.2f", padding=3, fontsize=14)

            ax.set_title(f"{metric} - {intervention}", fontsize=14, pad=10)
            ax.set_xlabel("Classifier")
            ax.set_ylabel(f"{metric} ± 95% CI")
            ax.set_ylim(0, 1)

            # Rotate x-axis labels if needed
            ax.tick_params(axis="x")

            sns.despine(ax=ax)

    fig.suptitle(
        f"Counterfactual Fairness Metrics - {causal_model_family} Causal Model",
        fontsize=16,
        y=0.98,
    )

    plt.tight_layout()
    out = output_dir / f"tuned_{causal_model_family.lower()}_cf_metrics.png"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✅ Saved {out}")


def create_summary_table(df: pd.DataFrame, output_dir: Path):
    """Create a summary table of mean PSR/NSR per causal model and classifier."""
    print("📊 Creating tuned CF metrics summary table...")
    # Calculate statistics manually to avoid pandas aggregation issues
    grouped = df.groupby(["classifier", "causal_model", "metric", "intervention"])[
        "rate"
    ]

    piv = pd.DataFrame(
        {
            "classifier": [],
            "causal_model": [],
            "metric": [],
            "intervention": [],
            "mean": [],
            "ci_lower": [],
            "ci_upper": [],
            "count": [],
        }
    )

    for name, group in grouped:
        if isinstance(name, tuple) and len(name) == 4:
            classifier, causal_model, metric, intervention = name
        else:
            continue
        piv = pd.concat(
            [
                piv,
                pd.DataFrame(
                    {
                        "classifier": [classifier],
                        "causal_model": [causal_model],
                        "metric": [metric],
                        "intervention": [intervention],
                        "mean": [group.mean()],
                        "ci_lower": [group.quantile(0.025)],
                        "ci_upper": [group.quantile(0.975)],
                        "count": [len(group)],
                    }
                ),
            ],
            ignore_index=True,
        )
    out = output_dir / "tuned_cf_metrics_summary.csv"
    piv.to_csv(out, index=False)
    print(f"✅ Saved summary table: {out}")


def main():
    print("🚀 Starting tuned CF metrics plotting...")
    print("=" * 60)

    base_path = Path(__file__).parent
    output_dir = base_path / "adult_tuned_causal_model_metrics"

    try:
        raw_df = load_tuned_cf_data(base_path)
        raw_df = rename_causal_models(raw_df)
        long_df = melt_cf_rates(raw_df)

        # Plot overall comparison
        plot_cf_metrics(long_df, output_dir)

        # Plot individual causal model families
        causal_model_families = ["Linear", "DCM", "CNF", "LGBM"]
        for family in causal_model_families:
            plot_single_causal_model_family(long_df, output_dir, family)

        create_summary_table(long_df, output_dir)
        print("\n🎉 All tuned CF plots generated successfully!")
        print(f"📁 Results saved to: {output_dir.absolute()}")
        print("=" * 60)
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
