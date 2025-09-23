# rewritten_code1.py

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

    hue_order = ["RF", "GB", "LR"]
    palette = tol_palette("vibrant")
    palette = [palette[1], palette[0], palette[2]]

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

    hue_order = ["RF", "GB", "LR"]
    palette = tol_palette("vibrant")
    palette = [palette[1], palette[0], palette[2]]

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
