import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_context("paper")
sns.set_theme(
    style="ticks",
    rc={
        "text.usetex": True,
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


def plot_score_variance(df, save_path=None):
    g = sns.catplot(
        data=df,
        x="knowledge",
        y="score_variance_by_individual",
        hue="classifier",
        hue_order=[
            "RF",
            "GB",
            "LR",
            # "GB_no_sensitive",
            # "RF_no_sensitive",
            # "LR_no_sensitive",
        ],
        kind="bar",
        dodge=0.8,
        estimator="mean",
        errorbar="pi",
        capsize=0.1,
        height=3,
        aspect=2,
    )
    g.set_axis_labels("Domain Knowledge Level", "Avg Variance of Score ± 95\% CI")
    g.legend.set_title("Classifier")
    sns.despine(fig=g.figure, top=False, right=False, left=False, bottom=False)

    if save_path:
        plt.savefig(save_path, format="pdf", dpi=300)
    plt.show()
    return g


def plot_counterfactual_metrics(data, interventions, save_path=None):
    g = sns.catplot(
        data=data,
        x="Knowledge",
        y="Rate",
        kind="bar",
        hue="Classifier",
        hue_order=[
            "RF",
            "GB",
            "LR",
            # "GB_no_sensitive",
            # "RF_no_sensitive",
            # "LR_no_sensitive",
        ],
        col="intervention",
        estimator="mean",
        errorbar="pi",
        col_order=interventions,
        row="metric",
        row_order=["PSR", "NSR"],
        height=3,
        aspect=1.4,
    )

    sns.despine(fig=g.figure, top=False, right=False)

    g.set_titles(template="{col_name}")
    for ax in g.axes.flatten():
        ax.set_title(ax.get_title().replace("_", " "), pad=5)

    for metric, ax in zip(g.row_names, g.axes[:, 0]):
        ax.set_ylabel(f"{metric} ± 95\% CI", rotation=90, labelpad=10, va="center")

    plt.ylim(0, 1)

    if save_path:
        plt.savefig(save_path, format="pdf", dpi=300)

    return g
