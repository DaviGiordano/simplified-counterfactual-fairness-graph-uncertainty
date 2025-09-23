import argparse
import json
import logging
import pickle
from pathlib import Path

from src.dataset.counterfactuals_wrapper import MultiWorldCounterfactuals
from src.dataset.load import load_dataset_col_trf
from src.graph.inspection import count_subdags_from_root
from src.utils import find_root

logger = logging.getLogger(__name__)


def parse_cli():
    p = argparse.ArgumentParser("Inspect multi-world counterfactuals and save results")
    p.add_argument(
        "--dataset",
        choices=[
            "adult",
            "adult_small",
            "synthetic",
            "compas",
            "bank_marketing",
            "credit_card",
            "law_school",
            "diabetes_hospital",
        ],
        required=True,
    )
    p.add_argument(
        "--knowledge",
        choices=["low", "med", "high", "forbidx1", "forbidx2"],
        required=True,
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_cli()
    BASE_PATH = find_root()
    CD_CONFIG_FPATH = BASE_PATH / "causal_discovery_params/boss.yaml"
    artifacts_path = BASE_PATH / f"output/{args.dataset}/{args.knowledge}"
    mw_counterfactuals_fpath = artifacts_path / "mw_counterfactuals.pkl"
    causal_worlds_fpath = artifacts_path / "causal_worlds.pkl"
    output_path = artifacts_path / "counterfactuals_summary"
    output_path.mkdir(parents=True, exist_ok=True)
    enc_dataset, col_trf = load_dataset_col_trf(args.dataset)

    with open(mw_counterfactuals_fpath, "rb") as f:
        mw_counterfactuals: MultiWorldCounterfactuals = pickle.load(f)

    with open(causal_worlds_fpath, "rb") as f:
        causal_worlds = pickle.load(f)

    # Save feature variances by individual
    feat_var_by_indiv = (
        mw_counterfactuals.counterfactuals.groupby(level="individual")
        .var()
        .iloc[0]
        .rename("feat_var_by_individual")
        .sort_values(ascending=False)
    )
    feat_var_by_indiv.round(10).to_csv(
        output_path / "feat_var_by_individual.csv",
        index_label="feat",
    )

    # # # Save counterfactuals quality
    # mw_counterfactuals.counterfactuals_quality.round(10).to_csv(  # type: ignore
    #     output_path / "counterfactuals_quality.csv"
    # )

    # Save unique subdags from sensitive feature
    dags = [cw.dag for cw in causal_worlds]
    dag_cnt = count_subdags_from_root(
        dags,
        enc_dataset.sensitive_name,
    )
    with open(output_path / "unique_subdags_from_sensitive.json", "w") as f:
        json.dump({str(k): v for k, v in dict(dag_cnt).items()}, f, indent=4)
