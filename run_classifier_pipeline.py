import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd

from src.causality.causal_world import CausalWorld
from src.classification.classify import fit_evaluate_classifier
from src.dataset.counterfactuals_wrapper import MultiWorldCounterfactuals
from src.dataset.load import load_dataset_col_trf
from src.log.logging_config import setup_logging
from src.utils import find_root

logger = logging.getLogger(__name__)


def parse_cli():
    p = argparse.ArgumentParser("Generate multi-world counterfactuals and pickle")
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
    p.add_argument(
        "--classifier",
        choices=[
            "LR",
            "RF",
            "GB",
            "LR_no_sensitive",
            "RF_no_sensitive",
            "GB_no_sensitive",
            # Add FAIRGBM options
            "FAIRGBM",
            "FAIRGBM_no_sensitive",
            "FAIRGBM_equal_opportunity",
            "FAIRGBM_predictive_equality",
        ],
        required=True,
    )
    # Add FAIRGBM-specific arguments
    p.add_argument(
        "--fairgbm-config",
        type=str,
        help="Path to FAIRGBM hyperparameter configuration file",
    )
    p.add_argument(
        "--fairgbm-trials",
        type=int,
        default=20,
        help="Number of hyperparameter tuning trials for FAIRGBM (advanced wrapper with Optuna)",
    )
    p.add_argument(
        "--fairgbm-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for FAIRGBM hyperparameter tuning (advanced wrapper)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_cli()
    BASE_PATH = find_root()

    model_metrics_output_dir = (
        BASE_PATH / f"output/{args.dataset}/model_metrics/{args.classifier}"
    )
    cf_metrics_output_dir = (
        BASE_PATH / f"output/{args.dataset}/{args.knowledge}/{args.classifier}"
    )
    cf_metrics_output_dir.mkdir(parents=True, exist_ok=True)
    mw_counterfactuals_fpath = (
        BASE_PATH / f"output/{args.dataset}/{args.knowledge}/mw_counterfactuals.pkl"
    )
    causal_worlds_fpath = (
        BASE_PATH / f"output/{args.dataset}/{args.knowledge}/causal_worlds.pkl"
    )
    current_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(cf_metrics_output_dir / f"log_{current_time}.log")

    with open(mw_counterfactuals_fpath, "rb") as f:
        mw_counterfactuals: MultiWorldCounterfactuals = pickle.load(f)
    with open(causal_worlds_fpath, "rb") as f:
        causal_worlds: list[CausalWorld] = pickle.load(f)

    enc_dataset, col_trf = load_dataset_col_trf(args.dataset)

    # Prepare FAIRGBM-specific arguments
    fairgbm_kwargs = {}
    if args.classifier.startswith("FAIRGBM"):
        if args.fairgbm_config:
            fairgbm_kwargs["config_path"] = Path(args.fairgbm_config)
        fairgbm_kwargs["n_trials"] = args.fairgbm_trials
        fairgbm_kwargs["n_jobs"] = args.fairgbm_jobs

    classifier, y_pred = fit_evaluate_classifier(
        args.classifier, enc_dataset, model_metrics_output_dir, **fairgbm_kwargs
    )
    mw_counterfactuals.evaluate_counterfactual_fairness(
        classifier,
        y_pred,
        enc_dataset.sensitive_test,
    )
    mw_counterfactuals.score_counterfactuals(classifier)

    score_var_by_indiv = (
        mw_counterfactuals.scores.groupby(level="individual")  # type: ignore
        .var()
        .rename("score_variance_by_individual")
    )

    score_var_by_indiv.to_csv(
        cf_metrics_output_dir / "score_var_across_cws_by_individual.csv"
    )

    mw_counterfactuals.cf_metrics.to_csv(cf_metrics_output_dir / "cf_metrics.csv")  # type: ignore
