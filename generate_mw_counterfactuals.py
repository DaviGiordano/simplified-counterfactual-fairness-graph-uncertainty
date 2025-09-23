import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd

from src.causal_discovery.causal_discovery import CausalDiscovery
from src.causality.counterfactual_graph_uncertainty import (
    generate_counterfactuals_from_worlds,
)
from src.dataset.load import load_dataset_col_trf
from src.load_parse.load import load_yaml
from src.log.logging_config import setup_logging
from src.utils import find_root

logger = logging.getLogger(__name__)


def parse_cli():
    p = argparse.ArgumentParser("Generate multi-world counterfactuals and pickle")
    p.add_argument(
        "--dataset",
        choices=[
            "adult",
        ],
        required=True,
    )
    p.add_argument(
        "--knowledge",
        choices=[
            "med",
        ],
        required=True,
    )
    p.add_argument(
        "--num_samples",
        type=int,
        help="Number of bootstrap samples",
        default=100,
    )
    p.add_argument(
        "--num_workers",
        type=int,
        help="Number of workers for counterf. generation",
        default=1,
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_cli()
    BASE_PATH = find_root()
    CD_CONFIG_FPATH = BASE_PATH / "causal_discovery_params/boss.yaml"
    KNOWLEDGE_MAP = {
        "adult": {
            "med": BASE_PATH / "data/adult/med_knowledge.txt",
        },
    }

    knowledge_fpath = KNOWLEDGE_MAP[args.dataset][args.knowledge]
    output_path = BASE_PATH / f"output/{args.dataset}/{args.knowledge}"

    current_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(output_path / f"log_{current_time}.log")

    configuration = load_yaml(CD_CONFIG_FPATH)
    enc_dataset, col_trf = load_dataset_col_trf(args.dataset)

    cd = CausalDiscovery(configuration, enc_dataset.X_train, knowledge_fpath)
    causal_worlds = cd.run_bootstrap_causal_discovery(
        col_trf,
        args.num_samples,
    )
    mw_counterfactuals = generate_counterfactuals_from_worlds(
        enc_dataset.X_enc_train,
        causal_worlds,
        enc_dataset.X_enc_test,
        enc_dataset.enc_sensitive_name,
        args.num_workers,
    )
    mw_counterfactuals.evaluate_counterfactuals_quality(
        enc_dataset.X_enc_test,
        enc_dataset.sensitive_test,
        args.num_workers,
    )

    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "mw_counterfactuals.pkl", "wb") as f:
        pickle.dump(mw_counterfactuals, f)
    with open(output_path / "causal_worlds.pkl", "wb") as f:
        pickle.dump(causal_worlds, f)

    logger.info("Finish: Counterfactual generation ended!")
