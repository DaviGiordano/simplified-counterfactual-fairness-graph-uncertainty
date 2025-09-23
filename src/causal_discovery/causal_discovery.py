#!/usr/bin/env python
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd
import pydot
import yaml
from sklearn.compose import ColumnTransformer

from src.causal_discovery.pytetrad.TetradSearch import TetradSearch
from src.causality.causal_world import CausalWorld
from src.graph.cpdag_to_dag import all_dags_from_cpdag
from src.graph.encode import encode_graph
from src.graph.inspection import (
    get_edge_frequencies,
    get_unique_cpdags_count,
    get_unique_dags_count,
)

logger = logging.getLogger(__name__)


class CausalDiscovery:
    """End-to-end orchestration: load data, configure Tetrad, run, and save."""

    def __init__(
        self,
        configuration: dict,
        df_train: pd.DataFrame,
        knowledge_fpath: Optional[Path] = None,
        sampling_seed: int = 42,
        verbose=True,
    ):
        logger.info("Initializing Causal Discovery..")
        logger.info(f"Configuration: \n{configuration}")
        logger.info(f"Data shape: {df_train.shape}")
        if knowledge_fpath:
            logger.info("Knowledge fpath: %s", knowledge_fpath)
            logger.info(f"Knowledge content:\n{knowledge_fpath.read_text(encoding='utf-8').strip()}")  # type: ignore

        self.configuration: dict = configuration
        self.df_train: pd.DataFrame = df_train
        self.knowledge_path: Optional[Path] = knowledge_fpath
        self.sampling_seed = sampling_seed
        self.verbose = verbose

        self.search: Optional[TetradSearch] = None
        self.elapsed_seconds: Optional[float] = None

    def run_bootstrap_causal_discovery(
        self,
        column_transformer: ColumnTransformer,
        number_bootstraps: int = 1,
    ) -> list[CausalWorld]:
        """
        Samples the data and runs the configured causal discovery algorithm.
        returns a list with elements:
            {data_index: pd.Series, cpdag:pydot.Dot, dag:nx.DiGraph}
        If each bootstrap yields N_b dags (varies with b), the list has number_bootstraps*N_b,
        """

        boostrap_results = []
        rng = np.random.default_rng(self.sampling_seed)
        logger.info(
            f"Starting bootstrap Causal Discovery with {number_bootstraps} boostrap samples.."
        )

        for idx in range(number_bootstraps):
            logger.info(f"Running bootstrap replica {idx+1}/{number_bootstraps}..")
            data_sample = self.df_train.sample(
                n=len(self.df_train),
                replace=True,
                random_state=rng,
            )
            result = self.run_causal_discovery(data_sample, column_transformer)
            boostrap_results.extend(result)

        if self.verbose:
            self.summarize_results(boostrap_results)

        return boostrap_results

    def run_causal_discovery(
        self,
        df_train: pd.DataFrame,
        column_transformer: ColumnTransformer,
    ) -> list[CausalWorld]:
        """
        Receives data to find graphs and returns a list with dicts that contain:
        {data_index: pd.Series, cpdag:pydot.Dot, dag:nx.DiGraph}
        If one cpdag has N dags in its MEC, we yield a list with N elements.
        Optionally, encodes the dags
        """
        logger.info(f"Running algorithm: {self.configuration['algorithm_name']}")

        search = self._get_configured_search(
            df_train,
            self.configuration,
            self.knowledge_path,
        )
        start = time.perf_counter()

        self._run_algorithm(
            search,
            self.configuration["algorithm_name"].lower(),
            self.configuration.get("algorithm_params", {}),
        )

        self.elapsed_seconds = time.perf_counter() - start
        cpdag = pydot.graph_from_dot_data(search.get_dot())[0]  # type: ignore
        result_dags = all_dags_from_cpdag(cpdag, all_nodes=list(self.df_train.columns))

        causal_worlds = [
            CausalWorld(
                dag=dag,
                enc_dag=encode_graph(dag, column_transformer),
                cpdag=cpdag,
                data_index=df_train.index,
            )
            for dag in result_dags
        ]

        logger.info(f"Causal Discovery CPDAG result:\n{search.get_string()}")
        logger.info(
            "Algorithm execution completed in %.2f seconds", self.elapsed_seconds
        )
        for i, causal_world in enumerate(causal_worlds):
            logger.info(
                f"Result DAG {i+1}/{len(result_dags)}:{causal_world.dag.edges()}"
            )
            # if column_transformer:
            #     logger.info(
            #         f"Result encoded DAG {i+1}/{len(result_dags)}:{causal_world.enc_dag.edges()}"
            #     )

        return causal_worlds

    def _get_configured_search(
        self,
        data: pd.DataFrame,
        configuration: dict,
        knowledge_path: Optional[Path],
    ) -> TetradSearch:
        """Create search instance and apply configurations."""
        # Create search
        search = TetradSearch(data)

        # Configure knowledge
        if knowledge_path:
            self._configure_knowledge(search, knowledge_path)

        # Configure test component
        test_name = configuration.get("test_name")
        if test_name:
            self._configure_test_or_score(
                search,
                test_name,
                configuration.get("test_params"),  # type: ignore
            )

        # Configure score component
        score_name = configuration.get("score_name")
        if score_name:
            logger.info("Configuring score component: %s", score_name)
            self._configure_test_or_score(
                search,
                score_name,
                configuration.get("score_params"),  # type: ignore
            )

        return search

    @staticmethod
    def _configure_knowledge(
        search: TetradSearch,
        knowledge_path: Optional[Path],
    ) -> Optional[Dict[str, str]]:
        """Load domain background knowledge from file and inject into the search."""
        if knowledge_path is None:
            return None

        if not knowledge_path.exists():
            logger.error("Knowledge file not found: %s", knowledge_path)
            raise FileNotFoundError(
                f"Knowledge file '{knowledge_path}' does not exist."
            )

        try:
            search.load_knowledge(str(knowledge_path))

        except Exception as err:
            logger.error("Failed to load knowledge: %s", err)
            raise RuntimeError(
                f"Unable to load knowledge file '{knowledge_path}': {err}"
            ) from err

        return {
            "path": str(knowledge_path),
            "content": knowledge_path.read_text(encoding="utf-8").strip(),
        }

    @staticmethod
    def _configure_test_or_score(
        search: TetradSearch,
        name: Optional[str],
        params: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Configure a test or score component with the given parameters."""
        if not name:
            return None

        valid_names = [
            "use_fisher_z",
            "use_conditional_gaussian_test",
            "use_degenerate_gaussian_test",
            "use_conditional_gaussian_score",
            "use_degenerate_gaussian_score",
        ]

        if name not in valid_names:
            raise ValueError(
                f"Unsupported method '{name}'. Choices: {list(valid_names)}"
            )

        logger.info("Configuring %s with params: %s", name, params)
        try:
            getattr(search, name)(**params)
        except Exception as err:
            logger.error(f"Setting {name} failed: {err}")
            raise RuntimeError(f"Unable to setup '{name}': {err}") from err

        return {"name": name, **params}

    @staticmethod
    def _run_algorithm(search: TetradSearch, name: str, params: Dict[str, Any]) -> None:
        """Execute the selected causal discovery algorithm."""

        valid_names = [
            "run_pc",
            "run_grasp",
            "run_dagma",
            "run_boss",
            "run_fges",
            "run_direct_lingam",
        ]

        if name not in valid_names:
            raise ValueError(
                f"Unsupported method '{name}'. Choices: {list(valid_names)}"
            )

        logger.info("Running algorithm %s with params: %s", name, params)
        try:
            getattr(search, name)(**params)
            logger.info("Algorithm %s completed successfully", name)
        except Exception as err:
            logger.error(f"Algorithm {name} failed: {err}")
            raise RuntimeError(f"Unable to run algorithm '{name}': {err}") from err

    @staticmethod
    def _save_graph(
        search: TetradSearch,
        output_path: Path,
    ) -> None:
        """Write both Tetrad's native graph string **and** a plain DOT file."""
        logger.info("Preparing graph output")
        graph_str = str(search.java)
        dot_str = search.get_dot()
        dot_path = output_path.with_suffix(".dot")
        try:
            logger.info("Writing Tetrad graph to %s", output_path)
            with output_path.open("w", encoding="utf-8") as fh:
                fh.write(graph_str)

            logger.info("Writing DOT graph to %s", dot_path)
            with dot_path.open("w", encoding="utf-8") as fh_dot:
                fh_dot.write(dot_str)

            logger.info("Successfully saved graph output")
        except Exception as err:
            logger.error("Failed to write graph output: %s", err)
            raise RuntimeError(
                f"Unable to write graph outputs ('{output_path}', '{dot_path}'): {err}"
            ) from err

    @staticmethod
    def summarize_results(results: list) -> None:

        all_cpdags: list[pydot.Dot] = []
        for result in results:
            all_cpdags.append(result.cpdag)

        logger.info(f"Number of unique CPDAGS: {get_unique_cpdags_count(all_cpdags)}")

        all_dags: list[nx.DiGraph] = []
        for result in results:
            all_dags.append(result.dag)
        logger.info(get_unique_dags_count(all_dags))
        logger.info(
            f"Frequency across {len(results)} dags:\n{get_edge_frequencies(all_dags)}"
        )
