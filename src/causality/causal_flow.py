"""
CausalFlow implementation using CausalMAF from causalflows library.
Based on the reference repository implementation.
"""

from __future__ import annotations

import math

# Note: CausalFlow implementation requires Python 3.11+ due to causalflows library compatibility
# For now, we'll create a placeholder implementation that raises an informative error
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from causalflows.flows import CausalMAF
from dowhy.gcm.divergence import auto_estimate_kl_divergence
from scipy.spatial import KDTree
from sklearn.model_selection import train_test_split

from src.utils import record_time


class CausalMAFModel:
    def __init__(
        self,
        graph: nx.DiGraph,
        columns: list[str],
        binary_root: str,
        hidden_features: tuple[int, ...] = (128, 128),  # Default from reference
    ):
        # 1) Topological order
        topo = list(nx.topological_sort(graph))
        if set(topo) != set(columns):
            raise ValueError("Graph nodes and columns mismatch.")
        self.columns = topo
        self.binary_root = binary_root
        self.col2idx = {c: i for i, c in enumerate(self.columns)}
        self.device = "cpu"

        A = nx.to_numpy_array(graph, nodelist=self.columns, dtype=int)
        np.fill_diagonal(A, 1)
        adjacency = torch.BoolTensor(A.astype(bool))

        # 2) Single-layer (abductive design is the default in CausalMAF)
        self.flow = CausalMAF(
            features=len(self.columns),
            context=0,
            adjacency=adjacency,
            hidden_features=list(hidden_features),  # MAF hypernet MLP sizes
        ).to(self.device)

    def _infer_binary_mask(self, df: pd.DataFrame) -> torch.BoolTensor:
        mask = []
        for c in self.columns:
            vals = pd.unique(df[c].dropna().astype(float))
            mask.append(set(vals).issubset({0.0, 1.0}))
        return torch.tensor(mask, dtype=torch.bool, device=self.device)

    def fit(
        self,
        df: pd.DataFrame,
        *,
        epochs: int = 200,  # Default from reference
        lr: float = 1e-3,  # Default from reference
        batch_size: int = 512,  # Default from reference
        val_split: float = 0.2,  # Default from reference
        patience: int = 50,  # Default from reference
        seed: int = 0,  # Default from reference
        warmup_steps: int = 1000,  # Default from reference
        verbose: bool = True,  # Default from reference
    ) -> dict[str, list[float]]:
        torch.manual_seed(seed)

        # Split
        from sklearn.model_selection import train_test_split

        df_tr, df_val = train_test_split(
            df, test_size=val_split, random_state=seed, shuffle=True
        )

        # Tensors in topo order
        x_tr = torch.from_numpy(df_tr[self.columns].to_numpy(np.float32)).to(
            self.device
        )
        x_val = torch.from_numpy(df_val[self.columns].to_numpy(np.float32)).to(
            self.device
        )

        # 3) Binary mask for dequantization
        binary_mask = self._infer_binary_mask(df_tr)
        bin_idx = torch.nonzero(binary_mask, as_tuple=False).squeeze(-1)

        opt = torch.optim.Adam(self.flow.parameters(), lr=lr)
        self.flow.train()

        train_losses: list[float] = []
        val_losses: list[float] = []
        best_state: Optional[dict] = None
        best_val: float = float("inf")
        no_improve = 0
        global_step = 0

        for ep in range(1, epochs + 1):
            # Shuffle
            perm = torch.randperm(len(x_tr), device=self.device)

            # Mini-batches
            mb_losses = []
            for i in range(0, len(x_tr), batch_size):
                xb = x_tr[perm[i : i + batch_size]]

                # 3) Dequantize binary variables (train only)
                if bin_idx.numel() > 0:
                    noise = torch.rand(
                        (xb.size(0), bin_idx.numel()), device=self.device
                    )
                    xb = xb.clone()
                    xb[:, bin_idx] = xb[:, bin_idx] + noise  # uniform dequantization

                loss = -self.flow().log_prob(xb).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                mb_losses.append(float(loss.detach().cpu()))

                # 5) Linear LR warmup
                global_step += 1
                if warmup_steps > 0 and global_step <= warmup_steps:
                    scale = global_step / float(warmup_steps)
                    for g in opt.param_groups:
                        g["lr"] = lr * scale

            # Average train loss over batches
            train_loss = float(np.mean(mb_losses))

            # 4) Eval mode for val loss
            with torch.no_grad():
                self.flow.eval()
                val_loss = float(-self.flow().log_prob(x_val).mean().cpu())
                self.flow.train()

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if verbose:
                print(
                    f"[epoch {ep:4d}/{epochs}] train={train_loss:.4f}  val={val_loss:.4f}"
                )

            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_state = {
                    k: v.cpu().clone() for k, v in self.flow.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    if verbose:
                        print(f"Early stop at epoch {ep} (best val={best_val:.4f})")
                    break

        if best_state is not None:
            self.flow.load_state_dict(best_state)

        return {"train": train_losses, "val": val_losses}

    @torch.no_grad()
    def nll(self, df_test: pd.DataFrame) -> float:
        x = torch.from_numpy(df_test[self.columns].to_numpy(np.float32)).to(self.device)
        self.flow.eval()
        return float(-self.flow().log_prob(x).mean().cpu())

    @torch.no_grad()
    def generate_counterfactuals(self, df_test: pd.DataFrame) -> pd.DataFrame:
        if self.binary_root not in df_test:
            raise ValueError(f"{self.binary_root!r} missing.")
        factual = torch.from_numpy(df_test[self.columns].to_numpy(np.float32)).to(
            self.device
        )
        idx = self.col2idx[self.binary_root]
        flipped = 1.0 - df_test[self.binary_root].astype(float).to_numpy(np.float32)
        cf = self.flow().compute_counterfactual(
            factual, idx, torch.from_numpy(flipped).to(self.device)
        )
        df_cf = pd.DataFrame(
            cf.cpu().numpy(), columns=self.columns, index=df_test.index
        )
        df_cf[self.binary_root] = flipped
        return df_cf[self.columns]

    @staticmethod
    def plot_losses(
        losses: dict[str, list[float]],
        test_loss: float,
        *,
        show: bool = True,
        save_path: str | None = None,
    ) -> None:
        """Plots train/val curves."""
        epochs = range(1, len(losses["train"]) + 1)
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, losses["train"], label="train", marker="o")
        plt.plot(epochs, losses["val"], label="val", marker="x")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(
            f"train and validation losses per epoch. (test loss = {test_loss:.2f})"
        )
        plt.legend()
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150)
        if show:
            plt.show()
        plt.close()


@torch.no_grad()
def evaluate_flow(flow_model, df_test, n_gen: int | None = None, k: int = 5):
    cols = flow_model.columns
    n_gen = n_gen or len(df_test)

    gen_np = flow_model.flow().sample((n_gen,)).cpu().numpy()
    df_gen = pd.DataFrame(gen_np, columns=cols)

    mse = {c: float(((df_test[c] - df_gen[c].mean()) ** 2).mean()) for c in cols}

    kl_vals = [
        float(auto_estimate_kl_divergence(df_gen[c].to_numpy(), df_test[c].to_numpy()))
        for c in cols
    ]
    summary = {
        "mean_mse": float(np.mean(list(mse.values()))),
        "overall_kl": float(np.mean(kl_vals)),  # same as evaluate_causal_model
    }
    return mse, summary


def fit_and_eval_flow(
    graph,
    df_train,
    df_test,
    binary_root,
    loss_png_path: Optional[Path] = None,
):
    times: dict[str, float] = {}
    flow = CausalMAFModel(
        graph=graph,
        columns=df_train.columns.tolist(),
        binary_root=binary_root,
    )
    with record_time("time_fit", times):
        losses = flow.fit(
            df_train,
            epochs=200,  # Default from reference
            lr=1e-3,  # Default from reference
            batch_size=512,  # Default from reference
            val_split=0.2,  # Default from reference
            patience=50,  # Default from reference
            seed=0,  # Default from reference
            verbose=True,  # Default from reference
        )
    with record_time("time_evaluate_model", times):
        test_nll = flow.nll(df_test)
        _, summary = evaluate_flow(flow, df_test, n_gen=len(df_test))
        summary["test_nll"] = test_nll

    if loss_png_path:
        flow.plot_losses(
            losses, summary["test_nll"], show=False, save_path=str(loss_png_path)
        )
    return flow, summary, times
