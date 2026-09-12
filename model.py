"""
Rete "vincolata dal connettoma": una recurrent network le cui connessioni
interne sono fissate dalla connettivita' reale del mushroom body (o, per il
confronto, da una maschera casuale con la stessa sparsita').

Biologia -> architettura:
  - input (le quantita' ordinate per prodotto) entra sui neuroni Kenyon (KC),
    gli unici che nel circuito reale ricevono l'input sensoriale (odore).
  - K passi di dinamica ricorrente attraverso la maschera di connettivita'.
  - l'azione si legge dai neuroni MBON (mushroom body output neurons),
    gli unici che nel circuito reale proiettano fuori dal mushroom body:
    una testa softmax indipendente per prodotto, su quante unita' servirne
    (0..MAX_COUNT).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

K_STEPS = 4  # passi di dinamica ricorrente prima della decisione


def load_circuit(data_dir="data"):
    neurons = pd.read_csv(Path(data_dir) / "mb_neurons.csv", dtype={"bodyId": str})
    edges = pd.read_csv(Path(data_dir) / "mb_edges.csv", dtype={"pre": str, "post": str})

    ids = list(neurons["bodyId"])
    idx = {b: i for i, b in enumerate(ids)}
    n = len(ids)

    adj = np.zeros((n, n), dtype=np.float32)
    for pre, post, w in edges[["pre", "post", "weight"]].itertuples(index=False):
        adj[idx[pre], idx[post]] = w
    adj = adj / (adj.max() + 1e-8)

    is_kc = neurons["type"].str.startswith("KC").to_numpy()
    is_mbon = neurons["type"].str.startswith("MBON").to_numpy()
    is_dop = neurons["is_dopaminergic"].to_numpy().astype(bool)

    return {
        "n": n,
        "adj": adj,
        "mask": (adj != 0).astype(np.float32),
        "is_kc": is_kc,
        "is_mbon": is_mbon,
        "is_dop": is_dop,
        "types": neurons["type"].to_numpy(),
    }


def random_mask_like(mask: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    flat = mask.flatten().copy()
    rng.shuffle(flat)
    return flat.reshape(mask.shape)


class ConnectomePolicy(nn.Module):
    """Per ciascuno dei n_items prodotti, una distribuzione categorica
    indipendente su {0, ..., max_count} unita' da servire."""

    def __init__(self, circuit: dict, n_items: int, max_count: int,
                 mask_override=None, normalize=False):
        super().__init__()
        n = circuit["n"]
        self.n = n
        self.k_steps = K_STEPS
        self.n_items = n_items
        self.n_choices = max_count + 1  # 0..max_count

        self.norm = nn.LayerNorm(n) if normalize else None

        mask = mask_override if mask_override is not None else circuit["mask"]
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))

        self.W = nn.Parameter(torch.randn(n, n) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n))

        kc_idx = np.where(circuit["is_kc"])[0]
        mbon_idx = np.where(circuit["is_mbon"])[0]
        if len(mbon_idx) == 0:
            mbon_idx = np.arange(n)
        self.register_buffer("kc_idx", torch.tensor(kc_idx, dtype=torch.long))
        self.register_buffer("mbon_idx", torch.tensor(mbon_idx, dtype=torch.long))

        self.input_proj = nn.Linear(n_items, len(kc_idx))
        self.output_proj = nn.Linear(len(mbon_idx), n_items * self.n_choices)

    def forward(self, order_counts: torch.Tensor) -> torch.Tensor:
        """order_counts: (batch, n_items) float, gia' normalizzato.
        Ritorna logits (batch, n_items, n_choices)."""
        batch = order_counts.shape[0]
        x = torch.zeros(batch, self.n, device=order_counts.device)
        x[:, self.kc_idx] = torch.relu(self.input_proj(order_counts))

        w_masked = self.W * self.mask
        for _ in range(self.k_steps):
            x = x @ w_masked + self.bias
            if self.norm is not None:
                x = self.norm(x)
            x = torch.tanh(x)

        out = x[:, self.mbon_idx]
        logits = self.output_proj(out)
        return logits.view(batch, self.n_items, self.n_choices)
