"""
Allena due reti identiche (stessa architettura, stessa sparsita') sul task
del bar con ordini a quantita' (es. 2 cappuccini + 3 cornetti): una con la
connettivita' reale del mushroom body, una con connettivita' casuale.

Nota di design: la ricompensa qui e' deterministica e piena informazione
(l'azione corretta = l'ordine stesso, nessuna incertezza nascosta da
scoprire per esplorazione). In questo regime REINFORCE e' matematicamente
overkill e troppo rumoroso per convergere in tempi ragionevoli su 4 teste
indipendenti (verificato: con reward puro resta bloccato al livello del
caso anche con batching ed entropy bonus, mentre la stessa architettura
allenata con cross-entropy supervisionata raggiunge accuracy 1.0 in poche
centinaia di step). Si usa quindi cross-entropy diretta sull'etichetta nota
(l'ordine) -> stesso segnale di correttezza, gradiente molto piu' pulito.
Il punteggio +1/-1 mostrato nell'app resta quello vero (corretto/sbagliato),
e' solo il metodo di ottimizzazione ad essere supervisionato.
"""
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bar_env import BarEnv, N_ITEMS, MAX_COUNT
from model import ConnectomePolicy, load_circuit, random_mask_like

STEPS = 3000
BATCH = 32
LR = 3e-3
LOG_EVERY = 20


def train_one(circuit, mask_override, seed, tag, out_dir: Path, normalize=False):
    torch.manual_seed(seed)
    env = BarEnv(seed=seed)
    policy = ConnectomePolicy(circuit, n_items=N_ITEMS, max_count=MAX_COUNT,
                               mask_override=mask_override, normalize=normalize)
    opt = torch.optim.Adam(policy.parameters(), lr=LR)

    running_avg_reward = 0.0
    log_rows = []

    for step in range(1, STEPS + 1):
        orders = [env.reset() for _ in range(BATCH)]
        order_t = torch.tensor(orders, dtype=torch.float32) / MAX_COUNT
        target = torch.tensor(orders, dtype=torch.long)  # (BATCH, n_items)

        logits = policy(order_t)  # (BATCH, n_items, n_choices)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))

        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            pred = logits.argmax(dim=-1)  # (BATCH, n_items)
            correct = (pred == target).float()
            reward = (correct * 2 - 1).mean().item()  # +1/-1 per item -> media

        running_avg_reward = 0.95 * running_avg_reward + 0.05 * reward
        if step % LOG_EVERY == 0:
            log_rows.append({"episode": step * BATCH, "running_reward": running_avg_reward, "tag": tag})

    out_dir.mkdir(exist_ok=True)
    torch.save(policy.state_dict(), out_dir / f"policy_{tag}.pt")
    return log_rows, policy


def main():
    circuit = load_circuit("data")
    out_dir = Path("checkpoints")

    print(f"Circuito: {circuit['n']} neuroni, {int(circuit['mask'].sum())} sinapsi, "
          f"{circuit['is_kc'].sum()} KC, {circuit['is_mbon'].sum()} MBON, "
          f"{circuit['is_dop'].sum()} dopaminergici")
    print(f"Task: {N_ITEMS} prodotti, 0..{MAX_COUNT} unita' ciascuno")

    all_logs = []
    rand_mask = random_mask_like(circuit["mask"], seed=0)

    conditions = [
        ("biologica", None, False),
        ("random", rand_mask, False),
        ("biologica_norm", None, True),
        ("random_norm", rand_mask, True),
    ]
    finals = {}
    for tag, mask, norm in conditions:
        print(f"\n--- training: {tag} ---", flush=True)
        logs, _ = train_one(circuit, mask, seed=0, tag=tag, out_dir=out_dir, normalize=norm)
        all_logs += logs
        finals[tag] = logs[-1]["running_reward"]
        print(f"    reward finale: {finals[tag]:.3f}", flush=True)

    with open(out_dir / "training_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "running_reward", "tag"])
        w.writeheader()
        w.writerows(all_logs)

    print(f"\nSalvati checkpoint e log in {out_dir}/")
    for tag, val in finals.items():
        print(f"Reward finale (media mobile) - {tag}: {val:.3f}")


if __name__ == "__main__":
    main()
