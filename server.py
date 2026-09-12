"""
Backend minimale per il gioco 3D: espone il modello gia' allenato (stesso
model.py/bar_env.py usati per l'esperimento) come API, e serve la pagina
Three.js statica. Un solo file, niente framework in piu' del necessario.

Avvio: ./moschina-env/Scripts/python.exe server.py
"""
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import queue_store
from bar_env import BarEnv, N_ITEMS, MAX_COUNT, ORDERS, describe_order
from model import ConnectomePolicy, load_circuit, random_mask_like

app = FastAPI()

circuit = load_circuit("data")
rand_mask = random_mask_like(circuit["mask"], seed=0)
CONDITIONS = {
    "biologica": (None, False),
    "random": (rand_mask, False),
    "biologica_norm": (None, True),
    "random_norm": (rand_mask, True),
}
POLICIES = {}
MODEL_INFO = {}
for tag, (mask, norm) in CONDITIONS.items():
    p = ConnectomePolicy(circuit, n_items=N_ITEMS, max_count=MAX_COUNT, mask_override=mask, normalize=norm)
    ckpt = Path("checkpoints") / f"policy_{tag}.pt"
    if ckpt.exists():
        p.load_state_dict(torch.load(ckpt, map_location="cpu"))
    p.eval()
    POLICIES[tag] = p
    MODEL_INFO[tag] = {
        "trained": True,
        "checkpoint_file": str(ckpt),
        "checkpoint_bytes": ckpt.stat().st_size if ckpt.exists() else None,
        "checkpoint_mtime": ckpt.stat().st_mtime if ckpt.exists() else None,
        "n_params": sum(t.numel() for t in p.parameters()),
    }

# ponytail: rete IDENTICA nell'architettura ma MAI allenata (pesi random) - serve
# come controprova: se questa sbaglia e la "biologica" no, sullo stesso identico
# codice, e' la dimostrazione che l'allenamento fa la differenza reale (non e'
# un trucco che rispecchia semplicemente l'input).
untrained = ConnectomePolicy(circuit, n_items=N_ITEMS, max_count=MAX_COUNT, mask_override=None, normalize=False)
untrained.eval()
POLICIES["untrained"] = untrained
MODEL_INFO["untrained"] = {
    "trained": False,
    "checkpoint_file": None,
    "checkpoint_bytes": None,
    "checkpoint_mtime": None,
    "n_params": sum(t.numel() for t in untrained.parameters()),
}

env = BarEnv()


class OrderIn(BaseModel):
    nickname: str
    items: list[int]  # 4 interi, uno per prodotto (vedi ORDERS)


@app.get("/api/next_customer")
def next_customer(tag: str = "biologica"):
    """Prende il prossimo cliente (uno vero in coda se c'e', altrimenti casuale),
    lo fa servire dalla policy scelta, e ritorna tutto cio' che serve alla scena 3D."""
    custom = queue_store.pop_order()
    nickname = None
    if custom is not None:
        env.set_next_order(custom.items)
        nickname = custom.nickname

    order = env.current_order
    policy = POLICIES.get(tag, POLICIES["biologica"])
    with torch.no_grad():
        order_t = torch.tensor([order], dtype=torch.float32) / MAX_COUNT
        logits = policy(order_t)
        probs = torch.softmax(logits, dim=-1)[0].tolist()  # (n_items, n_choices) - trasparenza: la distribuzione grezza, non solo la scelta finale
        action = logits.argmax(dim=-1)[0].tolist()

    _, reward, info = env.step(action, nickname=nickname)
    return {
        "nickname": info["nickname"] or "cliente casuale",
        "is_custom": nickname is not None,
        "order": info["order"],
        "served": info["served"],
        "order_text": info["order_text"],
        "served_text": info["served_text"],
        "all_correct": info["all_correct"],
        "per_item_correct": info["per_item_correct"],
        "items": ORDERS,
        "reward": reward,
        "probs": probs,  # probs[i] = [P(0 unita'), P(1), P(2), P(3)] per il prodotto i
        "tag": tag,
        "model_trained": MODEL_INFO.get(tag, {}).get("trained"),
    }


@app.get("/api/model_info")
def model_info():
    return MODEL_INFO


@app.post("/api/order")
def place_order(order: OrderIn):
    if len(order.items) != N_ITEMS or sum(order.items) == 0:
        return {"ok": False, "error": "ordine vuoto o malformato"}
    queue_store.push_order(order.nickname, order.items)
    return {"ok": True, "queue_length": queue_store.queue_length()}


@app.get("/api/queue")
def get_queue():
    return {
        "length": queue_store.queue_length(),
        "orders": [{"nickname": o.nickname, "items": o.items, "text": describe_order(o.items)}
                   for o in queue_store.peek_queue()],
    }


@app.get("/api/conditions")
def get_conditions():
    return {"tags": list(CONDITIONS.keys()), "items": ORDERS, "max_count": MAX_COUNT}


app.mount("/", StaticFiles(directory="web", html=True), name="web")
