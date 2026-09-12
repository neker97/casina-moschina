"""
Ambiente: un bar dove arrivano clienti con un ordine = quantita' per
ciascun prodotto (es. 2 cappuccini + 3 cornetti alla nutella). Azione della
mosca = quantita' che decide di servire per ciascun prodotto. Reward per
prodotto: +1 se il conteggio combacia esattamente, -1 altrimenti; media sui
4 prodotti (range -1..+1, stessa scala usata nell'esperimento precedente).

Bandit contestuale multi-item: niente movimento nello spazio, un cliente
per step.
"""
import random
from typing import List, Optional

ORDERS = ["caffe", "cappuccino", "cornetto_semplice", "cornetto_nutella"]
ORDER_LABELS_IT_SING = {
    "caffe": "caffe",
    "cappuccino": "cappuccino",
    "cornetto_semplice": "cornetto semplice",
    "cornetto_nutella": "cornetto alla nutella",
}
ORDER_LABELS_IT_PLUR = {
    "caffe": "caffe",
    "cappuccino": "cappuccini",
    "cornetto_semplice": "cornetti semplici",
    "cornetto_nutella": "cornetti alla nutella",
}
# retrocompatibilita' (usato in app.py per le etichette dei singoli item nel form)
ORDER_LABELS_IT = ORDER_LABELS_IT_SING
N_ITEMS = len(ORDERS)
MAX_COUNT = 3  # quantita' massima ordinabile per prodotto (0..MAX_COUNT)


def describe_order(counts: List[int]) -> str:
    parts = []
    for name, c in zip(ORDERS, counts):
        if c > 0:
            label = ORDER_LABELS_IT_PLUR[name] if c > 1 else ORDER_LABELS_IT_SING[name]
            parts.append(f"{c} {label}")
    if not parts:
        return "nessun ordine (?)"
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " e " + parts[-1]


class BarEnv:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.current_order: List[int] = self._random_order()

    def _random_order(self) -> List[int]:
        while True:
            counts = [self.rng.randint(0, MAX_COUNT) for _ in range(N_ITEMS)]
            if sum(counts) > 0:
                return counts

    def reset(self) -> List[int]:
        self.current_order = self._random_order()
        return self.current_order

    def set_next_order(self, counts: List[int]) -> None:
        """Inietta un ordine specifico (es. da un cliente esterno) invece del prossimo casuale."""
        self.current_order = list(counts)

    def step(self, action_counts: List[int], nickname: Optional[str] = None):
        order = self.current_order
        per_item_correct = [a == o for a, o in zip(action_counts, order)]
        reward = sum(1.0 if c else -1.0 for c in per_item_correct) / N_ITEMS
        info = {
            "nickname": nickname,
            "order": list(order),
            "served": list(action_counts),
            "order_text": describe_order(order),
            "served_text": describe_order(action_counts),
            "all_correct": all(per_item_correct),
            "per_item_correct": per_item_correct,
        }
        self.current_order = self._random_order()
        return self.current_order, reward, info
