"""
Coda condivisa tra tutte le sessioni Streamlit dello stesso processo server
(ogni tab/browser e' una sessione separata con il suo st.session_state, ma
tutte girano nello stesso processo Python -> un oggetto a livello di modulo
e' visto da tutti). Un lock basta, non serve un database per una serata.
"""
import random
import threading
from dataclasses import dataclass, field
from typing import List, Optional

_lock = threading.Lock()


@dataclass
class CustomerOrder:
    nickname: str
    items: List[str]  # sottoinsieme di bar_env.ORDERS


_queue: List[CustomerOrder] = []


def push_order(nickname: str, items: List[str]) -> None:
    with _lock:
        _queue.append(CustomerOrder(nickname=nickname.strip() or "Anonimo", items=list(items)))


def pop_order() -> Optional[CustomerOrder]:
    """Pesca un cliente a caso dalla coda (non FIFO): rende visibile che gli
    ordini reali inseriti vengono presi in ordine casuale, non in sequenza
    prevedibile, il che aiuta a capire che sono davvero letti dalla coda."""
    with _lock:
        if not _queue:
            return None
        idx = random.randrange(len(_queue))
        return _queue.pop(idx)


def queue_length() -> int:
    with _lock:
        return len(_queue)


def peek_queue() -> List[CustomerOrder]:
    with _lock:
        return list(_queue)
