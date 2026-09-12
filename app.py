"""
Casina Moschina - UI Streamlit
Guarda la rete vincolata dal connettoma del mushroom body servire clienti
al bar (ordini a quantita', es. 2 cappuccini + 3 cornetti), confronta il
suo apprendimento con la baseline a connettivita' casuale, e fai entrare
clienti veri da un altro browser/dispositivo con nickname e ordine a scelta.

Avvio: streamlit run app.py --server.address 0.0.0.0
"""
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

import queue_store
from bar_env import BarEnv, N_ITEMS, MAX_COUNT, ORDERS, ORDER_LABELS_IT, describe_order
from model import ConnectomePolicy, load_circuit, random_mask_like

st.set_page_config(page_title="Casina Moschina", page_icon="🪰", layout="centered")

ITEM_EMOJI = {
    "caffe": "☕",
    "cappuccino": "🥛",
    "cornetto_semplice": "🥐",
    "cornetto_nutella": "🍫",
}
ALL_TAGS = ["biologica", "random", "biologica_norm", "random_norm"]


@st.cache_resource
def load_everything():
    circuit = load_circuit("data")
    policies = {}
    rand_mask = random_mask_like(circuit["mask"], seed=0)
    conditions = [
        ("biologica", None, False),
        ("random", rand_mask, False),
        ("biologica_norm", None, True),
        ("random_norm", rand_mask, True),
    ]
    for tag, mask, norm in conditions:
        p = ConnectomePolicy(circuit, n_items=N_ITEMS, max_count=MAX_COUNT,
                              mask_override=mask, normalize=norm)
        ckpt = Path("checkpoints") / f"policy_{tag}.pt"
        if ckpt.exists():
            p.load_state_dict(torch.load(ckpt, map_location="cpu"))
        p.eval()
        policies[tag] = p
    log_path = Path("checkpoints") / "training_log.csv"
    log_df = pd.read_csv(log_path) if log_path.exists() else None
    return circuit, policies, log_df


circuit, policies, log_df = load_everything()

st.title("🪰☕ Casina Moschina")
st.caption(
    "Un barista con il cervello vero (parziale) di una Drosophila: la rete "
    "che serve i clienti ha la connettivita' presa dal circuito reale del "
    "mushroom body (dataset hemibrain, Janelia FlyEM)."
)

tab_sala, tab_prenota = st.tabs(["🍽️ Sala (guarda la mosca servire)", "📝 Entra come cliente"])

if "env" not in st.session_state:
    st.session_state.env = BarEnv()
    st.session_state.score = {t: 0.0 for t in ALL_TAGS}
    st.session_state.history = []

env = st.session_state.env


def serve_next(tag: str):
    """Se c'e' un cliente vero in coda lo usa, altrimenti un ordine casuale."""
    custom = queue_store.pop_order()
    nickname = None
    if custom is not None:
        env.set_next_order(custom.items)
        nickname = custom.nickname

    order = env.current_order
    policy = policies[tag]
    with torch.no_grad():
        order_t = torch.tensor([order], dtype=torch.float32) / MAX_COUNT
        logits = policy(order_t)
        action = torch.distributions.Categorical(logits=logits).sample()[0].tolist()

    _, reward, info = env.step(action, nickname=nickname)
    st.session_state.score[tag] += reward
    st.session_state.history.append({**info, "tag": tag, "reward": round(reward, 2)})
    return info


with tab_sala:
    tag = st.radio("Chi serve il banco oggi?", ALL_TAGS, horizontal=True,
                    help="'biologica'/'random' = connettoma reale vs sparsita' mescolata a caso. "
                         "'_norm' = con normalizzazione (approssima il controllo di guadagno "
                         "che nella mosca reale fa il neurone APL).")

    qlen = queue_store.queue_length()
    if qlen:
        st.info(f"👥 {qlen} cliente/i vero/i in coda — verranno serviti per primi.")

    st.subheader(f"Prossimo ordine: {describe_order(env.current_order)}")

    col1, col2 = st.columns(2)
    with col1:
        serve_click = st.button("🐝 Servi il prossimo cliente", use_container_width=True)
    with col2:
        auto_n = st.number_input("oppure servi N clienti di fila", min_value=1, max_value=200, value=20)
        auto_click = st.button("⏩ Servi N clienti", use_container_width=True)

    last_info = None
    if serve_click:
        last_info = serve_next(tag)
    if auto_click:
        for _ in range(int(auto_n)):
            last_info = serve_next(tag)

    if last_info is not None:
        chi = f"**{last_info['nickname']}**" if last_info["nickname"] else "cliente casuale"
        if last_info["all_correct"]:
            st.success(f"{chi} voleva: {last_info['order_text']} → la mosca serve: "
                       f"{last_info['served_text']} → tutto giusto! (+dopamina)")
        else:
            st.error(f"{chi} voleva: {last_info['order_text']} → la mosca serve: "
                     f"{last_info['served_text']} → non ci siamo del tutto")

    cols = st.columns(4)
    for col, t in zip(cols, ALL_TAGS):
        col.metric(t, round(st.session_state.score[t], 2))

    if st.session_state.history:
        st.subheader("Ultimi ordini serviti")
        hist_df = pd.DataFrame(st.session_state.history[-15:][::-1])
        hist_df["order"] = hist_df["order_text"]
        hist_df["servito"] = hist_df["served_text"]
        hist_df["cliente"] = hist_df["nickname"].fillna("casuale")
        st.dataframe(hist_df[["cliente", "order", "servito", "all_correct", "reward", "tag"]],
                     use_container_width=True, hide_index=True)

with tab_prenota:
    st.write("Mettiti in coda: la mosca ti servirà appena qualcuno preme 'servi' nella "
             "tab Sala (anche da un altro dispositivo sulla stessa rete).")
    nickname = st.text_input("Il tuo nickname", value="", max_chars=24)
    counts = []
    cols = st.columns(N_ITEMS)
    for col, name in zip(cols, ORDERS):
        with col:
            st.write(f"{ITEM_EMOJI[name]} {ORDER_LABELS_IT[name]}")
            c = st.number_input(f"qty_{name}", min_value=0, max_value=MAX_COUNT, value=0,
                                 label_visibility="collapsed", key=f"qty_{name}")
            counts.append(c)

    if st.button("☕ Ordina!", type="primary"):
        if sum(counts) == 0:
            st.warning("Ordina almeno qualcosa!")
        elif not nickname.strip():
            st.warning("Metti un nickname.")
        else:
            queue_store.push_order(nickname, counts)
            st.success(f"Ordine di {nickname} in coda: {describe_order(counts)}. "
                       "Vai nella tab 'Sala' (o fai guardare a qualcun altro) per vederlo servito!")

    pending = queue_store.peek_queue()
    if pending:
        st.write("**Coda attuale:**")
        for o in pending:
            st.write(f"- {o.nickname}: {describe_order(o.items)}")

st.divider()
st.subheader("Curve di apprendimento (training offline, REINFORCE)")
if log_df is not None:
    pivot = log_df.pivot(index="episode", columns="tag", values="running_reward")
    st.line_chart(pivot)
    st.caption("Reward medio (media mobile) durante l'allenamento: connettoma reale vs "
               "stessa sparsita' ma connessioni mescolate a caso, con/senza normalizzazione.")
else:
    st.info("Nessun log di training trovato: esegui `train.py` prima di avviare l'app.")

with st.expander("Dettagli sul circuito neurale usato"):
    st.write(f"- Neuroni totali: **{circuit['n']}**")
    st.write(f"- Sinapsi (connessioni non nulle): **{int(circuit['mask'].sum())}**")
    st.write(f"- Neuroni Kenyon (ingresso, ricevono l'ordine): **{int(circuit['is_kc'].sum())}**")
    st.write(f"- Mushroom body output neurons (uscita, decidono l'azione): **{int(circuit['is_mbon'].sum())}**")
    st.write(f"- Neuroni dopaminergici PAM/PPL1/PPL2 (segnale di rinforzo nella mosca reale): "
             f"**{int(circuit['is_dop'].sum())}**")
    st.caption("Fonte: hemibrain v1.2 (Janelia FlyEM / Scheffer et al. 2020), "
               "circuito del mushroom body (Li et al. 2020, eLife 62576).")
