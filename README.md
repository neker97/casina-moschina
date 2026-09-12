# 🪰☕ Casina Moschina

Progetto DLAI (BYOP) — categoria "artificial life" / "gaming and AI".

Una rete neurale la cui connettivita' interna e' presa dal circuito reale
del **mushroom body** di *Drosophila melanogaster* (dataset hemibrain,
Janelia FlyEM, v1.2, pubblico) impara a fare il barista: riceve l'ordine di
un cliente (quantita' di caffe, cappuccino, cornetto semplice, cornetto
alla nutella) e deve servire esattamente quelle quantita'.

Gira come un piccolo gioco 3D nel browser (Three.js + personaggi/cibo CC0
di Kenney.nl): i clienti entrano, ordinano, la mosca vola sullo scaffale,
prende i prodotti e li appoggia sul bancone con un badge della quantita'.
Si puo' anche entrare come cliente vero da un altro dispositivo sulla
stessa rete e mettersi in coda con un ordine a scelta.

## Domanda dell'esperimento

La connettivita' biologica reale (evolutasi per l'apprendimento associativo
olfattivo, non per fare i cappuccini) da' un vantaggio di training rispetto
a una rete della stessa dimensione e sparsita' ma con connessioni
posizionate a caso? Vedi `docs`/report per i risultati completi — in
sintesi: con REINFORCE (segnale di reward debole) la struttura conta e la
rete biologica ha bisogno di un meccanismo di normalizzazione (che
approssima il neurone APL reale) per non saturare; con apprendimento
supervisionato (segnale forte) la differenza si appiattisce, tutte le
condizioni risolvono il task.

## Come sapere che non sta "copiando" l'ordine

Il sospetto legittimo: se la risposta corretta e' sempre uguale all'ordine,
la rete potrebbe limitarsi a "passare" l'input in uscita senza calcolare
nulla. Due controprove verificabili:

1. **`untrained`** nel selettore di rete: stessa architettura, pesi mai
   allenati. Risponde sempre la stessa cosa fissa, sbagliando sempre —
   se fosse una scorciatoia funzionerebbe comunque, senza bisogno di pesi
   giusti.
2. I neuroni di ingresso (Kenyon cells, dove entra l'ordine) e quelli di
   uscita (MBON, da dove si legge la risposta) sono **fisicamente
   disgiunti** — zero sovrapposizione. Azzerando solo i pesi della rete
   ricorrente in mezzo (lasciando ingresso/uscita allenati intatti),
   l'accuratezza crolla da 100% a 0%: la risposta corretta passa
   obbligatoriamente per il calcolo ricorrente, non esiste una via diretta
   che lo aggira.

## Struttura

- `extract_mb_circuit.py` — estrae il sottografo del mushroom body dal
  connettoma hemibrain grezzo -> `data/mb_neurons.csv`, `data/mb_edges.csv`
- `model.py` — `ConnectomePolicy`: input sui neuroni Kenyon (KC), K passi di
  dinamica ricorrente mascherata dalla connettivita' (biologica o random),
  output dai mushroom body output neurons (MBON)
- `bar_env.py` — ambiente del bar (ordini a quantita', bandit contestuale)
- `train.py` — allena 4 condizioni (biologica/random × con/senza
  normalizzazione) con cross-entropy supervisionata, salva checkpoint e log
- `queue_store.py` — coda condivisa tra sessioni per i clienti veri
- `server.py` — backend FastAPI: espone il modello allenato come API e
  serve la pagina di gioco statica
- `web/` — frontend Three.js (gioco 3D) + asset CC0 estratti (Kenney.nl)
- `app.py` — versione precedente con UI Streamlit (non piu' mantenuta,
  lasciata per riferimento)

## Uso

```bash
python -m venv moschina-env
moschina-env\Scripts\pip install -r requirements.txt

# dati gia' inclusi nel repo (data/mb_neurons.csv, data/mb_edges.csv).
# Per rigenerarli da zero dal dataset grezzo:
#   curl -L -o data/hemibrain_v1.2.tar.gz https://storage.googleapis.com/hemibrain/v1.2/exported-traced-adjacencies-v1.2.tar.gz
#   tar -xzf data/hemibrain_v1.2.tar.gz -C data/
#   python extract_mb_circuit.py

moschina-env\Scripts\python train.py     # riallena da zero (checkpoint gia' inclusi nel repo)
moschina-env\Scripts\python -m uvicorn server:app --host 0.0.0.0 --port 8000
# apri http://localhost:8000
```

## Fonte dati

Hemibrain v1.2 (Scheffer et al. 2020) — tabella di adiacenza esportata via
neuprint-python, distribuita pubblicamente da Janelia Research Campus:
https://storage.googleapis.com/hemibrain/v1.2/exported-traced-adjacencies-v1.2.tar.gz

Circuito del mushroom body: Li et al. 2020, "The connectome of the adult
Drosophila mushroom body provides insights into function", eLife 62576.

Asset 3D: [Kenney.nl](https://kenney.nl) — Mini Characters, Food Kit
(licenza CC0).

## Nota per il report

Rientra nelle aree consentite dalle linee guida DLAI 2025/2026: **artificial
life** (architettura vincolata da un connettoma biologico reale) e **gaming
and AI** (ambiente/gioco simulato). Non e' un progetto di graph learning
(il grafo e' usato come maschera fissa di un'architettura allenata con
backprop standard, non come input a un algoritmo di apprendimento su grafi).
