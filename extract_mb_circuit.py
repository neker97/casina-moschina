"""
Estrae il circuito del mushroom body (centro di apprendimento associativo
della Drosophila) dal connettoma hemibrain ufficiale (Janelia FlyEM, v1.2).

Fonte dati: https://storage.googleapis.com/hemibrain/v1.2/exported-traced-adjacencies-v1.2.tar.gz
(dataset pubblico, nessuna autenticazione richiesta)

Output:
  data/mb_neurons.csv  -> bodyId, type, is_dopaminergic
  data/mb_edges.csv    -> pre, post, weight   (sottografo indotto, pesi = conteggio sinapsi)
"""
import csv
from pathlib import Path

RAW_DIR = Path("data/exported-traced-adjacencies-v1.2")
OUT_DIR = Path("data")

# Regioni anatomiche del mushroom body (emisfero destro, l'unico ricostruito
# per intero nell'hemibrain): calice, peduncolo, lobi alpha/alpha'/beta/beta'/gamma.
# Riferimento: Aso et al. 2014, Li et al. 2020 (eLife 62576).
MB_ROIS = {"CA(R)", "PED(R)", "aL(R)", "a'L(R)", "bL(R)", "b'L(R)", "gL(R)"}

# Tipi cellulari canonici del circuito (Kenyon cells, mushroom body output
# neurons, e i due cluster dopaminergici PAM/PPL1 che portano il segnale di
# rinforzo/punizione nella mosca reale). Filtra via i neuroni che toccano il
# MB solo di passaggio.
CANONICAL_PREFIXES = ("KC", "MBON", "PAM", "PPL1", "PPL2", "APL", "DPM")


def is_canonical(cell_type: str) -> bool:
    return cell_type.startswith(CANONICAL_PREFIXES)


def is_dopaminergic(cell_type: str) -> bool:
    return cell_type.startswith(("PAM", "PPL1", "PPL2"))


def main():
    # 1. neuroni con almeno una sinapsi in una ROI del mushroom body
    mb_touch = set()
    with open(RAW_DIR / "traced-roi-connections.csv", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for pre, post, roi, w in r:
            if roi in MB_ROIS:
                mb_touch.add(pre)
                mb_touch.add(post)

    # 2. tieni solo i tipi cellulari canonici del circuito (non passanti)
    types = {}
    with open(RAW_DIR / "traced-neurons.csv", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            body_id, cell_type = row[0], row[1]
            if body_id in mb_touch and cell_type and is_canonical(cell_type):
                types[body_id] = cell_type

    node_ids = set(types)
    print(f"Neuroni nel circuito MB (canonici): {len(node_ids)}")

    # 3. sottografo indotto: connessioni totali tra questi neuroni
    edges = []
    with open(RAW_DIR / "traced-total-connections.csv", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for pre, post, w in r:
            if pre in node_ids and post in node_ids:
                edges.append((pre, post, int(w)))

    print(f"Sinapsi (archi) nel sottografo: {len(edges)}")

    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "mb_neurons.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bodyId", "type", "is_dopaminergic"])
        for body_id, cell_type in sorted(types.items()):
            w.writerow([body_id, cell_type, int(is_dopaminergic(cell_type))])

    with open(OUT_DIR / "mb_edges.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pre", "post", "weight"])
        w.writerows(edges)

    n_dop = sum(1 for t in types.values() if is_dopaminergic(t))
    print(f"Neuroni dopaminergici (PAM/PPL1/PPL2) nel sottografo: {n_dop}")
    print("Scritto data/mb_neurons.csv e data/mb_edges.csv")


if __name__ == "__main__":
    main()
