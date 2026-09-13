# Casina Moschina: A Connectome-Constrained Neural Barista

*DLAI 2025/2026 — BYOP project. Category: artificial life / gaming and AI.*

> Nota: contenuto pronto da travasare nel template ufficiale del corso
> appena disponibile. Lunghezza attuale calibrata per ~2 pagine (1
> studente); tagliare/espandere secondo il numero di autori (+1 pagina a
> membro, per le regole del corso).

---

## Abstract

We ask whether the wiring diagram of a real biological neural circuit,
extracted from a public connectome dataset, provides any advantage over a
random network of matched size and sparsity when both are trained on a
task the circuit never evolved for. We build a recurrent network whose
connectivity mask is taken directly from the mushroom body circuit of
*Drosophila melanogaster* (hemibrain v1.2, Janelia FlyEM), and train it to
act as a "barista": given a customer's order (quantities of four items),
it must reproduce the exact quantities. We compare this biologically-
constrained network against a random-mask control of identical sparsity,
with and without a per-step normalization approximating the gain-control
role of the real APL neuron, and against an identical but untrained
network used as a sanity baseline. Under a weak learning signal
(REINFORCE), the biological mask underperforms the random one unless
normalization is added, at which point it solves the task near-perfectly
while the random mask degrades — suggesting the biological topology
specifically requires the gain-control mechanism it evolved with. Under a
strong learning signal (supervised cross-entropy), the difference vanishes
entirely: all four conditions reach 100% accuracy. We additionally verify,
via an ablation that zeroes the recurrent weights while keeping the
trained input/output layers intact, that the network's correct answers
are not a shortcut around the connectome-derived computation.

## 1. Motivation and research question

Connectome datasets (FlyWire, hemibrain) provide, for the first time, a
complete real wiring diagram of a biological brain circuit. A natural
question for representation learning / artificial life is: **does using a
real biological connectivity pattern as an architectural prior help a
neural network learn, compared to a random connectivity pattern of the
same size?** This is not obviously true — the mushroom body evolved for
associative olfactory learning, not for an arbitrary discrete decision
task, so any advantage would have to come from generic structural
properties (e.g. degree distribution, path lengths) rather than
task-specific tuning.

## 2. Data

We use **hemibrain v1.2** (Scheffer et al., 2020; distributed by Janelia
FlyEM), specifically the public "traced adjacencies" export
(`traced-neurons.csv`, `traced-roi-connections.csv`,
`traced-total-connections.csv`; 21,740 traced neurons, ~3.5M weighted
connections). We restrict to the **mushroom body** circuit — the
associative-learning center of the fly brain — by (a) keeping only
neurons with at least one synapse in a mushroom-body ROI (calyx,
peduncle, α/α′/β/β′/γ lobes) and (b) filtering to canonical mushroom-body
cell types (`KC*`, `MBON*`, `PAM*`, `PPL1*`, `PPL2*`, `APL`, `DPM`), which
removes passing-through neurons not part of the circuit proper. This
yields **2,308 neurons and 539,201 directed weighted synapses** (weight =
raw synapse count), including 1,927 Kenyon cells (KC, the circuit's only
sensory input), 58 mushroom body output neurons (MBON, the circuit's only
output), and 320 dopaminergic neurons (PAM/PPL1/PPL2 — the circuit's
real reinforcement signal, identified directly from cell-type
nomenclature, not from separate neurotransmitter-prediction data).

## 3. Model

`ConnectomePolicy` is a recurrent network over $N{=}2308$ units. A
trainable projection maps the input onto the **KC** subset only (the
circuit's real sensory entry point); the state then evolves for $K{=}4$
steps as
$$x_{t+1} = \tanh\big((x_t \cdot (W \odot M)) + b\big)$$
where $M \in \{0,1\}^{N\times N}$ is the fixed connectivity mask (real
mushroom-body adjacency, or a random permutation of it with identical
sparsity, as a control) and $W$ is trained end-to-end; only entries where
$M{=}1$ ever receive gradient signal. A trainable readout reads **only**
the **MBON** subset and maps it to the task output. Critically, KC and
MBON are disjoint (0 neurons in common): any signal reaching the output
must have propagated through at least one step of the masked recurrent
computation — there is no direct input→output path to shortcut. An
optional `LayerNorm` before each `tanh` approximates the global gain
control the real circuit obtains from the APL neuron (not modeled
explicitly).

## 4. Task and environment

A minimal bar environment: each "customer" orders a random quantity
(0–3) of four items (coffee, cappuccino, plain croissant, chocolate
croissant); the network must output the same four quantities. Reward is
+1/−1 per item, averaged. This is a **fully-observed, deterministic-label
bandit** — the correct action is always exactly the input — which lets us
compare two very different training signals on the same task.

## 5. Experiments

**Exp. 1 — weak signal (REINFORCE), single-item version.** On an earlier,
simpler single-item version of the task, we trained with per-step policy
gradient (single sample, no batching, no entropy bonus):

| condition | final reward |
|---|---|
| biological mask | 0.28 |
| random mask | **1.00** |
| biological mask + norm | **0.98** |
| random mask + norm | −0.28 |

Without normalization, the biological mask underperforms random; adding
normalization flips the picture entirely (biological nearly solves the
task, random degrades). We diagnosed the failure mode directly:
reachability analysis confirmed the KC→MBON signal path exists (57/58
MBON neurons reachable within 1 hop, 58/58 within 2), so the bottleneck is
not connectivity but **saturation** — the biological circuit's convergent
fan-in drives the recurrent `tanh` into a regime where output becomes
nearly input-independent, exactly the pathology the real APL neuron's
lateral inhibition prevents in vivo. The random-mask control has a
different in/out-degree distribution and does not suffer the same
collapse, explaining why it does not need (and is hurt by) the added
normalization.

**Exp. 2 — strong signal (supervised cross-entropy), full quantity task.**
On the full task (independent quantity per item, 4 items × 4 possible
counts), plain REINFORCE — even batched (32) with an entropy bonus —
failed to escape chance level ($\approx{-}0.5$) after tens of thousands of
samples: probing the untrained-forward behavior showed the network's
*argmax* output was **identical across wildly different inputs**,
indicating the recurrent dynamics were stuck at an input-invariant fixed
point that weak, high-variance policy-gradient updates could not escape.
Switching to direct supervised cross-entropy on the same architecture
(justified here because the correct label is always known — a REINFORCE
estimator with a perfect state-independent baseline reduces to exactly
this) resolved this immediately: **all four conditions (biological,
random, ± normalization) reach 100% accuracy** within a few thousand
examples (Fig. 1). Under a strong learning signal, the structural prior
no longer matters — the difference observed in Exp. 1 is specific to
weak/noisy training regimes.

*Figure 1 here: `fig_supervised_training.png` — reward vs. examples seen,
4 overlapping curves converging to 1.0.*

## 6. Verifying the network is not shortcutting the input

Because the correct action always equals the input, a legitimate concern
is whether the network is doing anything beyond copying. Two checks:

1. **Untrained control.** An identical network with random (never
   updated) weights was queried on the same inputs: it returns the exact
   same fixed output regardless of the order (0/30 correct, always
   identical), evidencing that correct behavior does not follow from the
   architecture alone and must be learned.
2. **Recurrent ablation.** On the trained biological network, zeroing
   only $W$ (the recurrent connectome weights) while keeping the trained
   input/output projections intact collapses accuracy from **30/30 to
   0/30**. Combined with the disjoint KC/MBON sets (§3), this shows the
   correct output is not reachable without the recurrent computation —
   there is no bypass.

## 7. Limitations

Synapse counts are used as unsigned weights (no excitatory/inhibitory
sign from neurotransmitter identity); the random-mask control uses a
single shuffle seed; only the mushroom body sub-circuit is modeled, not
the full 166k-neuron central nervous system; the task, while non-trivial
in its action space (256 joint outcomes), is deterministic and
fully-observed, which is why the interesting biological-vs-random
difference is visible only under a weak (RL) signal and not under
supervision.

## 8. Conclusion

A real biological connectome, used as a fixed architectural prior, is not
a free lunch: under weak training signal it actively underperforms a
random control unless paired with the specific gain-control mechanism
(normalization, standing in for the APL neuron) the circuit co-evolved
with — a concrete illustration that a biological structure and its
biological support machinery are a package deal. Under strong supervision
the effect disappears, and separately, ablation confirms that the
network's correct behavior genuinely depends on the connectome-masked
recurrent computation rather than shortcutting it.

## References

- Scheffer et al. (2020), *A connectome and analysis of the adult
  Drosophila central brain*, eLife 9:e57443 (hemibrain v1.2 dataset).
- Li et al. (2020), *The connectome of the adult Drosophila mushroom body
  provides insights into function*, eLife 9:e62576.
- Aso et al. (2014), *Mushroom body output neurons encode valence-specific
  memory*, eLife 3:e04580 (PAM/PPL1 dopaminergic cell types; APL role).

## Repository

Code, trained checkpoints (`.pt`), extracted connectome data, and the
Three.js demo: **https://github.com/neker97/casina-moschina**
