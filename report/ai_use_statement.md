# AI-use statement

*Not included in the 2-page limit, as per course guidelines.*

This project was developed in close collaboration with Claude (Anthropic),
via Claude Code, across an extended interactive session. AI assistance was
used throughout the project's lifecycle, not only for boilerplate code.
Specifically:

- **Data sourcing.** Claude searched for and identified the relevant
  public connectome datasets (hemibrain v1.2, FlyWire, MaleCNS), located
  the direct (unauthenticated) download URL for the hemibrain "traced
  adjacencies" export, and wrote the extraction script
  (`extract_mb_circuit.py`) that filters the raw connectome to the
  mushroom body circuit used in this project.
- **Architecture and training code.** Claude designed and implemented the
  `ConnectomePolicy` architecture (masked recurrent network with KC input
  / MBON output), the bar environment, and the training loop, in
  discussion with the student about the research question (biological vs.
  random connectivity).
- **Debugging that shaped the final method.** Two debugging episodes
  materially changed the project's design and are reported as findings,
  not hidden:
  1. An initial single-item REINFORCE experiment showed the biological
     mask underperforming a random one; Claude diagnosed the cause via a
     reachability analysis (ruling out a connectivity bottleneck) and
     identified saturation in the recurrent dynamics as the mechanism,
     leading to the normalization ablation reported in Exp. 1.
  2. When the task was extended to per-item quantities, plain REINFORCE
     (even batched, with an entropy bonus) failed to learn; Claude
     diagnosed this by probing the network's output on fixed test inputs
     and finding it was input-invariant, then proposed and implemented
     the switch to supervised cross-entropy (justified by the task's
     fully-observed, deterministic-label structure), which fixed
     training (Exp. 2). This diagnostic process is described in full in
     §5 of the report and is not disguised as a clean, one-shot result.
  3. A separate debugging pass on the 3D demo (not part of the ML
     results) fixed a Three.js skeleton-cloning bug that made spawned
     characters invisible, and a missing-texture bug from an
     under-copied shared asset file.
- **Verification experiments.** The untrained-network control and the
  recurrent-weight-zeroing ablation (§6 of the report) were proposed by
  Claude in response to the student explicitly asking how to verify the
  network was not merely copying its input, and were run and reported
  with their actual numeric outcomes.
- **3D demo.** The browser-based 3D visualization (Three.js scene, FastAPI
  backend serving the trained model, CC0 asset integration from Kenney.nl)
  was implemented by Claude at the student's request, as a way to make
  the trained model's behavior directly observable rather than only
  reported as numbers.
- **Report drafting.** This report's content (structure, prose, and the
  training-curve figure) was drafted by Claude from the project's actual
  logs and console outputs, then reviewed by the student.

All reported numbers (accuracies, parameter counts, ablation results) are
taken directly from real runs of the code in this repository, not
fabricated or estimated; they can be reproduced by cloning
https://github.com/neker97/casina-moschina and following its README.
