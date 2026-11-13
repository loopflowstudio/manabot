# scratch

- `mtg333-plan.md`: Prepared cutover and shared saves implemented; #513 merged
  as `6e20654c34`, capture recorder and experiment consumers ported. Current
  native rebuilt; same-input CPU oracle: 3,410 tensors exact. A40 retry
  `2f98203dd0a7` is measuring speed/profile; CUDA outputs exact, gradients
  ≤3.26e-9 but above repeat. Optimization follows its profile.
- `questions.md`: reversible assumptions. MTG-324's `plan.md`, `qa.html` and
  questions section arrived with main's #510 merge and were removed here; they
  remain at `f4e1822fa`.
- `mtg325-torch-backend-verification.md`: inherited baseline evidence and limits.
- `equivalence/`: shared CPU/CUDA oracle and launcher. Current captures are in
  `/tmp/mtg333/mana-*.pt`; CUDA run `047fa57e0e5e` retains its full captures.
- `speed/`: frozen benchmark and combined equivalence/speed/profile launcher.
  `run*/compare.txt` are MTG-325 evidence, not results of this Task.
  Integrated launch patches survive in `296c90f33` and `c150d5f25`, not the
  final diff. Latest launch: `cd7afaec86`, with a shared retained belief corpus; regenerate
  patches for changed code.

Full slice 1–6 receipts: `1f634d89b:scratch/mtg333-plan.md`.
Earlier uncommitted notes: `/tmp/mtg333-compress-before/`.
