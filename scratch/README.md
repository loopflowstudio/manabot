# scratch

- `mtg333-plan.md`: Prepared cutover and shared saves implemented; #513 merged
  as `6e20654c34`, capture recorder and experiment consumers ported. Current
  native rebuilt. CPU oracle: 3,410 tensors and five update cases exact;
  fresh Torch/MLX parity passes. All Task rentals are stopped.
- `speed/mtg333-results.txt`: full A40 comparison/profile and 5090 optimization
  receipts. Metadata traversal optimization improves damped throughput 5.6%.
  Final original-baseline check still has a 5.2% PPO regression; performance
  acceptance and the stricter CUDA repeat floor remain open for review.
- `questions.md`: reversible assumptions and exact provider refusal evidence.
  Upstream card notes remain in git at `9fe9eb1e7`.
- `mtg325-torch-backend-verification.md`: inherited baseline evidence summary;
  these are not MTG-333 results.
- `equivalence/`: controlled-input CPU/CUDA oracle and launcher. Current local
  receipts are in `/tmp/mtg333`; full CUDA captures remain in Etude.
- `speed/`: benchmark and launcher. `run*/compare.txt` are MTG-325 evidence.
  Generated patches are excluded from the final tree; launch commits
  `cd7afaec86`, `e6cd7f59cc`, `46e3db32d7` retain their exact bytes.

Gate still needs the broader suites, installed wheel and Python 3.12 checks,
and the full-file ordering hang investigation. Full slice 1–6 receipts:
`1f634d89b:scratch/mtg333-plan.md`.
