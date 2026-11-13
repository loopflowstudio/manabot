# scratch

- `mtg333-plan.md`: state, binding facts, open review findings and receipts.
  Prepared cutover, shared saves and #513/#514 integration are implemented.
  Fresh CPU oracle: 3,410 tensors and five updates exact. Two measured host
  optimizations retained; metadata attempt discarded. PPO remains 7.2% slower
  in the final same-card check; speed acceptance, strict CUDA repeat floor and
  the order-dependent test timeout remain for review. Broader checks are at gate.
- `speed/mtg333-results.txt`: full A40 and 5090 speed tables.
- `questions.md`: reversible assumptions taken without asking.
- `mtg325-torch-backend-verification.md`: inherited baseline evidence; not
  MTG-333 results.
- `equivalence/`: controlled-input CPU/CUDA oracle and launcher. Local
  receipts are in `/tmp/mtg333`; full CUDA captures remain in Etude.
- `speed/`: benchmark and launcher. `run*/compare.txt` are MTG-325 evidence.
