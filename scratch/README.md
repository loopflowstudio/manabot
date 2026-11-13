# scratch

- `mtg333-plan.md`: state, binding facts, open review findings and receipts.
  Prepared cutover, shared saves and #513/#514 integration are implemented.
  Fresh CPU oracle: 3,410 tensors and five updates exact. Bounded PPO host
  prefetch improves median throughput 10.5% over parent and 1.7% over main
  `de39d601fb` on one RTX 5090, three processes per arm; main/head ranges
  overlap. Rental terminated. Strict CUDA repeat floor, host preparation/load
  costs, threaded profile limits and the order-dependent timeout remain for
  review; broader checks are at gate. Concept cleanup (2026-11-13) now passes host rows from the collector
  and uses `inference` as the only statement of preparation purpose. The fresh
  3,410-tensor CPU oracle and five-case update comparison remain exact; focused
  tests passed (63; two CUDA cases deselected). Concept review 2 directs one
  more cleanup, not yet implemented: the torch rollout and callers inside
  `no_grad` state preparation purpose literally instead of from grad mode.
- `speed/mtg333-results.txt`: full A40 and 5090 speed tables.
- `questions.md`: reversible assumptions taken without asking.
- `mtg325-torch-backend-verification.md`: inherited baseline evidence; not
  MTG-333 results.
- `equivalence/`: controlled-input CPU/CUDA oracle and launcher. Local
  receipts are in `/tmp/mtg333`; full CUDA captures remain in Etude.
- `speed/`: benchmark and launcher. `run*/compare.txt` are MTG-325 evidence.
