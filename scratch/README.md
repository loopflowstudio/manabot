# scratch

- `mtg333-plan.md`: state, open findings, binding facts, gate acceptance and
  receipts. Prepared cutover, shared saves, the #513 port, CUDA equivalence,
  speed/profile and the optimize pass are done. Open for review: PPO is 5.2%
  slower on the final same-card check, and CUDA gradients exceed the repeat
  floor while meeting 1e-8. Main's #514 (PPO speedup) is not merged.
- `speed/mtg333-results.txt`: full A40 and 5090 speed tables.
- `questions.md`: reversible assumptions taken without asking.
- `mtg325-torch-backend-verification.md`: inherited baseline evidence; not
  MTG-333 results.
- `equivalence/`: controlled-input CPU/CUDA oracle and launcher. Local
  receipts are in `/tmp/mtg333`; full CUDA captures remain in Etude.
- `speed/`: benchmark and launcher. `run*/compare.txt` are MTG-325 evidence.
