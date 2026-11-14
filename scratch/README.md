# scratch

- `mtg333-plan.md`: current design, decisions, findings and receipts.
  Prepared cutover, shared saves, #513/#514 integration, bounded PPO prefetch,
  role-based preparation and callback shutdown fixes are implemented. Review 4
  now returns configured backend objects from `resolve`, names the update-only
  surface `UpdateBackend`, and makes `backend.load_policy` the shared loader.
  Direct compress removes duplicate loaders/selectors, unused layout state
  and redundant array conversions. Fresh CPU oracle: 3,410 tensors and five
  updates exact; 149 focused tests pass; captured Torch/MLX parity passes.
  CUDA repeat-floor finding is accepted in review 4; host preparation/load
  costs and threaded-profile limits remain visible. No rental is live.
  Broader gate, Python 3.12/3.13 installed-wheel checks and CI remain.
- `speed/mtg333-results.txt`: full A40 and 5090 speed tables.
- `questions.md`: reversible assumptions taken without asking.
- `mtg325-torch-backend-verification.md`: inherited baseline evidence; not
  MTG-333 results.
- `equivalence/`: controlled-input CPU/CUDA oracle and launcher. Local
  receipts are in `/tmp/mtg333`; full CUDA captures remain in Etude.
- `speed/`: benchmark and launcher. `run*/compare.txt` are MTG-325 evidence.

## After landing

Start MTG-343 once #519 merges (agreed 2026-11-14: "oh sure yeah that works"); its acceptance is torch-free imports of the update rules and rollout code (comment on MTG-343).
