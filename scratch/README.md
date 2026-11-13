# scratch

- `mtg333-plan.md`: slices 1–6b implemented; `105e7e2aa` merged. Fresh CPU
  oracle: 3,410 tensors and five update cases exact. A6000 CUDA outputs exact,
  gradients ≤3.73e-9 but above the repeat floor. Speed/optimization remain
  incomplete: the queued rental is stuck until its 19:15:52 UTC deadline; see
  the plan's "Next actions".
- `questions.md`: reversible assumptions. MTG-324's `plan.md`, `qa.html` and
  questions section arrived with main's #510 merge and were removed here; they
  remain at `f4e1822fa`.
- `mtg325-torch-backend-verification.md`: inherited baseline evidence and limits.
- `equivalence/`: shared CPU/CUDA oracle and launcher. Current captures are in
  `/tmp/mtg333/mana-*.pt`; CUDA run `047fa57e0e5e` retains its full captures.
- `speed/`: frozen benchmark and combined equivalence/speed/profile launcher.
  `run*/compare.txt` are MTG-325 evidence, not results of this Task.
  Integrated launch patches survive in `296c90f33` and `c150d5f25`, not the
  final diff. Launch the latter commit or regenerate patches for changed code.

Full slice 1–6 receipts: `1f634d89b:scratch/mtg333-plan.md`.
Earlier uncommitted notes: `/tmp/mtg333-compress-before/`.
