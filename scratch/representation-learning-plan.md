# V7 learning comparison — completed implementation

2026-11-04 · Task 5da5f4 · results stop at demo/review; no model promotion.

All eight replicates completed on `92becff6091912a3ec83ae8fdd4e1c2347ff1d45`:
128 updates and 4,096 games each, 4,056,651 learner decisions total. All 1,024
held-out games completed and replayed. GRU minus pooled score was −0.98 points
(95% paired-seed interval [−10.13, +8.18]); training-time ratio 1.06. The declared
advantage criterion failed. This is inconclusive, not equivalence or a winner.
See `experiments/state-representation-v7.md` and the published manifest-backed
receipts in `experiments/data/state-representation-v7-5da5f4/`.

Both arms use the delivered v7 contract; effect and continuation reduction change
jointly. No temporal memory, ABI/default, native rules or frozen pilot evidence
changed. The scientific source archive is retained. The maintained runner is
`experiments/state_representation_v7.py`, matching the v7 tracking name; its
Etude page shows training diagnostics separately from the scientific comparison.

## Delete — do not maintain

- Removed the stale v5 scientific declaration and unexecuted three-arm schedule.
  `experiments/state_representation.py` became `state_representation_v7.py`;
  historical source/receipts retain the original filename.
- Removed `manabot.verify.representation.Measurement`, `prove`, `main` and its
  three-arm timing command. It predates v7 and has no remaining consumers.
  Keep `source_effect_root` and `shared_effect_root`: current transport/model
  tests still need these native fixtures. Old proofs remain at `9dc32125f`.
- Removed the completed `runs(canary=True)` allocation path. Keep `recipe(smoke=True)`
  for bounded learner tests and `runs(revision)` for cohort/report discovery.
  Calibration receipts stay excluded and bound to their original sources.
- Consolidated completed continuation and PR419 verification notes here and in
  `questions.md`; full notes remain at `9dc32125f:scratch/`.

No remaining deletion targets. Frozen recipes, protocol hashes, run order,
allocation limits and receipts must survive simplification unchanged.

## Cost and remaining boundary

Jack Heart authorized $25 total, separate from MTG-213. L40/L40S were unavailable;
4090 launch `8dd337797e0a` refused capacity without acquisition. The whole-cohort
CPU fallback was frozen before training: two balanced four-process batches,
one thread and 16 GiB per run. Evaluation followed training, four endpoints at
once with two active games each. No retries or score-based extensions.

Prior L4 bill: $0.07258259388618171 (supersedes $0.07083924 estimate). No new rental
charges; keep the $6 ancillary reserve pending invoices. Provider-name checks
confirmed both relevant pods absent. Etude retains refused-session metadata
without a provider receipt; its terminate operation skips refused sessions.
No Task training/evaluation process remains; MTG-213 allocations are untouched.

Check: `uv run --extra dev --extra artifacts --extra evaluation pytest tests/training/test_state_representation_study.py tests/eval/test_experiment_files.py tests/model/test_state_records.py tests/verify/test_shared_effect_transport.py -q` → 38 passed; Ruff and diff checks passed. Exact comparison preserves scientific/smoke recipes, scoring/protocol, all eight job declarations and retained recipe digests. Gate owns broader acceptance; demo/review owns result judgment.

Earlier artifact tracking check passed; Etude report admitted all 8 runs and
784 metrics. `uv run --extra dev ty check` could not start because ty is absent.
No further training is needed for this step.
