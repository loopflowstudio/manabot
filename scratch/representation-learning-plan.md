# V7 learning comparison — completed implementation

2026-11-04 · Task5da5f4 · results stop at demo/review; no model promotion.

All eight fixed replicates completed on92becff609:128updates and4,096games each,
4,056,651learner decisions total. All1,024held-out games completed and replayed.
GRU minus pooled score=-0.98points;95%paired-seed interval[-10.13,+8.18]. The
training-time ratio is1.06. The declared advantage criterion failed; this is
inconclusive, not proof of equivalence or a winner. The readable report is
experiments/state-representation-v7.md; full receipts live under
experiments/data/state-representation-v7-5da5f4/ and are published by manifest.

The fixed v7 comparison jointly changes effect and continuation token reduction.
No temporal memory, ABI/default, native rules or frozen pilot evidence changed.
The exact scientific source archive is retained. Post-run code only moves the
experiment to its v7 tracking name and adds the ordinary Etude diagnostic page.

## Delete — do not maintain

- Stale v5 scientific declaration and unexecuted three-arm schedule: replaced by
  explicitly named v7 pooled/GRU recipes. Excluded canaries retain their source.
- experiments/state_representation.py: moved to the single maintained owner
  experiments/state_representation_v7.py, matching retained experiment names.
  Model tests and the source-audit consumer now reference that owner. The frozen
  scientific commit still contains its original filename; no receipts rewritten.

## Cost and remaining boundary

L40/L40S unavailable;4090 launch8dd337797e0a refused capacity without acquisition.
Whole-cohort CPU fallback was frozen before training: two balanced four-process
batches, one thread and16GiB per run. Evaluation followed training, with four
endpoints at once and the same two-game envelope per endpoint. No retries.
Prior L4 bill=$0.07258259388618171; no new rental charges; reserve$6 for ancillary
costs within$25 total. Correct provider-name lookup confirms both relevant pods
absent. Etude's refused-session metadata persists without a provider receipt;
its terminate operation deliberately skips refused sessions. No task-owned
training/evaluation process remains. MTG-213 allocations are untouched.

Check: study/declaration/report-discovery tests4passed, untracked-data check1passed;
Ruff passed; real Etude report CLI admitted all8runs and784metrics. Local `uv run
--extra dev ty check` could not start because ty is absent. Gate owns broader checks;
demo/review owns the result judgment. No further training or score-based extension.
