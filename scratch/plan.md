# MTG-317 plan

`Execution.precision` (`float32`/`tf32`/`bf16`, `manabot/training/precision.py`)
already exists on main, always serialized, with tests in
`tests/training/test_precision.py`. The deliverable is the L4 measurement,
report and default decision.

## Built
- `experiments/cuda_precision.py`: capture (one float32 update of frozen
  w160l8, own hooks: `l4_step_profile.insitu` no longer runs on main — its
  `make_actor` lacks `bucketed_shapes`, its learner replay calls the old
  `observation_tensors(obs, device)`), then 5 arms × 4 groups, verdict in
  `analyze`. CPU smoke: `uv run python -m experiments.cuda_precision smoke` passes.
- `experiments/cuda_precision.md`: rule committed at `71c044113` before the run.
- Not Etude-discovered (no report()/eval()), so no test_experiment_files edit.

## Launch
- Launcher DB is schema 13 → control with `etude-sdk==0.2.15` overlay
  (locked 0.2.16 refuses the DB); wrapper `/tmp/mtg317-etude.sh`.
- 15:49 UTC: `etude launch --rent l4-etude` refused (no L4 stock); queued as run
  `f71ee505f7cc`, cohort `mtg317-precision-1`, `--max-usd 3`, job 2.0 h / $1.50;
  `etude wait mtg317-precision-1` drives it.

## Result
- Run `f71ee505f7cc` succeeded, ~$0.45. Verdict: TF32 unresolved, BF16
  rejected, recommendation float32. No code/default change; report is the
  deliverable. Data published (`experiments/data/cuda_precision/`).
- Steer (comment ddd63c9e, 2026-11-12) applied: keep verdicts; state the ~3e-4
  policy-loss reference plainly; absolute table beside ratios; say the rule was
  too strict and why; next test = short real learning A/B at w384l8 (same
  seeds, a few hundred updates, strength vs frozen anchor).

## Realign (2026-11-12)
- Branch already contains origin/main; no upstream change to the forward.
  Sibling steer f5c9dda9 (MTG-318 CUDA graphs, MTG-319 live-token attention
  touch the same forward): this branch changes no training code, so only
  "rebase on main before publishing" applies.
- docs/training.md numbers aligned with the report (BF16 loss within 0.6%,
  BF16 2–2.7× at w384l8). The steer's "within 0.4%" is its wording; the
  report's measured maximum is 6.4e-3.

## Remaining
- Rebase on main, delete `scratch/` in the pre-land commit, publish.

## Compress (2026-11-12)
- `experiments/cuda_precision.py` measurement path left as run at `71c044113`
  (the freeze); only `charts` was added after. docs/training.md's results
  paragraph cut to decision + pointer; the method lives in the report.
