# MTG-153 — evaluation anchors across retained checkpoints

Status: implementation and admission checks; **full experiment unrun**. This
protocol is a draft until committed before the scientific attempt. Jack Heart
approved the cohort and Mini-only, eight-hour, $0, one-attempt limits on
2026-10-19. The numeric predictions and recommendation rule below are draft
analysis choices, not accepted strength conclusions. No strength claim or anchor
recommendation follows from smoke games.

## Question and prediction

Does heuristic order retained checkpoints while leaving more headroom than
scripted greedy? Does a fixed trained checkpoint provide a useful second axis?

Working numeric predictions, declared before the full run: heuristic will give
init < mean Mini 10k < mean GPU 25k in at least three of four cells; its GPU
mean will be below 0.90 in every cell. The frozen checkpoint's self-score will
lie within [0.35, 0.65] in each cell. These are hypotheses, not admission gates.
An honest contrary result is retained. Any game failure, replay mismatch,
identity mismatch, missing deal or exhausted budget stops the single attempt;
there is no automatic relaunch or analysis over survivors.

## Frozen population and deals

Score initialization from Mini seed 12261; Mini 10k seeds 12261/12262/12263;
GPU 25k seeds 12261/12262. Freeze Mini 12261 at 10k as the checkpoint anchor,
chosen by seed rather than strength. Both anchors play all six checkpoints,
including the fixed checkpoint against itself under distinct player identities.
Also play greedy vs heuristic, greedy vs frozen, heuristic vs frozen.

Each pairing has 25 deals × eight legs = 200 games; 15 pairings = 3,000 games.
Initialization uses its original MTG-137 initialization family; endpoints use
their original endpoint family. Anchor-only pairs use the Mini 12261 endpoint
family. Admission resolves every checkpoint by size and SHA-256 and every
historical greedy monitor by the SHA-256 in MTG-137's `analysis.json`. Identical
retained copies are interchangeable bytes; no artifact is moved or modified.

Greedy is not rerun against the scored cohort. Its original rows supply paired
differences on the same deals. Policy RNG differs with the opponent's arena
identity, so this pairing controls deals rather than individual policy samples.

The checkpoint opponent uses stochastic CPU inference, one thread, one pass,
batch size one, with explicit repeated-deck admission for mirrors. A checkpoint's
setup, world and observation ABI must be admitted. Neither code anchor changes.
The prepared cohort records source, native binary, registrations, artifact hashes,
coordinates, historical baseline hashes and deal families. Execute checks them.

## Analysis and decision

Report LvA first, then AvL, LvL, AvA, from the scored player's perspective.
Resample the 25 whole deals (10,000 bootstrap replicates, seed 12290), retaining
the two correlated seats in each cell. Report checkpoint score intervals, paired
differences from historical greedy, descriptive group-mean ordering, training
seed ranges and max-minus-min spread, and headroom (1 − score) per checkpoint.
Anchor-only pair scores use the named first player's perspective. Self-play is
a noisy sanity check, not a forced 0.5 score or evidence of policy quality.

Training seed counts remain three Mini and two GPU samples. Deal intervals do
not establish method-level confidence; initialization has one seed. The strongest
confound is opponent-specific nontransitive strength: rank agreement on two
anchors cannot establish general skill. A later independent checkpoint cohort
and additional opponents would discriminate it.

The report's provisional recommendation keeps both measured anchors for future
studies and historical greedy for continuity. For the daily regression it selects
the anchor ordering the most cells by group mean, breaking ties in favor of the
cheaper heuristic. This rule does not claim statistically resolved ordering.
Review headroom and uncertainty before adopting it.

The MTG-152 Study recorder reconstructs targeting from replay-validated Commands;
no policies are resampled. Counts and fractions retain cell, seat, source, side,
declines and unresolved targets. These are dependent descriptive choices, not
misplay rates. Per-game Study records and targeting JSON remain beside traces.

## Execution

Mini only; at most eight hours and $0, one scientific attempt. Smoke is one game
per pairing (15 games), without strength analysis. Admission and smoke failures
remain in their own directories. A scientific attempt claims an exclusive marker
beside its cohort file, even if it fails; changing its output directory cannot
resume or relaunch it. Preparing a new cohort does not grant relaunch approval.

The original `prepared` and `smoke` directories retain earlier evidence and
cannot be overwritten. The following fresh paths admit and smoke current source
with the matching w6 runtime; a fresh cohort does not authorize another scientific
attempt after one has been claimed:

```sh
uv run python -m experiments.runners.evaluation_anchors prepare \
  --retained /Users/jack/src/fantasia.measure-learning-progress-across-simple/.runs/mtg122 \
  --analysis /Users/jack/src/fantasia.fc765e-run-corrected-world/.runs/baseline-final/analysis.json \
  --out .runs/evaluation-anchors/prepared-02
uv run python -m experiments.runners.evaluation_anchors smoke \
  --cohort .runs/evaluation-anchors/prepared-02/cohort.json \
  --out .runs/evaluation-anchors/smoke-02
# Commit the prediction/protocol before the one authorized scientific attempt.
uv run python -m experiments.runners.evaluation_anchors run \
  --cohort .runs/evaluation-anchors/prepared-02/cohort.json \
  --out .runs/evaluation-anchors/scientific
uv run python -m experiments.runners.evaluation_anchors report \
  --cohort .runs/evaluation-anchors/prepared-02/cohort.json \
  --out .runs/evaluation-anchors/scientific
```

Report is read-only. A failed, running or smoke attempt returns status without
strength estimates. Source edits require fresh admission before execution.

Implementation check (2026-10-19): all 15 one-game pairings completed and replayed,
with zero failures and Study targeting artifacts for every game (28.31 seconds).
Focused tests also exercise full synthetic report analysis, missing evidence,
corrupt checkpoint admission, incomplete deals and refusal to relaunch. This is
engineering evidence from before the subsequent simplification. Reconciliation
on 2026-10-19 confirmed 15 Study and targeting artifacts and the original cohort
binding, but its source digest differs from current code. Fresh admission and a
current-source smoke remain outstanding; the 3,000-game scientific evaluation
remains unrun.
