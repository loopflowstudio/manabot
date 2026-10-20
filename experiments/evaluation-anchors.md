# MTG-153 — evaluation anchors across retained checkpoints

Status: **complete**. One scientific attempt, 3,000/3,000 games, zero failed or
mismatched games (2026-10-20). Jack Heart approved the cohort and a single
attempt on 2026-10-19; with the Mini occupied by MTG-122, he approved running it
on a rented RunPod CPU pod instead (8 vCPU cpu3c, $0.24/h, $3 cap; about $1.15
spent). Predictions and the recommendation rule below were committed before
the attempt (50178682).

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

Execution record. The protocol's Mini-only, $0 limit was replaced by the
approved pod: 50178682 at `/opt/m`, managym built in release for linux x86_64,
retained checkpoints and MTG-137 `analysis.json` copied byte-for-byte, then
`prepare`, `smoke` (15/15 games, all replays passed, 61 s) and one `run`
(3,000 games, 12,388 s, serial at one thread). Prepared cohort, smoke, rows,
traces, Study records, targeting and `report.json` are retained locally in
`.runs/mtg153-pod/`. The pod was deleted after copy-back.

## Results

Scores are the scored player's win rate, 25-deal bootstrap 95% intervals.
Headroom is 1 − score. Δ is the paired difference from that checkpoint's
historical greedy rows on the same deals.

**LvA (learner Lessons vs anchor Allies)** — the headline cell:

| checkpoint | vs heuristic | Δ greedy | vs frozen 12261@10k | Δ greedy |
|---|---|---|---|---|
| init | 0.00 [0.00, 0.00] | +0.00 | 0.02 [0.00, 0.06] | +0.02 |
| Mini 10k 12261 | 0.30 [0.16, 0.46] | −0.10 | 0.34 [0.22, 0.48] | −0.06 |
| Mini 10k 12262 | 0.22 [0.10, 0.36] | −0.24 | 0.34 [0.20, 0.50] | −0.12 |
| Mini 10k 12263 | 0.24 [0.12, 0.36] | −0.06 | 0.18 [0.08, 0.30] | −0.12 |
| GPU 25k 12261 | 0.52 [0.36, 0.66] | −0.12 | 0.42 [0.30, 0.54] | −0.22 |
| GPU 25k 12262 | 0.46 [0.32, 0.60] | −0.16 | 0.34 [0.24, 0.44] | −0.28 |

Group means and training-seed ranges, all cells:

| anchor | cell | init | Mini 10k (range) | GPU 25k (range) | ordered |
|---|---|---|---|---|---|
| heuristic | LvA | 0.00 | 0.25 (0.22–0.30) | 0.49 (0.46–0.52) | yes |
| heuristic | AvL | 0.20 | 0.89 (0.84–0.92) | 0.88 (0.86–0.90) | no |
| heuristic | LvL | 0.04 | 0.51 (0.46–0.58) | 0.52 (0.52–0.52) | yes |
| heuristic | AvA | 0.04 | 0.47 (0.36–0.56) | 0.55 (0.52–0.58) | yes |
| frozen | LvA | 0.02 | 0.29 (0.18–0.34) | 0.38 (0.34–0.42) | yes |
| frozen | AvL | 0.10 | 0.75 (0.70–0.80) | 0.66 (0.60–0.72) | no |
| frozen | LvL | 0.04 | 0.51 (0.40–0.58) | 0.62 (0.56–0.68) | yes |
| frozen | AvA | 0.02 | 0.47 (0.42–0.52) | 0.52 (0.42–0.62) | yes |

Anchor-only pairings (first-named player's score):

| pairing | LvA | AvL | LvL | AvA |
|---|---|---|---|---|
| greedy v heuristic | 0.00 [0.00, 0.00] | 0.80 [0.68, 0.90] | 0.12 [0.04, 0.20] | 0.22 [0.12, 0.32] |
| greedy v frozen | 0.00 [0.00, 0.00] | 0.58 [0.44, 0.72] | 0.14 [0.06, 0.24] | 0.34 [0.24, 0.44] |
| heuristic v frozen | 0.10 [0.02, 0.18] | 0.74 [0.62, 0.86] | 0.68 [0.54, 0.80] | 0.52 [0.42, 0.62] |

Predictions:

- Heuristic orders init < Mini 10k < GPU 25k in ≥3 of 4 cells: **held** (3/4;
  AvL is a tie, 0.89 vs 0.88).
- Heuristic GPU mean < 0.90 in every cell: **held**, barely in AvL (0.88;
  GPU 12262 alone reaches 0.90).
- Frozen self-score within [0.35, 0.65] in each cell: **failed** in LvA (0.34)
  and AvL (0.70). The mirror cells were in range (LvL 0.40, AvA 0.52). LvA and
  AvL sum to 1.04, so the miss is the Allies deck advantage between identical
  policies, not an instrument fault; the prediction should have been stated for
  mirror cells only.

Reading:

- LvA is where the anchors differ. Against heuristic, every GPU seed
  (0.46–0.52) scores above every Mini seed (0.22–0.30), a 0.24 gap in group means,
  with headroom of about 0.5 at 25k. Against the frozen checkpoint the ranges
  meet (Mini 0.18–0.34, GPU 0.34–0.42) and the gap is 0.09.
- AvL saturates heuristic: trained checkpoints score 0.84–0.92 with upper bounds
  of 0.94–0.98, so by the MTG-152 usability rule (≤0.85, upper < 0.95) heuristic
  is not usable in AvL. The frozen checkpoint keeps AvL usable (0.60–0.80) but
  orders it backwards (10k 0.75 > 25k 0.66).
- Heuristic is the strongest of the three anchors except as Lessons against the
  frozen checkpoint's Allies (0.10). Greedy wins no LvA game against either anchor.
- Both anchors score lower than greedy on nearly every trained checkpoint and
  cell (Δ −0.04 to −0.44). This agrees with MTG-135's protocol017 panel (9,152
  games, heuristic−greedy gap nearly constant at −0.18 to −0.29), so heuristic
  adds headroom while keeping greedy's ordering; it does not separate checkpoints
  that greedy could not.

Recommendation. The report rule ties at 3/3 ordered cells and breaks toward
heuristic for the daily regression. The LvA evidence supports that independently.
Use heuristic as the primary anchor, report LvA first, and flag AvL as
saturated. Keep the frozen Mini 12261@10k checkpoint as a secondary anchor where
an unsaturated AvL or a learned-policy opponent matters; it costs a model forward
pass per decision. Keep scripted greedy frozen for continuity. Five trained
seeds and two anchors do not establish general skill or a method-level ordering.

Targeting counts by cell, seat and source are in `report.json` under
`targeting`. They describe choices and are not misplay rates.
