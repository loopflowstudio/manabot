# TLA frontier self-play at two model sizes (c5fba4)

Exploratory. How far does self-play on the 134-deck 17Lands TLA trophy corpus
push play strength at the current recipe size (w64), and does a network about
17× larger in compute parameters (w160l8: width 160, 8 attention layers, 8
heads) get further on the same budget? One training seed per size, so the
reading is descriptive, not a controlled claim that one size is better in
general.

**Result.** At this budget, neither size is ahead. Each run had about USD32
and 54 hours on one L4. Small reached 1,952 updates; Large lost its rental
after 931, and its last stored checkpoint is update 925. Large's final checkpoint
scores 0.53 against Small's checkpoint from the same training time and 0.52
against Small's final; both intervals include 0.5. Against the heuristic, the
finals score 0.70 (Large) and 0.72 (Small). Large is stronger at equal updates
(0.62–0.66), but each Large update costs about 2.5× the time. The one
suggestion in Large's favor is its trend at equal training time: behind
through update 400 (0.39–0.48), level from 600 on (0.50–0.53). Training Large
for longer is the follow-up.

## Design

- **Source.** Training and every scored evaluation pass ran from revision
  `ab2b757d9` (world w17, card digest `0798b2c7…`, policy input v9).
- **Training.** Both sizes ran the same damped self-play rule, population,
  four producers, FP32 precision and 54-hour wall. Each size's damping horizon
  was fitted from a five-update calibration: 2,474 updates for w64 and 999
  for w160l8. Checkpoints were exported every 25 updates.
- **Evaluation.**
  - 16 fixed random 17Lands deck pairs × 4 crossed legs = 64 games per
    comparison. A draw counts as half a win.
  - Intervals are 95% bootstrap intervals that resample whole deck pairs.
  - Comparisons were played at shared update milestones (25 … 6,400) and at
    each run's endpoint.
  - **Half training time:** the same learner's latest checkpoint at or before
    half its elapsed training hours.
  - **Heuristic:** the first-64 heuristic.
  - **Fixed updates:** Large vs Small at the same update.
  - **Fixed training time:** Large vs Small's last checkpoint at or before
    Large's elapsed training hours.
- **Not measured.** No held-out-deck claim; the population is the training
  corpus.

## Runs and cost

2026-11-13: The report retains the recorded machine-loss outcome for
`5b7d8fa7b38c` in the experiment file, so reading its endpoint no longer needs
the original launcher database. Stored runs and matches are unchanged.

Launcher-reported cost.

| Run | What | Outcome | Cost |
| --- | --- | --- | ---: |
| `59718fe34945` | Small (w64) training | succeeded, 1,952 updates | USD31.45 |
| `5b7d8fa7b38c` | Large (w160l8) training | machine lost; 931 updates stored, last checkpoint u925 | USD32.91 |
| `tla-frontier-c5fba4-paired-v1-{3e5506948dd8, f8bfbdb47bf6, 239dcecb44df, 7d7987b33266}` | evaluation | 880 matches | USD0.59 |
| `tla-frontier-c5fba4-cpu-split-v1-{db4bcf749222-capacity-1, d977da726324, 53dd18c20036}` | evaluation | 240 matches | USD0.24 |
| `tla-frontier-c5fba4-gpu-split-v1-{76f6eba7fa0c-capacity-1, 4c5dde665ee0}` | evaluation, void (below) | 0 games played | USD0.04 |
| calibration, rehearsal and vocabulary-check runs | preflight | — | about USD6.5 |

Total about USD71.7 against the USD110 ceiling. The kept evaluation is 1,120
matches, or 4,480 games. Every game is valid and attempted.

## Results

Scores are for the first-named model. Each row is 64 games.

| Comparison | Score | 95% interval |
| --- | ---: | ---: |
| Large u925 (final) vs Small u1952 (final) | 0.52 | 0.41–0.63 |
| Fixed training time: Large u600 vs Small u1200 | 0.50 | 0.42–0.59 |
| Fixed training time: Large u800 vs Small u1525 | 0.53 | 0.45–0.61 |
| Fixed training time: Large u925 vs Small u1700 | 0.53 | 0.44–0.61 |
| Fixed updates: Large u600 vs Small u600 | 0.62 | 0.53–0.72 |
| Fixed updates: Large u800 vs Small u800 | 0.66 | 0.55–0.77 |
| Large u925 vs heuristic | 0.70 | 0.61–0.80 |
| Small u1952 vs heuristic | 0.72 | 0.61–0.81 |
| Large u925 vs itself at half training time (u450) | 0.56 | 0.47–0.66 |
| Small u1952 vs itself at half training time (u1125) | 0.56 | 0.50–0.64 |
| Large u925 vs its initialization | 1.00 | 1.00–1.00 |
| Small u1952 vs its initialization | 1.00 | 1.00–1.00 |

- **Against the heuristic,** both sizes climb from 0.11 at initialization.
  Large leads from update 300 to 600. Large flattens at 0.70 from 600;
  Small reaches 0.72 by 1,600.
- **At half training time,** each learner's late checkpoints still beat their
  half-time selves (0.50–0.64), so neither had plainly stopped improving.
  Updates 25 and 50 have no half-time point: no trained checkpoint had been
  saved by half their training time.

Charts: [published report page](data/tla-frontier-c5fba4/report/tla_frontier.html)
(four charts: model vs itself at half training time, model vs heuristic, Large
vs Small at fixed training time, Large vs Small at fixed updates).

## Changes that keep the question

- **2026-11-11.** Large's uploader blocked on a TLS socket with no timeout,
  so storage stopped at update 931 while the pod trained on past 1,014. The
  launcher then recorded the rental as lost. Large's endpoint is its last
  stored checkpoint, update 925.
  - A local copy of the pod's records through update 1,014 is kept with the
    operator records, but it isn't in storage.
  - The Etude fix (request timeouts and retries) is EXP-47, merged as
    `9f96e49`.
- **2026-11-11.** Large-model evaluation moved to a GPU serving lane, with
  Small-only matches staying on CPU. Both GPU passes that played
  (`76f6eba7fa0c-capacity-1`, `4c5dde665ee0`) were refused before any game:
  - The frontier registration takes its observation-ABI hash from a fresh
    default observation space, `31747d…`.
  - Every frontier checkpoint loads as `83f8c1…`.
  - The GPU server compares the two; CPU play does not.

  Etude recorded both passes as succeeded. **2026-11-12:** both passes were
  voided, and their 160 matches were replayed on CPU under the original
  tournament in `cpu-split-v1-53dd18c20036`. Earlier capacity refusals were
  retried under `-capacity-N` names.
- **2026-11-12.** The experiment moved to the current experiment setup:
  - The protocol document and JSON, the digest check in `frozen()`, and the
    dedicated tests were removed.
  - The protocol values are inline in `experiments/tla_frontier.py`.
  - `experiments/regimes/tla-frontier-c5fba4-w64.json` is kept, because the
    QA smoke run, `tests/training/test_builders.py` and the MPS regression
    test load it as the reference recipe.
  - The heuristic's source identity no longer hashes the protocol file. A
    re-plan from main therefore names new matches; reproduce stored results
    from `ab2b757d9`.

## Limits

- One seed per size.
- 64 games per point resolve differences of about ±0.1.
- Large's run ended about 75 updates short of its fitted horizon of 999, and
  Small's ended well short of its horizon of 2,474.
- Both learning schedules were fitted for their own run lengths, so
  comparisons at equal updates compare different schedule positions.
