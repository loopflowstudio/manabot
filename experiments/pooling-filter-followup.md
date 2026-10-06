# Pooling × advantage floor — ETU-106

Jack Heart authorized this bounded follow-up on 2026-10-06 after merged PR #226
(`8755ca0f`). The [completed screen](value-token-screen.md) remains immutable:
its equal-update ordering differs from its sparse common-cost comparison, both
pooling alternatives remain unresolved, and its exposure accounting does not
establish why scores or advantages differed. This follow-up changes no default,
frozen screen contract or ETU-91 artifact. No paid compute is authorized.

## Question and controls

Does removing the absolute advantage floor change learning differently for
validity-masked mean and value-token aggregation? Cross `masked_mean/value_token`
with `min_advantage=0.01/0`. Keep the screen's width64, depth1, four attention
heads, post-normalization, scalar critic, Ataraxos move learning, inclusive .75
quantile, actor_critic filter scope, semantic Allies/Lessons setup, four streams,
64 transitions per stream and one CPU thread. Recipes derive from the original
committed screen plan; admission checks the entire resulting recipe, including
all learning settings. History, categorical outcomes and action domain stay fixed.
The existing source adaptations are documented in [Ataraxos fidelity](../docs/ataraxos.md).

Prospective predictions: removing the floor increases token optimizer exposures
by at least 50%, but the endpoint token floor effect minus masked floor effect
has absolute mean below five percentage points. These are uncertain predictions,
not implied by the previous result. Token attention changes shared policy
representations; even a measured interaction cannot identify a critic-only cause
or causally explain the previous cohort.

## Calibration and hard budget

One allocation: **28,800 seconds total**, at most **21,600 for calibration and
training**, at most **7,200 for evaluation and reporting**. No retries, count
increases after admission, replacement seeds or truncated successful arms.
The parent supervisor bounds whole child process groups even if native work
fails to return. Failed exports, games, replay or reporting retain their receipts
and stop the campaign. A checkout lock prevents simultaneous follow-up campaigns;
exclusive output creation prevents restarting over earlier attempts.

Before strength is inspected, all four arms run two linked 20-update stages
at calibration seed 10620. Each has a 400-second run watchdog and two 190-second
stage watchdogs; the calibration phase has an 1,800-second ceiling including
startup and exports. Calibration checkpoints are never scored or continued into
scientific seeds. Calibration failures consume allocation and stop admission.

Use the slowest whole-run seconds per update, including exports, with 25% timing
headroom. After subtracting actual calibration time from 21,600 seconds, select
the largest total update count in {400, 500, 600, 700, 800} that fits all twelve
runs. Two equal linked stages export midpoint and endpoint. **400 total updates
is the minimum useful screen**, an explicit executive choice: below half the
original exposure, this sparse two-checkpoint interaction would have too little
learning to justify its fixed evaluation cost. Failure to fit 400 stops with the
measured limiting rate and feasible count. 800 and checkpoints 400/800 are targets.
Each scientific run receives an equal share of the remaining training watchdog;
counts are not adapted to its performance or realized speed.

The old cohort's 3,224 evaluation seconds for 1,800 games imply approximately
4,300 seconds for 2,400 games, or 6,450 with 50% headroom. This motivates the
unchanged two-hour reserve; it is not a new-world runtime guarantee. Training
calibration does not score or select candidates. Any evaluation timeout fails the
cohort without shrinking the fixed 100-game cells. Require 3 GiB projected output
plus a 4 GiB disk reserve; the two-hour diagnostic checks remaining updates at the
slowest observed rate with 25% headroom and stops on infeasibility or low disk.

## Frozen cohort and interpretation

Fresh paired training seeds: **10621, 10622, 10623**. Arm IDs in canonical order:
masked/.01, masked/0, token/.01, token/0. Execution orders are **0123, 3210, 2301**.
Three seeds cannot perfectly balance four positions; the reversed first pair
and rotated third reduce order confounding without claiming to eliminate host
contention. Resolved recipes, counts, order, calibration hashes, landed commit,
native/source/content/setup/ABI identities are retained before scientific work.

Each raw checkpoint plays **100 games** against source-pinned scripted greedy:
25 deals **961160–961184**, all four deck/seat legs and shared candidate/reference
action-seed aliases. Exactly 24 cells / 2,400 games; no calibration evaluations,
extra opponent, EMA replicate, best-seed selection or score-dependent stopping.
All games must terminate and exact-replay. Independent training seeds are the
experimental units; the 2,400 games are not 2,400 training replicates.

The existing executed notebook/report retains per-seed midpoint/endpoint scores,
transition and cumulative training-hour curves, measured inference costs and
common-cost last-available-checkpoint comparisons on overlapping support only.
`factorial.json` adds zero-minus-.01 effects within each pooling and the difference
of those effects, paired before resampling. Percentile 95% intervals use 10,000
draws with RNG seed 10624, resampling three training seeds and 25 common whole
deal blocks across all four arms. It records retained rows, empty-filter updates,
actor/critic exposures, absolute advantages and recorded value losses; full
per-update diagnostics, including selection-group residuals, remain in run receipts.
Value loss is the recorded final timestep loss, not a whole-update MSE estimate.

A floor effect merits a confirmatory follow-up only if endpoint mean gain is at
least five points, all three seeds are nonnegative, the common-cost comparison
agrees and every game replays. An all-negative effect averaging at most minus
five points rejects that floor change for this recipe; otherwise unresolved.
The interaction is descriptive with three seeds, not an automatic promotion rule.
Neither outcome changes defaults or proves general strength, calibration,
control competence or the chapter's human-challenger outcome.

## Launch and monitoring

Land the software through Loopflow before pinning the scientific source. From
this retained checkout, with a clean landed HEAD:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m experiments.runners.run_pooling_filter --campaign LANDED_COMMIT --out .runs/etu106-pooling-filter
```

Run under a durable local process supervisor. `supervisor.json` records its PID,
start time, phase, cost and failure; `active-child.json` and `child.log` retain the
child handle/output. `calibration.sqlite`, all four calibration run directories,
`calibration.json` and `resolved-plan.json` precede `study/`. The study keeps its
existing VerifyStore, run/checkpoint receipts, command traces, plots and notebook.
The parent enforces the total and phase ceilings; `study/phase.json` records the
evaluation deadline. Never edit imported sources or rebuild the extension while
this campaign runs. Offline regeneration after completion uses:

```bash
uv run python -m experiments.runners.run_pooling_filter --report-only --out .runs/etu106-pooling-filter/study
```

Software preparation is not empirical completion. ETU-106 remains open.
