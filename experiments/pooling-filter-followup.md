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

The following result closes this bounded campaign; ETU-106 remains open.

## Result — completed 2026-10-06

**No value-token or floor-removal strength benefit was demonstrated. This is
not evidence of equivalence.** Both floor interventions remain unresolved under
the registered rule: neither reaches the five-point, all-seeds-nonnegative bar,
and neither meets the all-negative rejection rule. No default changes.

The supervisor admitted **600 total updates**, with checkpoints at 300/600,
from the four-arm calibration; the target 800 did not fit its conservative
projection. Twelve scientific runs completed with no replacements or retries.
All 24 comparisons completed exactly 100 terminal, replay-verified games each:
**2,400 games, zero failures or truncations**, and zero replay mismatches or
private exposures. These are three training seeds, not 2,400 method replicates.

Scores below are percentages against the frozen scripted-greedy anchor. Seed
order is 10621/10622/10623. Each entry retains all three seeds.

| Pooling / floor | 300 updates | 600 updates | Endpoint mean |
| --- | --- | --- | ---: |
| Masked / .01 | 36 / 43 / 40 | 40 / 40 / 38 | 39.33% |
| Masked / 0 | 40 / 39 / 38 | 41 / 39 / 39 | 39.67% |
| Token / .01 | 35 / 30 / 34 | 38 / 30 / 44 | 37.33% |
| Token / 0 | 33 / 44 / 33 | 37 / 40 / 32 | 36.33% |

The registered paired bootstrap resamples three training seeds and 25 common
whole deal blocks together across arms (10,000 draws, seed 10624). Differences
and 95% percentile intervals are in percentage points:

| Contrast | Midpoint mean [interval] | Endpoint seed differences | Endpoint mean [interval] |
| --- | --- | --- | --- |
| Masked: zero − .01 | −0.67 [−5.67, 4.33] | +1 / −1 / +1 | +0.33 [−4.00, 4.33] |
| Token: zero − .01 | +3.67 [−4.33, 13.33] | −1 / +10 / −12 | −1.00 [−11.33, 10.33] |
| Token floor effect − masked floor effect | +4.33 [−7.00, 17.33] | −2 / +11 / −13 | −1.33 [−12.67, 11.00] |

The predicted absolute endpoint interaction below five points occurred in the
point estimate; its wide interval does not establish a small underlying effect.
The token exposure prediction also held: floor zero increased endpoint exposures
by 154.3%, 101.8% and 145.3%. That did not reliably improve playing score.

### Collection, exposure and cost are different comparisons

Every endpoint collected **153,600 learner transitions** over 600 updates
(256 transitions/update). Native decisions and finished training games vary
with behavior. Retained rows, optimizer exposures and actor/critic exposures
coincide in this one-pass actor_critic recipe; equal update counts do not mean
equal effective optimization work.

| Pooling / floor | Endpoint exposures by seed | Empty-filter updates by seed | Cumulative training seconds by seed |
| --- | --- | --- | --- |
| Masked / .01 | 33,594 / 35,440 / 33,713 | 0 / 0 / 0 | 1,224.85 / 1,397.06 / 1,269.56 |
| Masked / 0 | 38,400 / 38,400 / 38,400 | 0 / 0 / 0 | 1,295.80 / 1,472.26 / 1,335.00 |
| Token / .01 | 15,102 / 19,028 / 15,652 | 14 / 2 / 9 | 947.41 / 1,060.02 / 1,210.48 |
| Token / 0 | 38,400 / 38,400 / 38,400 | 0 / 0 / 0 | 1,313.64 / 1,330.75 / 1,345.56 |

The saved common-cost window is only **726.97–947.41 seconds**, using the last
available checkpoint without interpolation. Mean scores over that window are
39.67 / 39.00 / 33.00 / 36.67% in the table's arm order. At its final instant,
token/.01 seed 10621 advances to its endpoint, making that arm's common-horizon
mean 34.00%; the other means stay unchanged. Thus token floor removal has a
positive sparse common-cost point difference while its equal-update endpoint
difference is negative. Neither comparison supports a consistent benefit.
Two checkpoints cannot locate a learning curve between observations or establish
an equal-exposure strength effect. The notebook retains both transition and
training-hour plots; the compact data retains their exact coordinates.

The campaign charged **20,303.74 seconds (5.64 hours)** within eight hours.
Calibration cost 322.35 seconds; the four run costs were 84.10/86.87/61.13/84.60
seconds in canonical arm order. Scientific TrainingRun totals were 15,206.05
seconds; evaluation cost 4,757.99 seconds, including 301.03 seconds of replay.
The remaining 17.36 seconds cover supervision/reporting/other overhead, not
unaccounted free training. Stage collection, learning, export and diagnostic
costs remain separately available. Observed candidate inference seconds per
call, pooled over each arm's 600 evaluation games, were 7.40/7.35/6.91/6.93 ms.
These include the recorded player call path and host conditions; one CPU thread
does not match inference cost or isolate architectural throughput.

### Reproduction, provenance and next decision

The scientific source was `59f16e0df9b516f56858c73ad1be88716fc9401f`, after
software PR #236. [Compact evidence](data/pooling-filter/evidence.json) binds
all **225 retained files** by SHA-256 and size, exact resolved configurations,
calibration, code/native/world/content/setup/observation/action identities,
run seeds and receipts, all raw/EMA export identities, the 24 scored raw
checkpoints, per-deal/per-leg scores, paired uncertainty, exposure diagnostics
and timing coordinates. Raw and EMA siblings are not independent replicates;
only raw checkpoints were scored. The native world is w4; exact digest bindings,
not that short label alone, determine reproducibility.

Full private receipts, per-update diagnostics, traces, VerifyStore databases,
plots and executed `study/analysis.ipynb` remain in
`/Users/jack/src/etude.test-ataraxos-inspired-model-representations/.runs/etu106-pooling-filter`.
The backup at `/Users/jack/etu106-evidence/pooling-filter-20261006` was rechecked
against every source file during this evidence export. Neither path is replaced
by this compact public projection; preserve both. There were no failed attempts
inside this campaign. The earlier screen and its failures remain separate,
immutable evidence; no ETU-91 run was restarted or reinterpreted.

Regenerate the compact evidence without training or rewriting source artifacts:

```bash
uv run python -m experiments.runners.export_pooling_filter --source .runs/etu106-pooling-filter --backup /Users/jack/etu106-evidence/pooling-filter-20261006 --output /tmp/pooling-filter-evidence.json
cmp experiments/data/pooling-filter/evidence.json /tmp/pooling-filter-evidence.json
```

Three seeds, one scripted opponent, two checkpoints and shared encoder changes
limit the conclusion. A value token changes policy representations as well as
critic aggregation; the factorial cannot identify a critic-only mechanism.
Exposure differences describe the intervention's work, not a causal explanation
of the prior screen or a proof that advantages caused weak play. Neither this
result nor the original screen establishes general strength, control competence,
model equivalence or human-challenger acceptance.

ETU-106 remains open. Explicit recent-event input is the next accepted feature
priority, with categorical outcomes and capacity interactions still unmeasured
here. Any further scored comparison needs its own bounded protocol/allocation;
this closeout authorizes no new campaign. Learned recurrent memory remains
deferred. Check: compact export/backup hashes and byte-identical regeneration passed;
both paired bootstraps reproduced from compact scores; 14 focused tests and
Ruff/format/diff checks passed.
