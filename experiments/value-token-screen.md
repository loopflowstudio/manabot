# Scalar value-token screen — ETU-106

Jack Heart requested this screen on 2026-10-05 after PR #222 merged.
The frozen cohort completed on 2026-10-06; results and retained evidence follow below. The separate `value-token-screen` /
`screening` profile preserves the eight-arm value-model smoke and the existing
three-anchor scientific protocols. ETU-91's checkout, cohort and allocation are
untouched. No paid compute is included.

## Question and fixed contrast

Does changing aggregation improve learning against the fixed scripted opponent
at this small capacity, without changing information, learning rules or action
coverage? Historical mean, validity-masked mean and learned value token all use
width 64, four heads, one post-normalized attention layer and a scalar critic.
All other resolved settings are equal: Ataraxos move learning, semantic
Allies/Lessons observations, authored sideboards, ordinary flat decisions,
current-policy self-play, CPU and one thread. The shared value token also changes
policy representations; this cannot isolate a critic-only effect. No history,
compound decoder, belief sampler, categorical head or additional depth is added.

Each of seeds **10611, 10612, 10613** runs every arm. Initial shared parameters
use the same seed, but added token parameters and different policies cause later
RNG trajectories to diverge. Alternating forward/reverse arm order reduces one
order confound; it does not remove laptop contention.

Each run has two linked **400-update** stages, four streams and 64 transitions
per stream per update: midpoint at 102,400 learner transitions and end at 204,800.
Continuation retains the optimizer and collector. These are update-count
midpoint/end checkpoints, not assertions of equal elapsed time. Raw checkpoints
alone enter evaluation; EMA exports do not add replicates. Across nine runs,
7,200 updates and 1,843,200 learner transitions are scheduled.

Every checkpoint plays **100 games** against the source-pinned `scripted_greedy`
player: 25 deals **961060–961084**, each with all four deck/seat legs. The same
deals and candidate/reference action-seed aliases pair arms, seeds and
checkpoints. Total: **18 cells / 1,800 games**, no candidate-versus-candidate
matches, random anchor, PUCT anchor or extra endpoint cohort. The arena retains
terminal outcomes, exact replay, commands, failures and inference latency.
Evaluation deals are reserved separately from training seed streams; this does
not promise that no training game shares an opening hand.

## Allocation and stop policy

The hard ceiling is **eight laptop hours**, with at most **six hours of training**
and **two hours of evaluation plus reporting**. Unused training time cannot expand
evaluation. Each run has a 2,400-second watchdog; each stage has 1,190 seconds,
leaving setup/export margin. Watchdog expiry is failure, not a shortened successful
arm. Counts are fixed before execution and never increased to fill spare time.
The short retained ETU-106 smoke motivates 400 updates per half; extrapolation
is provisional and does not establish a long-run six-hour runtime.

At **two hours from launch**, `diagnostic-2h.json` records completed updates,
remaining-training projection, disk reserve and continue/stop. During training,
project remaining updates at the slowest observed run's seconds per completed
update, with 25% headroom. Stop if that forecast exceeds the six-hour training
ceiling, no completed-update rate is available, a run failed, or free disk falls
below 4 GiB. The check does not inspect scores or select a winning arm. If already
in evaluation, it records training completion and checks the same reserve.
The Python timer is delivered when the interpreter regains control; a native call
can delay it. Existing collector deadlines additionally bound rollout work.

Before launch, require the 2 GiB projected output allowance plus 4 GiB reserve.
This is a conservative reservation, not a measured peak. Stop immediately on any
training/export error, invalid/incomplete/replay-failing arena cell, game timeout
(120 seconds), command cap (10,000), phase deadline or total deadline. Finite JSON
receipts reject nonfinite diagnostics. Preserve all failed/partial attempts and
artifacts; there is no automatic retry, seed replacement, count adjustment or
training resume. A diagnostic stop requires a separately frozen next allocation.
If the process dies abruptly, elapsed cost is unknown until reconciled; never
infer unused budget from the last receipt.

## Prediction and interpretation

Prospective prediction: the absolute mean paired endpoint score differences
versus historical pooling will be below **5 percentage points for the token** and
below **3 points for masked mean**. This is a hypothesis, not prior evidence.

Report every seed at both checkpoints, learner-transition curves and cumulative
training-hour curves. At common observed cost use the last checkpoint available;
no interpolation, extrapolation or comparison outside overlapping cost support.
The existing report supplies per-arm training-seed/common-deal bootstrap intervals;
three seeds give descriptive uncertainty. Game-level intervals alone cannot
establish an architecture effect. Evaluation/replay costs remain separately
visible and counted toward the overall allocation.

A candidate merits a confirmatory follow-up only if its endpoint mean paired
score improvement is at least 5 points, no seed is negative, the shared-cost
comparison agrees, all games replay, and measured inference seconds per decision
are at most 1.25 times historical. A mean loss of at least 5 points in all-negative
seeds rejects that candidate for this recipe. Otherwise keep it unresolved.
These are screening dispositions, not automatic default changes or admission.
The candidate-to-historical paired differences must use matching seed/deal blocks;
per-arm confidence intervals are not difference intervals. Midpoint is diagnostic,
not a candidate-selection opportunity; the same fixed evaluation deals are reused,
so there is no untouched confirmation set.

One thread is a serving envelope, not matched inference compute: token attention
costs more. A scripted opponent can reward exploitable behavior. This screen
cannot establish general strength, exploitability, control competence, categorical
superiority, human-challenger success, or an isolated value-head mechanism.

## Resolved plan and exact launch

The committed [resolved plan](plans/value-token-screen.json) binds recipes, seeds,
deals, native extension, engine/model/content/setup/ABI sources, runner sources,
analysis notebook and dependency declarations. It is a same-checkout prospective
plan; loading it after source or native-extension drift fails before training.
It does not certify another machine's installed dependencies. Keep this checkout's
imported sources and native extension stable throughout a launched cohort.

From the repository root, the executed launch command was:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m experiments.runners.run_value_screen --plan experiments/plans/value-token-screen.json --out .runs/etu106-value-token-screen
```

The command ran at source commit `2bc013da0f442997d33d1219878d85a048ea23c6`.
The existing output is immutable evidence, not a restart destination. The frozen
plan is retained unchanged; integration changes its bound sources, so it is not
a launchable plan on the delivery head. No additional campaign is authorized.

To resolve a new prospective plan without training after a deliberate source
change (retain the previous plan and review its changed identities first):

```bash
uv run python -m experiments.runners.run_value_screen --write-plan /tmp/value-token-screen-next.json
```

Offline analysis, including failed attempts with available receipts:

```bash
uv run python -m experiments.runners.run_value_screen --report-only .runs/etu106-value-token-screen
```

Software validation uses bounded fixtures: plan rejection, unchanged legacy
restrictions, full 18-cell scheduling, early arena failure, source-drift rejection,
two-hour forecast stop/continue and repeatable executed report/notebook outputs.
Fixture games and checkpoints are synthetic; they supply no strength or timing
measurement. The completed real cohort is distinguished below.

Gate (2026-10-05): the affected screen/value/study/architecture-recipe suites
passed **31 tests**; focused Ruff checks/format and `git diff --check` passed.
The committed resolved plan reproduces current source/runtime identities without
training. CI owns its broader matrix; no Rust or model code changed here.


## Completed screen (2026-10-06)

All nine runs completed both 400-update stages: three paired training seeds,
1,843,200 learner transitions, 18 raw-checkpoint evaluations of 100 games each.
Inspection of every saved arena row and comparison receipt confirms all **1,800
terminal games passed exact replay**, with zero failures, truncations or recorded
integrity violations. Every stage has 400 diagnostics and 102,400 transitions;
optimizer exposure totals agree with the sum of retained samples. No replacement
attempt, additional training or evaluation was run during delivery.

Scores below are percentages against the fixed scripted opponent; columns list
seeds 10611, 10612, 10613. The same 25 deals and all four deck/seat legs are reused
at both checkpoints, as frozen above.

| Aggregation | Midpoint scores | Endpoint scores | Endpoint mean | Paired endpoint difference vs historical |
| --- | --- | --- | ---: | --- |
| Historical | 37, 28, 28 | 43, 33, 43 | 39.67% | — |
| Masked | 35, 28, 29 | 41, 44, 41 | 42.00% | −2, +11, −2 points; mean +2.33 |
| Token | 27, 36, 29 | 30, 35, 39 | 34.67% | −13, +2, −4 points; mean −5.00 |

Post-run descriptive paired seed/common-deal bootstrap 95% intervals for the
endpoint differences are **[−6, +12] points** (masked) and **[−14, +3.33] points**
(token). These resample three training seeds and 25 common deal blocks, keeping
four legs together and pairing arms; 10,000 draws use NumPy default_rng seed
10606. This analysis choice is post-run, not an added prospective criterion.
The retained notebook's per-arm intervals are not paired-difference intervals.
Three seeds cannot establish a general architecture effect.

The masked prediction (absolute mean gap below 3 points) held; the token
prediction (strictly below 5 points) missed at the boundary. Neither candidate
meets the frozen promotion criterion. Token's mean loss reaches 5 points, but
one seed is positive, so the all-negative-seeds rejection criterion also fails.
**Keep both unresolved; promote neither and change no default.**

### Cost and optimizer exposure investigation

The study receipt charges **18,467.19 seconds (5.13 hours)**, within eight hours:
15,230.85 seconds for run receipts and 3,224.35 seconds for evaluation, with
11.99 seconds of remaining orchestration/reporting overhead. Replay is 187.97
seconds within evaluation, not additional cost. The two-hour diagnostic continued
at 3,334/7,200 updates with a conservative 19,219.34-second training projection
and without inspecting strength. Actual training remained below six hours.

The common observed training-cost interval is 995.43–1,364.68 seconds. At its
upper cutoff, last-available checkpoint scores average historical 31.00%, masked
30.67%, token 34.00% (only token seed 10613 has reached its endpoint). Thus the
endpoint ordering is not an equal-cost ordering; sparse checkpoints do not allow
interpolation or a general speed claim. Measured pooled candidate latency across
both checkpoints is historical 7.553, masked 7.697, token 7.508 ms/decision.
The token/historical ratio is 0.994, within the 1.25 screen threshold, but these
are observed mixed-decision costs on this host, not matched FLOPs or controlled
microbenchmarks. The retained notebook includes transition and training-hour curves.

| Aggregation | Total optimizer sample exposures per seed | Empty-filter updates per seed |
| --- | --- | --- |
| Historical | 50,569; 50,718; 50,441 | 0; 0; 0 |
| Masked | 45,763; 46,237; 48,378 | 0; 0; 0 |
| Token | 15,339; 20,940; 21,853 | 14; 7; 10 |

These are samples used by the optimizer, not optimizer calls or collected
transitions. Every arm still collects 204,800 transitions. Frozen
`ataraxos.py::update_move_iteration` trains both actor and critic only on retained
rows and skips empty timesteps. `selection.py::selected_moves` retains absolute
advantages at or above `max(75th percentile, 0.01)`. With 256 rows, the quantile
alone retains at least 64; lower counts establish that the minimum threshold
binds. Saved diagnostics match exposures exactly and record 31 empty token
updates. The token therefore receives substantially fewer actor **and** critic
sample exposures under this fixed rule, alongside lower learning time.

This identifies the accounting path, not why token advantages are smaller or
why endpoint scores differ. Changed shared policy representations, critic scale,
visited states and stochastic trajectories remain possible contributors. The
receipts do not retain full per-update advantage tensors for counterfactual
refiltering. No filter intervention or matched-exposure experiment was conducted;
do not claim a proven architectural defect, better calibration, causal starvation,
or a cheaper equivalent learner. A separately authorized diagnostic could freeze
rollout batches and compare advantage distributions and floor sensitivity before
another architecture claim. History, categorical/depth contrasts and remaining
ETU-106 science remain open.

### Provenance and preservation

The compact [result receipt](data/value-token-screen-result.json) retains all 18
checkpoint digests and metrics, per-run exposure counts, source/runtime identities,
paired analysis method and a SHA-256 manifest of all **142 evidence files**.
The protocol digest is
`b1d9e7a0a7c21b1a2dfd6c7bd68c4c2fff1b0d195dfb8a289f2e35dcf295ec80`.
The frozen world is w4, with exact engine/content/setup/ABI identities in that
receipt and `resolved-plan.json`; no measurements transfer to integrated sources.

The full evidence remains at
`/Users/jack/src/etude.test-ataraxos-inspired-model-representations/.runs/etu106-value-token-screen`:
VerifyStore SQLite, every run receipt, raw/EMA exports, arena registrations,
compressed command traces, resolved plan/recipes/protocol, diagnostic, executed
notebook, plots and reports. Before integration, all 142 files were copied and
SHA-256 verified at `/Users/jack/etu106-evidence/value-token-screen-20261006`;
the sibling `value-token-screen-20261006.sha256.json` records those hashes.
Keep the original checkout and absolute paths usable; the backup does not rewrite
embedded paths. No original evidence was regenerated, pruned or deleted.

This single scripted-opponent screen does not establish general strength,
exploitability, control competence, human-play success or an isolated critic
mechanism. ETU-91 is unchanged. Jack Heart authorized delivery; that is not human
code-review approval or completion of ETU-106's remaining scientific work.

Delivery check (2026-10-06): `uv run pytest tests/training/test_value_screen.py tests/training/test_value_models.py tests/training/test_study.py tests/training/test_architecture_recipes.py tests/training/test_capacity_study.py -q` — 39 passed; focused Ruff/format and `git diff --check` passed. CI verifies the integrated head.
