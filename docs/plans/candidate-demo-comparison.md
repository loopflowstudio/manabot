# Candidate versus demo opponent: comparison protocol

**2026-09-29 · ETU-84 · Intelligence KR 2 · Proposal awaiting Jack Heart's
acceptance.** Jack requested autonomous drafting. He accepts or changes this
protocol in the design review of the comparison Task. This document authorizes
no experiment; no training or scored game was run for ETU-84. The human
challenger protocol for KR 3 is separate.

## Question and scope

Does the existing local training recipe produce manabots that outperform the
stock Etude Fantasia demo opponent across complete GW Allies versus UR Lessons
games, on the corrected Learn/Lesson world, within the same inference resource
allowance and the declared 10% tolerance for measured inference cost?

The [approved budget plan](strongest-manabot-training.md) sets the scope; the
[operator workflow](../local-trained-challenger.md) supplies the recipe and
artifact owners. The [prior evidence](../evidence/trained-challenger-2026-09-25.md)
supports feasibility only. Prediction, to commit before training:
**mean candidate score at most 0.50 against Search-64**. Eight teacher games are
a small imitation corpus; a negative result is plausible and useful. This is
a hypothesis, not a measured forecast or a reason to lower the success bar.

## Baseline, candidates and freeze

**Baseline: stock Search-64.** Freeze `SearchVillain` from
[`etude/villain.py`](../../etude/villain.py): 64 simulations **per legal action**,
16 compatible worlds × 4 random terminal rollouts, 2,000 steps per rollout,
first maximum in legal-offer order. These are the unconfigured server's defaults,
not PUCT's total-root simulation budget. Pin source, native implementation and
seed derivation. Comparing a server-selected checkpoint instead requires an
amendment with its exact admissible digest; the pre-correction seed-79 checkpoint
is not an eligible substitute.

**Candidates: five independent executions of the existing local recipe.**
Use [`scripts/train_challenger.py`](../../scripts/train_challenger.py), after
its corrected-world binding is accepted, with eight complete Search-64 teacher
games per execution, alternating deck assignment and retaining both seats'
labels. Use the existing hard-action BC objective, width 64, four attention
heads, eight epochs, Adam learning rate 0.001, batch 64, and 20% whole-game
validation (currently two of eight games). These are the demonstrated operator
settings, selecting flat search from the budget plan's alternatives. Each run
starts fresh with its own corpus; no architecture change, value objective,
search at candidate inference, or sweep. Pin expanded defaults in each recipe.

Select the final epoch of every execution, even when validation declines.
Evaluate all five outputs; there is no best-seed selection. The unit of the
primary claim is this procedure across independent training seeds. A claim
about any one checkpoint reports only that checkpoint's conditional result.

Before training, commit the accepted protocol, prediction, seed schedule,
recipe and budgets. Development then settles the provisional counts and limits
below. Before final evaluation, commit an accepted freeze addendum with:

- Exact corrected world/version, source commit and retained source bytes,
  native extension digest, content/pack manifest, main decks and formal
  sideboards, rules/setup identities, Observation/action ABI, dependency lock,
  Python/Torch versions, M4 Max hardware and worker configuration.
- Baseline and Random/Passive control registrations and source identities;
  all five checkpoint digests, candidate manifests, training/validation game
  identities, data digests, selection rule and every attempt receipt.
- Accepted sample count, complete ordered game schedule and policy RNG seeds,
  compute profile receipts, analysis implementation identity and this document's
  revision. No unresolved identity or provisional numeric limit at scoring.

Corrected rules acceptance includes the named sideboards and public Learn
knowledge described in the [Learn/Lesson record](../rules/learn-lesson.md).
Matching tensor dimensions or loading an old checkpoint is insufficient.
The runtime must preserve every legal offer and use only the acting viewer's
information. Retain hidden-world-swap, live/headless and exact-replay evidence
for the configured players. Seeds used for those proofs belong to development.

## Deals, seeds and the four-game block

Use these proposed seed reservations. The values are arbitrary reproducible
identifiers, not lucky seeds; spacing follows the trainer's `seed + game_index`
derivation. Audit actual expanded seeds and related game identities, not just
top-level labels, for disjointness before the freeze.

| Purpose | Proposed seeds | Basis |
|---|---|---|
| Independent training runs | 84,010,000; 84,010,100; 84,010,200; 84,010,300; 84,010,400 | Five runs; spacing 100 prevents overlap among each run's eight consecutive engine deals. The same run seed initializes model, optimizer order and teacher RNG through the existing recipe. |
| Internal validation | Two whole games selected by the shipped splitter inside each eight-game corpus | Existing 20% split, rounded to two. Both seats and all decisions of a game stay together; validation labels never enter optimizer updates. |
| Development/cost calibration | 84,020,000–84,020,003 | Four blocks per candidate/baseline pair; small complete-game cost sample, not a strength estimate. |
| Development controls | 84,020,100–84,020,103 | Four blocks for baseline versus each control; use the first block for each candidate versus each control. |
| Final evaluation | 84,030,000–84,030,063 | 64 untouched blocks shared across the five candidate comparisons; provisional cost-limited count below. |
| Analysis RNG | 84,040,000 | Fixed reproducibility seed for resampling, unrelated to engine or policy draws. |

Training corpora, their internal validation games, development games and final
games are distinct. Do not use final roots for profiling, tuning, choosing a
checkpoint, curriculum or debugging before scoring. If final deals become
development evidence, retire that entire cohort and register fresh deals.
Do not pool a game's other seat, replay, fork or relabelled trajectory into
another split. Retain corpus/attempt identity alongside game indexes, since
indexes 0–7 repeat across executions.

For every final deal seed and candidate, run exactly these four games:

| Leg | Seat 0 (on the play) | Seat 1 |
|---|---|---|
| A | Candidate, UR Lessons | Baseline, GW Allies |
| B | Baseline, UR Lessons | Candidate, GW Allies |
| C | Candidate, GW Allies | Baseline, UR Lessons |
| D | Baseline, GW Allies | Candidate, UR Lessons |

A/B share the exact seed and ordered deck/sideboard setup; C/D share their
own reversed setup. Only player assignments swap within each pair. This gives
identical initial deals within each pair. Reusing a seed across reversed deck
orders does **not** promise identical per-deck physical hands. Check initial
root/setup digests and engine seed propagation; all four legs remain one
statistical block. Use the same blocks for every candidate.

Freeze separate per-player policy seeds using the existing arena `derive_seed`
mechanism and registered identities. Keep each player's RNG stream stable
across the paired legs, resetting it at each game; seed Torch explicitly for
stochastic checkpoint play. A shared deal seed alone does not initialize that
player. Verify real demo/arena parity, including forced decisions and search
call counters, before freezing; do not invent a replacement player.

Run blocks in seed order, rotating the starting candidate and starting leg by
block index. Complete all candidates for a block before the next block to
balance temporal load. Keep this schedule regardless of outcomes.

## Local budget and sample size

All compute uses the existing **M4 Max / 128 GiB**, CPU, **one game worker and
one Torch thread**, sequential training runs. One worker follows the measured
operator recipe and avoids assuming four-worker scaling. No paid compute,
provisioning or cloud service. Record setup, source capture, generation,
training, export/reload, validation, inference and analysis separately.

The two retained old-world eight-game runs took **123.462 and 130.496 worker
seconds**, at about 1.5 GB RSS. Five runs at that range would take
**617.310–652.480 seconds**.
This motivates five seeds rather than the budget plan's initial three. The
24-game attempt failed at **600.177 seconds**, so it does not justify a larger
corpus. These records lack complete source capture and controlled concurrency;
they establish neither corrected-world throughput nor evaluation cost.

| Limit or count | Proposal and basis |
|---|---|
| Training | Five attempts, each ≤600 worker seconds and ≤4 GiB sampled process-tree RSS; ≤1 hour total operator time including source capture/export. Existing caps; the extra ten minutes beyond 5 × 600 s covers supervisor/capture overhead, not extra training. |
| Setup/admission | ≤1 active hour for locked environment/build and existing compatibility/parity checks. Engineering repairs that do not fit return to implementation before any scored launch. |
| Development | ≤1 active hour for 152 games: 80 candidate–Search (5 × 4 blocks × 4), 32 Search–control (2 × 4 × 4), and 40 candidate–control (5 × 2 × 1 × 4). Counts are provisional pending complete-game cost receipts. Random/Passive results are diagnostics only. |
| Final | **64 blocks per seed, 256 games per candidate, 1,280 games total**, ≤6 active hours including recording and replay. Provisional: 16.875 seconds/game before fixed overhead. The calibration below must establish feasibility. |
| Total | ≤24 elapsed hours from setup to report and ≤10 active local hours, including a protected 1-hour failure/analysis reserve: 1 setup + 1 training + 1 development + 6 final + 1 reserve. This selects the approved plan's local one-day envelope; it does not promise completion. Reserve cannot buy more training attempts or scored games. |
| Incremental cash | ≤$2 total, below the plan's strict $10 ceiling; $0.40 protected (20% reserve). Planning estimate: 0.15 kW incremental power × $0.40/kWh × 10 h = $0.60. Use measured power and actual tariff where possible; otherwise retain labelled estimates. Stop if the conservative revised estimate exceeds $2. |

Settle final volume from complete-game receipts, not browser timing or engine
SPS. Measure the four development blocks for **each** candidate against Search,
including recording and replay on this hardware. Let `T` be the largest observed
seconds per four-game block and `H` the measured fixed final startup/report
overhead. Admit 64 blocks only if:

```
1.25 * T * 5 * 64 + H <= 21,600 seconds
```

The 25% timing margin is a conservative planning choice for a small pilot,
not a confidence bound. If that test fails, retain the cost result and propose
a new count/budget before scoring; do not silently reduce 64 or consume the
reserve. An accepted amendment must give its precision consequence. If even
the development cohort cannot finish in its hour, the launch is infeasible.

Sixty-four blocks are a precision/cost compromise: for a bounded block score,
the worst-case standard error is at most `0.5 / sqrt(64) = 0.0625` within one
checkpoint, roughly a 12.3-point normal 95% half-width. This is planning
arithmetic, not a promised CI, and pairing may reduce variance. Five independent
training seeds permit a cross-seed estimate but remain a small sample. This
design can detect a clear gain; it does not guarantee resolving five points.

## Matched inference and operational limits

Both players get the same CPU/thread/worker allocation, batch size one, no
cross-game batching, no background analysis, and no persistent cache across
games. The candidate makes one stochastic policy forward pass per decision
(`deterministic=False`, as exported). Search keeps its full frozen Search-64
algorithm. Count encoding, belief/setup work if any, inference, action selection
and search in decision timing; report model loading separately.

The following limits are provisional until checked on development receipts and
accepted in the freeze. Enforce them equally without fallback actions. If
Search-64 cannot qualify, amend the envelope before scoring; keep its algorithm.

| Limit | Proposal and basis |
|---|---|
| Decision latency | p95 ≤100 ms and a 1-second hard limit: the budget plan's deployment target plus a tenfold tail allowance. |
| Player-process RSS | ≤1 GiB, inherited from the budget plan. |
| Scored game | Stop at 10,000 semantic Commands or 300 wall seconds, whichever comes first: the Rules complete-game parity proposal's command cap plus a proposed stall watchdog. |
| Search rollout-cap hits | ≤1% of executed rollouts, inherited from the budget plan's scaling limit. Sum native `cap_hits` and divide by summed native `simulations`; do not average per-decision percentages. Apply separately to each training corpus, development Search games and final Search games. A cap hit is not a rules draw. |

Use the existing arena matched-root profiler on development replay roots:
**16 warm-up and 128 measured roots**, inherited from its existing profile.
Pool roots from the 80 candidate–Search development games. Select roots
round-robin across deck × observed decision-kind groups, using canonical
game/revision order within each group, without replacement. Sort groups by
deck key then decision-kind identifier, skipping exhausted groups. Use the
first 16 for warm-up and the next 128 for measurement; retain the exact root
list and record absent kinds. All five candidates and Search use this same
ordered root list. Insufficient roots fail profiling rather than duplicating
observations. Measure each of these six players **three times**, serialized,
rotating order; three passes expose timing spread cheaply without claiming a
timing CI. Charge all profiling and replay work to the development hour.
Use each player's median total decision CPU time and median p95 wall latency
across passes. Require candidate CPU time and p95 latency each **≤1.10 ×
Search's**, plus the common envelope. Ten percent is the existing arena's
latency tolerance, proposed here also for CPU timing noise. Preserve raw passes
and process RSS, and report p50/p95/max latency by decision kind in actual games.

The comparison matches **allowed resources**: a cheaper candidate may leave
some unused. Report realized CPU/wall seconds and their ratio, without claiming
equal consumed compute, FLOPs or simulations. Search augmentation requires a
separate treatment and protocol.

## Analysis and improvement criterion

For a completed game, candidate score is 1 for a win, 0 for a loss, and 0.5
only for an authoritative rules draw. A missing winner alone does not establish
a draw. Define `x[s,b]` as the mean of the four game scores in training seed `s`,
deal block `b`; `m[s]` is that seed's mean across 64 blocks, and `M` is the
equally weighted mean of the five `m[s]`. No weighting by game length, number
of decisions, deck win rate, or how often a policy was callable.

Report all five seed scores, each of the four deck/seat cells, each deck pooled
over seats, wins/draws/losses, completion and failure counts, and total costs.
Use **95% intervals** throughout, a conventional uncertainty level proposed
before measurement:

- Cross-seed interval: `M ± 2.776 * sample_sd(m) / sqrt(5)` (Student t,
  four degrees of freedom). This describes training variability conditional
  on the fixed final deals; state the small-sample/approximate-normal limitation.
- Conditional on each checkpoint: percentile bootstrap of whole four-game
  blocks. Resample globally shared block indexes when comparing seeds, keeping
  all legs together. Never bootstrap individual decisions or treat 256 games
  as 256 independent deals.
- Combined sensitivity interval: crossed bootstrap, independently resampling
  five training-seed indexes and 64 block indexes, retaining the complete
  Cartesian submatrix. This includes both sources of variation and their
  shared-deal dependence, approximately; it does not cure having five seeds.

Use **10,000 resamples**, the reserved analysis seed, and empirical 2.5/97.5
percentiles for the bootstrap intervals. This is cheap offline arithmetic and
gives more stable tails than the arena smoke's small resample count. The existing
arena analysis owner supplies resampling and summaries. Per-cell intervals are
descriptive; the procedure across five seeds is the single primary comparison.

Declare **full-game improvement** only if every condition holds:

1. All five planned training attempts produce admissible final checkpoints and
   the entire accepted final cohort completes with valid receipts and replay.
2. `M ≥0.55`, and the lower endpoints of **both** the cross-seed t interval and
   combined bootstrap interval are **>0.50**. Five points is a proposed minimum
   useful gain over the incumbent; 0.50 is equal paired performance. These are
   decision thresholds, not derived historical effects.
3. Every seed has `m[s] >0.50`, and each deck's pooled candidate score is
   **≥0.50**. These conservative safeguards prevent a single seed or deck from
   carrying an otherwise losing result; per-seat outcomes remain visible.
4. The compute/cost limits hold, with zero illegal actions, omitted offers,
   hidden-information exposures, root mutations or replay mismatches. Zero is
   an integrity requirement, not an estimated acceptable defect rate.

Report competence separately: opportunities and outcomes for Learn choices,
discard/retrieval/decline, holding interaction, passing with a playable land,
target selection and combat; retain exact decision/trace references for observed
failures. Replay every game. Do not infer a correct strategic line merely from
the selected action or a win. Existing competency fixtures may supply diagnostic
context, but old-world scores do not enter this cohort. Random/Passive controls
check gross behavior and legal completion; their win rates cannot guide tuning
or promotion, or satisfy KR 2.

Classify a valid cohort missing the strength thresholds as negative or
inconclusive. An incomplete cohort, cost overrun or integrity failure makes
the comparison infeasible or invalid.

## Failures, exclusions and retention

Retain **every scheduled and attempted execution**, including rejected loads,
training timeouts, missing outputs, exceptions, stalls and interrupted games.
List planned but unstarted games explicitly. Existing `recipe.json`,
`sources.zip`, per-game shards, `phases.json`, `receipt.json`, `candidate.json`
and arena traces/receipts remain authoritative. A report indexes these by digest
and attempt ID; it is not a second receipt store. Product validation uses the
existing AttemptStore with `automated_validation` origin; bot evaluation records
use `bot_evaluation`. No human-play claim follows from either.

A candidate crash/timeout is an operational failure, not a rules loss or draw.
Preserve its prefix and cause; report completed-only scores as descriptive,
plus bounds assigning every unresolved scheduled game 0 and then 1. Neither
bound can override the complete-cohort requirement.

A demonstrably external interruption may be retried **once per affected game**
under identical frozen bytes/seeds/configuration, within the original caps.
One retry is a proposed bounded recovery allowance, not another random draw.
Retain and link both attempts; only the authoritative completed replay supplies
the game score. Disagreement between deterministic replays or a policy/engine
defect invalidates that comparison and stops dependent scoring. A training
restart is a new attempt and is outside this five-attempt cohort.

Exclude only duplicate ingestion of the same trace identity and unrelated
origins/cohorts established by the frozen manifest, with counts and reasons.
Never exclude gameplay-caused failures or losing blocks. Do not replace training
seeds, select checkpoints from final results, relabel draws, alter sample size
after seeing scores, or stop early for success. World, ABI, recipe, baseline or
player changes require a new accepted freeze and fresh final deals; retain old
attempts and their cost.

## Existing consumers and launch gaps

Read-only inspection at base `48910ab4d17bd8150f0d4fd16c98299cfc7335d9`
identified these implementation prerequisites for the comparison Task:

| Existing owner | Required binding before scoring |
|---|---|
| `scripts/train_challenger.py`, `manabot/sim/distill.py` | The runner constructs main-deck-only `MatchHypers`. Consume the accepted authored matchup including sideboards, retain actual train/validation identities, and obtain corrected-world repeat receipts with exact source capture. |
| `load_checkpoint_agent`, `CheckpointVillain`, candidate admission | Prove ordinary corrected-world schema/content binding and full legal-offer preservation through the actual demo. No loader bypass or state-dict port for old evidence. |
| `manabot/arena/match.py`, `models.py`, existing arena runner | `play_cell` hardcodes `INTERACTIVE_DECK`; the frozen contract admits w2 and two legs. Extend the existing owner for corrected authored matches and four-leg blocks, new registrations/schedules and honest failed-attempt retention. Do not run or rewrite the frozen historical contract as though it covered this matchup. |
| `manabot/arena/players.py`, `etude/villain.py` | Bind exact Search-64 and Random/Passive semantics, stochastic policy seeding and per-game reset; verify parity on common roots including forced choices. Demo Search skips its RNG call counter on a single offer while `FlatMCPlayer` currently increments; nominally equal settings alone do not prove parity. |
| `manabot/arena/profile.py`, `rating.py`, `replay.py` | Reuse root profiling and replay; bind CPU timing, watchdogs, four-leg/cross-seed analysis and correct authoritative-draw handling in these owners. Existing rating filtering of truncated rows cannot decide this protocol's full-cohort result. |

ETU-84 changes documentation only. ETU-14, ETU-21, ETU-31 and ETU-55 remain
retired or deferred.
