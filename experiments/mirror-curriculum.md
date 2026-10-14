# Mirror curriculum (ETU-125)

The 120-update laptop pilot below is completed historical evidence. Jack Heart
authorized a separate fresh 10,000-update comparison on 2026-10-08; its protocol
is recorded in the long-comparison section below. The old eight-hour cutoff
applies only to the pilot.

Jack Heart authorized this overnight laptop screen on 2026-10-07, with no rental,
ETU-128 training, Mini displacement or GPU-cohort change. One allocation runs
from 2026-10-08 06:00 UTC through 14:00 UTC, including preparation, calibration,
interruptions, evaluation and reporting. A bounded idle-sleep assertion applies
only to this job; closed-lid execution is not promised.

## Frozen question and predictions

Does including mirrors improve Lessons learning and target cross-deck strength?
Prediction, before scoring: mirror-inclusive training improves Lessons versus
greedy Allies by 5 percentage points at the common endpoint, with at most a
5-point loss for Allies versus greedy Lessons. Treat the result as exploratory:
three independent seeds, small evaluation cohorts and two simple opponents cannot
establish general strength. A positive retention decision requires a positive
paired-seed mean Lessons cross-deck gain with a 95% seed-bootstrap lower bound
above zero and no more than a 5-point average Allies cross-deck regression.
Otherwise retain the negative/unresolved result. Game intervals resample whole
deals and do not substitute for independent training seeds.

First diagnose the unchanged ETU-118 seed-11851 update-2500 checkpoint, SHA-256
`55d3cfb0420f900c7bfcbc7341ebd3924c71dd48b951bc0b31021c0c59ce1090`.
Prediction: both policy decks gain at least 20 points when the greedy opponent
switches from Allies to Lessons. The four-cell matrix tests this directly;
matchup effects and imperfect opponent quality prevent unique causal attribution.
No claim about self-targeting is made without inspecting retained Commands.
The supplemental diagnostic uses five paired deals per greedy cell (10 games),
two per random cell (4 games), in namespace 1912500000. These small cells reserve
learning time; their intervals can remain broad. Source changes in the supplemental
instrument are recorded separately; checkpoint bytes, full rules/content/input
binding and historical primary cohorts remain unchanged.

## Training intervention

The saved baseline is the Mini's width64/depth2/heads4 semantic encoder,
masked-mean scalar critic and Ataraxos move settings, with history off and all
current authored decklists/sideboards. Both arms use 12 streams × 32 learner
transitions per update (384), one CPU learner thread, float32, identical
initialization seeds and iteration-based schedules. The stream geometry differs
from Mini's 4×64 but is identical between the randomized arms; this is not a
replication of Mini's training rate or learning curve.

- A: cross-deck only. Half the streams start Lessons, half Allies; learner
  seats alternate within each setup, giving each deck/play-order stratum 1/4.
- B: four streams each Lessons mirror, Allies mirror and cross-deck. Each
  mirror has both learner seats; cross-deck has both deck orders and seats.
  Thus matchup probabilities are 1/3 each and deck marginals are 1/2 each.

The sampling unit is a learner transition: every stream supplies the same count
per update. Fixed strata persist through native terminal auto-reset. Completed
games and opponent decisions can be unequal because game lengths differ; do not
claim game-balanced sampling. Explicit pack selection keeps mirror semantic IDs
and the complete compiled content manifest equal to the cross-deck world.

Initialization pairs seeds 12551/12552/12553 across arms; order is AB, BA, AB.
Calibration seed 12550 never enters scientific results. Twenty updates per arm
measure whole execution cost, including recovery export. Freeze a common even
multiple-of-20 target, at most 400 updates, from the slower calibration with 1.6×
headroom. Reserve evaluation from measured diagnostic seconds/game with 1.6×
headroom, at least 2,400 seconds, and 600 seconds for final reporting. Require at least 100 updates. Timing-only admission tries, in order:
three seeds with 10 greedy/4 random games per cell; two seeds with those counts;
three seeds with 6 greedy/2 random games per cell; two seeds with reduced counts.
Two seeds always means seeds 12551/12552. Reduced counts or two seeds are labeled
pilot. If none fits, stop. The smaller-cell fallback was registered after the
cross-deck timing calibration (394.925 seconds for 20 updates), before mirror
calibration completed or either scientific arm ran. Outcomes do not enter admission.
A two-seed result is descriptive and cannot trigger the retention decision.
The exact plan is recorded before training. No outcome chooses counts or seeds.

## Evaluation and reporting

All arms use the same eight-leg deal block: both policy seats in each of the
four policy-deck/opponent-deck cells. Greedy and uniform legal random source
identities, artifact hashes, actual seat decks and exact replay travel with rows.
Each initialization, midpoint and endpoint receives the same timing-admitted
per-cell counts (10 greedy/4 random preferred; 6/2 pilot fallback). Monitoring namespaces 1912510000 and 1912520000 are
held out from training and shared across arms/seeds/checkpoints. They are repeated
screen evaluations, not untouched final-confirmatory deals. Initialization bytes
are the frozen/no-update control: all gains subtract this same policy's paired
outcomes under the identical frozen evaluator and RNG plan; no fake optimizer
run is used for the control. Aggregates, when shown, weight all four cells equally.

The editable `comparison.ipynb` generates the read-only `comparison.html` using
PR262's report owner, with four cells prominently separated. Show per-cell counts,
deal-cluster uncertainty and gains over initialization; compare paired arm effects
across seeds separately. Costs and progress retain transitions, updates, native
decisions and elapsed time. Calibration and diagnostic results are labeled
separately from randomized training. No conclusion is available while seeds are
pending or the primary cells are incomplete.

## Stop and recovery

TrainingRegime/VerifyStore own optimizer, EMA, collector and recovery snapshots.
The local independent service owns an immutable absolute allocation deadline and
process-tree teardown. It cannot extend the allowance on reconnect or sleep.
The existing one-host/one-evaluator leases prevent overlap with ETU-126. Any
legality, nonfinite, semantic, replay or budget failure retains its attempt and
suppresses the affected estimate. No automatic failed-job replacement occurs.
Same-host checkpoint recovery is explicit and must fit the original deadline;
recovery failure stops that attempt. No model selection, paid compute or promotion
follows automatically from this screen.

## Execution evidence

The supplemental diagnostic completed all 56 games with exact replay. Counts:

| Policy deck | Greedy Allies | Greedy Lessons | Random Allies | Random Lessons |
| --- | ---: | ---: | ---: | ---: |
| Allies | 4/10 | 8/10 | 3/4 | 4/4 |
| Lessons | 1/10 | 7/10 | 3/4 | 3/4 |

Within-deck greedy-opponent switches favor Lessons by +40 [10,70] points for
the Allies policy and +60 [50,80] for the Lessons policy (95% whole-deal bootstrap,
five paired deals). This small sample supports the opponent-quality hypothesis;
deck matchup effects remain confounded, and no unique mechanism or self-targeting
claim follows. The randomized curriculum effect remains unavailable.

[Hash-bound raw diagnostic rows](data/etu125/diagnostic.json) retain registration,
source, setup, timing and replay identities. Original traces, producer snapshot and
checkpoint remain under `.runs/etu125-overnight`. The diagnostic's final report
call failed on missing arguments after all games completed; the corrected editable
notebook regenerated HTML from saved results without rerunning any game.

Both timing calibrations completed: cross-only 394.970 seconds and mixed
391.435 seconds for 20 updates each, seed 12550, source `b6cd4405`.
The [frozen plan](data/etu125/plan.json) admits a **two-seed pilot**, seeds
12551/12552, 120 updates (46,080 learner transitions) per arm, midpoint 60,
and 6 greedy / 2 random games per cell at all three checkpoints. Four runs use
AB/BA order. Each run has a 3,911.711-second ceiling; evaluation reserves
7,705.485 seconds. The original absolute deadline overrides every inner allowance.
The [calibration receipt](data/etu125/calibration.json) and
[allocation](data/etu125/allocation.json) retain exact values. Post-calibration
changes add admission checks and timing-based cohort selection/reporting; the
measured collector, learner, model and recovery work remain unchanged.

Independent scientific launch is prepared under
`com.manabot.etu125-training-1`; launch receipt, heartbeat and worker log live in
`.runs/etu125-overnight/training-service/`. TrainingRun and Experiment receipts
under `.runs/etu125-overnight/science/` bind the final source, world, artifacts
and actual progress. The create-once notebook and HTML live at
`.runs/etu125-overnight/comparison.{ipynb,html}`. Launch is only established by
that service's receipt and observed updates, not this preparation record. Python focused checks passed 41 tests after repairing malformed
pack admission and rerunning the source-sensitive recovery case against stable
source; the preceding suite passed 70 other checks. All 60 native library tests
passed in debug. These checks and the diagnostic are not curriculum evidence.

## Fresh 10,000-update comparison (authorized 2026-10-08)

Jack Heart explicitly authorized four local CPU runs: two curricula × paired
seeds 12551/12552, ordered AB/BA. This is independent of Mini ETU-118, the GPU
capacity cohort and ETU-128. No paid compute or remote job change is authorized.
The completed pilot remains at `.runs/etu125-overnight`, with its original source,
allocation, four 120-update runs and all 56 frozen-policy diagnostic games intact.
Pilot Lessons cross-deck endpoint differences were 0 and −16.67 points (mean
−8.33 points); six-game cells and two seeds do not establish a curriculum effect.
This negative descriptive result is not replaced by the longer cohort.

### Frozen intervention and analysis

Fresh initialization, full compiled pack/decklists, width64/depth2/heads4,
masked-mean scalar value, history off, Ataraxos move learning, 12 streams × 32
learner transitions, one CPU learner thread and float32 remain identical to the
pilot recipe and across arms. One update means 384 learner transitions, including
empty-filter updates. Each run targets 3,840,000 transitions in 10,000 updates.
The learner's global collection/update iteration owns the unchanged learning-rate
and entropy schedules; linked export stages do not reset Adam, EMA or the clock.
The current normalized-log numerical safeguards and effect diagnostics remain on.

Transition strata and exact sampling probabilities remain as specified above:
A cross-deck only, B one-third each Allies mirror/Lessons mirror/cross-deck.
Both have one-half deck exposure and balanced learner seat/play order, not
necessarily equal completed game counts. No pilot checkpoint is continued.

Primary prediction remains +5 points for Lessons versus greedy Allies with
no more than −5 points for Allies versus greedy Lessons. Report each paired seed
and their mean; two-seed bootstrap intervals are descriptive, unstable measures
and cannot authorize a retention decision. Whole-deal bootstrap cell/gain intervals
keep both seats together and quantify conditional evaluation noise separately.
Do not establish general strength, unique opponent causality or human-play success.

Evaluate raw weights at 0, 2,500, 5,000 and 10,000 updates. At every milestone,
each of the four matchup cells receives **100 greedy games (50 paired deals)**
and **10 random games (five paired deals)**. Greedy deals start at 1,912,610,000;
random at 1,912,620,000. These namespaces are separate from the pilot, held out
from optimization and shared across arms, seeds and milestones. They are repeated
screen deals, not untouched confirmation. Freeze actual opponent source/registration
identities with the source receipt. Initialization supplies each paired seed's
frozen/no-update control; gains use the same deals/seats. Any aggregate weights
all four cells equally. There are 6,400 primary and 640 random games in total.

Use the existing notebook/dashboard, per-cell counts, replay receipts and numerical
health panels. Show updates/transitions and recorded training/active time separately
from total allocation and additive worker/evaluator cost. Common-sample comparisons
use identical milestones; common-time comparisons use only the latest observed
checkpoint at/before a shared cutoff, with sparse resolution explicit. Never
interpolate an unseen score or rank unequal latest checkpoints as equal-cost.
Missing or failed primary cells suppress affected estimates; retain every attempt.

### Engineering admission and recovery

Before scientific launch, six profiled updates per arm on this laptop completed
in 34.79/35.39 seconds. Recursive process-tree inspection consumed about 20 seconds
per arm (over a million macOS process-table queries), a supervision overhead
confound rather than evidence of weak model hardware throughput. Sampling RSS once
per second preserves per-step deadline checks. The separate 20-update-per-arm
launchd calibration measures this repair at the actual Background/Nice=10 priority;
its exact timing and allocation decisions belong in the receipt below.

The old one-million-step recovery journal would stop well before 10k updates.
The new recipe explicitly bounds it at 20 million native decisions, preserving
original seed/action replay and observation equality checks. Immutable recovery
snapshots occur at update 1, every 250 updates and stage boundaries. A failure
can repeat up to 249 updates; ancestor costs remain charged. The first stage ends
at 2,500, the second at 5,000, the last at 10,000. Snapshot storage grows with
history; reserve disk and stop explicitly if exhausted. No checkpoint selection,
scientific failure replacement or schedule migration is automatic.

The launchd service survives terminal/agent exit. Its idle-sleep assertion cannot
make a closed laptop execute CPU work. Sleep pauses useful work and consumes the
same immutable calendar allocation; continuous watchdogs include it. Ordinary
wake resumes the existing process. After process death, use the existing explicit
Experiment `--resume --recover ORDINAL` path with the same source, recipe, host,
store and deadline, preserving the stopped attempt. Source/runtime drift fails
admission; never reset elapsed allowances to make recovery fit.

### Admitted allocation and source freeze

The retained [calibration receipt](data/etu125/long-calibration.json) binds both
engineering attempts and exact source/native identities. Under Background/Nice=10,
the repaired arms took 217.30/207.05 seconds for 20 updates. Under Standard/Nice=10,
they took 43.50/40.93 seconds. The selected placement is **Standard/Nice=10**,
one serial learner plus one bounded evaluator, one CPU thread each. These were
sequential short probes under differing host loads, not a clean hardware benchmark.
The profiler identifies process scanning; scheduling-class sensitivity is measured,
but its exact hardware mechanism is not established.

The slow selected short-run rate projects **24.17 training hours** over all four
10k runs. The historical diagnostic rates project another **22.91 evaluator hours**
for 6,400 greedy plus 640 random games; no unmeasured scheduling speedup is assumed
for evaluation. Growing journals/diagnostics, longer learned games and contention
can increase either estimate. A roughly two-day awake-work projection is not a
promise of sustained throughput or a few-hour experiment.

The [new plan](data/etu125/long-plan.json) and
[allocation](data/etu125/long-allocation.json) freeze:

- 48 hours per TrainingRun, including its setup/exports and sleep-inclusive
  watchdog; stage shares are 25%, 25%, 50%, with finalization reserve.
- 48 evaluator hours, at most three hours per complete checkpoint/anchor cohort.
- 48 additional worker hours reserved for explicit stopped-attempt recovery,
  plus six hours for preparation/reporting/cleanup. Total additive process cap:
  294 hours; at most 240 of these can be learner/recovery time, not a target.
- Fourteen calendar days beginning with the first new calibration:
  **2026-10-08 11:31:07 UTC to 2026-10-22 11:31:07 UTC**. Time away/sleep consumes
  this allowance. Runs stop at their 10k target even if allowances remain.
- 40 GiB planned evidence storage and a hard 20 GiB free-disk reserve. Sampled
  learner RSS remains capped at 15 GiB; snapshots contain private hidden state.

The expired pilot allocation is neither modified nor extended. All new engineering
attempts count within the new calendar window and remain retained, alongside their
failed-check logs. Only the new four-run cohort enters curriculum estimates.
A failed scientific attempt is not replaced; explicit same-source recovery retains
its lineage and spent time. No paid compute, Mini or GPU operation occurs.

Launch receipts and logs live under `.runs/etu125-10k/training-service/`; canonical
Experiment/TrainingRun records and raw arena/replay evidence live under
`.runs/etu125-10k/science/`. The editable `comparison.ipynb` generates read-only
`comparison.html`; `contrasts.json` holds counts, initialization gains, paired-seed
endpoint and sparse common-time contrasts. Refresh every 30 minutes, retaining
notebook edits. Source admission and actual launch progress are recorded in
`source-freeze.json` and `launch-evidence.json` in that directory, not inferred
from this protocol. No 10k learning result exists at preregistration.

Focused recovery/numerical/matchup checks passed, including exact sparse recovery;
authoring/execution follow-up passed 30 checks, including notebook regeneration.
The earlier sparse-test guard and missing authoring-owner failures were repaired
before scientific launch. Existing Rust and full CI results on PR264 predate these
Python-only follow-ups; no new native behavior is claimed.

## Delivery reconciliation (2026-10-10)

Jack Heart requested PR264 conflict repair, green CI and landing. Before changing
the source-bound checkout, inspection found no remaining cohort process. The
Experiment export retains four completed trajectories (attempts 0, 1, 2, 5),
two disk-reserve failures (3, 4), and overall `incomplete` status. Monitoring
contains 43 completed records and one incomplete record. These counts do not
certify a complete scientific cohort; reconciliation of scheduled evaluations,
costs, replay and final analysis remains separate from software delivery.

Frozen `9649f42f` source is archived at `.runs/etu125-10k/frozen-source/`, alongside
`pre-delivery-notes.patch`. Original attempts, recovery logs, notebooks and
contrasts remain unchanged. The current integration is not an admitted source
for resuming those runs. The retained notebook/HTML and endpoint contrasts are
provisional until the incomplete evidence is reconciled.

Main's mirror-only arena and this comparison now share `matchup_mode`: `cross`
retains historical legs 0–3, `mirrors` selects 4–7, and `all` selects 0–7. The
frozen monitoring protocol's `include_mirrors` selects `all`; it does not alter
historical protocol identities. Both paths use the same deck/seat layout and
explicit checkpoint roster admission.
