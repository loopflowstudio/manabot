# Current learning baseline — ETU-118

Status: the selected 3 × 10,000-update mini cohort launched on 2026-10-07 after
current-source calibration and admission, and all three seeds completed training
on 2026-10-09. Development-deal monitoring shows every seed improving against both
anchors; the pre-set criteria are **not yet judged** because the held-out final
comparison was still running when the results below were written. Historical
failures remain charged. No baseline promotion or daily-test validation has occurred.

## Sustained mini cohort results (2026-10-09)

Read at 2026-10-09 10:09–10:19 UTC from a JSON-only copy of the mini run
directory (`.runs/etu118-mini-20261007-1/science-20261009` in this checkout;
`experiment.json` SHA-256 `a6f05d03…cdbf530`, `monitoring/dashboard.json`
`f5f53796…2882b0c9`). No process was signalled and `experiment.sqlite` was not
opened. "From files" below means a value copied from those records; "derived"
means arithmetic over their rows, done for this report and not by the frozen
analysis.

### State of the run

- From files: `experiment.json` is `completed`, three attempts `completed`, no
  error, downtime and uncertain seconds both 0. The monitoring queue finished
  90 of 90 cohorts with none pending.
- The experiment stayed `running` for about 12 minutes after the last learner
  finished (09:56:50 UTC) only because the last queued monitoring cohorts were
  still playing. Nothing was stalled.
- The coordinator (mini PID 86658) then entered the predeclared final comparison
  on its own, as `run` is written to do. `final-result.json` is `running` from
  10:09:03 UTC, bound to plan `46aabdb0…034bb7`. It plays 24 cells of 400 games:
  greedy first, then random; each of initialization, 800, 1240 and 10000 for each
  seed. The first cell covered 49 of 100 deals in 9.6 minutes, so roughly 20
  minutes per cell and about 8 hours in all (inferred from that one cell; near
  18:00–20:00 UTC). No final cell had completed at the time of writing.
- The supervisor's HTML refresh stopped at 2026-10-08 19:21 UTC after 298
  successful refreshes because it had used 3480.5 of its 3600-second reserve; this
  is the designed bound, not a failure. `finalize` refreshes once more at its end.
- If the final comparison is interrupted, the retained-evidence resume is, from
  the frozen bundle on the mini (do not run while PID 86658 lives):
  `cd /Users/jack/src/etude/.runs/etu118-mini-20261007-1/source && .venv/bin/python -m experiments.runners.sustained_baseline finalize ../plan.json ../science --resume`

### What ran

From files (stage totals in each seed's `training-dashboard.json`, attempt process
seconds in `experiment.json`); rates derived.

| Seed | Updates | Learner transitions | Optimizer exposures | Self-play games | Active hours | Transitions/s | s/update |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11851 | 10,000 | 2,560,000 | 624,379 | 34,623 | 12.98 | 54.8 | 4.67 |
| 11852 | 10,000 | 2,560,000 | 631,502 | 33,720 | 14.16 | 50.2 | 5.10 |
| 11853 | 10,000 | 2,560,000 | 636,151 | 35,995 | 13.21 | 53.9 | 4.75 |

Seeds ran one after another on one learner thread with the evaluator overlapping.
The filter retained 24.4–24.8% of collected rows for optimization and no update
was skipped. Learning rate followed the absolute-iteration schedule from 1e-4
through update 1240 down to 1.99e-5 at update 10000. Collection and learning each
took about half of training time. Whole experiment: 146,016.7 elapsed seconds
(40.56 hours), 171,708.6 additive process seconds including 26,431.3 evaluator
seconds, against the 514,044.9-second schedule allowance and the 86,400-second
monitoring reserve. Actual pace was 4.7–5.1 s/update against the admitted
conservative 8.455.

### Learning curves on monitoring deals

Each cell is one 100-game cohort (25 deals × 4 deck/seat legs), policy sampling
one stochastic forward pass. Per-seed brackets are the saved 95% deal-block
bootstrap interval (from files). The mean and the pooled interval (resampling the
25 shared deals, conditioning on the three trained seeds) are derived. All 9,000
monitoring games terminated, replayed exactly, and had no truncations or draws.

Win rate (%) against scripted greedy:

| Update | 11851 | 11852 | 11853 | Mean | Pooled deal interval |
| --- | --- | --- | --- | --- | --- |
| 0 | 30 [24, 36] | 31 [25, 37] | 30 [24, 36] | 30.3 | [24.0, 36.3] |
| 400 | 41 [36, 46] | 42 [37, 46] | 28 [21, 35] | 37.0 | [32.3, 41.3] |
| 620 | 43 [38, 47] | 46 [42, 50] | 39 [33, 44] | 42.7 | [39.0, 46.0] |
| 800 | 44 [39, 49] | 44 [39, 49] | 40 [34, 45] | 42.7 | [38.3, 46.7] |
| 1240 | 47 [43, 51] | 46 [41, 50] | 44 [39, 49] | 45.7 | [41.7, 49.3] |
| 2500 | 46 [42, 49] | 47 [43, 51] | 45 [41, 48] | 46.0 | [42.7, 49.0] |
| 5000 | 47 [41, 54] | 53 [49, 57] | 45 [39, 51] | 48.3 | [44.3, 52.7] |
| 7500 | 55 [47, 63] | 52 [46, 59] | 56 [50, 63] | 54.3 | [48.7, 60.0] |
| 10000 | 53 [47, 59] | 51 [43, 59] | 59 [52, 66] | 54.3 | [49.7, 59.0] |

Win rate (%) against uniform legal random (separate deals 1911185000–1911185024):

| Update | 11851 | 11852 | 11853 | Mean | Pooled deal interval |
| --- | --- | --- | --- | --- | --- |
| 0 | 47 [40, 54] | 44 [36, 52] | 46 [39, 53] | 45.7 | [38.3, 52.7] |
| 400 | 67 [60, 74] | 70 [62, 78] | 64 [56, 72] | 67.0 | [62.3, 72.0] |
| 620 | 64 [58, 71] | 68 [61, 75] | 71 [64, 78] | 67.7 | [63.7, 71.7] |
| 800 | 64 [58, 70] | 76 [69, 82] | 76 [70, 82] | 72.0 | [68.0, 76.3] |
| 1240 | 76 [70, 82] | 71 [64, 78] | 78 [71, 85] | 75.0 | [70.7, 79.3] |
| 2500 | 81 [76, 86] | 79 [71, 86] | 78 [71, 84] | 79.3 | [74.0, 84.7] |
| 5000 | 88 [81, 94] | 84 [76, 90] | 77 [68, 86] | 83.0 | [78.0, 88.0] |
| 7500 | 86 [81, 91] | 85 [78, 92] | 87 [82, 92] | 86.0 | [82.0, 90.0] |
| 10000 | 89 [83, 94] | 89 [83, 94] | 86 [79, 92] | 88.0 | [83.7, 92.0] |

Greedy was also scored at roughly hourly exports (20, 22 and 21 cohorts per
seed). Averaging each seed's cohorts from update 5000 on gives 50.6, 51.9 and
52.7%, against 45.9, 47.3 and 47.0% for updates 1240–4999 (derived). Single
cohorts in that late range span 45–59%, so one endpoint cohort is uncertain by
several points.

### The headline number is mostly a deck matchup

Derived from the same rows. Wins per leg are per seed (11851/11852/11853, each
out of 25) and pooled out of 75. Seat is the recorded `player_a_seat`; which seat
acts first was not checked.

Against scripted greedy:

| Update | Allies, seat 0 | Allies, seat 1 | Lessons, seat 0 | Lessons, seat 1 | Allies | Lessons |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 13/13/13 = 39 | 16/17/16 = 49 | 1/1/1 = 3 | 0/0/0 = 0 | 58.7% | 2.0% |
| 800 | 18/19/17 = 54 | 23/23/20 = 66 | 3/2/3 = 8 | 0/0/0 = 0 | 80.0% | 5.3% |
| 1240 | 22/21/20 = 63 | 23/23/23 = 69 | 2/2/1 = 5 | 0/0/0 = 0 | 88.0% | 3.3% |
| 2500 | 22/21/21 = 64 | 23/23/21 = 67 | 1/3/2 = 6 | 0/0/1 = 1 | 87.3% | 4.7% |
| 5000 | 19/23/18 = 60 | 22/23/21 = 66 | 5/3/4 = 12 | 1/4/2 = 7 | 84.0% | 12.7% |
| 7500 | 20/21/22 = 63 | 23/23/23 = 69 | 8/7/6 = 21 | 4/1/5 = 10 | 88.0% | 20.7% |
| 10000 | 21/19/21 = 61 | 24/22/24 = 70 | 7/4/10 = 21 | 1/6/4 = 11 | 87.3% | 21.3% |

Against random:

| Update | Allies, seat 0 | Allies, seat 1 | Lessons, seat 0 | Lessons, seat 1 | Allies | Lessons |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 15/14/15 = 44 | 15/15/15 = 45 | 9/8/9 = 26 | 8/7/7 = 22 | 59.3% | 32.0% |
| 800 | 24/24/24 = 72 | 24/24/24 = 72 | 5/14/12 = 31 | 11/14/16 = 41 | 96.0% | 48.0% |
| 1240 | 25/24/24 = 73 | 25/24/24 = 73 | 15/11/13 = 39 | 11/12/17 = 40 | 97.3% | 52.7% |
| 5000 | 25/25/23 = 73 | 25/24/21 = 70 | 20/15/17 = 52 | 18/20/16 = 54 | 95.3% | 70.7% |
| 10000 | 25/24/25 = 74 | 25/24/24 = 73 | 20/21/20 = 61 | 19/20/17 = 56 | 98.0% | 78.0% |

What this shows:

- An untrained policy already wins 59% piloting Allies against either anchor, and
  2% piloting Lessons against greedy. The 30% starting point is the average of
  those two, not a measure of weak play in general.
- Piloting Allies against greedy saturated near 87–88% by update 1240 and did not
  move afterwards. Every point of greedy gain after 1240 came from piloting
  Lessons: 5/150 wins at 1240, 19/150 at 5000, 32/150 at 10000.
- Against random, Lessons piloting rose steadily from 32% to 78%; Allies was
  already at 96% by update 800.
- Seat matters less than deck. Against greedy the policy did better in seat 1
  with Allies (70 vs 61 of 75 at the end) and in seat 0 with Lessons (21 vs 11);
  pooled over decks the seats are level (54.7% vs 54.0%).

### Seed-to-seed spread

- At the endpoint the three seeds span 51–59% against greedy (SD 4.2 points) and
  86–89% against random (SD 1.7). Late Lessons-versus-greedy wins differ most:
  8, 10 and 14 of 50.
- The three initialization cohorts are not independent measurements. Different
  initial weights, same deals and action seeds: game outcomes agree in 99–100 of
  100 games against greedy and 97–99 against random (derived). The initial score
  is effectively one 100-game sample, so "every seed gained" is weaker evidence
  than three independent starting points would give.
- Trained endpoints agree on 83–87 of 100 outcomes between seeds, so the seeds
  did diverge in play.

### Pre-set criteria

The criteria are defined on the held-out final deals 1911184000–1911184099, 400
games per cell. Those cells are still being played, so **no criterion has passed
or failed**. The table applies the same rule (mean of per-seed gains; 95%
percentile bootstrap over the three seeds, generator seed 118 as in
`sustained_baseline_result.analyze`) to the monitoring cohorts instead. This is a
derived preview on inspected development deals with a quarter of the games. It
is not the predeclared test and cannot substitute for it.

| Criterion (greedy unless noted) | Threshold | Monitoring-deal gains per seed | Mean | Seed interval | Held-out verdict |
| --- | --- | --- | --- | --- | --- |
| Gain over initialization | ≥ 10, every seed > 0, lower bound > 0 | +23, +20, +29 | +24.0 | [+20, +29] | pending |
| Gain over update 800 | ≥ 5, lower bound > 0 | +9, +7, +19 | +11.7 | [+7, +19] | pending |
| Gain over update 1240 | ≥ 5, lower bound > 0 | +6, +5, +15 | +8.7 | [+5, +15] | pending |
| Random change from initialization | ≥ −5 | +42, +45, +40 | +42.3 | [+40, +45] | pending |

On monitoring deals all four would be met. The gain over initialization and the
random criterion have wide margins and are unlikely to reverse. The two
later-reference criteria are less secure: seed 11852's gains of +7 and +5 points
are inside the ±5–8 point interval of a single 100-game cohort, and the pooled
deal-block interval for the gain over 1240 is [+4.0, +13.7]. Whether training
past update 5000 still helps against greedy is not established here: gains from
5000 to 10000 are +6, −2 and +14.

### Limits

- Monitoring deals were visible throughout the run and are 25 deals per cohort.
  The held-out comparison is the evidence the protocol accepts; it is unfinished.
- Three seeds. With n = 3 the seed bootstrap's lower bound equals the smallest
  seed's gain, so "positive lower bound" adds nothing beyond "positive in every
  seed". The interval says little about other seeds.
- Scripted greedy is a weak and lopsided reference. It loses 59% of games to an
  untrained policy when it holds Lessons and beats that policy 98% of the time
  when it holds Allies, so
  results against it mostly measure which deck the policy holds. These records
  cannot separate deck imbalance from greedy piloting Lessons badly.
- Random is weaker still, and the Allies legs against both anchors are near
  ceiling, leaving only the Lessons legs able to show further progress.
- All evaluation is sampled play by raw weights against two fixed anchors on one
  two-deck matchup. Nothing here measures strength against people, against other
  trained policies or with other decks.
- The local HTML regeneration used this checkout's report code at `05da79a9`,
  not the frozen bundle `ba0ae3e9`; the mini's last own refresh is kept beside it
  as `comparison.mini-20261008T1221.html`.

### Still open

Held-out final comparison and its `conclusion.json`; the three demo-loaded
executions with both deck assignments; S3 model publication (credentials were
unavailable at launch and have not been rechecked); W&B projection for seeds
11852 and 11853 (not checked); daily-test derivation and independent validation. None of these
is claimed.

## Current mini execution direction (2026-10-07)

Jack Heart selected ETU-105's completed actor-and-critic-filtered masked-mean scalar
width64/depth2/heads4, no-history current-self recipe for three independent seeds
× 10,000 updates. This supersedes the first-screen depth1 candidate and automatic
12,800–102,400 horizon selection below. The choice is informed by early gains in
all three ETU-105 seeds and no demonstrated equal-cost payoff for actor-only
filtering. It is a conservative experiment, not a promoted baseline or an optimality
claim. Optional all/none Ataraxos screens belong to ETU-105 and do not gate this run.
ETU-116 duration and ETU-82 repeated training/export/play outcomes remain here;
ETU-85 retains demo-opponent/human comparison. Software delivery alone cannot
complete ETU-118.

### Handoff, placement and retained allocation

The mini is `jack@100.96.227.95`, repository `/Users/jack/src/etude`; orchestration
uses `lf ssh`. A remote supervisor must survive laptop closure. Jack Heart released
ETU-105 CPU after all twelve final/random evaluation cells completed. Read-only
inspection confirmed exit-zero/completed rows in
`.runs/etu105-mini-filter-scope-final-20261007-1/supervisor.json`, SHA-256
`52970cabb661dc96d2d434d2ef5ec0d24b0095d5544b6496274fa883908b9548`.
At 2026-10-07 16:44:39 UTC none of its twelve child PIDs appeared and neither
shared experiment/evaluator lease had an open owner. Fresh lease acquisition still
owns launch admission. ETU-103 GPU work and ETU-105 retained source/evidence remain
untouched. Host-local serial mini execution needs no shared-laptop fairness redesign.

The original 604800-active-second ceiling retains 3030.011508249935 seconds of
preparation, leaving **601769.9884917501 seconds before new charges**. There is no
fresh week, paid provisioning or duplicate laptop cohort. Known pause/sleep is
separate; unknown restart gaps remain conservative charges. Experimental setup,
matched-prefix checks, calibration, failures, training, monitoring, recovery and
final reporting all consume this balance. Charge maximum elapsed/additive process
occupancy once per attempt and retain both observations, not their sum.

Mini free space at inspection was 60,449,624 KiB (57.65 GiB). The earlier 195.33 GiB
readmission was on the laptop. Neither that laptop projection nor this mini free
space observation replaces current mini source/runtime/storage admission. Preserve
all originals and a 4 GiB reserve; insufficient capacity never authorizes deletion
of another campaign's evidence.

### Calibration amendment fixed before execution

Jack Heart delegated amendment within the remaining week. This execution design
reserves at most **7200 new active seconds** for one selected-recipe mini calibration
and preparation: unchanged timing seeds 11841–11843 × 32 updates, 1800 seconds per
learner, 1200 evaluator seconds (200 per initial/endpoint four-game cohort), and
600 seconds for bounded matched-prefix/source checks and coordination/reporting.
The 1800-second watchdog replaces the unsuccessful 220-second bound, not the
learning recipe or sustained endpoint. Actual costs are charged; unused reserve
remains available. Full consumption leaves 594569.9884917501 seconds before other
new charges. No automatic retry or overwrite of prior attempts is authorized.

This amendment replaced the old 3600-second exploration ceiling for this new
attempt only; all prior charges and historical receipts remain unchanged. The
runner implements the 7200-second allowance; the execution receipt below records
its completed calibration and preparation charges.
Calibration scores cannot select recipes, seeds or endpoints. Jack Heart's roughly
31-hour training estimate is a planning estimate, not current-source calibration.

### Exact recipe and schedule semantics

The source-bound authority is the actor-critic regimes in
[data/etu105/filter-scope-mini.json.gz](data/etu105/filter-scope-mini.json.gz), with
[original source provenance](ataraxos-technique-screen.md#sources-and-existing-evidence).
The complete learning configuration, model, authored decks/sideboards, observation
capacity, raw current-self behavior and Adam ownership carry forward. Four streams
× 64 transitions remain 256 collected rows/update: 2.56M per 10,000-update run.
All original ETU-105 artifacts remain immutable; its parent-discovered Git commit
is not the exported source identity.

The serialized historical recipe says `schedule_clock=run_elapsed_budget`, but
its Ataraxos move rule actually calls `learning.rates(iteration)`. Retained
learning diagnostics use the one-based collection/update count, 1–620 then
621–1240, with no reset and including skipped updates. LR and tau ignore the
wall-budget fraction on this path. Extending the watchdog therefore need not
stretch this schedule. Preserve the serialized meaning and actual behavior;
changing a field to imply equivalence is insufficient.

The 2026-10-07 read-only audit compared all 3,720 retained actor-critic update rows
against current `AtaraxosMoveLearning.rates`: LR/tau, iteration sequence, linked
620+620 stages and 256-row collection settings agree. It performed no training
and does not establish source-wide or recovery-path equivalence.

The first stages remain linked 620 + 620 updates. Learner, Adam, EMA, live collector
and iteration continue at the boundary. Later linked stages reach 2500, 5000, 7500
and 10000. Scoring at 400/800 uses monitoring exports, not inserted stage boundaries.
Current initialization/recovery differs from the historical source; a bounded
matched-prefix comparison of learner/RNG/collector/continuation behavior is required
before launch. Formula agreement is not complete training equivalence. Statistical
reproducibility is the contract, not byte-identical stochastic runs across hosts.

### Frozen cohort, monitoring and endpoint criterion

Training seeds are 11851–11853, independent of timing and ETU-105 seeds. Freeze
source/native/runtime/world, full resolved recipe, initial checkpoints, exact
opponent/inference identities, deals, clocks, budgets and stopping rules before
scoring. Current-source mini calibration must admit the fixed 10,000-update horizon
using the existing conservative 1.5× slowest timing rule and measured retained
storage, including format-3 recovery and evaluation. Failure cannot silently shorten
or enlarge the cohort. The monitoring and final/reporting reserves remain 86,400
seconds each within the original week; remaining learning is divided across seeds.

Greedy monitoring uses deals 1911183000–1911183024 × all four deck/seat legs:
100 games at initialization, updates 400/620/800/1240/2500/5000/7500/10000 and roughly
hourly exports where practical. Duplicate artifact/coordinate triggers are scored
once. A separately scheduled live random diagnostic uses 1911185000–1911185024,
100 games at initialization and stage endpoints. Both run during training through
the shared evaluator; opponent-specific curves must remain separate. These are
inspected development deals. No monitoring score selects a checkpoint or stops
training automatically; a flat curve prompts review.

Untouched final deals 1911184000–1911184099 retain four legs each against both
anchors at initialization, 800, 1240 and 10000 (24 complete cells). Preserve the
historical criteria: mean greedy gain over initialization at least 10 points,
positive in every seed and positive lower 95% paired training-seed bootstrap bound;
at least 5 points mean gain over update800 with positive lower paired-seed bound;
random mean regression no worse than 5 points from initialization. Add the same
5-point/positive-lower-bound requirement versus update1240 to demonstrate progress
beyond the selected recipe's early endpoint. This prospective addition does not
weaken any historical threshold. Every failure and incomplete cell remains visible
and suppresses acceptance; three seeds remain exploratory.

Retain all checkpoints/seeds, paired changes, seed and deal uncertainty, plateaus,
regressions, transitions, optimizer exposures, throughput, losses, entropy and full
phase costs. Complete-state recovery preserves committed updates/exports; repeated
post-snapshot work retains cost. Manual pause commits a learner boundary and drains
the current bounded cohort. Physical lid closure remains untested. Three ordinary
demo-loaded executions and complete games with both deck assignments are still
required; arena replay alone does not satisfy that acceptance.

Experiment, TrainingRun/VerifyStore and the arena retain execution authority.
Create-once editable notebook cells generate read-only HTML with live progress,
strength and freshness; W&B is a metric projection. The existing
[S3 contract](../docs/training.md#monitoring) and
`manabot/training/artifacts.py` own private model publication and verified readback.
The contract is now located; mini credentials and actual archival remain unverified.
Model publication is explicit for stopped/completed runs, not a new live recovery
service. Daily-test derivation and independent validation follow sustained evidence.

### Implementation boundary

The runner now consumes `sustained_baseline_source()` in `experiments/ataraxos_mini.py`, extracted
unchanged from all three retained actor-critic recipes. It fixes 10000 updates,
620/1240 linked early boundaries, explicit 400/800 exports, separate live random
monitoring and 24 final cells. The 7200-second calibration amendment is implemented.
Ataraxos move recovery admits the historical serialized clock because its learning
rates use absolute iterations; other elapsed schedules remain unsupported.

Detached supervision refreshes the editable notebook every five minutes when
available, with a 120-second attempt cap and 3600-second cumulative report reserve
inside the unchanged 86400-second final/report allocation. Failed refreshes remain
visible and never restart training. Seed and paired-deal gain intervals are saved
separately; the deal intervals condition on the fixed trained seeds.

### Mini execution receipt (2026-10-07)

The [calibration manifest](data/current-baseline-mini-calibration.json) and
[byte-preserving archive](data/current-baseline-mini-calibration.zip) retain 73 JSON
receipts: source comparison, all calibration runs/cohorts, failed launch intent,
allocation and frozen plan. Full native checkpoints/databases remain on mini at
`/Users/jack/src/etude/.runs/etu118-mini-20261007-1`, with a local retained copy under
this Task's `.runs/etu118-mini-20261007-1/retained`. Do not delete either source or
evidence directory while training or archival remains open.

Frozen execution source is `ba0ae3e9ef9b99df7f84d4e0fa43cc640d5341f6`, with the
exact native library and a separate locked Python 3.12 environment. The four-update
historical/current prefix used two linked two-update stages, the same 256-row
learning recipe and one runtime. Batches, learner/Adam states, Torch/minibatch RNGs
and diagnostics matched exactly at every update, including continuation and extra
monitoring exports. World/setup identities agree; engine/source hashes differ.
This is a bounded equivalence check, not full 1240-update reproduction.

All three timing seeds completed 32 updates. Six initial/endpoint cohorts completed
24 games with exact replay. The attempt used 435.142165 elapsed seconds and
521.313901 additive process seconds, including 102.759537 evaluator seconds.
The admitted conservative rate is 8.455201 seconds/update: 70.46 projected training
hours across 30,000 updates, not the earlier roughly 31-hour planning estimate.
Storage projection is 6,198,501,227 bytes against 59,870,167,040 free, retaining the
4 GiB reserve. This is short-run extrapolation, not a promise of long-run speed.

The partial local bundle failed on the repository's `CLAUDE.md` symlink and remains
retained; internal file links are now materialized. A first mini calibration
coordinator failed before any learner because its supplied PATH omitted `/usr/sbin`
and boot admission could not find `sysctl`. Its evidence remains in `calibration/`,
charged 128.339317 seconds conservatively. Correcting only the launch environment
created `calibration-2/`; no learning seed was restarted. Precalibration preparation
was 615.473018 seconds versus its planned 600-second overhead subreserve. This
operational overrun and the separate PATH failure are disclosed and charged;
actual new preparation remained within the 7200-second amendment, without a new
allocation. All earlier laptop failures remain charged too.

Preparation totals 4355.137744 seconds, including a conservative 60-second
admission/tracker allowance, leaving 600444.862256 active seconds of the original
604800. Plan SHA-256 is
`46aabdb0ca584d62cc2264fde9766914750d3befd696970d0428678bae034bb7`.
It freezes the exact recipe, seeds 11851–11853, monitoring/final deal identities,
linked milestones, analysis, source/runtime, time and storage bounds.

The launch SSH connection returned after starting supervisor PID 86655 at
2026-10-07 17:35:22 UTC. Its independent coordinator is PID 86658. The supervisor
has refreshed the create-once notebook into `science/comparison.html`; refresh
attempts/costs are in `science.supervisor.json`. The first 100-game greedy
initialization cohort completed with exact replay while learning continued;
random monitoring is separate. The [launch verification receipt](data/current-baseline-mini-launch.json) records
seed 11851 at 128 committed updates, 32,768 learner transitions and 7,634 optimizer
sample exposures after disconnect. `load_update` verified the format-3 recovery
artifact at iteration 128 against canonical VerifyStore. Both 100-game initialization
cohorts completed with exact replay. A consistent SQLite backup, run export and
cohort receipts are retained in `launch-proof/`. Seeds 11852/11853 remain scheduled.
These counts prove real execution, not improvement.

S3 publication remains unavailable: the default AWS chain has no credentials and
configured `softmax` credentials fail token retrieval. No login or credentials
were changed. W&B accepted all 128 retained diagnostic rows in
[the seed-11851 projection](https://wandb.ai/loopflow-studio/etude/runs/training-ed8e425c3663727830077360).
That is a verified backfill; it is not an unattended publisher for later seeds.
The detached supervisor continues notebook/HTML refreshes. Sustained
improvement, final scoring, three demo-loaded executions, model archival and
independent daily-test validation remain unfinished; this launch does not promote
a baseline or complete ETU-118.

## Weekly-first scope revision (2026-10-06)

Jack Heart clarified WEEKLY → DAILY during the bounded full-game run. Establish a
serious sustained improving trajectory first, then validate a short daily test
against it. This supersedes any baseline-promotion wording in the original debug
protocol below. Its frozen thresholds still score that debug cohort; they cannot
promote a baseline, choose a recipe for fast early gains, or complete ETU-118.
The original debugging cohort and its thresholds remain intact. Sustained hardware/horizon/budget require exact
recipe calibration and coordination with ETU-116. Jack Heart subsequently
authorized a week on the laptop, subject to the recovery requirements below. The current executable protocol and retained outcomes are below.

## Frozen positive control

Question: does the historical masked-mean candidate learn to direct lethal damage
at the opponent through the ordinary RL path?

`uv run python -m experiments.runners.current_baseline --plan experiments/regimes/current-baseline-positive.json`
exports the Experiment declarations, seeds, held-out deals and thresholds. Execution:
`uv run python -m experiments.runners.current_baseline --run .runs/etu118-positive-1`.
Use a fresh destination; existing attempts are never overwritten or retried.

The starting recipe is the exact ETU-106 historical scalar baseline with only
masked pooling selected: width64, heads4, depth1, no history, Ataraxos move,
quantile .75, minimum advantage .01, actor-and-critic filtering, LR upper bound
.0001, policy/value traces .5/.8, collection KL .1 and action-type reference.
Architecture and learning settings remain fixed. Workload becomes 100 updates,
four streams, 16 terminal transitions per stream (6,400 per execution).

`lethal-target-v1` keeps the exact authored Allies/Lessons decks and semantic
catalog. The UR player alternates seats. The native scenario surface clears hands,
sets both life totals to 1–3, adds 3–4 existing Mountains to UR's battlefield and
puts its existing Igneous Inspiration in hand. A fixed preparation casts it;
the learner chooses between the two legal player targets. Fixed priority passing
then lets the engine resolve the spell to terminal. No reward is fabricated.
This is a target-choice task with scripted preparation/continuation, not a learned
complete-game policy. Changing seat reverses the winning positional action index;
visible life/mana, physical card identities and library order vary with seed.
The model sees ordinary viewer-safe encoded observations, no answer feature.

Seeds 11801, 11802, 11803 each run learning and no-update collection controls.
Initialization and frozen endpoints must have identical weights and paired
sampled evaluation outputs. Raw trained exports must change weights and reload
through ordinary admission. Each evaluated artifact plays 64 held-out seeds
1911180000–1911180063 in both seats (128 terminal games). Action RNG is matched
across artifacts; every selected action and terminal digest is replayed from its
root. Evaluation games quantify checkpoint noise; three training seeds are the
method-level replicates. No seed replacement or checkpoint selection.

Prediction and pass rule, fixed before scoring: **each seed** reaches at least
85% sampled terminal wins and improves at least 25 percentage points over its
initialization and frozen control. All games must terminate legally with exact
replay. Failures, empty filtering and negative results stay in the record; failure
does not authorize changing this threshold. Constant action-index play achieves
50% on the seat-balanced task. Report each seed plus paired-seed uncertainty;
three seeds do not establish transfer or broad robustness.

One learner CPU thread, 900 seconds per complete attempt including reload/scoring,
3,600 seconds aggregate including all failed attempts. The regime reserves 780
seconds for training with a 760-second stage bound. No paid compute or other
campaign intervention. The native build is a setup operation, not training.
Host load fell from 41.73 to 10.46 during preparation; these runs cannot calibrate
uncontended throughput. A new full-game allocation must be frozen before that
level; positive-control failure requires diagnosis before full-game debugging.

## Execution and evidence ownership

Experiment resolves complete regimes and existing execute_regime / VerifyStore
owns learning, artifacts and costs. SeatRoutedCollector, NetOpponentTrainer and
the existing Ataraxos update are unchanged in their estimator meaning. A serial
native-root buffer implementation supplies genuine terminal transitions and
respects paused streams. The asynchronous Experiment monitor only admits ordinary
Allies/Lessons games, so this root evaluator is explicit rather than relabeling
fixture outcomes as arena games. Retained JSON binds checkpoint hashes, every
root/action/terminal digest, failures and elapsed cost.

## Debugging acceptance and remaining outcome

The original positive-control and short full-game cohorts completed; their results
are retained below. The editable notebook generates the debugging evidence HTML.
These checks cannot promote the candidate. Sustained learning, followed by an
independently validated daily test, remains the main outcome.

## Frozen full-game debugging protocol

After the complete positive control passes, the next bounded level uses the
unchanged authored Allies/Lessons setup and masked-mean learning settings, fresh
seeds 11821–11823, and **128 updates × 4 streams × 64 transitions**. The environment
returns to ordinary native full-game reset/step; the training opponent is fixed
uniform random, not current-self. This isolates learning before self-play.
Initialization and full-length no-update controls remain paired per seed.

Freeze 12 new held-out deals 1911181000–1911181011, all four deck/seat legs, for
48 games per checkpoint against source-pinned scripted greedy. This opponent is
an evaluation reference, not the training opponent. Three seeds × initialization,
trained and frozen endpoints means 432 complete arena games. Each policy uses
one CPU thread, one stochastic forward pass and the same action-seed aliases.
The arena retains canonical Commands, exact replay, truncations and failures.
No search, best-checkpoint selection or training on evaluation deals.

Allocate at most **2,700 seconds** total for this level and **900 seconds per
seed/control attempt**, including evaluation; cap training at 600 seconds per
run / 580 per stage. Positive-control costs plus this allocation must fit the
original 3,600-second exploratory ceiling. Stop without replacements on failure.
These are conservative execution allowances, not measured throughput promises.

Prediction: the mean paired trained-minus-initialized score is at least 10 points.
The original debugging criterion requires that mean gain, positive gain in every seed, identical paired
frozen/initial outputs, all games replaying, and a positive lower 95% paired-seed
bootstrap bound. Report every seed and seat/deck leg regardless of the result.
Three seeds and one scripted reference are only a reproducible small-matchup
learning control, never a chapter-strength or self-play claim. Failure preserves
the narrow target-task result; neither debugging outcome permits baseline promotion.

## Retained failure and protocol amendment (2026-10-06)

The first attempt at commit `14597e3e`
stopped after 52.36744591698516 seconds. Seed11801 reached 100% from 50%; its
frozen control stayed at 50%. Seed11802 failed after six updates because seed22243
surfaces Discard before Priority. No third seed ran. These partial observations
cannot satisfy the cohort rule. Originals remain in `.runs/etu118-positive-1`.

Preparation now legally executes initial non-priority decisions until Priority
before injecting; no deal is dropped or reseeded. The fixed seeds, 100 updates,
held-out cohort and thresholds are unchanged. Explicit recovery is a fresh full
cohort, not continuation or replacement of the failed seed, with all previous
52.36744591698516 seconds deducted from the 3,600-second ceiling:

`uv run python -m experiments.runners.current_baseline --run .runs/etu118-positive-2 --prior-result .runs/etu118-positive-1/result.json`

The amendment follows observed partial results and is not independent confirmation.

## Sustained-run ownership and daily validation

Jack Heart's 2026-10-07 mini decision supersedes the earlier laptop placement
and automatic horizon selection. The accepted cohort is three 10,000-update
seeds of the ETU-105 actor-and-critic-filtered, depth2 masked-mean scalar recipe.
ETU-116 duration and ETU-82 repeated training/export/play outcomes are consolidated
here. Calibration, source freeze and training completed; the dated mini receipts
above own their identities and costs. ETU-125 owns the next Mini workload;
no automatic ETU-118 extension or duplicate laptop study follows.

The original seven-active-day allocation remains nonrenewing, including every
preparation failure, monitoring, recovery and final/reporting charge. Complete
learner, optimizer, EMA, RNG and collector recovery retains attempt lineage and
committed work. Bounded interruption/resume checks passed; physical lid closure
remains untested. Unknown restart gaps remain conservative charges. The observed
40.56-hour training/monitoring experiment is distinct from additive process cost
and the still-unsettled final/reporting cost.

The frozen protocol above retains initialization, early and later milestones,
fixed greedy/random anchors, paired deck/seat/deal blocks, separate development
and final deals, and all seed/checkpoint identities. Its thresholds remain fixed.
Monitoring improvements alone cannot establish held-out acceptance or ordinary
demo admission. Operational stalls remain distinct from statistical plateaus;
manual checkpoint-safe pause is implemented, with no automatic plateau stop.

After an improving sustained trajectory exists, daily-test selection can inspect
candidate prefixes and evaluation subsets against later outcomes. Reserve
independent repeats for validation rather than fitting and testing the same three
seeds. Include optimizer-disabled, reward-perspective and terminal/bootstrap
regressions as candidates; test which the short measurement actually detects.
Do not claim a shorter test predicts long-run plateaus, capacity rankings or
architecture improvements merely because its early score increases. The present
root checks detect terminal reward/sign errors and lack of optimizer improvement;
they have **no established predictive relationship** to sustained strength.

As reconciled on 2026-10-11, the available final-result snapshot is still the
2026-10-09 running record, not a fresh execution-status observation. Final analysis
and full cost settlement, three ordinary demo-loaded executions with both deck
assignments, verified S3 archival, remaining W&B projections, and independent
daily-test validation remain unestablished. No daily threshold or promoted baseline
is accepted. Software delivery does not complete ETU-118.


## Retained debugging results

The corrected target cohort completed in 137.39071470798808 seconds after the
retained 52.36744591698516-second failed attempt. All three seeds rose from 50%
to 100%; frozen controls stayed at 50%. The degenerate three-seed bootstrap gain
interval is [50, 50] points, not evidence of population certainty. All 1,152
root evaluations replayed exactly.

The short full-game cohort at `6a2da188` completed in 2028.4034482080024 seconds.
For seeds 11821/11822/11823, initial scores were 31.25/33.33/33.33% and trained
scores 37.50/41.67/39.58%; frozen controls matched initialization. Gains were
6.25/8.33/6.25 points, mean 6.94 [6.25, 8.33] by paired training-seed bootstrap.
All 432 games replayed exactly. The predeclared 10-point mean-gain criterion
**failed**. This is retained negative debugging evidence, not sustained self-play
acceptance; training used a fixed random opponent and only 128 updates. Individual
checkpoint deal-cluster intervals remain in the original monitor records.
Total experimental debugging cost including the failure was 2218.161608833 seconds.
Software builds/tests and report generation are separate preparation work.

The byte-preserving [JSON bundle](data/current-baseline-debugging.zip) and
[hash manifest](data/current-baseline-debugging.json) retain all 27 result/run/monitor
records. Full private checkpoints, SQLite and replay tapes remain under this
checkout's `.runs/etu118-{positive-1,positive-2,full-game-1}`. Do not relocate or
remove those originals during delivery. The editable
[notebook](study/current-baseline.ipynb) reads the bundle and generates HTML;
refresh never overwrites notebook edits. Regenerate with:

```bash
uv run --extra notebook python -m experiments.runners.current_baseline_report experiments/data/current-baseline-debugging.zip --output .runs/etu118-report/published.html
```

## Historical laptop sustained proposal (superseded 2026-10-07)

The serious candidate keeps the masked-mean scalar architecture and Ataraxos
move-learning controls above, with current-self collection. Absolute iteration
learning schedules do not stretch with the selected horizon. Three new independent
seeds are 11851, 11852 and 11853. Timing-only calibration uses 11841–11843, 32
updates each, one learner thread, one evaluator and a 1,050-second total cap;
per-learner attempts are capped at 240 seconds. This cap plus retained debugging and the failed timing attempt below fits the
original 3,600-second exploration ceiling. Calibration scores
cannot select a recipe or horizon. Recovery and restart checks precede it.

`experiments.runners.sustained_baseline` constructs an ordinary Experiment and
uses the shared runner. Candidate endpoints are 12,800, 25,600, 51,200 or 102,400
updates. Select the largest fitting a conservative measured rate (1.5 times the
slowest whole-run average or 95th-percentile update interval), measured retained
storage projection and 4 GiB disk reserve. Smaller horizons do not silently
satisfy this sustained protocol. Preserve 400/800 and doubling milestones,
initial raw weights, and hourly monitoring exports. Raw weights alone are scored;
EMA exports remain retained, without favorable checkpoint selection.

One nonrenewing 604,800-active-second ceiling includes recorded debugging,
calibration, training, monitoring, recovery and final evaluation/reporting. Reserve
86,400 seconds for monitoring and 86,400 for final evaluation/reporting; divide
remaining learning allowance equally among seeds. These are maximum allowances,
not a requirement to spend a week. Sleep and safe-pause downtime are separate.
Unknown abrupt-restart gaps are conservatively charged and identified as estimates.
No automatic retry, fresh allocation, paid compute or mini displacement.

Repeated greedy monitoring uses 25 fixed deals 1911183000–1911183024, all four
seat/deck assignments, with deal-cluster uncertainty. These inspected games are
not held-out endpoint evidence. Final evaluation uses 100 separate paired deals
1911184000–1911184099, all four assignments, against source-pinned scripted greedy
and uniform legal random at initialization, update800 and the preselected endpoint.
Every game must terminate legally and replay; incomplete cohorts suppress claims.
No replacements, exclusions, outcome-based stopping or endpoint selection.

Promotion requires the primary greedy endpoint's mean gain over initialization
at least 10 points, positive gain in every seed and a positive lower 95% paired
training-seed bootstrap bound. It must also improve at least 5 points on average
over update800 with positive lower paired-seed bound, demonstrating progress beyond
the existing short screen. The secondary random mean must not regress more than
5 points from initialization. Retain per-seed and deck/seat results and both
checkpoint deal uncertainty and training-seed uncertainty separately. Three seeds
remain exploratory; passing does not establish human-challenger acceptance.
Plateaus, regressions and failures remain results; thresholds will not be weakened.

No automatic statistical plateau rule is selected. The dashboard must expose
strength versus active training cost, update/evaluation times, throughput and lag,
with manual checkpoint-safe pause/resume. An operational stall is missing updates,
checkpoints or failed evaluation, not a claim inferred from RL loss. A daily test
is selected and independently validated only after this trajectory is established.


The historical interface examples below describe the superseded laptop proposal.
The selected mini cohort was admitted using the implemented protocol and frozen
bundle recorded above; these examples are not its recovery commands:

```bash
uv run python -m experiments.runners.sustained_baseline calibrate .runs/etu118-calibration-3 --preparation-seconds 2789.1133701669414
uv run python -m experiments.runners.sustained_baseline freeze .runs/etu118-calibration-1 experiments/regimes/current-baseline-sustained.json --preparation-seconds ACTUAL_RETAINED_COST
uv run python -m experiments.runners.sustained_baseline run experiments/regimes/current-baseline-sustained.json .runs/etu118-weekly-1
uv run python -m experiments.runners.sustained_baseline pause .runs/etu118-weekly-1
uv run python -m experiments.runners.sustained_baseline run experiments/regimes/current-baseline-sustained.json .runs/etu118-weekly-1 --resume --recover STOPPED_ATTEMPT_ORDINAL
uv run python -m experiments.runners.sustained_baseline report .runs/etu118-weekly-1
```

`ACTUAL_RETAINED_COST` includes debugging and every calibration attempt, with
parallel evaluator occupancy retained separately; it is not a new allocation.
Use the frozen source bundle for the serious run and every recovery. Its source
manifest binds the original commit and exact tracked/native bytes. It lives
inside this checkout's `.runs`, has no Git metadata and launches no second Task.
Ordinary review commits cannot change its imports or restart admission. Model
exports, optimizer state and replay tapes remain private evidence, not notebook
inputs or player-facing files.

The shared editable `comparison.ipynb` is created once in the run directory.
`report` executes its existing cells to write read-only HTML, preserving edits.
Individual curves show progress before all seeds reach common milestones;
comparisons still use their shared milestones. Saved JSON carries current progress
between report refreshes. Last saved update, evaluation lag, throughput, uncertainty,
failed attempts and costs distinguish stale execution from a statistical plateau.
No raw-loss stopping rule is active. A requested learner pause also waits for the
current evaluation cohort to finish; it is bounded, not instantaneous.

A completed sustained cohort automatically enters the fixed endpoint comparison
inside the reserved final budget. Its plan hash marks those held-out rows as a
predeclared comparison, never arena promotion authority. Failed/incomplete cells
suppress acceptance. A later `finalize PLAN RUN --resume` retains existing attempts
and budget; it does not silently replace failed games. The derived conclusion
reports criteria met/not met, leaving human-challenger and daily-test acceptance
explicitly separate. No result has yet established either.

### Timing-only calibration amendment (2026-10-06)

The original 64-update probe at `975ee9dd` exceeded its first 240-second learner
cap before a complete update checkpoint was committed. No subsequent seeds ran.
The failed attempt remains in `.runs/etu118-calibration-1`: 240.11953008300043
seconds elapsed and 255.35032220798894 additive learner/evaluator seconds. Charge
the larger amount once; retain incomplete training state and the initial evaluation.
No score was used to choose this amendment.

The fresh timing cohort uses the same three calibration seeds and exact recipe,
32 updates each, unchanged 240-second learner caps and a 1,050-second total cap
with 300 seconds reserved for six bounded four-game evaluator cohorts. Debugging
plus failure plus this maximum totals 3523.511931040989 seconds, within 3,600.
This shortens a throughput probe, not the sustained endpoint or learning criteria.
No fresh allocation or claim follows from the failed timing attempt.

### Storage admission failure and current boundary (2026-10-06)

The amended calibration at `784dcedd` completed seed11841's 32 updates in
200.77795141597744 training seconds, then stopped the second learner when disk
space crossed the shared 4 GiB reserve. The third seed did not run. The retained
attempt used 269.494022333005 elapsed seconds and 315.2050481259648 additive
learner/evaluator seconds. Its incomplete cohort cannot freeze a sustained plan.
The first timeout's partial update work remains uncommitted; neither attempt is
silently resumed, replaced or treated as successful calibration.

All experimental preparation so far is **2788.716979166954 seconds**, charging the
larger elapsed/process total per calibration attempt and retaining prior debugging.
The [calibration archive](data/current-baseline-calibration.zip) and
[hash/cost manifest](data/current-baseline-calibration.json) preserve 23 original
records/logs/notebooks. Full SQLite, model and recovery files remain in
`.runs/etu118-calibration-{1,2}`. Both coordinators and their workers stopped.
The amended attempt's editable notebook regenerated its HTML from retained data.

A timing-only projection from the single completed seed puts the smallest planned
12,800-update × three-seed cohort at **29.879 GiB**, plus the 4 GiB free-space
reserve. This is provisional, not admitted calibration. Repeating the growing
historical diagnostics inside every immutable recovery snapshot dominates the
projection; compressed current-game prefixes alone do not bound total retained
storage. Free space was approximately 1.5 GiB when execution stopped. No unrelated
files, campaign evidence or mini processes were changed.

Before another calibration or sustained launch, storage must satisfy admission.
Recovery format 3 now replaces diagnostic copies with count/digest bindings to
the original writer's canonical VerifyStore prefixes. Loading verifies and restores
those exact rows, preserving stage-local offsets, complete learning state and
exports. Completed-stage diagnostics use the same references. Retained format 2
snapshots remain unchanged; this does not admit old sources for new training.

The [offline storage measurement](data/current-baseline-storage.json) and
[reproduction script/receipt](data/current-baseline-storage.zip) re-encode the
retained 32-update state without training or changing original artifacts. The
snapshot falls from 2,134,089 re-encoded bytes to 2,084,252; changing just the
reference count to 12,800 adds two bytes. The latter isolates format growth and
is not a trained long-horizon state. Under the existing conservative allowances
(130 snapshots per seed, 1 MiB current-game headroom each, four growing run-record
copies, and 2 GiB evaluation reserve), 12,800 updates × three seeds now projects
**6.361 GiB plus 4 GiB free-space reserve**. Measured free space was **6.220 GiB**;
admission still fails. The original 29.879 GiB projection remains historical.
Complete current-source three-seed calibration is still required, and freeze now
rejects calibration using the old snapshot format. The 0.3963909999874886-second
offline measurement is charged separately, bringing preparation including this
measurement to **2789.1133701669414 seconds**; the original 2788.716979166954-second
experimental ledger is unchanged. No calibration or sustained learner started.
Merely freeing enough for another short probe would not admit the full projected
cohort. The seven-day authorization remains valid; no frozen sustained plan,
source bundle or sustained learner has been created. Do not shorten the main
horizon into a toy test, reset spent budget or claim baseline/daily acceptance.

Verification: affected Python gate 39 passed; focused monitoring/report/authoring
and abrupt-restart checks 44 passed; allocation checks 2 passed. Seven native
debug checks and all-target/all-feature Clippy passed. These bounded proofs do not
establish a physical lid-close test, sustained throughput or learning success.


### Consolidated continuation and storage readmission (2026-10-06)

Jack Heart requested resumption after consolidating ETU-116 duration measurements
and ETU-82 repeated training/export/play acceptance into ETU-118. Preserve the
reserved mini sequence and separately authorized mini allocation; use one admitted
trajectory for overlapping questions. Three complete independent executions must
load through the ordinary demo and complete both deck assignments. Publish model
bytes through the S3 contract and retain notebook/W&B reporting. The contract was
unlocated at this earlier readmission; the current mini section resolves that gap.
ETU-85 retains the demo-opponent and exploratory human comparison.

[Fresh storage admission](data/current-baseline-storage-readmission.json) records
209735249920 free bytes against 6830273768 projected plus 4294967296 reserve.
It supersedes the capacity shortage, not the incomplete timing evidence. The
format-3 calibration retains all three timing seeds and 32 updates, with a total
cap of 810.8866298330586 seconds: the original 3600-second exploration ceiling
minus 2789.1133701669414 already charged. It may fail within that bound; failures
remain evidence and do not grant a new allowance. No sustained horizon is frozen.

Calibration readmission reserves 150 seconds for bounded evaluation and divides
the remaining allowance across three learners (220.2955432776862 seconds each).
The exact 32-update recipe and seed cohort stay fixed; monitoring is timing-only.
Jack Heart selected ETU-105 filter scope after the completed ETU-103 mini screen;
the separate mini duration proposal is superseded and will not run concurrently.

Compact recovery verification: 28 selected recovery/allocation checks passed,
including exact state, abrupt exit, multistage completion, setup-failure lineage
and missing/changed diagnostic rejection. Three amended admission checks passed.
These are software checks, separate from experimental preparation charges.


### Format-3 calibration attempt 3 (2026-10-06)

At committed source `221a7f74`, `.runs/etu118-calibration-3` acquired the shared
evaluator lease and began the unchanged 32-update recipe. Initialization evaluation
completed during training in 20.412291165994247 process seconds. The first learner
hit its 220.2955432776862-second cap; the supervisor retained a failed attempt at
220.48584691699943 process seconds and stopped with two seeds pending. There is
no committed 32-update checkpoint. The pre-kill TrainingRun export says running;
the completed supervisor failure receipt is authoritative for this process exit.
This is incomplete timing evidence, not a live learner or permission to retry.

The [attempt manifest](data/current-baseline-calibration-3.json) and
[byte-preserving archive](data/current-baseline-calibration-3.zip) retain receipts,
logs and editable notebook. Full SQLite and private state remain in the original
directory. Charged cost is 240.89813808299368 seconds (additive process occupancy,
larger than 220.5782599170052 elapsed). Preparation is now 3030.011508249935 seconds;
569.9884917500649 remain under the original 3600-second exploration ceiling and
601769.9884917501 under the seven-active-day total. Software tests remain separate.

No complete current-source calibration means no frozen horizon, immutable execution
bundle or sustained launch. The observed first-seed timeout makes a complete
three-seed retry within the remaining exploration allocation unsupported. A revised
calibration allocation/protocol must be explicitly fixed within the remaining week
before another attempt; no cost reset or shortened toy endpoint is inferred.
Sustained execution also needs the shared-evaluator coordination described in the
working design, live random-diagnostic scheduling, and the existing S3 publication
contract resolved. At that time ETU-105 owned the next mini workload; its subsequent
completion/release and the new duration direction are recorded above. No duplicate ran.
