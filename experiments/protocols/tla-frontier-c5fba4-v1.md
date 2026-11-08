# TLA frontier self-play at two model sizes (c5fba4 v1)

**Status: accepted design, not frozen.** The 2026-11-06 design review returned the
reviewed draft for revision, and the revised design was accepted the same day with
the one-producer calibration pod and the fact-embedding check (item 5 under
[Before freeze](#before-freeze)). The JSON protocol, regime and runner on this
branch follow it except where that section lists remaining software. Task
30df0c's input consolidation (PR #437) merged on 2026-11-07, and the parameters
below are recounted on it. Two values come only from calibration: each size's
damping horizon, w64's as `iteration_schedule.steps` in its regime JSON and
w160l8's as `large.expected_updates` in the protocol JSON.
Calibration, rehearsal and the fact-embedding check ran before MTG-241 on
card digest `6a818cbcc0c48e0d9b6abb42fee4cbea288312a271e0e9514b5683f0ea3d1cd9`;
no scored run has launched. The November 8 pre-freeze amendment moves
`runtime_world.card_digest` to MTG-241's composable-execution world
`0798b2c7` (full digest in the JSON). Retained preflight results keep their original
world identity. Recalibrate on this world before freezing; the fact-embedding
verdict remains the preflight reading, not a scored result on the amended world.

Once frozen, this protocol is fixed before any scoring. The runner is
`experiments/tla_frontier.py`; the bytes are
[the JSON protocol](tla-frontier-c5fba4-v1.json) and the w64 regime
`experiments/regimes/tla-frontier-c5fba4-w64.json`. The w160l8 regime is that
regime with the protocol's width, depth, heads and damping horizon. `frozen()`
rejects a change to the regime's hash, the runtime world or the derived tensor
shapes. A changed world before launch needs a new protocol version, not edited
bytes.

## Question

How far can self-play on the frozen 17Lands TLA trophy population push play
strength at the current recipe size, and does a network about 17× larger in
compute parameters (17.4×; 13.6× in total parameters) get further on the same
budget? This is a frontier push with one seed per size, not an ablation grid.

The run also moves the default training recipe toward what is best for USD10–100
runs. The defaults it establishes are four observation producers for
deck-source training and a damping horizon fitted to the run's expected
updates. Frozen regimes and experiment identities keep their bytes.

## What is trained

Both sizes use MTG-213 v10's recipe unchanged except where listed:

- **Population.** The same 134 frozen whole lists (corpus SHA256
  `21736009…`), sampled independently and uniformly per seat. No held-out-deck
  claim follows.
- **Learning rule.** The damped policy rule MTG-213 inherited from MTG-183's
  Elemental study (λ_policy 0.5, λ_value 0.8, quantile filter,
  action-type-uniform reference, learning rate 0.5 / position^1.1 clamped to
  [5e-6, 1e-4], τ 0.05 / position^0.3, EMA 0.999), 256 streams × 64 transitions,
  both seats learning. The open review of this rule for full TLA (MTG-213
  comment 81ec3f17) is closed for this protocol by keeping it. The frontier
  question is about scale, and changing the rule would confound the first
  reading of it.
- **Damping horizon, fitted per size.** Each size declares
  `iteration_schedule.steps` equal to its expected updates, so its schedule spans
  the run it will actually have. The reference stays the rule's 10,000 updates.
  Without a horizon, the position is the raw update count, and a run of about
  100–1,300 updates barely moves along a curve built for 10,000. The learning
  rate stays clamped at 1e-4 until position 2,305, so it would never decay, and τ
  would end near 0.006 rather than 0.003. With the horizon, update `u` of a size
  with `steps` S sits at position 1 + 9,999 × (u − 1)/(S − 1). The rate leaves its
  1e-4 clamp about 23% of the way in and ends near 2.0e-5, and τ ends near 0.0032,
  for either size.

  Until calibration, the JSON holds provisional horizons: 1,296 for w64 (v10's
  150 s/update) and 129 for w160l8 (the size gate's 1,500 s/update).
  Calibration's five updates are too few for the horizon to matter.
  Expected updates come from calibration. For each size, S = min(10,000,
  ⌊194,400 / s⌋), where s is calibration's median seconds per update over updates
  2–5 at the frozen producer count on the chosen GPU. The fit is not sensitive to
  error at the end of the run, where the curve is flat. A size that reaches only
  55% of S, the 1.8× slow-host case, ends at position about 5,500 with a rate near
  3.8e-5. A size that passes S holds the reference end. Calibration times a fresh
  policy, and later games may run longer, so a size falling short of S is
  expected and reported, not corrected.
- **Producers.** Both sizes collect with four observation producers
  (`execution.workers` 4), 64 streams each. With one producer, collection
  occupies one CPU thread, which MTG-213 v10 measured at 75% of a 150-second
  update. Roster sampling draws each game's decks from its stream's native seed,
  and producer streams keep global seeds, so the draws do not depend on the
  producer count. That is an acceptance check of the software, not an
  assumption. Calibration reads the step phase at four producers, and a third
  pod runs w64 at one producer, so the readings show what four producers buy.
- **Input.** Policy input v9 (v8 layer-state layout plus naming candidates),
  512 local facts, 128 cards and 96 permanents per seat, 64 policy action
  slots with the accepted matched first-64 overflow for every policy. The
  permanent capacity rises from v10's 64 because agent permanents overflowed
  it in MTG-213; it is `max_permanents_per_player` in the selected
  `ObservationSpace`, from which every buffer derives. Slots add no parameters;
  calibration reads what they cost per update.
- **Input architecture.** Both sizes use the consolidated input of Task 30df0c:
  fact and effect ids embed through the card encoder's exact token vocabulary,
  and sources join the card table, replacing the six hashed 4096-row tables.
  It merged as PR #437 on 2026-11-07, and the freeze pins it.
- **Failed games.** `game_errors.recover_capacity` stays on, as in v10 and as
  decided for MTG-252. An engine error or a capacity overflow (for example
  `continuation relations 65 > 64`, or a permanent past 96) aborts that one
  game; its samples are discarded and training continues. The two stay
  labelled apart, `engine_error` and `resource_limit`, in each run's game-error
  state. The rolling stop is unchanged: a run ends when more than 25% of the
  last 100 settled games failed, after 100 games. The analysis reports both
  labels' counts per size.
- **Sizes.** Only width, depth, heads and horizon differ. Two arms:

  | Label | Width | Attention layers | Heads | Compute parameters | Total parameters |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | w64 | 64 | 2 | 4 | 307,010 | 412,482 |
  | w160l8 | 160 | 8 | 8 | 5,341,922 | 5,605,602 |

  Compute parameters exclude the embedding lookup tables (the card token table
  and the fact-kind, effect-detail-kind and name tables). They were counted with
  PR #437 merged, at `563ae37dd`; the freeze revision's tests check the ratio.
  The ratio is 17.4× (13.6× in total parameters), inside the requested 10–100×,
  and the large arm grows in depth as well as width. The counts taken before
  consolidation (385,538 and 5,538,242, 14.4×) excluded six 4096-bucket hashed
  tables that PR #437 removed. The earlier draft's w256 (4 layers, 8 heads) had
  8,663,554 compute parameters on that pre-consolidation base, 22.5×. W256 is
  the fallback: if w160l8 misses calibration's speed gate, a new protocol
  version replaces it with w256 before any scoring.
- **Budget per size.** One GPU allocation of at most USD38 and 56 hours; the
  regime wall budget is 54 hours (194,400 s). The update cap is 10,000, the
  horizon's ceiling, so the wall is the stop that binds. Training exports the
  last complete raw learner when it stops. The sizes therefore receive equal
  wall-clock compute, not equal updates. Training seed 21501 for both. No
  automatic retry. Every command in this protocol runs through one control
  prefix. The nested Doppler projects supply provider and artifact access. Until
  the freeze the control client is Etude 0.2.14, because the locked 0.2.12 cannot
  read the launcher's Vast provider rows; the freeze moves it and the job project
  to Etude main ([Before freeze](#before-freeze) item 3). The prefix is an array,
  not a function, so `caffeinate` can run it:

  ```bash
  control=(doppler run -p etude -c prd -- doppler run -p manabot -c prd
      --preserve-env=RUNPOD_API_KEY -- uv run --extra artifacts --extra evaluation
      --with "etude-sdk[s3,eval]==0.2.14")
  ```

  Both sizes train at once, so comparisons appear as they go. A cohort's
  default is one machine at a time, so the launch names two:

  ```bash
  "${control[@]}" etude launch experiments/tla_frontier.py --revision <frozen> \
      --rent l4 --cohort tla-frontier-c5fba4 --parallel 2 --max-usd 76
  ```
- **Exports.** Each size exports its raw learner every 25 updates and at every
  milestone. Milestones and endpoints are played against the heuristic. Every
  export is a candidate for the w64 side of an equal-hours pair.
- **Hardware.** Both sizes run on the same GPU model, chosen at launch and
  recorded before any score is read; different models void the equal-compute
  reading. The protocol rents the `l4` on-demand, and `mtg252-a40` only if L4
  stock is empty, in which case that name replaces `l4` in every command of the
  launch and calibration reruns on it. USD38 over 56 hours admits at most
  USD0.67/hour, which covers the USD0.59/hour L4 and A40 of earlier receipts but
  not the USD0.96/hour RTX 5090. Each run records its host (`TrainingRun.host`:
  CPU model, schedulable CPUs, quota) and its step phase. The analysis reports
  both, and neither voids a reading. A slow pod is kept: equal wall at one price
  is still equal dollars.

## Calibration

Before the training runs, at a merged revision carrying the software below,
`experiments/tla_frontier_calibration.py` runs each size for five updates
on the chosen GPU, under its own experiment and regime ids, so it is never a
frontier run. One pod per size at four producers, at most USD0.55 for w64 and
USD0.90 for w160l8, and a third pod running w64 at one producer, at most
USD0.55: USD2 together, from the reserve. The one-producer pod differs from the
w64 pod only in `execution.workers`; it sets nothing and gates nothing. `l4` allows
two machines at once, so the third pod queues behind the first two.

```bash
"${control[@]}" etude launch experiments/tla_frontier_calibration.py \
    --revision <merged> --rent l4 --cohort tla-frontier-c5fba4-calibration \
    --parallel 2 --max-usd 2
"${control[@]}" python -m experiments.tla_frontier_calibration --source <store>
```

It reads, per size, seconds per update (median over updates 2–5), the expected
updates S that horizon fitting would take from it, CUDA peak, peak host memory,
median rows per update, the step phase, the pod's host and its game-error counts
by label. Each stored run is
named by its launcher id, so a reading shows its job name. Host memory and rows
add no gate. They show how far the regime's 40 GiB stop is from a fresh
policy's reading, with rows wider for 96 permanents.

The readings set two things before freeze:

1. **Size gate.** W160l8 launches at 1,500 s/update or less, S ≥ 129. If it
   cannot fit or reads above that line, a new protocol version replaces it with
   w256 (4 layers), still before any scoring, and calibrates that.
2. **Horizons.** Each size's S from its reading.

A gate reading depends on its host: a slow pod reads up to 1.8× above a fast
one, so the reading shows the pod's step phase and CPU beside it.

### Evaluation rehearsal

A game past its limit is stored as a delivered match and is never replayed, so
the first rented evaluation pass must not be a scored one. Once the calibration
runs have ended, calibration plays one pass of its own through the same job as a
frontier pass: `tla_frontier.play`, a `cpu32` rental, 28 workers, the journal in
`/dev/shm`, the 600-second limit and the USD2 pass ceiling. Its two comparisons
on the frozen pairs, 32 matches, keep every worker busy: `w160l8`'s last
checkpoint vs its update 0, the heaviest shape, and `w160l8`'s last vs
`w64`'s last. Runs are selected by their `tla-frontier-c5fba4-calibration-*`
regime tags, which also proves a tag on a really delivered checkpoint. These
games sit under calibration's experiment and an environment of their own, so
they touch no frontier slot, match key or score.

```bash
"${control[@]}" etude eval experiments/tla_frontier_calibration.py \
    --source <store> --rent cpu32
"${control[@]}" etude wait <run>
"${control[@]}" python -m experiments.tla_frontier_calibration \
    --source <store> --rehearsal
```

`etude eval` returns at launch and prints the pass's run id, so `etude wait`
holds the read until the pass is over; read earlier, it sees no games.

The reading shows games, invalid games, median and slowest game seconds and the
cost of the passes that stored a match; a pass that stored none still billed, and
its cost comes from `etude status <run>`. The rehearsal gate is a rough line: no game
killed or left invalid except by a capacity abort, and the slowest under 300
seconds, half the limit. Otherwise `game_seconds` or the size changes in a new
protocol version, before any scoring. A capacity abort is labelled, counted and
bounds a reading, as under Failed games: no game limit or size removes one. Its
failure text gives the seat's card rows by zone, its token count and its commonest
tokens, and its Command journal stays in the pass's `arena/` output, so
`manabot.sim.replay.replay_prefix` rebuilds the position. The policies are nearly fresh, so the reading shows what a
pass does on a rental, not how long a trained policy's games run. If `w160l8`
stored nothing past update 0 the rehearsal refuses to play, because the size
gate has already failed. A rehearsal pass that fails or delivers nothing
recovers as under [Failed pass](#failed-pass), with `C5FBA4_PASS_ATTEMPT` set on
the `etude eval` line above. A match it stored as failed stays failed, so that
reading fails this gate.

## Launch sequence

1. Items 1, 2 and 4–6 under [Before freeze](#before-freeze) merge on main. PRs
   #435 and #437 have merged, and the parameter counts above are taken with both.
2. Operator checks. Every rented on-demand has a startup limit, or a dud pod on a
   supply with none holds its whole window. `l4` has 900 seconds; set the others
   where absent. `cpu32` allows one machine, which fits one pass at a time.

   ```bash
   "${control[@]}" etude ondemand startup cpu32 --seconds 900
   "${control[@]}" etude ondemand startup mtg252-a40 --seconds 900
   ```

3. The record coverage sweep on the revision calibration runs from:
   `uv run python -m manabot.verify.record_coverage --games 200 --seed 0`.
   Every card and id must map; it is rerun on the frozen revision. Its seed 1
   ran about 25× slower per decision on very large boards, a game-time and
   per-update cost risk that calibration's seconds per update and the
   rehearsal's slowest game read; it is not a vocabulary failure.
4. Calibration, then the evaluation rehearsal.
5. The fact-embedding check (item 5 under [Before freeze](#before-freeze)), from
   the same merged revision, once calibration has read w64's cost.
6. Both gates, and each size's horizon written into the JSON protocol and regime.
   Freeze at the merged revision that carries them and item 3.
7. Execution review.
8. The training launch, then the evaluation driver.

## Evaluation schedule

Every comparison plays the same 16 fixed pairs (deck seed 21590001, deals
21591000–21591015), four legs per pair swapping decks and seats: 64 games. The
heuristic is the matched first-64 heuristic. Shared update milestones are 25,
50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600, 2400, 3200, 4800 and
6400.

- At update 0 and every reached milestone: each size vs the heuristic.
- **Equal hours.** At every w160l8 milestone and at its endpoint: w160l8 vs w64's
  last checkpoint at or before the same hours. A pair that is also an
  equal-update or endpoint pair is played once, under that comparison's name, and
  both slots read its games. Hours are a checkpoint's wall time
  since its run's first checkpoint, update 0, so pre-training setup counts for
  neither size. The pair is fixed once w64 has stored a checkpoint later than
  that hour, or has ended. If w64 ended earlier than that hour, the slot is
  unreached. The rule never gives w64 more time than w160l8, and gives it up to
  one export interval less, which the analysis reports in hours.
- **Equal updates.** At every milestone both sizes reached: w160l8 vs w64.
- At each endpoint: that size vs the heuristic (unless the endpoint is a
  milestone) and vs its own update-0 initialization.
- Once both runs end: w160l8 endpoint vs w64 endpoint, the comparison at equal
  wall-clock compute when both finished at the wall.

A size's endpoint is the last stored checkpoint of its ended run, including a
failed run, so its reach is still played. A run has ended when the worker
recorded its end or the launcher closed it with any outcome but `succeeded`.
Only a worker writes a stored end, so both `eval`s and the analysis command ask
the launcher about each run still reading `running`. A `succeeded` run with no
stored end is still delivering and stays running. A running run the launcher
does not know raises, so these reads need the launcher, and a read away from it
fails rather than taking a dead pod for a live one.

Evaluation runs on rented CPU through Etude while training continues, so
progress is visible as it goes. Each pass is one 32-vCPU rental with 28
workers, capped at USD2 and 1.75 hours plus setup across up to three attempts;
cumulative evaluation spend is capped at USD14. Etude retries a machine lost or
refused after admission, resuming the same run and skipping stored matches, so
no game result can be replaced; training keeps no automatic retry. A create
refused before admission is not retried: see
[Evaluation driver](#evaluation-driver).

Each game has a 600-second wall limit, enforced by killing the game and counting
it failed. MTG-213's 120 seconds could penalize the larger network for its own
latency on a loaded rental; a real fresh-w256 match on a development Mac ran
about 0.05 s per decision, so the limit may be generous, and the rehearsal reads
it on a rental. The arena journals in memory and copies its files out afterward,
because its per-record sync takes about 0.1 s on a rented container disk and
counts against that limit.

Milestones past a size's endpoint stay in the schedule as unreached slots; failed
or truncated games stay failed in their scheduled slots and are never replaced,
so no pass uses Etude's `--retry-failed`.

### Evaluation driver

One command drives every pass, from a checkout of the frozen revision on an
always-on host beside the controller. It starts with the training launch. If that
host is a laptop, prefix it with `caffeinate -is`, and check from `etude status`
that the controller is ticking. A booted pod ends itself at its deadline or
budget, but only a ticking controller ends a pod that never booted and hands
passes on:

```bash
"${control[@]}" etude eval experiments/tla_frontier.py --source <store> \
    --rent cpu32 --watch 14400
```

Every pass must run from that revision: a later commit changes `seats.source`,
which analysis flags as a mixed cohort. The four-hour period batches both sizes'
new milestones, so each milestone is played within one period of arriving.

Expect about 45 comparisons over about 15 rentals, for about USD11 with setup,
if w64 reaches about 3,000 updates and w160l8 about 400. Those reaches are guesses
from v10's 150-second update, three quarters of it collection, before four
producers and #435; calibration replaces them, and the count follows the reached
milestones. Etude does not enforce
the USD14 cap across passes: the operator reads spend after each pass and stops
the loop when it reaches USD14. If the projection runs over, milestones 75 and
150 are the first dropped, by recorded operator decision before their games.

Any launcher error ends the loop, silently for a create refused before admission,
which leaves no ledger entry to find. Restarting it is safe, because stored
matches are skipped, and a stopped driver costs only timeliness: every
checkpoint stays playable. So check the process is alive whenever spend is read.
Stop it once both runs have ended and a pass reports nothing new.

To see progress at any time, draw the report from the same checkout, with
launcher access, since it asks `eval` what is wanted:

```bash
"${control[@]}" etude report experiments/tla_frontier.py --source <store>
```

The report is descriptive curves by update, the equal-hours curve drawn at
w160l8's updates because Etude's report axes are checkpoint clocks;
`experiments/tla_frontier_analysis.py` owns every reading below. A relaunched
pass adds a second tournament to its selector. Each shows the same curves,
whichever pass played a match.

Reading progress changes nothing. No run is cancelled, relaunched or extended
because of a score. A stop for cost, a dead run or any outside reason is a
recorded decision that keeps its evidence.

Etude's launch ledger, `~/.etude/experiments/launched.json`, is one file for
every experiment on the computer, and any entry for a failed pass raises
`launched_pass_failed` and ends the watch loop. Before starting the loop, check
each entry's status there, including other experiments'.

### Failed pass

A pass's launcher id is a digest of its pending matches, and the launcher refuses
an id it knows, including one from a refused create. So a pass that delivered
nothing cannot launch again as it stands, and the endpoint pass has no new
milestone coming to change its matches. The recovery is:

1. Look at why it failed, from `etude status <run>` and its logs.
2. Remove its entry from the ledger, if it has one.
3. If a refused create is holding `cpu32`'s one-machine limit, raise the limit.
4. Set the next attempt and restart the loop. The first relaunch is attempt 2.

```bash
"${control[@]}" etude ondemand max cpu32 2
C5FBA4_PASS_ATTEMPT=2 "${control[@]}" etude eval experiments/tla_frontier.py \
    --source <store> --rent cpu32 --watch 14400
```

The variable suffixes the tournament's name (`-attempt-2`), which changes the
pass's id and nothing scored: names sit outside match keys and environments. The
attempt plays exactly the matches no pass attempted, leaves failed games failed,
and analysis reads one comparison across both passes. A milestone delayed this
way is still played.

## Lost run

A run whose rental dies stops at its last export, which is its endpoint. It keeps
its learning and strength readings, its equal-hours pairs up to that point and
its state, hours and cost in the report. It has no scale reading. Until the
controller declares the pod lost, the size's endpoint stays unknown. This
version builds no continuation, replacement or restart; any further training is a
new protocol version.

## Reading

`experiments/tla_frontier_analysis.py` produces these readings. Scores are the
candidate's mean game score with 95% whole-pair bootstrap intervals (seed
21592001), for comparisons whose 16 pairs all completed. Head-to-head rows put
w160l8 first. A milestone past an ended run's endpoint is unreached and expects
no games; any other absent comparison leaves that reading unknown.

A failed game narrows a reading rather than erasing it. Once every pair of a
comparison has a match receipt, including a failed match with no game rows, the
comparison also reports `worst` and `best`: the same statistic with each unscored
leg counted as 0 and as 1, bootstrapped with the same indices. Nothing is
replayed, replaced or dropped from the denominator, because failures are not
neutral: limits and kills fall on long games. With no failures all three agree.
Any unscored game leaves the analysis incomplete, with an issue naming the count.

- **Learning**, per size: established when `worst`'s lower bound is above 0.5,
  rejected when `best`'s lower bound is at or below 0.5, and unknown between.
- **Strength**, per size: the milestone curve and endpoint vs the heuristic,
  with the same bounds.
- **Scale**: the endpoint head-to-head is primary, with the same bounds, when
  both runs finished at the wall. The equal-hours curve shows the same
  comparison through training, with each pair's two hours. Equal-update
  head-to-heads describe sample efficiency. With a lost run, scale is absent and
  the equal-hours curve stands as a descriptive reading up to the loss.

The analysis runs where the launcher is, because a lost pod's end lives there:

```bash
"${control[@]}" python -m experiments.tla_frontier_analysis \
    --source <store> --out <analysis.json>
```

The analysis also reports each size's horizon S beside the updates it reached,
its producer count, host and step phase, and the run's settled game attempts
with their `engine_error` and `resource_limit` counts and discarded samples, read
from `TrainingRun.game_errors`. A run that hits the rolling stop ends there; its
endpoint is its last export, as for any ended run.

Each comparison and the whole analysis also total the first-64 cutoff: decisions
and dropped actions, over the games that returned a count. A game killed at its
limit returns none, so those totals can be low. An overflow is accepted behavior:
a nonzero count is reported, adds no issue and leaves `complete` unchanged. The
totals span both players, so they support no per-policy rate.

Report every attempt, failure, resource-limit discard, unreached milestone and
cost. One seed per size supports a descriptive frontier reading, not a controlled
claim that one size is better in general.

### Fact contribution

Neither the hashed nor the catalog fact embedding has learning evidence, and
both fact output layers start at zero, so this descriptive reading asks whether
training uses them. It trains nothing, plays no scored game and gates nothing;
it is reported beside learning. `experiments/tla_frontier_facts.py` fixes one
sample of positions from the 16 frozen pairs: one random-action game per pair
(leg 0, the pair's deal seed, action seed 21593001, forced decisions skipped),
keeping every fourth decision up to 32 per pair; a game past a capacity stops
there. Every checkpoint reads the same positions. For each object a valid fact
row or a shared-effect membership names, it reports the norm of the fact
contribution added to that object, in total and per fact kind, and the norm of
the shared-effect contribution, each relative to the object's embedding before
facts: the median and mean over (position, object) rows, with their counts.
The shared-effect reducer has no zero-initialized output, so its update-0
reading is the baseline its trained readings are read against.

It covers each vocabulary-check arm at every export (update 0 included) and
each frontier run at update 0, every reached milestone and its endpoint. A
fact reading near zero after training is a finding that neither design is
being used. It runs on CPU where the analysis runs:

```bash
"${control[@]}" python -m experiments.tla_frontier_facts \
    --experiment tla_frontier_vocabulary --source <store> --out <facts.json>
"${control[@]}" python -m experiments.tla_frontier_facts \
    --experiment tla_frontier --source <store> --out <facts.json>
```

## Before freeze

Software, merged on main after PRs #435 and #437, each with its acceptance
check:

1. **Roster sampling with 2–4 producers** (`manabot/training/rollout/producers.py`,
   `collection.py`, `regime.py`). `ShardedVectorEnv` forwards training decks and
   the uniform roster to every shard and concatenates `current_player_configs` and
   exposure. The one-producer restrictions on deck sources and
   `elemental-uniform` go. Check: with one seed, per-stream roster draws, exposure
   counts and the collected batch under a fixed policy agree for 1, 2 and 4
   producers on CPU; every committed regime file round-trips to the same bytes.
2. **Defaults for new runs** (`manabot/recipes/builders.py`, `docs/training.md`,
   `docs/learning.md`). New damped self-play recipes take their expected updates
   and declare `IterationSchedule(steps=expected)`; deck-source and Elemental
   recipes default to four producers. New self-play recipes recover capacity
   overflows (`GameErrorPolicy(recover_capacity=True)`), so an overflow aborts
   one labelled game rather than the run. Pydantic field defaults stay, so saved
   configurations keep their meaning. Check: every existing experiment module's
   regime ids and dumps are unchanged, and a fresh recipe's dump carries the
   four producers, the fitted horizon and `recover_capacity`.
3. **Checkpoint wall time in `ExperimentState`** (Etude PR #86, no patch release:
   `Checkpoint.wall_ms`, from `training_checkpoints`). `eval` and the analysis
   take equal-hours pairs from `tla_frontier.checkpoint_hours`, which reads no
   hours until this lands, so those slots stay unplayable and the analysis
   reports them missing. No manabot clock stands in. The freeze change moves the
   job project and the control client to `etude-sdk` from main together, because
   `Record` forbids extra fields: a pass sent with `wall_ms` fails on a job
   pinned to 0.2.12. Check: the Etude test reads `wall_ms` on a stored checkpoint,
   match keys are unchanged, and `checkpoint_hours` rebases it per run.
4. **Frontier runner** (`experiments/tla_frontier*.py`, protocol JSON, regime,
   tests). Producers and horizons per size from the protocol, update cap 10,000,
   the milestones above, the equal-hours schedule and its analysis, evaluation
   USD14, the w160l8 size and label, 96 permanents per seat, the recounted
   parameters, calibration's expected-updates reading, and per-label game-error
   counts in calibration and the analysis. Check: `uv run --extra dev --extra evaluation --extra artifacts
   pytest tests/eval/test_tla_frontier.py tests/eval/test_tla_frontier_vocabulary.py
   -q`, the equal-hours schedule tests over fixture checkpoint hours (including
   one pair played once under both slots), and an analysis test whose fixture
   runs carry both labels in `game_errors.by_cell`.

5. **Catalog against hashed fact ids** (accepted with the design). The field is
   Task 30df0c's: PR #437 lets fresh regimes select `record_vocabulary: hashed`,
   and this item's tests rerun on the frozen revision. Before the freeze,
   `experiments/tla_frontier_vocabulary.py` trains two arms of the w64
   regime that differ only in `ModelConfig.record_vocabulary`, catalog and
   hashed: same recipe, seed 21501, decks, four producers, 75 updates, exports
   at 25, 50 and 75, a 15,600-second wall and at most USD3 and 4.6 hours each on
   the frontier's GPU: the length recorded after calibration read w64 at
   167 s/update, which leaves room for 200. Both arms carry the
   regime's provisional horizon of 1,296. At that horizon, and at calibration's 1,164,
   the rate stays at its 1e-4 clamp through update 150, and the horizon only moves
   τ, equally in both arms.

   **Extension (Jack Heart, 2026-11-07).** Catalog's pod read 1.6× slow per row
   (host step spread 8.4 against 1.2), so it stopped at update 64 and the
   75-update reading was unread. Jack Heart authorized up to USD10 more to
   extend an inconclusive check to 150 updates, past the USD100 Task ceiling if
   it is used. Each arm continues from its first segment's final raw/EMA/Adam
   exports to update 150 under its own `-to150` regime id, exporting at 100 and
   125: a 21,000-second wall and at most USD4 and 6.1 hours each, which covers
   catalog's 86 updates at its slow pod's 234 s. Once both extensions end at
   update 150, one pass on the 16 frozen pairs plays each arm's update 150
   against its own update 0, and catalog's update 150 against hashed's, catalog first,
   under an environment of its own. The criterion, fixed here, fails catalog when
   any of these holds:

   1. **Head-to-head:** the catalog-vs-hashed `best` upper bound is below 0.5,
      so catalog loses under every failed-leg outcome.
   2. **Learning:** hashed's learning against its initialization is established
      (`worst` lower bound above 0.5) and catalog's is rejected (`best` lower
      bound at or below 0.5).
   3. **Learning curve:** catalog's median `value_loss` over updates 76–150 is
      more than 1.10 times hashed's.

   Otherwise catalog passes. A reading missing an input (an arm short of update
   150, an unplayed comparison or an absent loss) is unread, which is no pass.
   The same comparisons at updates 25, 50 and 100 are played as exports arrive;
   they are descriptive and enter no rule.
   A failure, or an unread check, returns the embedding choice to review before
   the freeze. Commands, from the merged revision:

   ```bash
   "${control[@]}" etude launch experiments/tla_frontier_vocabulary.py \
       --revision <merged> --rent l4 --cohort tla-frontier-c5fba4-vocabulary \
       --parallel 2 --max-usd 4
   "${control[@]}" etude eval experiments/tla_frontier_vocabulary.py \
       --source <store> --rent cpu32
   "${control[@]}" etude wait <run>
   "${control[@]}" python -m experiments.tla_frontier_vocabulary --source <store>
   ```

6. **Fact-contribution reading** (`experiments/tla_frontier_facts.py`, its
   tests; declared under [Fact contribution](#fact-contribution)). Check:
   `uv run --extra dev --extra evaluation --extra artifacts pytest
   tests/eval/test_tla_frontier_facts.py -q`: the positions repeat exactly, a
   fresh model's fact contribution is zero, weighted fact output registers per
   kind, and the reader covers the frontier and vocabulary checkpoints it names.

Calibration, the rehearsal and the fact-embedding check run from the merged
revision of items 1, 2 and 4–6; they do not need item 3. The freeze is a second,
small change carrying both horizons, the regime hash and the Etude move, and it
waits for item 3, because equal-hours slots read `checkpoint_hours`, which is
runner code fixed at the freeze.

## Budget

The Task's ceiling is USD100 including setup, failures, retries and evaluation:
2 × USD38 training + USD14 evaluation = USD90, leaving USD10 for calibration
and the rehearsal (USD1.51 spent), the fact-embedding check (USD6 for its two arms
and a USD2 pass ceiling, about USD5.50 expected), setup and failure costs. Those
leave USD0.49 of the reserve at every ceiling. Evaluation is about 18% of training's ceiling. Any retry needs
a recorded decision that keeps the total within USD100. Evaluation never runs on
a laptop.
