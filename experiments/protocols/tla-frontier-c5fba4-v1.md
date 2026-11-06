# TLA frontier self-play at two model sizes (c5fba4 v1)

This protocol is fixed before any scoring. The runner is
`experiments/tla_frontier.py`; the bytes are
[the JSON protocol](tla-frontier-c5fba4-v1.json) and the w64 regime
`experiments/regimes/tla-frontier-c5fba4-w64.json`. The w256 regime is that
regime with the protocol's width, depth and heads. `frozen()` rejects a change
to the regime's hash, the runtime world or the derived tensor shapes. A changed
world before launch needs a new protocol version, not edited bytes.

## Question

How far can self-play on the frozen 17Lands TLA trophy population push play
strength at the current recipe size, and does a network about 15× larger in
compute parameters get further on the same budget? This is a frontier push with
one seed per size, not an ablation grid.

## What is trained

Both sizes use MTG-213 v10's recipe unchanged except where listed:

- **Population.** The same 134 frozen whole lists (corpus SHA256
  `21736009…`), sampled independently and uniformly per seat. No held-out-deck
  claim follows.
- **Learning rule.** The damped policy rule MTG-213 inherited from
  MTG-183's Elemental study (λ_policy 0.5, λ_value 0.8, quantile filter,
  action-type-uniform reference, learning rate 0.5 / iteration^1.1 clamped to
  [5e-6, 1e-4], EMA 0.999), 256 streams × 64 transitions, both seats learning.
  The rule's schedule is absolute in iterations, so the update target is a cap,
  not a schedule input. The open review of this rule for full TLA (MTG-213
  comment 81ec3f17) is closed for this protocol by keeping it: the frontier
  question is about scale, and changing the rule would confound the first
  reading of it.
- **Input.** Policy input v9 (v8 layer-state layout plus naming candidates),
  512 local facts, 128 cards and 64 permanents per seat, 64 policy action
  slots with the accepted matched first-64 overflow for every policy.
  Game-local resource recovery as accepted for v10 (25% rolling threshold).
- **Sizes.** Only width, depth and heads differ:

  | Label | Width | Attention layers | Heads | Compute parameters | Total parameters |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | w64 | 64 | 2 | 4 | 647,682 | 1,958,402 |
  | w256 | 256 | 4 | 8 | 9,712,130 | 14,955,010 |

  Compute parameters exclude the five 4096-bucket hashed lookup tables, whose
  bucket count is unchanged; their width follows the model width. The ratio is
  15.0×, inside the requested 10–100× and toward its low end, because the
  larger size's update cost on a GPU has not been measured.
- **Budget per size.** One GPU allocation of at most USD38 and 56 hours; the
  regime wall budget is 54 hours (194,400 s) and the update cap 3,000. Training
  stops at whichever comes first and exports the last complete raw learner.
  The sizes therefore receive equal wall-clock compute, not equal updates.
  Training seed 21501 for both. No automatic retry. Every command in this
  protocol runs through one control prefix. The nested Doppler projects supply
  provider and artifact access. Etude 0.2.14 is the control client only: the
  locked 0.2.12 cannot read the launcher's Vast provider rows, and workers keep
  0.2.12. It is an array, not a function, so `caffeinate` can run it:

  ```bash
  control=(doppler run -p etude -c prd -- doppler run -p manabot -c prd
      --preserve-env=RUNPOD_API_KEY -- uv run --extra artifacts --extra evaluation
      --with "etude-sdk[s3,eval]==0.2.14")
  ```

  Both sizes train at once, so equal-update comparisons appear as they go; a
  cohort's default is one machine at a time, so the launch names two:

  ```bash
  "${control[@]}" etude launch experiments/tla_frontier.py --revision <frozen> \
      --rent l4 --cohort tla-frontier-c5fba4 --parallel 2 --max-usd 76
  ```
- **Exports.** Each size exports its raw learner every 25 updates and at every
  milestone. Only milestones and endpoints are played; the other exports bound
  what a lost rental can cost, to 25 updates: about 1 hour at 150 s/update and
  10.4 hours at the 1,500 s/update launch gate.
- **Hardware.** Both sizes run on the same GPU model, chosen at launch and
  recorded before any score is read; different models void the equal-compute
  reading. The protocol rents the `l4` on-demand, and `mtg252-a40` only if L4 stock
  is empty, in which case that name replaces `l4` in every command of the launch
  and calibration reruns on it. On retained Elemental runs A40 was no faster at
  the same price, and every A40 host read slow. This regime has one worker, so
  one CPU thread sets the pace. At a fixed GPU label update time moved 1.6–1.8×
  with the host, and vCPU count explained none of it. The check therefore records
  each pod's host speed: its median
  `time/collection/environment_step_and_encoding` per row over updates 2–6, a
  phase with no network work, so w64's pod compares directly with w256's. It also
  records `host.cpu_model` from `training/run.json` when the revision has it.
  A difference is reported and does not void the reading: equal wall at one
  price is still equal dollars. USD38 over 56 hours admits at most USD0.67/hour,
  which covers the USD0.59/hour L4 and A40 of earlier receipts but not the
  USD0.96/hour RTX 5090, where the dollar ceiling would stop training first.
- **Calibration.** Before the training runs, and after this protocol merges,
  `experiments/tla_frontier_calibration.py` runs each size for five updates on
  the chosen GPU: at most USD1 and 1.2 hours per size, USD2 in total, inside the
  reserve:

  ```bash
  "${control[@]}" etude launch experiments/tla_frontier_calibration.py \
      --revision <frozen> --rent l4 --cohort tla-frontier-c5fba4-calibration \
      --parallel 2 --max-usd 2
  "${control[@]}" python -m experiments.tla_frontier_calibration --source <store>
  ```

  It reads seconds per update, CUDA peak, peak host memory, median rows per
  update and host speed (updates 2–5 of its five), under its own experiment and
  regime ids, so it is never a frontier run. Each stored run is named by its
  launcher id, so a reading shows its job name. Host memory and rows add no gate;
  they show how far the regime's 40 GiB stop is from a fresh policy's reading.
  W256 launches at 1,500 s/update or less, about 130 updates in 54 hours. That
  reach is an upper estimate: calibration times a fresh policy, and MTG-183
  slowed 1.55× between windows (confounded by a host change), which would leave
  about 83 updates. A size at the gate therefore has head-to-heads at 25, 50 and
  75, and at 100 only if it holds its calibrated speed. If it cannot fit or reads
  above that line, the size changes in a new protocol version, still before any
  scoring. A gate reading depends on its host: a slow pod reads up to 1.8× above
  a fast one, so read the size's cost beside its pod's host speed.
- **Evaluation rehearsal.** A game past its limit is stored as a delivered match
  and is never replayed, so the first rented evaluation pass must not be a scored
  one. Once both calibration runs have ended, calibration plays one pass of its
  own through the same job as a frontier pass: `tla_frontier.play`, a `cpu32`
  rental, 28 workers, the journal in `/dev/shm`, the 600-second limit and the USD2
  pass ceiling. Its two comparisons on the frozen pairs, 32 matches, keep every
  worker busy: w256's last checkpoint vs its update 0, the heaviest shape, and
  w256's last vs w64's last. Runs are selected by their
  `tla-frontier-c5fba4-calibration-*` regime tags, which also proves a tag on a
  really delivered checkpoint. These games sit under calibration's experiment and
  an environment of their own, so they touch no frontier slot, match key or score.

  ```bash
  "${control[@]}" etude eval experiments/tla_frontier_calibration.py \
      --source <store> --rent cpu32
  "${control[@]}" python -m experiments.tla_frontier_calibration \
      --source <store> --rehearsal
  ```

  The reading shows games, invalid games, median and slowest game seconds and the
  pass's cost. The second launch gate is a rough line like the first: all 128 games
  valid, so none killed, and the slowest under 300 seconds, half the limit.
  Otherwise `game_seconds` or the size changes in a new protocol version, before
  any scoring. The policies are nearly fresh, so the reading shows what a pass does
  on a rental, not how long a trained policy's games run. If w256 stored nothing past
  update 0 the rehearsal refuses to play, because the first gate has already failed.
  A failed rehearsal pass leaves a launch ledger entry, and clearing it, as under
  [Evaluation driver](#evaluation-driver), rehearses that recovery too.
- **Host screen.** Over 54 hours a slow pod costs a size up to about 40% of its
  updates, so before any scoring the operator reads each pod's host speed and
  replaces a slow one. This is a declared operator step, not an automatic retry:
  jobs keep one attempt, and any retry after the screen is a new protocol version.

  - Read each pod once it passes update 6, while it is below update 25, and start
    the evaluation driver only after both reads, so no game exists when it acts.
    Only a pod below the first milestone can be replaced: an ended run past it
    stays its size's run, and a second run of that size stops evaluation and
    analysis. W64 can pass update 25 in about an hour, before a slow w256 passes
    update 6, so w64 is read against calibration's pods and whatever else the
    store holds. A pod found slow only later is kept and reported.

    ```bash
    "${control[@]}" python -m experiments.tla_frontier_calibration \
        --source <store> --screen
    ```

  - A pod reads slow above 1.3× the lowest host speed among this launch's pods
    and calibration's, so both pods can read slow against calibration. That line
    is rough, taken from Elemental's bands; TLA has no anchor of its own.
  - Cancel a slow pod with `etude cancel <run>` and do not delete it. The worker
    takes Etude's SIGTERM and records the run as failed at its last export. Confirm
    that from `state` in the screen's output before relaunching, because `size_run`
    admits one live run per size. A pod lost outright records no end, so its size
    is not relaunched; any retry is then a new protocol version. Otherwise relaunch
    that size once, alone and with no cohort. Its job carries the USD38 ceiling, and
    `--max-usd` is a cohort limit that Etude refuses without a cohort:

    ```bash
    "${control[@]}" etude launch experiments/tla_frontier.py --revision <frozen> \
        --size <w64|w256> --rent l4
    ```

  - A size that reads slow twice keeps its second run. A replacement starts from
    update 0, and the cancelled pod's cost, at most about USD1.50, comes from the
    reserve.
  - A run that ends below the first milestone is set aside. It is no size's run,
    plays nothing, and analysis lists it, since every attempt is reported.
  - A pod that never started stores no run and trained nothing, so it is not the
    screen's one replacement. Relaunch that size with the command above, and report
    the launcher's record of the pod with the attempts. A refused cohort launch
    leaves no cohort and boots nothing that stays, so it reruns unchanged.
  - The screen holds only if the step phase is steady on one pod. If calibration
    shows `step_spread` above 1.3× on either pod, there is no screen for this
    launch and host speed is reported only.

## Launch sequence

1. Operator checks. Every rented on-demand has a startup limit, or a dud pod on a
   supply with none holds its whole window: for training, a size silently not
   training for up to 56 hours. `l4` has 900 seconds; set the others where absent.
   `cpu32` allows one machine, which fits one pass at a time.

   ```bash
   "${control[@]}" etude ondemand startup cpu32 --seconds 900
   "${control[@]}" etude ondemand startup mtg252-a40 --seconds 900
   ```

2. Calibration, then the evaluation rehearsal.
3. Both launch gates: w256 at 1,500 s/update or less, and the rehearsal's.
4. The training launch.
5. The host screen.
6. The evaluation driver.

## Evaluation schedule

Every comparison plays the same 16 fixed pairs (deck seed 21590001, deals
21591000–21591015), four legs per pair swapping decks and seats: 64 games. The
heuristic is the matched first-64 heuristic. Shared update milestones are 25,
50, 75, 100, 150, 200, 300, 400, 600, 800, 1000, 1200, 1600, 2000 and 2400.

- At update 0 and every reached milestone: each size vs the heuristic.
- At every milestone both sizes reached: w256 vs w64 at equal updates.
- At each endpoint: that size vs the heuristic (unless the endpoint is a
  milestone) and vs its own update-0 initialization.
- Once both runs end: w256 endpoint vs w64 endpoint, the comparison at equal
  wall-clock compute when both finished.

A size's endpoint is the last stored checkpoint of its ended run, including a
failed run, so its reach is still played.

Evaluation runs on rented CPU through Etude while training continues, so
progress is visible as it goes. Each pass is one 32-vCPU rental with 28
workers, capped at USD2 and 1.75 hours plus setup across up to three attempts;
cumulative evaluation spend is capped at USD12. Etude retries only a refused or
lost machine, resuming the same run and skipping stored matches, so no game
result can be replaced; training keeps no automatic retry.

Each game has a 600-second wall limit, enforced by killing the game and counting
it failed. MTG-213's 120 seconds would penalize the larger network for its own
latency: one thread takes about 0.4 s per w256 decision against 0.15 s at w64,
across games averaging 180 decisions. The arena journals in memory and copies
its files out afterward, because its per-record sync takes about 0.1 s on a
rented container disk and counts against that limit.

Milestones past a size's endpoint stay in the schedule as unreached slots; failed
or truncated games stay failed in their scheduled slots and are never replaced,
so no pass uses Etude's `--retry-failed`.

### Evaluation driver

One command drives every pass, from a checkout of the frozen revision on an
always-on host beside the controller. It starts after the host screen. If that
host is a laptop, prefix it with `caffeinate -is`, and check from `etude status` that
the controller is ticking. A booted pod ends itself at its deadline or budget, but
only a ticking controller ends a pod that never booted and hands passes on:

```bash
"${control[@]}" etude eval experiments/tla_frontier.py --source <store> \
    --rent cpu32 --watch 14400
```

Every pass must run from that revision: a later commit changes `seats.source`,
which analysis flags as a mixed cohort. The four-hour period batches both sizes'
new milestones, so each milestone is played within one period of arriving.
Expect about 36 comparisons over about 15 rentals, for about USD8.50 with setup
included and unmeasured, assuming no dud pods. Etude does not enforce the USD12
cap across passes:
the operator reads spend after each pass and stops the loop when it reaches
USD12. That stop and the USD12 reserve cover pods that never boot but still bill.

To see progress at any time, draw the report from the same checkout:

```bash
"${control[@]}" etude report experiments/tla_frontier.py --source <store>
```

The report is descriptive curves by update; `experiments/tla_frontier_analysis.py`
owns every reading below.

Reading progress changes nothing. Past the host screen, no run is cancelled,
relaunched or extended because of a score. A stop for cost, a dead run or any
outside reason is a recorded decision that keeps its evidence and has no scale
reading.

Etude's launch ledger, `~/.etude/experiments/launched.json`, is one file for
every experiment on the computer, and any entry for a failed pass raises
`launched_pass_failed` and ends the watch loop. Before starting the loop, check
each entry's status there, including other experiments'. After a failed pass,
look at why it failed, remove its entry from that file and restart the loop. A
milestone delayed this way is still played; its failed games are not replayed.

## Reading

`experiments/tla_frontier_analysis.py` produces these readings. Scores are the
candidate's mean game score with 95% whole-pair bootstrap intervals (seed
21592001), for comparisons whose 16 pairs all completed. Head-to-head rows put
w256 first. A milestone past an ended run's endpoint is unreached and expects
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
- **Scale**: the equal-compute endpoint head-to-head is primary, with the same
  bounds; equal-update head-to-heads describe sample efficiency.

A failed size keeps its learning and strength readings at its last stored
checkpoint, reported beside the run's state. Its endpoint head-to-head is still
reported, but not as equal compute. The update cap is a second stop, so a size
reaching it spent less than the wall budget. The scale reading therefore needs
both runs finished with neither endpoint at the cap, and is otherwise absent,
with an issue saying so.

Each comparison and the whole analysis also total the first-64 cutoff: decisions
and dropped actions, over the games that returned a count. A game killed at its
limit returns none, so those totals can be low. An overflow is accepted behavior:
a nonzero count is reported, adds no issue and leaves `complete` unchanged. The
totals span both players, so they support no per-policy rate.

Report every attempt, failure, resource-limit discard, set-aside run, unreached
milestone and cost. One seed per size supports a descriptive frontier reading, not a
controlled claim that one size is better in general.

## Lost rental

A size whose rental dies stops at its last export. By default that is its
endpoint: it keeps its learning and strength readings and has no scale reading.
A run whose pod dies without recording an end stays `running`, so its endpoint
stays unknown until the launcher records the end. The operator reads its stored
checkpoints meanwhile.

Any further training is a recorded decision within the unspent part of that
size's USD38. It continues from the last export under a new protocol version and
does not restart: a restart cannot be afforded late and would compare a shorter
run. The precedent is MTG-183's w3 to w4, through
`LearnerContinuation.from_monitoring`: raw weights, the EMA reset to them and a
fresh Adam, a declared discontinuity.

This version does not build that path, and two limits stand in its way.
`size_run` admits one run per size, so the new version must read a chain of runs.
`stage_monitoring` in `experiments/elemental_long.py` needs the parent's
`run.json`, which a truly dead pod may not have delivered; this is unproven.

## Budget

The Task's ceiling is USD100 including setup, failures, retries and evaluation:
2 × USD38 training + USD12 evaluation = USD88, leaving USD12 for calibration
(USD2), the evaluation rehearsal (a USD2 ceiling, about USD0.50 expected),
cancelled pods at the host screen (about USD3 at most), setup and failure costs.
Any retry needs a recorded decision that keeps the total within USD100.
Evaluation never runs on a laptop.
