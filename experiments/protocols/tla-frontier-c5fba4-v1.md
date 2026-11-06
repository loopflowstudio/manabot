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
  Training seed 21501 for both. No automatic retry. Both sizes train at once,
  so equal-update comparisons appear as they go; a cohort's default is one
  machine at a time, so the launch names two:

  ```bash
  doppler run -p etude -c prd -- doppler run -p manabot -c prd \
      --preserve-env=RUNPOD_API_KEY -- uv run --extra artifacts --extra evaluation \
      --with "etude-sdk[s3,eval]==0.2.14" etude launch experiments/tla_frontier.py \
      --revision <frozen> --rent <gpu> --cohort tla-frontier-c5fba4 \
      --parallel 2 --max-usd 76
  ```

  The nested Doppler projects supply provider and artifact access. Etude 0.2.14
  is the control client only: the locked 0.2.12 cannot read the launcher's Vast
  provider rows, and workers keep 0.2.12.
- **Hardware.** Both sizes run on the same GPU model, chosen at launch and
  recorded before any score is read; different models void the equal-compute
  reading. Etude's GPU templates set a minimum vCPU count, not an exact one, and
  collection dominates update cost, so the same check records both pods' vCPUs.
  A difference is reported and does not void the reading: equal wall at one
  price is still equal dollars. USD38 over 56 hours admits at most USD0.67/hour,
  which covers the USD0.59/hour L4 and A40 of earlier receipts but not the
  USD0.96/hour RTX 5090, where the dollar ceiling would stop training first.
- **Calibration.** Before the training runs, and after this protocol merges,
  `experiments/tla_frontier_calibration.py` runs each size for five updates on
  the chosen GPU: at most USD1 and 1.2 hours per size, USD2 in total, inside the
  reserve. It reads seconds per update and CUDA peak, under its own experiment
  and regime ids, so it is never a frontier run and plays no comparison.
  W256 launches at 1,500 s/update or less, about 130 updates in 54 hours, which
  leaves head-to-heads at 25, 50 and 100. If it cannot fit or reads above that
  line, the size changes in a new protocol version, still before any scoring.

## Evaluation schedule

Every comparison plays the same 16 fixed pairs (deck seed 21590001, deals
21591000–21591015), four legs per pair swapping decks and seats: 64 games. The
heuristic is the matched first-64 heuristic. Shared update milestones are 25,
50, 100, 200, 300, 400, 600, 800, 1000, 1200, 1600, 2000 and 2400.

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
always-on host beside the controller:

```bash
doppler run -p etude -c prd -- doppler run -p manabot -c prd \
    --preserve-env=RUNPOD_API_KEY -- uv run --extra artifacts --extra evaluation \
    --with "etude-sdk[s3,eval]==0.2.14" etude eval experiments/tla_frontier.py \
    --source <store> --rent <cpu32> --watch 14400
```

Every pass must run from that revision: a later commit changes `seats.source`,
which analysis flags as a mixed cohort. The four-hour period batches both sizes'
new milestones, so each milestone is played within one period of arriving.
Expect about 15 rentals at USD0.45, setup included and unmeasured, for about
USD7, assuming no dud pods. Etude does not enforce the USD12 cap across passes:
the operator reads spend after each pass and stops the loop when it reaches
USD12. That stop and the USD12 reserve cover pods that never boot but still bill.

To see progress at any time, draw the report from the same checkout:

```bash
doppler run -p etude -c prd -- doppler run -p manabot -c prd \
    --preserve-env=RUNPOD_API_KEY -- uv run --extra artifacts --extra evaluation \
    --with "etude-sdk[s3,eval]==0.2.14" etude report experiments/tla_frontier.py \
    --source <store>
```

The report is descriptive curves by update; `experiments/tla_frontier_analysis.py`
owns every reading below.

Etude's launch ledger, `~/.etude/experiments/launched.json`, is one file for
every experiment on the computer, and any entry for a failed pass raises
`launched_pass_failed` and ends the watch loop. Before starting the loop, check
each entry's status there, including other experiments'. After a failed pass,
look at why it failed, remove its entry from that file and restart the loop. A
milestone delayed this way is still played; its failed games are not replayed.

A run whose pod dies without recording an end stays `running`, so its endpoint
stays unknown until the launcher records the end. The operator reads its stored
checkpoints meanwhile.

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

Report every attempt, failure, resource-limit discard, unreached milestone and
cost. One seed per size supports a descriptive frontier reading, not a
controlled claim that one size is better in general.

## Budget

The Task's ceiling is USD100 including setup, failures, retries and evaluation:
2 × USD38 training + USD12 evaluation = USD88, leaving USD12 for calibration
(USD2), setup and failure costs. Any retry needs a recorded decision that keeps
the total within USD100. Evaluation never runs on a laptop.
