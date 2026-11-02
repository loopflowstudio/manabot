# Evaluating manabots

How manabot measures checkpoints: replay-checked arena games, the shared
inference path that serves checkpoints to them, Etude benchmarks and Duels,
17Lands-style card data, tactical checkpoint scenarios and frozen-policy
attacks. Training-time monitoring that calls this machinery is described in
[training.md](training.md); how to read experiment reports is in
[experiment-metrics.md](experiment-metrics.md).

- [Ownership and arena](#ownership-and-arena)
- [Quick local evaluation](#quick-local-evaluation)
- [Shared inference](#shared-inference)
- [Continuing an evaluation](#continuing-an-evaluation)
- [Performance evidence and calibration](#performance-evidence-and-calibration)
- [Card data](#card-data)
- [Checkpoint scenarios](#checkpoint-scenarios)
- [Frozen-policy attacks](#frozen-policy-attacks)
- [Open direction](#open-direction)

## Ownership and arena

Simulation (`manabot.sim`) owns serving, native games and bounded replay.
Evaluation (`manabot.eval`) owns admission (`eval.admission`), requests and
results (`eval.evaluation`, `eval.results`), paired cells (`eval.match`),
ratings (`eval.rating`, seat-aware Bradley-Terry with paired-block bootstrap)
and callback transport (`eval.endpoint`). Training monitoring is an adapter in
`manabot.experiment.checkpoint_evaluation`; local execution imports neither it
nor callback transport. `manabot.eval.models` holds only the identities
`EvaluationKey` (the arena a game was played in) and `MatchRow` (one played
leg); wire field names keep their recorded meanings because frozen evidence
carries them. Tests load the INT-6 fixture's key and anchors through
`tests/eval/arena_fixture.py`.

One deal is one complete block: four correlated deck/seat legs by default,
eight with mirrors. Legs are not independent samples; uncertainty is a
paired-deal bootstrap over whole deals, and a single deal has none. Only
terminal, replay-verified games score. Draws, crashes, timeouts and failed
replays are retained by outcome and never earn credit.

Working anchors are the heuristic player and frozen checkpoints. The code
anchors `scripted_greedy` and `random` (`manabot.sim.anchors`, frozen v1
sources) remain admissible for compatibility and diagnostics, not as working
anchors.

## Quick local evaluation

Score a current-world flat checkpoint against the heuristic on the selected
Allies/Lessons setup:

```sh
uv run python -m manabot.eval.local \
  --checkpoint .runs/small/policy.pt --opponent heuristic \
  --deal-seed 101 --out .runs/local-check --seconds 300 --workers 2
```

- `--opponent` also accepts another local `.pt`, `scripted_greedy` or `random`.
- Repeat `--deal-seed` for more blocks; `--include-mirrors` plays all eight legs;
  `--matchup` takes an `AuthoredMatchup` JSON for another admitted roster. The
  checkpoint must already admit that setup; nothing is converted.
- `--workers`, `--batch-rows`, `--resident-mib`, `--game-seconds` and
  `--max-commands` set limits. `--seconds` bounds the whole lifetime:
  checkpoint preparation (in a killable child), serving startup, play and replay.
- `--request request.json --out NEW` repeats a saved request in full. Repeated
  seeds are not new samples.

It needs no training records, credentials or the `evaluation` extra, uses CPU
with one Torch thread per process, and never rents hardware, enters an Etude
queue or starts a callback endpoint. Exit 0 means a complete replayed cohort;
1 means an incomplete attempt. An admission timeout retains
`admission-failure.json`. Results are labelled `local-check-not-admission`:
a quick check, not promotion or held-out evidence.

Typed callers use `manabot.eval.local.checkpoint_request` and
`manabot.eval.evaluation.evaluate_local(request, limits=..., out_dir=...)`.
An unknown training seed stays `None`, never inferred from deals.

`evaluate(request, out_dir=..., inference=...)` is the single execution entry
point shared by the local CLI, training monitoring and Etude benchmark/Duel
callbacks. Both public evaluators reject an existing output directory, empty or
not, before writing anything. `evaluation.json` retains the request, effective
execution, rows, status, elapsed time and paired-deal intervals; incomplete
cohorts emit no interval. Journals, Commands, replay and performance records
use arena formats under `<out>/panel`. Lower down, `eval.match.play_cell` plays
explicit leg subsets.

Etude entry points are `manabot.eval.benchmark` (one Etude match is one deal
with all eight legs; Etude owns match records, resume and uncertainty) and
`manabot.eval.duel`.

## Shared inference

`manabot.sim.execution` declares a local allocation in three layers:

- `EvalCompute` (`shared-eval-v2`): device and dtype, i.e. how a served
  policy's numbers are computed. This is part of registration identity.
- `EvalAllocation`: active games, batch rows, batching delay, resident-model
  bytes, request timeout, catalog cache and residency.
- `EvalExecution`: absolute Unix deadline plus instrumentation switches.

Allocation and attempt settings live in receipts, not registration hashes:
batch composition already varies with arrival order and every row samples from
its own decision seed, so more games or larger batches keep Etude model,
benchmark and match identities. Changing device, dtype or contract version
creates new identities. `seed_compute_envelope_id` preserves the original seed
authority across serving changes. `shared-eval-v1` (through PR #389) also
hashed allocation settings; its records stay readable as `EvalComputeV1` /
`EvalExecutionV1` but are not reused or continued under v2 (Jack Heart,
2026-10-28).

With an execution, flat stateless checkpoints are served; without one,
`play_cell`, `evaluate` and the monitor play them scalar in the game child
under their original registrations. Served and scalar players of one checkpoint
therefore have distinct identities. Retiring scalar checkpoint play awaits a
CPU measurement of served play after direct game connections (Jack Heart,
2026-10-28). Code and search players always play in their game child.

```python
from pathlib import Path
import time

from manabot.eval.evaluation import EvaluationRequest, evaluate
from manabot.sim.execution import EvalExecution
from manabot.sim.serving_protocol import InferenceRuntime

execution = EvalExecution(device="cpu", max_active_games=4, max_batch_rows=16,
                          max_delay_seconds=0.005, resident_model_bytes=1 << 30,
                          deadline_unix=time.time() + 300)
with InferenceRuntime(execution, Path(".runs/eval/allocation")) as inference:
    result = evaluate(EvaluationRequest(...), out_dir=Path(".runs/eval/panel"),
                      inference=inference)
```

Each allocation needs a new directory; concurrent local callers may share one
runtime and its game slots. `evaluate_checkpoint(..., execution=...)` runs a
monitor panel this way; `benchmark.evaluate` and `duel.evaluate` connect Etude's
spawned callbacks to one authenticated Unix-socket endpoint, so Etude worker
count cannot multiply the allocation's limits. Each invocation retains its
receipts under a fresh `allocation/<attempt>/`.

**Processes.** One allocation runs one serving child (`manabot.sim.serving`),
the only process that loads served weights. `PolicyKey` groups requests by
checkpoint, observation/action ABI, world, device, dtype, determinism and
implementation; matchup is not a model input, so games seating one checkpoint
share its batches across matchups (setups are still validated at admission).
Batches flush on row cap or delay. Each game child talks to the service over
its own socket (`InferenceClient`); the supervisor keeps lifecycle, deadlines,
failure classification and the Command journal, retaining each Command before
native execution. CPU and CUDA share one serial service with one Torch thread;
CPU-specific engine work waits for a calibration showing it saturated.

Children start through `manabot.sim.child`, never `multiprocessing` spawn.
Only search players (flat-MC, PUCT, local update) load Torch in a game child;
replay children never do. `manabot.sim.threads` starts children single-threaded
and enforces thread budgets (4 game/replay, 16 serving) as infrastructure
failures; `effective_cpus()` honors affinity and cgroup quotas.

**Failure.** Every inference call is capped by the request timeout, the game's
remaining wall time and the allocation deadline. An unanswered request, dead
service or failed handoff is an infrastructure failure that fails the
allocation, never a player forfeit; unstarted identities go to
`unattempted.json`. The deadline also bounds replay: incomplete replay is
unavailable, not a mismatch. Replay verifies Commands, not cross-device policy
equivalence.

## Continuing an evaluation

```sh
uv run python -m manabot.eval.local --status .runs/local-check/panel
uv run python -m manabot.eval.local --continue .runs/local-check/panel --seconds 300
```

Continuation plays only the missing deals, rerunning any partial deal whole;
only the deadline may change locally. Typed callers use
`evaluation_status(directory)` and `continue_evaluation(directory,
execution=...)`, or pass an existing runtime. Device, dtype, batching and
residency must match; a changed request needs a fresh evaluation. Relocated
checkpoints may be passed explicitly and are rechecked by digest and size.

The first `evaluation.json` freezes identity and request; later attempts write
under `attempts/<sequence>/` without replacing earlier traces, and completed
blocks are immutable. Filesystem claims admit one writer (not distributed
leases), so move an evaluation with its whole directory. Re-entering an Etude
benchmark or Duel continues a failed match; a cached success adds no sample.

## Performance evidence and calibration

Cells write `performance.json`, `rows.json`, journals and traces; allocations
write `execution.json`, `allocation.json` and `serving.json`.
`diagnostic_batches` adds profiler traces, never mixed with ordinary
throughput. Cell, allocation and evaluator elapsed times are separate
intervals; none includes CLI checkpoint preparation or billing, and receipts
carry no cost. Etude cache reuse adds no measurement.

```sh
uv run python -m experiments.report_prototype.build \
  --eval-receipt .runs/eval/panel --out .runs/eval/report
uv run --extra evaluation python -m manabot.eval.serving_benchmark \
  --plan .runs/eval-serving/plan.json --out .runs/eval-serving/result
```

The report viewer never sums overlapping clocks;
`./scripts/verify-eval-performance-report` is its headless proof (needs locked
`academy/ui` dependencies and Playwright Chromium). The serving benchmark's `BenchmarkPlan` requires the variants
`cold_scalar`, `resident_only`, `resident_catalog` and `batched`, three
interleaved repetitions, tolerances, thresholds, a memory cap and a deadline. It
reports whether declared thresholds were met, not an adoption decision; CPU
fixture tests establish correctness only.

## Card data

17Lands-style card statistics from evaluation games. A card evaluation is an
Etude eval run naming an **experiment**; all runs of an experiment pool in one
kept Etude database (`ETUDE_HOME/statsdb`). Each match plays one deal's eight
legs. The native recorder runs during Command replay, outside observations,
rule hashes, RNG and search clones.

```sh
uv run --extra evaluation python -m manabot.eval.cards evaluate --spec evaluation.json --out .runs/cards
uv run --extra evaluation python -m manabot.eval.cards data --experiment EXP --source .runs/cards [--html page.html | --json]
uv run --extra evaluation python -m manabot.eval.cards audit --experiment EXP --source .runs/cards --replay
```

`CardEvaluationSpec` (`manabot/eval/card_evaluation.py`) names experiment and
run, registrations, checkpoint paths, unique deal seeds, limits and an optional
`AuthoredMatchup` (default Lessons/Allies). Resuming requires identical players
and conditions. `data` filters with `--subject`/`--opponent` (name, Etude
identity or 8+ characters of checkpoint digest), `--position`, `--deck`,
`--opponent-deck`, `--run`, `--commit`; `--rebuild` rereads all blocks.

Any evaluation calling `play_cell` can deliver card data: pass
`card_evidence_dir`, merge `trace_card_tables(...)` tables and blobs into its
Etude `Result`, and declare `rollups=CARD_ROLLUPS` (`manabot.eval.card_metrics`).

**Columns.** GP (games with the card in deck), OH (opening-hand copies), GD
(copies drawn later), GIH = OH + GD with GIH WR = winning qualifying copies /
qualifying copies, GNS (in deck, never in hand), IWD = GIH WR − GNS WR in
points. Counts are copy-weighted; tutors, bounce and generated cards are not
draws. Only won/lost games with a passed replay are eligible; there is no
implicit minimum sample. These are associations, not causal card values. The
GIH interval is a deterministic 95% percentile bootstrap over whole deal seeds,
one draw of deal weights per filter selection; it needs two deals.

**Tables.** Each match delivers `card_plays` (per game and player),
`card_counts`, `cards` and the shared `plays`; `seat` indexes `Match.players`,
`game_seat` is the native seat (0 on the play). Queries read only `CARD_ROLLUPS`
(`card_deals`, `game_deals`) through `manabot.eval.card_query.card_db`, never
card rows; a changed rollup takes a new name. Games pool across source commits,
which are provenance and a filter. A crashed or unreplayed game is an
ineligible row; its deal's other games still count. Evidence is a private
blob per game read with Etude's `read_blobs`; `etude convert-eval` makes older
per-file runs readable.

**Arena logs.** `cards export-log --run RUN --game GAME --viewer 0|1 --mapping
mapping.json --out game.log` writes `arena-card-log-v1` (`eval.arena_log`).
`CardMapping` binds content keys to retained Scryfall responses with verified
`arena_id`; export refuses missing mappings and never joins by name. Logs omit
opponent private cards and library order, are synthetic, and must never be
submitted to 17Lands; `tests/fixtures/seventeenlands` pins the upstream client.

**Scale.** MTG-206 (2026-10-27): 5,000 player-games and 90 cards rendered 363
filter combinations in 16.01 s wall; a $0.03 L4 smoke audited 96/96 games
clean. The `Card evaluation scale proof` workflow runs
`manabot.eval.cards verify --games 1024` on CPU as an engineering proof.
Details: [evaluation.md#card-data at a74f31828](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/card-evaluation.md).

## Checkpoint scenarios

Score retained tactical positions S1–S5 with the ordinary checkpoint player,
without training:

```sh
uv run python -m manabot.verify.checkpoint_scenarios --checkpoints checkpoints.json --out .runs/tactics-001
uv run python -m manabot.verify.checkpoint_scenarios --out .runs/tactics-001 --report-only
```

`checkpoints.json` maps scenario names (`s1_counter_the_bomb`,
`s2_hold_the_wipe`, `s3_bolt_the_threat`, `s4_race_vs_block`,
`s5_hold_up_quench`) to checkpoint paths; omitted scenarios are unsupported.
Each checkpoint must bind that scenario's exact deck pair from
`manabot.verify.scenarios.SCENARIOS`, so an Allies/Lessons checkpoint cannot
score them; a mismatch is a failed admission, not a tactical loss. Belief
checkpoints are unsupported because injected roots lack prior history. The
runner uses CPU, one thread, deterministic actions, `--seed 2` and
`--max-steps 1000` by default, and updates no weights.

Each premise is revalidated first (failure: unsupported). A score requires the
reference behavior and its resolved effect; technical failures record no
score. These are local behaviors against a scripted continuation, not
strength, and must not be pooled. `results.json` and `traces.jsonl.gz` retain premises, attempts, digests
and Commands; `--report-only` regenerates the report by replay without
weights. Tests: `tests/verify/test_checkpoint_scenarios.py`.

## Frozen-policy attacks

Bounded adversarial evaluation, not exact exploitability. An `AttackPlan`
(`manabot/experiment/attacks.py`) fixes target checkpoint bytes and SHA-256,
producer seeds, at least two independent attacker seeds, a strictly increasing
`cumulative_updates` ladder, shared `final_deal_seeds`, a one-stage self-play
`template` regime and time budgets (`total_seconds`, `evaluation_seconds`,
`game_seconds`, `max_commands`) plus a predeclared `prediction`. Each
target/attacker pair becomes an ordinary TrainingRegime against the frozen
target; rungs continue one attacker.

```sh
uv run python -m manabot.experiment.attack_execution --plan attack.json --out .runs/attack-001
```

Requires a fresh output directory, the selected matchup and a POSIX main
thread (it owns a SIGALRM deadline and refuses an active timer). Literal seed
reuse across producers and attackers is rejected. Every attacker faces its
target before training and at every rung on all four legs; only terminal,
replay-verified games count. A failed cell stops the cohort, and results keep
`plan.json`, `result.json`, `report.md`, `training.sqlite`, run directories and
arena evidence. Final paired improvement bootstraps over attacker seeds and is
exploratory. A failed bounded attacker never certifies equilibrium. The
real-PPO positive control raised score against a deliberately passive toy
target from .59375 to 1.0 in 64 updates. History:
[evaluation.md#frozen-policy-attacks at a74f31828](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/frozen-policy-attacks.md).

## Open direction

Jack Heart's accepted direction (2026-10-24) makes checkpoint-versus-checkpoint
evaluation at growing model sizes the primary capacity workload; small local
checkpoint-versus-heuristic checks keep the same semantics without training,
cloud, queue or callbacks. The accepted cloud startup policy targets at least
two hours of estimated eligible useful work on the chosen host/model mix before
a server starts, releasing idle compute when work finishes. That gate belongs
in Etude's launch scheduler and is not implemented. Compute policy, a durable
Etude priority queue, adaptive allocation and further serving optimizations
remain unselected designs; see
[evaluation.md at a74f31828](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/shared-evaluation.md).
