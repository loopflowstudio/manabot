# b015d5 training throughput experiment — direction (2026-11-06)

Jack Heart's direction. b015d5 itself is the experiment Task; no separate Task.

- Work on training performance at the system level: machine shape and training
  regime. Optimizing the engine is not where this Task's energy goes.
- Budget: USD 50 total, a hard cap covering training, evaluation, failures and
  machine setup. Do not use more than the question needs.
- Agents may run subgraphs on this Task as needed.
- Merge PR #435 once CI is green (Jack Heart via c5fba4's design review,
  2026-11-06); c5fba4 freezes its protocol only after #435 lands.
- Validate first that collection is the right bottleneck. Retained fast-L4 receipts
  read about 17.6 s collecting and 15.1 s learning, predating PR #435's fixes, so
  collection may be only about half an update and the removable idle share is a
  bound, not a measurement. Regime arms depend on this result.

## Candidate arms, in order

1. Measure the update on real CUDA with PR #435's attribution: collection, learner,
   idle stream share, GPU and CPU use. Cheapest step; gates the rest.
2. Seeds per machine (Jack Heart: worth experimenting with): 1, 2, 4 and more seeds
   on one machine; seconds per update, dollars per learner decision, GPU/CPU use and
   host memory, on named hosts. A packed seed must learn as it would alone. One
   256-stream run peaked near 9-10 GB host memory on laptop CPU device; recheck on
   CUDA. Check whether Etude can launch several seeds on one pod; fix gaps in Etude.
3. Stop waiting for the slowest game (Jack Heart: a valid experiment, once collection
   is validated). Two-seat idle slots are .63-.67 of stream slots, at most 29-34% of
   wall (07c490). This reopens two-seat exact quota, which an earlier steer on 07c490
   left unbuilt. Refilling finished streams is the alternative; check its
   short-game bias.
4. Overlap collection and learning: one-update-stale policy, needs a second core.
5. Multi-producer `elemental-uniform`: owned by Task c5fba4, which changes
   `collection.py` so deck-source roster sampling supports 2-4 producers after #435
   lands. Do not duplicate it here; read its result. The one-producer rule looks like an
   implementation limit: `RootRoster.pair` samples decks from each game's seed alone;
   `ShardedVectorEnv` lacks `set_uniform_roster`, `set_training_decks` and
   `current_player_configs`, and `SeatRoutedCollector` and `TrainingRegime` reject the
   combination. Producers speed only stepping, about 20% of an update, and compete
   with packing for cores.

## Measure

Strength per dollar and per hour, not seconds per update: a regime can make updates
faster and learn less from each. Compare arms on the same host; hosts differ
1.6-1.8x in single-thread speed. Retain `TrainingRun.host`. Current recipe is
ten-deck Elemental. Do not rewrite frozen worlds, observation versions or recipes:
a new regime is a new declared variant. Task 07c490 owns the one-seat closure study.

## Design status (2026-11-06)

Owning artifact: `experiments/protocols/training-throughput-b015d5-v1.md`, accepted
by Jack Heart at experiment-design review (comment 14e6cc54) with fixed gates
replaced by step-at-a-time choice; no spending authority beyond execution review.
Nothing has launched.

- Exists: step-1 probe software (bench `--json`/`--recipe`/`--cuda-counts`,
  `experiments/training_throughput.py` phase driver and samplers) with its CPU
  acceptance test.
- Step 1: `probe` job, 2 L4 + 1 A4000 runs; phases solo, pack-2, pack-4, CUDA
  counts. Reports the phase tree, idle share, CPU/GPU use, peak memory and
  $/M learner decisions; Jack Heart chooses the next step from those readings.
- Blocked on: PR #436 (07c490: `elemental_variants.elemental_recipe`,
  `ELEMENTAL_PAIRS`) is open and BLOCKED, Integration failing (checked
  2026-11-06). Step 1 freezes when it lands or the probe pins its exact commit.
  Re-quote L4 and A4000 stock before launch.
- Later: step 2 packing needs Etude machine slots (Etude Task 8ffc3e, approved);
  steps 3-4 build after c5fba4's `collection.py` change. Proposed reservation
  $24.40 of $50.
- Check 2026-11-06: `uv run --extra dev pytest tests/training/test_training_throughput.py -q` — 3 passed.

## Concept review (2026-11-06, graph step, no person): done in implement

The three directions from reviewing `8d178f40e`..`73e82311f` are implemented:

1. Per-update rows are retained. Bench takes `--out DIR` (still temporary without it),
   and the probe passes `seed-N/run`, so each child's Etude metrics run (per-update
   `time/*`, idle and learner rows) is published next to `receipt.json`. The CPU
   acceptance test reads every child's `metrics_run` under its folder with
   `read_updates` and requires one `time/*` row per update. A child's folder is about
   40 MB on CPU at 8 streams and 2 updates (mostly 6 MB checkpoints), so the checkpoints
   stay in the copy.
2. The protocol and the bench docstring state P4 as launches and syncs per update (run
   total / updates). Per-forward and per-minibatch figures are upper bounds only.
3. The launch lines use `--rent l4` / `--rent a4000`.

Check: `uv run --extra dev pytest tests/training/test_training_throughput.py -q` passed (3).

Unchanged: Step 1 still waits on PR #436, which was OPEN as of this review. Phases, seeds,
the memory skip rule, the failure semantics and the budget match the protocol.
