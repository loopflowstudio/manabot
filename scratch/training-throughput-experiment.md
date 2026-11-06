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

## Design (experiment-design pass, 2026-11-06)

Owning artifact: `experiments/protocols/training-throughput-b015d5-v1.md`, a draft
awaiting Jack Heart's review in the Task conversation. Nothing has launched. Summary:

- Step 1 is a measurement-only `probe` job, one Etude run per machine: 2 L4s
  (6 vCPU / 48 GB) and 1 A4000 (16 vCPU / 48 GB). Phases run in sequence on the
  same host: solo, pack-2, pack-4, then a CUDA kernel/sync count. It reads the
  phase tree, idle share, CPU cores, GPU busy share, peak memory and $/M learner
  decisions. Gates G1 (collection >= 50% and idle >= .50), G2 (overlap bound
  >= 25%) and G3 (pack-2 gain >= 1.6 with memory headroom) decide steps 2-4.
- Packing needs no strength runs: separate processes under a synchronous update
  learn the same by construction. A CPU digest and the packed seeds' metric
  spread check it. Production packing needs an Etude machine-slots PR; Etude
  runs one attempt per machine at a time (`etude-sdk/docs/launcher.md`, "no
  run-count packing setting").
- Step 3 is two-seat exact quota with per-seat carry-over (quota ~190 by
  calibration). Refill-with-discard is built only if an offline bias check of
  game lengths stays within 5%.
- Step 4 is one-update-stale overlap, as a learner thread.
- Strength arms: 4 seeds (26101-26104), 75 updates, exports every 5 updates,
  07c490's 50 `ELEMENTAL_PAIRS` vs heuristic. Primary clocks are wall time and
  dollars; secondary is learner decisions; paired by seed.
- Proposed reservation $24.40 of the $50 cap; the rest is unallocated.

Dependencies before freeze: PR #436 (07c490) must land, because it carries
`elemental_variants.elemental_recipe` and `ELEMENTAL_PAIRS`. The probe software
(bench `--json`/`--recipe`, `experiments/training_throughput.py`) is not yet
written. Steps 3 and 4 build after c5fba4's `collection.py` change.

## Design direction (Jack Heart, 2026-11-06)

- No strict gates: take one step at a time and choose the next from the readings, putting energy where it pays.
- Etude machine slots approved to build and ship: Etude Task 8ffc3e.
- PR #436 (07c490) supplies the control recipe; its CI repair is in progress.
