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
