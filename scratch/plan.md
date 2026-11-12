# MTG-318 plan

## Built

- `experiments/collection_tail_graphs.py` (commit `8b753ade3`, pushed): `job`
  runs insitu (3 updates, frozen w160l8 regime on the current world, bucketed
  on; per-call phases, recording split into append/pack/rest with page faults,
  store growth and GC pauses; tail-weighted kept inputs), probe (sync sites,
  dynamo explain, whole-forward capture attempt), replay of arms eager /
  bucketed / compiled / trunk-graph at w160l8 and w384l8 plus a bucketed
  repeat, then analyze -> `summary.json` with `decision` per pre-registered rule.
- Smoke: `ETUDE_OUTPUT_DIR=/tmp/x uv run python -m experiments.collection_tail_graphs job --smoke`
  passed every stage on CPU (2026-11-12).

## Findings so far (from code and MTG-301 evidence, not yet the L4 run)

- Coalescing has nothing to merge: `_collect` makes one policy call per
  environment step over every active stream (MTG-301 in situ: 1,717 calls =
  1,717 steps). The tail is complete-game closure with few active streams.
- MTG-301 bucketed replay traces: small calls spend ~5 ms host in `prepare`
  (host bucket measurement) against ~1 ms unbucketed; that is most of the
  1.38x small-call cost.
- CPU dynamo explain (random init, 1 row): ~20 breaks: `named_modules` in
  admission binding, `pack_padded_sequence`/PackedSequence GRUs, `len(rows)==0`
  in facts, `len(live_positions)==0` in shared effects (CPU-only `used` columns
  too).
- Observation rows are ~217 KB each; 49,943 rows/update -> ~11 GB store per
  update, allocated fresh (np.empty) each update and doubled on growth. Prime
  suspect for the recording gap: first-touch page faults and growth copies.

## Result (run `1a2e6cb86345`, ~$0.27, 2026-11-12)

- No path accepted (rule 1): compiled 1.32x/1.19x, trunk graph 1.17x/1.06x on
  calls <= 8 rows (w160l8/w384l8); whole-forward capture fails at
  `buckets.py:307`; 23 host syncs per bucketed forward remain. Coalescing: 0
  calls removable (calls == steps). Recording 2.7-4.6 s/update on this host.
- Report written in `experiments/collection_tail_graphs.md`; data published
  under `experiments/data/collection_tail_graphs/` (manifest committed).
- No `Execution` switch, no docs change: the pre-registered rule says the report
  is the deliverable.
- Checks: `tests/infra/test_experiment_data_untracked.py` and experiment
  discovery test pass.
