# MTG-273 — collection slowdown

The inherited b015d5/C1/C2 notes belong to another Task. Their complete text is
retained at `757bb6257ef07cf2536ad7f7ec24f0f7309628c6:scratch/questions.md`.

## Scope and reversible choices

The Task asks for the cause of the width-64 frontier slowdown and a measured
repair. No specific implementation design was present. Use a bounded metric
writer fixture, retained frontier evidence and source inspection; do not run a
training campaign, induce load, change frozen runs or touch other Tasks' processes.

The existing SDK carries immutable author files under `artifacts/`. Use that
public storage contract for one content-addressed outcome table per update,
with one digest reference in its existing metric row. This preserves the SDK's
resume/duplicate-update selection, shipment and relocation. Keep the existing
flat outcome API as an offline reader projection; never write those names again.
Historical scalar outcome metrics remain readable without rewriting evidence.

## Evidence and limits

- The frontier owner's retained note at
  `/Users/jack/src/manabot.c5fba4-frontier-self-play/scratch/frontier-evaluation-direction.md`
  reports: “The first native training reader expanded a1.303GB permanent metrics
  manifest to about14GB resident memory and was stopped.” Its index names
  107736 matchup metrics (134 × 134 × 2 × 3). This is another run's observation,
  not a measurement performed here or proof of all collection slowdown.
- Installed Etude 0.2.15 `train.manifest.write_manifest` loads the entire
  permanent manifest even on `permanent=False`; permanent writes rebuild all
  block descriptions. `Run._fold` calls it in the background writer thread.
- Bounded actual-SDK fixture: 16 pairs × 2 seats × 3 counts, one update per
  permanent block. At 1/4/8 blocks, manifest bytes were 62,645 / 202,629 / 389,276;
  non-permanent refresh tracemalloc peaks were 292,833 / 1,093,580 / 2,160,384
  bytes, elapsed 0.00178 / 0.00442 / 0.00804 seconds. No games or load generation.
  An initial fixture used a non-hex writer ID and selected no blocks; discarded.
- This reproduces history-dependent metric indexing, not the complete original
  CUDA run or an attribution of every extra collection second. The SDK also has
  residual history-dependent indexing for ordinary low-cardinality metrics.

## Open acceptance

`scratch/mtg-273.md` owns the implemented approach, retained checks and remaining
acceptance. The original-run causal profile and post-repair long-run RSS/time
evidence remain missing. If retained evidence cannot settle them, the exact
bound and placement of further ordinary execution remain for demo; the small
index fixture does not authorize or substitute for a campaign.
