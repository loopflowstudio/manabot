# TLA trophy pilot: implementation proof, GPU launch pending

As of 2026-11-03, the corpus, independent per-seat sampler, GPU job and paired
evaluation runner are implemented. **The requested GPU pilot has not started.**
The earlier missing-credential claim was based on the wrong Doppler project.
Jack Heart corrected it on 2026-11-03: `manabot/prd` supplies R2 artifact
credentials and `etude/prd` supplies the RunPod key. No secret was copied or
changed. Provider access uses both existing configurations:

```sh
doppler run --project manabot --config prd -- doppler run --project etude --config prd -- uv run etude availability l4 --timeout 20
```

The latest continuation, prompted by Jack Heart's 20:03 UTC capacity update,
also failed admission. L4 launch `949ad568707d` / requested run `9453ce06ef46`
quoted $0.59/hour (maximum $0.7375), then lost capacity during source
preparation. The source is `b8835f63b4e7132ae6fd3f86cc73cefa6716e57f`.
Etude's existing `source.freeze` then prepared request `d8ff03724b8a` once,
removing that preparation delay from the next quote-to-admission interval.
At 20:15 UTC an A4000 alternative quoted $0.25/hour (maximum $0.3125 under
the unchanged $1 job cap). Its selection was recorded before submission,
within Jack Heart's alternative-GPU authorization. RunPod refused the actual
create: no A4000 was available in any data center. Launch `95efe5e43df8`
recorded no acquired machine. No hardware deviation actually ran.

At **20:16 UTC**, reconciliation again confirmed zero provider pods, no
unsettled GPU sessions, no pilot cohort and **$0 cumulative pilot spend**.
No retry or queued work remains. The [continuation audit and prepared request](
data/mtg213/launch-audit-20261103-resume/receipt.json) retain the exact source,
quotes, refusal and cleanup evidence; `prepared-run.json` beside it retains
the typed Etude request for a future authorized continuation after fresh
capacity and cost checks. A positive catalog quote did not establish rentable
capacity even with source preparation completed. No learner milestone was sent.

At 19:35 UTC, all six configured GPU launch quotes rejected capacity: `l4`,
`l4-etude`, `a4000`, `a4500`, `mtg188-a6000` and `mtg188-policy-l4`.
Earlier at 19:28 UTC, L4 and A6000 briefly admitted quotes at $0.59/hour,
with a $0.7375 maximum including 900 seconds of setup. The actual L4 submission
(launch `f17df4be888e`, requested run `74e6147280ac`, exact source
`d6ad09e42412aca7c6a235f5acfd7db1cd20051c`) then failed capacity admission
at 19:33:20 UTC, after source preparation. No run or machine session was created.

The discrepancy is not a stricter job hardware requirement: Etude 0.2.12's
availability and launch quote both call the same `RunPodMachines.offer` /
`RunPodAPI.quote` with the same template. All configured GPU templates have no
data-center restriction. Provider catalog replies changed between sequential
reads; their upstream cause remains unknown, and a quote reserves no capacity.
The final read confirmed zero provider pods, no unsettled GPU sessions and no
pilot cohort. **Paid compute remains $0; no teardown is outstanding.**
The [launch audit](data/mtg213/launch-audit-20261103/receipt.json) retains quotes,
timestamps, failed-admission diagnostics and allocation reconciliation.

The [frozen protocol](protocols/tla-trophy-213-v1.md) reserves at most $1 for
its first L4 attempt within Jack Heart's $10 total cap. His later direction
authorizes an available configured GPU alternative within that cap, with any
hardware/allocation deviation recorded before scoring. No alternative allocation
ran; the frozen corpus, recipe and analysis criteria remain unchanged.
Current configured options are exhausted at the recorded observation time,
not permanently unavailable. No queued work or retry loop was left running.

The snapshot and local proof are published in the artifact store:

```sh
uv run --extra artifacts python -m manabot.data fetch experiments/data/mtg213
```

The corpus contains 1,000 recorded 17Lands trophy events, with 134 eligible
whole decks and 866 rejected lists. Source bytes, source-event/build identities,
seventh-win main/sideboard lists and native rejection reasons are retained.
See [corpus.json](data/mtg213/corpus-v1/corpus.json) and
[source.json](data/mtg213/corpus-v1/source.json). Training and evaluation both
use the frozen 134-entry population. No unsupported card was substituted or
removed, and no additional event replaced a rejected deck.

## Local implementation proof

This separate CPU proof used the production regime executor, full 64×2 model,
current v5 input and the whole eligible population, reduced to two streams and
one update. It is not a GPU pilot result or learning-rate comparison.

| Measurement | Observed |
|---|---:|
| Completed collection/update iterations | 1 |
| Learner / environment decisions | 199 / 199 |
| Completed / crashed / abandoned training games | 2 / 0 / 0 |
| Optimizer exposures / optimizer steps | 50 / 38 |
| Collection / learning time | 0.383 s / 41.351 s |
| Stage wall / CPU time | 43.094 s / 42.906 s |
| Valid / expected evaluation games | 12 / 12 |
| Evaluation failures / truncations / replay failures | 0 / 0 / 0 |
| Paid compute | $0 |

The learning phase, especially backward/gradient clipping (39.639 seconds),
dominated this tiny CPU proof. Simulation speed is not a measure of learning.
T6's separately owned referenced-definition learner work is not included here.

The raw endpoint is admitted and retained as
[self-play-raw.pt](data/mtg213/local-proof-v1/self-play-raw.pt), SHA256
`576efea65a2f8b1d61d6ca34585af2093afe601f1818110675c9a2fa09a553cf`.
Initialization, EMA and Adam state are also retained, with the
[TrainingRun](data/mtg213/local-proof-v1/run.json), game-attempt ledger,
Etude metrics and evaluation blocks, semantic Command traces and replay
receipts. [source.json](data/mtg213/local-proof-v1/source.json) names the base
commit and retained source patch/native binary. Their source and native digests
were checked against the original run receipt.

One sampled pair/deal, four policy/deck/seat legs, produced these **validation
observations only**: initialization vs heuristic 0/4, updated policy vs heuristic
0/4, updated policy vs initialization 3/4. The pair is the first frozen pair;
this local proof is explicitly separate from the GPU cohort. One update and
one correlated deal provide no credible improvement or pace estimate. The
predeclared 32-game/checkpoint GPU comparisons remain entirely unplayed.

## Remaining acceptance

Once a bounded launch quote admits available capacity and existing allocations
are checked, the committed, capped GPU job remains to be run. The paid-training
milestone requires observed learner updates. Retain initialization, periodic and endpoint
comparisons, attempted/failed/missing denominators, uncertainty, decisions,
compute and wall time. GPU training and Elemental-relative learning pace remain
unresolved. The implementation's focused native/debug, corpus, recovery and
paired-evaluation checks passed; broader gate/CI checks and shipping remain
separate from this implementation proof.

The Etude score curves are descriptive. The companion analysis reads retained
Etude blocks and training checkpoints to evaluate the frozen criterion:

```sh
uv run --extra artifacts --extra evaluation python -m experiments.tla_pilot_analysis --source r2://manabot/jobs --out .runs/mtg213-analysis.json
```

It reports each checkpoint/anchor and all eight pair identities, valid W/D/L,
expected/attempted/failed/missing games and matches, and checkpoint clocks.
Attempted games is a lower bound when a failed callback left no per-game
receipt: its failed match is counted and its unknown games remain missing.
Every frozen milestone remains expected even for an early endpoint; an absent
successful endpoint retains two slots with unknown update coordinates.

The 2,000 seeded bootstrap resamples move entire four-leg pairs together and
reuse the same pair draws across checkpoints. The JSON includes heuristic gains
relative to initialization, their 95% intervals, and endpoint vs initialization
intervals. `success` is true only when the complete schedule meets all three
frozen criteria; false is a complete negative result, and null with `issues`
means no qualified learning claim. Different evaluation source versions also
withhold qualification. Each training run is analyzed separately. Controlled
fixtures establish this analysis behavior, not a scientific result for the
unstarted GPU pilot; no frozen protocol criteria or historical evidence changed.
