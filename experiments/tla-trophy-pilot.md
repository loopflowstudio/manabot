# TLA trophy pilot: implementation proof, GPU launch blocked

As of 2026-11-03, the corpus, independent per-seat sampler, GPU job and paired
evaluation runner are implemented. **The requested GPU pilot has not started.**
Etude reports `RUNPOD_API_KEY is not set`, both directly and through the
configured `manabot/prd` Doppler environment. No rental or job was created;
paid compute spend is $0. The [frozen protocol](protocols/tla-trophy-213-v1.md)
reserves at most $1 for its first L4 attempt within Jack Heart's $10 total cap.
Authentication was not changed. The blocker was reported in the Task conversation.

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

Once provider credentials are available to the existing Etude launcher, run the
committed, capped GPU job and report a paid-training milestone only after actual
learner updates are observed. Retain initialization, periodic and endpoint
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
