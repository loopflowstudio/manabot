# TLA trophy pilot: capacity failures before learning

V5 source is frozen as `97a2cfb963d7e3a3b2816f31c45dda72f34cdddd`, prepared
run `e8c76bc2c06f`. Its A6000 quote was refused; at 06:37:17 UTC L4, A4000
and A4500 also reported no capacity. **V5 has not been submitted or rented**.
No GPU learner update is yet established. All earlier rentals are ended.


V4 run **baa8671d6480**, source `0b6c2bb854ba`, passed CUDA numerical admission
on an RTX A6000 but failed before learning at **61 card objects plus sideboard
against capacity 60**. Exact replay identifies stream 22 after 300 actions:
40 original cards, five generated tokens and 16 sideboard cards. Its 238
completed games supplied no optimizer update. The rental ended; estimated full
rental cost is $0.04314118, cumulative billed plus estimated cost **$0.21368949**.
All four reservations total $2.95 conservatively held against the same $10 cap.

V4 initialization evaluation completed 32 valid games (**2 wins / 30 losses**),
with 256 scheduled games still missing: complete=false, success=null. All
receipts remain under [v4 evidence](data/mtg213/gpu-v4-20261104/summary.json).
The prospective [v5 contract](protocols/tla-trophy-213-v5.md) increases the
existing card axis to 128 and corrects native buffer sizing. It retains v7,
the 134 entries, learner and evaluation schedule, and requires fresh initialization.


Retained **v4** consumed delivered PR #419's v7 shared-effect inputs.
The [new contract](protocols/tla-trophy-213-v4.md) binds all tensor capacities,
fresh initialization and the unchanged 134-entry population. Retained 140/270
positions pass native scalar/vector/Python parity; compatibility and overflow
checks pass. This preparation does not establish GPU learning. At 05:59 UTC,
all three earlier pilot rentals are ended, provider inventory is empty, and
fresh bills total **$0.17054831** (v3: $0.09962018). Conservatively retain the
prior **$2.2125** reservations until the next launch is reconciled.


The retained **v3/256 GPU attempt also failed before its first learner update**.
Nine Fire Nation Drill activations over fifteen permanents produced **270 distinct
fact rows**, exceeding 256. The rental is terminated. The fresh initialization
completed all 32 baseline games, but there is no trained checkpoint or learning
claim. Frozen v1/v2/v3 protocols and evidence remain unchanged.

## V3 continuation (2026-11-04 UTC)

Run `c216c6be7037` used merged PR #416 commit
`3b122de9078c985f846a0f3c5a3a42ebca3b1182`, world w17, the v3 card digest,
the unchanged 134-entry corpus and v6/256. Etude **0.2.13** restored launcher
inventory decoding in an isolated client; the frozen worker retained **0.2.12**.
Fresh provider inventory and bills preceded launch. The L4 quote was $0.59/hour,
$0.7375 maximum reservation and a $1 cohort ceiling within the existing $10 cap.

The L4 passed 32 CUDA numerical checks and its optimizer probe. Collection then
failed with `facts 270 > 256`; the attempt ledger retained **171 completed games**,
but **zero learner updates** incorporated their samples. A bounded replay of
stream 251, seed 31552, action 208 matches the retained pre-command frame and the
same error through native scalar, native vector and Python encoders. Nine
timestamped `LoseKeywords` records per opposing permanent each name hexproof and
indestructible: **9 × 15 × 2 = 270** distinct rows. A diagnostic 512-row buffer
preserves all rows identically in native/Python encoding. That diagnostic neither
changes the frozen pilot capacity nor establishes sufficient capacity for training.
See the [replay receipt](data/mtg213/gpu-v3-20261104/capacity-replay.json) and
[retained stream](data/mtg213/gpu-v3-20261104/capacity-stream.jsonl).

The fresh initialization checkpoint completed eight paired matches / 32 games:
**3 wins, 29 losses**, all valid, with no failures or truncations. Its numerical
result happens to match v1; these are separate v3 executions and evidence.
The [analysis](data/mtg213/gpu-v3-20261104/analysis.json) retains all 288 scheduled
games, **256 missing**, no trained endpoint, `complete=false`, `success=null`.
Cloud discovery stalled during a network outage before playing any match;
evaluation then used a byte copy of this run's tracking prefix and the canonical
CLI on the verified source. No failed match was retried.

The outage also delayed cleanup. The launcher confirms this rental ended at
**02:26:33 UTC**. Its full estimated rental cost, including idle cleanup time,
is **$0.09958535**; the smaller per-run allocation estimate is not the whole bill.
Fresh provider bills for the two earlier rentals total **$0.07092813**, making
cumulative billed-plus-estimated compute **$0.17051348**. Conservatively retain
all three reservations (**$2.2125**) against $10 until final provider settlement.
The final inventory contained one differently named pod absent from this
launcher's database; it was left untouched. The advertised `etude-result.json`
object was again missing; every other listed output was downloaded and
hash-verified. [Summary and cost accounting](data/mtg213/gpu-v3-20261104/summary.json),
[settlement](data/mtg213/gpu-v3-20261104/settlement.json) and
[missing-output receipt](data/mtg213/gpu-v3-20261104/missing-outputs.json) retain
these limits.

The 256-row assumption is invalidated. The approved shared-effect repair and
remaining acceptance are below. V3 source, inputs, attempts and evidence remain
frozen; no unchanged retry, truncation or deck exclusion is justified.

## Historical v1 result (2026-11-03)

The paid pilot reached CUDA collection but **completed no learner update**.
A legal state needs 140 fact rows; frozen policy input v5 has 128. Initialization
is admitted and evaluated, but there is no trained endpoint or learning-pace claim.
The A6000 allocation failed its timing fixture; the later L4 allocation exposed
the fact-capacity overflow.
The retained 2026-11-03 20:48 UTC settlement confirms both GPU rentals torn
down, with no provider pods or unsettled sessions at that read.

The [v2 repair](protocols/tla-trophy-213-v2.md) introduced v6/256; generic
fresh v6 defaults to 512 and frozen v5 stays at 128. Publication then integrated
[PR #415](https://github.com/loopflowstudio/manabot/pull/415), whose five-card
repairs changed the native digest. [V3](protocols/tla-trophy-213-v3.md) bound that
runtime while preserving the 134 decks, input capacity, recipe and budget.
Its executed result is above. V2 source remains at
`d655d37996d2702cd5fee6ac37db11d855e87640`.

### Paid attempts

| Run | Hardware | Result | Estimated cost |
|---|---|---|---:|
| `d8ff03724b8a` | RTX A6000 | 31 CUDA tests passed; timing fixture failed before training | $0.03275634 |
| `40709cd93f90` | L4 | 32 CUDA tests and optimizer probe passed; first collection failed | $0.03233182 |

Both quoted $0.59/hour with a $0.7375 maximum reservation and $1 allocation cap.
**Estimated compute at that read: $0.06508817**, with $1.475 reserved pending
billing. The v3 accounting above supersedes these estimates against the $10 cap.
The second operational cohort is `mtg213-pilot-v1-correction1`, capped at
$1, because Etude cannot append to an existing cohort. This did not create a new
scientific population or change the schedule. [Settlement](data/mtg213/gpu-corrected-20261103/settlement.json)
confirms teardown at 20:48:27 UTC. V1 ended without another retry.

The original prepared source `b8835f63b4e7132ae6fd3f86cc73cefa6716e57f` was
verified against all 1,327 bundled files and rebuilt in an isolated archive.
Its w17 card digest `17735e21aa9d59ebf17319f7ee43e16921189fde2f8886e38bfefc6dcca51e68`
matches the frozen corpus, and all 134 entries admit. Current HEAD's different
native digest was not substituted. The [first attempt](data/mtg213/gpu-attempt-20261103/settlement.json)
retains the failed gate: learning took 8.9029 seconds and diagnostics 8.6478,
violating an invalid hardware-dependent ordering assertion.

Prospective repair source `2adbd0758e40518947daf91569e1938b060d4346` derives from
that exact parent with only the timing test changed. All native, learner, corpus,
recipe and evaluator bytes remain identical. The fixture now charges controlled
executor durations and checks exactly 10.25 seconds of successful-plus-failed
learning independently of persistence overhead. It passed all three CPU variants
in both the Task and historical archive, then the full CUDA gate on the L4.
The [source correction](data/mtg213/gpu-corrected-20261103/source-correction.json)
was selected before retry or scoring; original requests and evidence are unchanged.

### Retained capacity counterexample

L4 collection failed after 6.343 seconds, before an optimizer update. The attempt
ledger records 13 completed games before the batch failure; none of their samples
were incorporated into a learner update. Settled learner/decision counters remain
zero, so those counters are not a measure of all attempted simulation work.

Stream 181, seed 31482, action 87 reproduces the same error at the retained
pre-command frame. Seven legal activations of **The Fire Nation Drill** remove
hexproof and indestructible from ten opposing permanents, retaining
7 × 2 × 10 = **140 fact rows**. Native and Python encoders both reject them.
The [single-stream input](data/mtg213/gpu-corrected-20261103/capacity-stream.jsonl),
[replay result](data/mtg213/gpu-corrected-20261103/capacity-replay.json) and
[effect counts](data/mtg213/gpu-corrected-20261103/capacity-observation-summary.json)
retain the exact decks, sideboards, seed, actions, buffer shapes and runtime IDs.
Replay used the same source/world on macOS; native binary bytes differ from Linux.

Encoding errors are deliberately fatal in `step_isolated_into`; they are not
quarantined gameplay errors. This invalidates the pilot's capacity assumption.
No facts were truncated, frozen shapes widened, decks removed, actions hidden,
or failing games replaced. This failure required a shared representation decision;
the accepted prospective v6 contract is described above. The launcher listed
`etude-result.json` as an output whose
object was absent; [the missing-output receipt](data/mtg213/gpu-corrected-20261103/missing-outputs.json)
retains that gap. Other outputs were downloaded and hash-verified.

### Scheduled initialization evaluation

The retained initialization is [self-play-initial-raw.pt](data/mtg213/gpu-corrected-20261103/training/self-play-initial-raw.pt),
SHA256 `456a9a225fceb72034b5d5a244378244f18592b42f77b88f655ae7d281f72f7b`.
The same pinned source/native world played all eight frozen pairs and four legs
against the fixed heuristic: **3 wins / 0 draws / 29 losses**, 32 valid games,
32 replay passes, no failed or truncated games. The whole-pair bootstrap score
is .09375, 95% interval [0, .1875]. This is initialization strength, not learning.

The [analysis](data/mtg213/gpu-corrected-20261103/analysis.json) retains the complete
288-game schedule: 32 attempted/valid, 256 missing, no trained endpoint,
`complete=false`, `success=null`. All milestones and both unknown-endpoint
comparisons remain expected. Raw Etude blocks, traces, replays and checkpoint
records are retained under the [evaluation directory](data/mtg213/gpu-corrected-20261103/evaluation/).
No failed match was retried and no game was replaced.

Two execution defects were repaired without rewriting these observations:
Etude's runpy discovery needs a canonically imported callback, and its table
projection can omit nullable columns when all values are null. The frozen run
used its unchanged callback through canonical SDK invocation; the corrected
offline reader consumed its sealed blocks with the same frozen criterion.
[Reader provenance](data/mtg213/gpu-corrected-20261103/analysis-source.json) binds
that reader's exact bytes. The discovery regression and clean/all-failed nullable
column cases pass through focused tests. Local CLI use also names
`ETUDE_OUTPUT_DIR` explicitly, as documented in `manabot/README.md`.

## Corpus and earlier attempts

The [frozen protocol](protocols/tla-trophy-213-v1.md) and
[corpus](data/mtg213/corpus-v1/corpus.json) preserve 1,000 historical seven-win
17Lands events: 134 eligible whole lists, 866 rejected, with source-event/build
IDs, main/sideboards and native rejection reasons. No unsupported card was
substituted or removed. Training and evaluation use the same eligible population.

Earlier capacity refusals cost $0 and acquired no machine. The
[original audit](data/mtg213/launch-audit-20261103/receipt.json) and
[continuation audit](data/mtg213/launch-audit-20261103-resume/receipt.json)
retain every quote, failed submission and cleanup observation. Provider access
uses nested Doppler `manabot/prd` (artifacts) then `etude/prd` (RunPod), without
copying or printing credentials. All experiment bytes are manifest-published:

```sh
uv run --extra artifacts python -m manabot.data fetch experiments/data/mtg213
```

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
periodic and endpoint GPU comparisons remain unplayed; initialization results
are reported above.

## Implementation validation and remaining acceptance

[PR #416](https://github.com/loopflowstudio/manabot/pull/416) merged at
`3b122de9078c985f846a0f3c5a3a42ebca3b1182` with green
[CI](https://github.com/loopflowstudio/manabot/actions/runs/41752523278).
The subsequent focused corpus, pilot and discovery verification passed 38 tests;
that exact merged source was frozen for v3. Earlier validation, including failed
combined runs and their bounded-fixture repairs, remains at
`7b8bb32731205981b6be09097eb4ca4b39b3864f:experiments/tla-trophy-pilot.md`.
Those implementation checks establish neither GPU learning nor strength.

Operator comment 78f4c226 selected 256 after Jack Heart's question about that
capacity; be5b1394 relays his confirmation of the immediate larger-capacity
hotfix. The earlier 512 choice was also the operator's, and 0c17ad58 was an
operator-issued run reason. The 140-row replay passed at 256 and a controlled
258-row fixture failed as intended; v3's later 270-row failure demonstrates
why those checks did not establish sufficient capacity for training.

Jack Heart approved the shared-effect fixed-tensor repair (comment 42a12145):
encode effects once and route them to affected object embeddings before attention.
Task 5da5f4 owns that prerequisite. After delivery, bind a prospective pilot with
fresh initialization and evaluation on the same 134 decks, reconciling costs
against the existing cumulative $10 cap before allocation. Preserve v1/v2/v3
evidence; the new run cannot fill v3's 256 missing evaluation slots.

GPU learner updates and trained checkpoints remain missing. No paid-training
milestone or learning/Elemental-pace claim is supported; MTG-213 is incomplete.
