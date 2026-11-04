# TLA trophy pilot: collection capacity failure before learning

The paid pilot reached CUDA collection but **completed no learner update**.
A legal state needs 140 fact rows; frozen policy input v5 has 128. Initialization
is admitted and evaluated, but there is no trained endpoint or learning-pace claim.
The A6000 allocation failed its timing fixture; the later L4 allocation exposed
the fact-capacity overflow.
The retained 2026-11-03 20:48 UTC settlement confirms both GPU rentals torn
down, with no provider pods or unsettled sessions at that read.

The subsequently accepted [prospective v2 repair](protocols/tla-trophy-213-v2.md)
selects policy input v6 and an explicit 256 fact rows for the pilot; generic
fresh v6 configuration defaults to 512. It preserves v5 and all
results below. Local publication checks passed; CI, source freeze and a new
capped training/evaluation attempt remain outstanding. No GPU learner update
has yet been observed.

Publication then integrated [PR #415](https://github.com/loopflowstudio/manabot/pull/415),
whose five-card repairs change the native digest. The active prospective runner
is now [v3](protocols/tla-trophy-213-v3.md), preserving the same 134 decks,
v6/256 input and scientific/budget settings. V2 protocol bytes and its validated
source remain unchanged. The local validation counts below describe the
pre-sync v2 source; combined-runtime validation is recorded separately.

## Paid attempts (2026-11-03)

| Run | Hardware | Result | Estimated cost |
|---|---|---|---:|
| `d8ff03724b8a` | RTX A6000 | 31 CUDA tests passed; timing fixture failed before training | $0.03275634 |
| `40709cd93f90` | L4 | 32 CUDA tests and optimizer probe passed; first collection failed | $0.03233182 |

Both quoted $0.59/hour with a $0.7375 maximum reservation and $1 allocation cap.
**Total estimated paid compute: $0.06508817.** Provider billing is not settled;
conservatively retain both reservations ($1.475) against Jack Heart's $10 total
cap. The second operational cohort is `mtg213-pilot-v1-correction1`, capped at
$1, because Etude cannot append to an existing cohort. This did not create a new
scientific population or change the schedule. [Settlement](data/mtg213/gpu-corrected-20261103/settlement.json)
confirms teardown at 20:48:27 UTC. V1 ended without another retry; a prospective
v2 allocation requires fresh billing reconciliation within the same total cap.

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

## Retained capacity counterexample

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

## Scheduled initialization evaluation

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

## Remaining acceptance

Operator comment 78f4c226 selects 256 rows for the pilot, following Jack Heart's
question about 256, under the existing authorization. The earlier 512 was also
an operator engineering choice; comment 0c17ad58 was an operator-issued run
reason, not separate human confirmation. The retained failure replay now encodes
all 140 distinct facts in parity at 256 through native scalar/vector and Python
paths, while both v5 encoders retain the original overflow. A controlled fixture
rejects 258 facts at the 256 bound. The new local receipt is
`.runs/mtg213/v6-256-capacity-replay.json`; the earlier 512 receipt is unchanged.
This allocation does not establish capacity for every legal state or a measured
end-to-end speedup.

Pre-sync v2 publication validation covered 510 distinct passing Python tests (2 skipped),
895 passing debug Rust tests (2 ignored), Ruff, Rust fmt/clippy and a rebuilt
native extension. The numerical fixture now uses a deterministic four-decision
quota instead of waiting for complete games under its extreme policy. Two
recovery checks rejected a commit change during validation and passed when
rerun with HEAD fixed; recovery admission remains unchanged. These are local
implementation checks, not GPU or strength results.

After integrating PR #415, v3 validation covers **532 distinct passing Python
tests (2 skipped)** and **905 passing debug Rust tests (2 ignored)**, with Ruff,
fmt/clippy and a rebuilt native extension. These are grouped runs and successful
rechecks, not one uninterrupted green process. The combined Python process hit
collection deadlines in the numerical-accounting fixtures; those now explicitly
bound their requested decision quotas. A later complete-game timing fixture
was slow in that combined process, which was stopped; the remaining 212 cases
ran in a fresh process. Two recovery cases restored all state exactly but assumed
a game would finish within a one-seat quota. The completion assertion now applies
to two-seat batches, where completion is guaranteed; all four seat/learner cases
pass. Production collection and recovery admission are unchanged. CI remains
the check of the configured combined workflow.

The new runtime also replays the retained 87-action prefix to the same frame:
all 140 facts agree through scalar/vector/Python encoding at 256, and v5 still
overflows. The separate local receipt is `.runs/mtg213/v3-capacity-replay.json`;
earlier replay receipts remain unchanged. This is capacity-repair evidence only.

CI and an exact source freeze remain before the next capped attempt.
That attempt requires fresh reconciliation of both $0.7375
reservations and provider billing, new v3 initialization and the unchanged
evaluation schedule. Actual GPU learner updates, periodic/endpoint checkpoints
and learning/Elemental-pace evidence remain unmet. No paid-training milestone
was sent; the Task is not complete.

Gate review of PR #416 found one CI fixture mismatch: Python's fresh v6/512
configuration was compared to explicit native v5/128. The corrected text proof
now checks matching configurations at v5/128, pilot v6/256 and the fresh default;
production encoding is unchanged. Focused validation passed **237 tests**, with
one CUDA-only check skipped, plus Ruff and Python/Rust formatting. The original
CI main group completed 900 other tests with one skip; debug Rust, engine budget,
long integration, package/install and Academy checks passed at `feab62822`.
CI on the fixture correction remains required before landing. These checks
establish implementation behavior, not GPU progress or learning gain.
