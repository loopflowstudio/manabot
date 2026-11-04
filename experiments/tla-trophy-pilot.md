# TLA trophy pilot: v9 learned, permanent capacity stopped the run

V9 Run **2171156493ca** executed the frozen source
`6da60684cb628ec9b431dbbb9d7dc91348743622` on a Vast RTX 5090 on
2026-11-04. CUDA numerical admission passed. It completed **5 learner updates,
983 optimizer minibatch steps, 251,040 learner transitions and 1,291 games**,
then failed during collection at **41 agent permanents against capacity 40**.
The [training receipt](data/mtg213/gpu-v9-20261104/training/run.json) retains
394.18 stage seconds: 273.48 collecting and 119.87 learning. These clocks
exclude part of the rental. No trained checkpoint exported before failure;
only initialization is retained. MTG-213 remains incomplete.

The [exact replay](data/mtg213/gpu-v9-20261104/capacity-replay.json) identifies
stream 43, seed 32624, after 371 actions, with a matching pre-command frame.
Scalar native, vector native and Python encoding reject the same 41-permanent
position. It contains 25 Phoenix Fleet Airships, 11 lands and five other
permanents. The Airship definition has a conditional end-step token-copy
ability; this capacity witness alone establishes no rules defect. A diagnostic
64-permanent allocation encodes all 41 with scalar native/Python equality.
That is a repair witness, not a universal bound or a changed v9 protocol.
Continue with a prospective capacity repair and fresh initialization; never
rewrite v9 or infer a checkpoint from its progress counters.

[Paired evaluation](data/mtg213/gpu-v9-20261104/analysis.json) completed all
32 initialization games against the matched-support heuristic: **7 wins,
25 losses**, no failed or truncated games. Score .21875, 95% whole-pair
bootstrap interval [.125, .3125]. The frozen 288-game schedule retains
**256 missing games**, `complete=false`, `success=null`. A final evaluation
pass found no additional playable comparisons. No trained-policy gain or
Elemental-relative learning pace is established. MTG-183's .065 → .304 at
update 50 and .441 at 100 remain historical context: different rules,
observations, populations, anchors, hardware and variable complete-game batch
sizes prevent a controlled pace comparison. There is no trained TLA score to
compare at a matched decision coordinate.

The [cutoff receipt](data/mtg213/gpu-v9-20261104/training/failures/action-cutoff-00c266bf52a9409b91a4aa72a926c44d.json)
records 302,941 consumed decisions, including the failed collection, and zero
action overflows/dropped actions. It is an aggregate across participating
policies, not a learner-only count. This run's stopping error is permanent
capacity, separate from the accepted action cutoff.

The [launch](data/mtg213/gpu-v9-20261104/launch/launch-20261104-220427.json)
used 32 vCPUs, 255GB RAM and USD0.962962963/hour. Jack Heart's accepted
USD1.25 allocation/USD0.97-hour placement amendment replaced the earlier
USD1 limit; cumulative USD10 remained unchanged. Etude also checks the job's
spending ceiling against setup plus execution, so the placement copy raised
`Job.max_usd` to USD1.25. SDK `job_identity`, Run ID, source, configuration,
recipe and schedule remained equal to the original prepared Run. The quote
reserved USD1.187654321. No source was frozen again.

[Settlement](data/mtg213/gpu-v9-20261104/settlement.json) confirms the pilot
rental ended at **22:13:47 UTC** and instance 59168321 is absent. The final
inventory identifies an unrelated `state_representation_gpu` rental, which was
left untouched. Whole-rental estimated cost is **USD0.148237115**;
cumulative billed-plus-estimated cost is **USD0.475343889**, with conservative
reservations **USD5.101645926 of USD10**. Vast billing remains unavailable;
these estimates are not final invoices. All available outputs were fetched;
`etude-result.json` is absent because the training command failed.

Verification on main `3de3a0ebf7b20ae764e0b423089730c5bc710cb6` found only
documentation changes after merged PR #423; executable code and the frozen
protocol match. The 106 focused checks passed, as did real local v9 runtime
admission. The GPU failure is new evidence and a concrete **fix-forward** gap:
ship a prospective permanent-capacity repair, retain this failed attempt and
complete trained-export/evaluation acceptance. See the
[summary receipt](data/mtg213/gpu-v9-20261104/summary.json).

## V9 preflight history


V9 Run **2171156493ca** freezes merged source
`6da60684cb628ec9b431dbbb9d7dc91348743622`, the original 134 decks and matched
first-64 legal support for all participants. It remains **unsubmitted**.
The fresh 20:59–21:00 UTC recheck on 2026-11-04 briefly obtained compatible
L4 and A6000 whole-rental quotes of USD0.7375 each. Both launch-time rechecks
then refused capacity **before submission**. RTX 5090 quoted
USD1.1481481481481481/hour, above the USD0.81 ceiling; RTX 3090 and A4000
also refused capacity. The attempt stopped blocked instead of repeating
unchanged requests. [Recheck evidence](data/mtg213/gpu-v9-recheck-20261104/outcome.json)
retains the quotes, launch refusals and accounting. The
[earlier preflight](data/mtg213/gpu-v9-preflight-20261104/outcome.json)
retains the prepared Run and initial refusals; no source was prepared again.

The final 21:00:52 UTC inventory confirms no Vast instances, RunPod pods or
unsettled launcher hardware. The independently owned representation rental
was untouched and disappeared independently between reads. Accounted prior
full-rental cost remains **USD0.32710677365251912**, with
**USD3.9139916049382716** conservatively reserved against USD10;
**new paid compute is USD0**. Provider billing for the earlier Vast rental
remains unavailable, so its full-rental estimate is retained. The USD1 allocation
cap is unchanged. V9 still needs GPU execution, an admitted trained checkpoint,
paired evaluation and the qualified Elemental comparison. A new capacity signal
is needed before another unchanged launch attempt.

## Previous executed pilot: v5

V5 Run **e8c76bc2c06f**, frozen source
`97a2cfb963d7e3a3b2816f31c45dda72f34cdddd`, ran on a Vast **RTX 5090** on
2026-11-04. CUDA numerical admission passed. Training completed **3 rollout
updates, 585 optimizer minibatch steps and 149,260 learner transitions** before
the next collection failed: **256 offered actions exceeded the frozen capacity
of 64**. Settled batches contain 771 completed games. This establishes actual
GPU learning execution, not improvement in game strength or a completed pilot.

The [training receipt](data/mtg213/gpu-v5-20261104/training/run.json) records
212.43 seconds, including 142.77 seconds collecting and 68.86 seconds learning.
These are stage clocks, not the whole paid window. No trained checkpoint was
exported before failure; the first scheduled milestone was not reached. The
initialization checkpoint and private failure capture are retained. Missing
trained checkpoints cannot be scored or replaced by the final in-memory state.
The [initialization evaluation](data/mtg213/gpu-v5-20261104/analysis.json)
completed eight pairs / four legs against the fixed heuristic: **5 wins / 27
losses**, 32 valid games, no failures or truncations. The whole-pair bootstrap
score is .15625, 95% interval [.03125, .28125]. All 288 scheduled games remain
accounted for: **256 missing**, `complete=false`, `success=null`. This measures
initialization only; no trained checkpoint is available for comparison.

Vast integration used the published isolated Etude 0.2.14 control client,
leaving the frozen worker's 0.2.12 dependency, v7 input, 134-deck population and
recipe unchanged. Allocation: 24 vCPUs, 63GB RAM, 80GB disk, free transfer,
$0.7816148148148148/hour and $0.9639916049382715866666666667 reserved inside the
$1 ceiling. The public image is pinned Linux amd64 / CUDA 12.8.1; runtime
admission is separately retained in
[numerical validation](data/mtg213/gpu-v5-20261104/numerical-validation.json).

[Settlement](data/mtg213/gpu-v5-20261104/settlement.json) confirms rental deletion
at 17:33:51 UTC, no Vast instances, no RunPod pods and no unsettled hardware.
Prior billed costs are $0.213831719709560274. The new full-rental estimate is
$0.113275053942958846, with provider billing still unavailable, for cumulative
billed-plus-estimated **$0.32710677365251912**. Conservative reservations remain
**$3.9139916049382716 against $10**. The per-run allocation estimate of $0.0905
excludes some rental time and is not the full cost.

The launch milestone was reported to the primary Task conversation. Its first
terminal observation was at 17:33:51 UTC (the progress comment's 17:32 timestamp
was approximate and is corrected here). The
[retained-prefix replay](data/mtg213/gpu-v5-20261104/capacity-replay.json)
identifies stream 103, seed 32172, after 76 actions. The pre-command frame
matches; native scalar, vector and Python encoders all reject the same 256
offered actions at capacity 64. A diagnostic 512 configuration retains all 256
in Python, but the frozen native buffer path still enforces 64. This is evidence
that a configuration-only retry is insufficient, not a new pilot selection.
The historical v8 contract below binds the buffer repair, X-cost rules repair
and explicit first-64 legal-action cutoff; executed v5 evidence remains frozen. Compared with Elemental, this
pilot now establishes GPU execution on trophy pairs, but supplies no trained policy for a
learning-gain or cost-to-strength comparison. Different hardware, input size and
population also prevent interpreting raw throughput as a controlled comparison.

The [v5 contract](protocols/tla-trophy-213-v5.md) retains v7, the 134 entries, learner
and evaluation schedule, with a 128-card axis and corrected native buffer sizing.
It requires fresh initialization.

## V4 result (2026-11-04 UTC)

Run **baa8671d6480**, source `0b6c2bb854ba`, passed CUDA numerical admission
on an RTX A6000 but failed before learning at **61 card and sideboard objects
against capacity 60**. Exact replay identifies stream 22 after 300 actions:
40 original cards, five generated tokens and 16 sideboard cards. Its 238
completed games supplied no optimizer update. The rental ended; its provider
bill is **$0.043283410370349884**, included in the 09:03 inventory and
cumulative accounting above. This supersedes the earlier whole-rental estimates;
the frozen v4 receipts retain those estimates.

V4 initialization evaluation completed 32 valid games (**2 wins / 30 losses**),
with 256 scheduled games still missing: complete=false, success=null. All
receipts remain under [v4 evidence](data/mtg213/gpu-v4-20261104/summary.json).

The [v4 contract](protocols/tla-trophy-213-v4.md) consumed PR #419's delivered
v7 shared-effect inputs and bound all tensor capacities. Retained 140/270-fact
positions pass native scalar/vector/Python parity, compatibility and overflow
checks. At the 05:59 UTC pre-launch settlement, the three earlier rentals had
ended and bills totaled **$0.17054831** (v3: $0.09962018). The current accounting
above includes the subsequent v4 and v5 rentals.

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

The 256-row assumption is invalidated. V3 source, inputs, attempts and evidence remain
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
policy input v5 and the whole eligible population, reduced to two streams and
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
Task 5da5f4 delivered it in PR #419, merged as
`fb5ad2c1372f9c967b550f0b8ae0f84883f75563`; v4 consumed it and v5 retains it.
V5 subsequently reached GPU learning, as reported above. An admitted trained
checkpoint, complete paired evaluation and Elemental-relative learning analysis
remain outstanding. Any successor needs an explicit prospective contract and
fresh inventory, billing and quote reconciliation within the $1 allocation and
cumulative $10 caps. Preserve all executed versions; new games cannot fill their
missing evaluation slots.

### Card-capacity repair publication checks (2026-11-04 UTC)

PR #420 merged at 08:15:17 UTC as
`0a5f2ad0fb55030f2b8caa2d7e403e3a3876dd8a`. The merged implementation passed
50 focused pilot, shared-effect transport, 128-card CPU learner/export/reload
and experiment-discovery checks. The prepared Run retains its frozen source.

On the synced repair branch, Rust formatting, all-target/all-feature Clippy and
the complete debug workspace tests pass (`CARGO_PROFILE_DEV_OPT_LEVEL=1`, two
build jobs). Python validation passes **391 tests, one skipped**: the complete
model suite, active training, current-world arena admission and monitoring,
pilot/discovery/regime checks, retained shared-effect and card-capacity replays,
and artifact tracking/CI selection/release checks. Changed Python files pass
Ruff lint and formatting; the diff has no whitespace errors. All 230 added
artifact manifest entries match retained local bytes, with existing entries
unchanged. The earlier release rebuild and atomic extension replacement remain
the native runtime used for these Python checks.

These checks establish the retained 61-card repair and CPU learner/export/reload
behavior at 128 cards. They do not execute the prepared v5 GPU Run, establish a
universal capacity bound, or supply missing trained-checkpoint evaluations.

## Superseded v6 buffer repair (2026-11-04 UTC)

The [v6 protocol](protocols/tla-trophy-213-v6.md) binds current world/content and
73/56/38 object widths with 512 actions. Native scalar/vector buffers now derive
all configurable capacities from the supplied arrays; scalar companion validation
finishes before copying any output. V6 prepared the runner and analysis;
v5 source, protocols and evidence remain frozen. This prepares fresh training
and evaluation on the same 134 decks; it supplies no new learner or cost result.

## Superseded v7 rules repair (2026-11-04 UTC)

The [v7 protocol](protocols/tla-trophy-213-v7.md) supersedes unsubmitted v6 Run
`bcde93e9ff8c` (source `ad2c3a010f53`), which incurred no rental or scoring.
Operator comment a85a89e8 required examining the 256 offers before continuation.
They were **illegal X choices**, not 256 distinct payable strategies: an airbent
Wan Shi Tong selected X from its printed cost but paid fixed {2}. CR 107.3b
requires X=0. Shared selected-cost handling fixes announcement and payment;
any-type mana retains X, and free casts retain additional waterbend X when the
printed mana cost has no X. The original v5 observations remain frozen evidence
of that failure, rather than a rules-correct action contract.

The repaired retained prefix agrees across scalar/vector/Python paths at
non-default capacities. A separately retained historical observation tests the
256-row encoder boundary without generating or playing those illegal actions.
Directed debug tests check payment, counters and draws. V7 selected fresh
initialization, the same 134 decks and recipe,
512 actions and unchanged evaluation and budget limits. Source freezing completed as Run **af3661790ac5** at
`b5406bb33abd98870af6ca07c3beed63a92b7c1c`. The
[source receipt](data/mtg213/gpu-v7-preflight-20261104/source.json) records its
world, shapes and protocol/regime hashes. It was **never submitted or scored**.

At 18:44:18 UTC, fresh [inventory/cost reconciliation](data/mtg213/gpu-v7-preflight-20261104/inventory.json)
found no remaining provider rentals or unsettled hardware. Cumulative accounted
cost remains $0.32710677365251912 and conservative reservations
$3.9139916049382716. [Actual quote refusals](data/mtg213/gpu-v7-preflight-20261104/refusals.json)
blocked execution: Vast RTX 5090 exceeded its $0.81/hour ceiling, and the
configured RunPod L4 alternative reported no available machines. No new cost,
initialization result or trained-policy result exists. Continuation needs
compatible capacity under the unchanged <=$1/$10 limits; the scientific
acceptance remains unfinished.

## Superseded v8: explicit legal-action cutoff (2026-11-04)

Jack Heart selected 64 action slots with deterministic first 64 legal actions and
logging (comment 3b0f8826). The [v8 protocol](protocols/tla-trophy-213-v8.md) binds
that approximation for training and learned-policy evaluation, preserving input-v7
shared effects, 128 cards, 134 decks, recipe and budgets. Prepared v7 Run
`af3661790ac5` remains unsubmitted and is superseded. No v8 training or evaluation
result is claimed by this implementation update.

Directed native checks establish that ordinary Wan Shi Tong with seven untapped
basics offers/payments match X=0..5, while direct X=6/255 submissions reject with
the complete rules-state witness unchanged. Airbend and any-type-mana casts
also reject unavailable X. Prefix tests preserve original native Command mapping,
scalar/vector/Python parity, masks, collection and bounded replay diagnostics.

V8 was frozen as Run `f47be2d3c0d8`, source
`900c207502bd28757d7cf66196bd464458e39402`, but remains unsubmitted.
At 19:10–19:12 UTC, the bounded RTX5090 quote refused its USD0.81/hour ceiling;
L4 and RTX3090 quotes refused available capacity. No paid allocation or new
evaluation occurred. The final inventory at 19:11:43 found no live rentals or
unsettled hardware. Historical accounted cost remains USD0.32710677365251912,
with conservative reservations USD3.9139916049382716 of the USD10 cap.
Preflight receipts live under `experiments/data/mtg213/gpu-v8-preflight-20261104/`.
The prepared Run remains bound to that source. Subsequent collector and
diagnostic fixes require a successor source freeze before execution; these
preflight receipts do not establish current availability.

Local gate validation of the enabling repair passed 310 focused tests, including
buffer parity, replay, cutoff/collection, admission, monitoring and discovery;
the updated CI integration shard collected 1,038 tests and Ruff passed. Cutoff
samples now bind the current actor/frame and share scalar/vector decision-kind
labels; vector reseeding preserves exact diagnostic totals. The unchanged native
sources retain the preceding 957-pass debug validation (two ignored) and Clippy
result. Platform CI and the engine performance comparison remain delivery
checks. These software checks supply no trained checkpoint or learning result.

## Prospective v9: integration and matched support (2026-11-04)

Jack Heart selected the same first-64 legal prefix for the heuristic as for
learned policies (comment ffd29028). The [v9 protocol](protocols/tla-trophy-213-v9.md)
binds that benchmark and the integrated T7 runtime. V8's prepared source and
evidence stay frozen; its runtime guard rejects the new catalog. V9 keeps the
134-deck population, learning recipe, schedule and budget, and requires fresh
initialization and evaluation. No new paid execution occurred during CI repair.

The heuristic restricts candidates before scoring and fallback. A constructed
native position has 67 legal choices: the full heuristic prefers Shivan Dragon
at index 65, while the matched heuristic selects Hill Giant at original index 0
and executes that exact Command. The full native offer set stays intact.
Aggregate cutoff diagnostics cover all participating policy decisions, including
the heuristic; they cannot establish actor-specific or learner-only rates.
