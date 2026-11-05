# TLA trophy pilot: trained v10 endpoint and paired evaluation

V10 Run **ae76c17951c3** executed merged source
`fe83a6b068dace2187613e1bf124f9677b077ff3` on a RunPod NVIDIA L4 on
2026-11-05 UTC. CUDA numerical admission passed. The bounded job succeeded and
exported its learner state at **16 updates, 2,954 optimizer steps and 754,114
learner transitions**. Training paused normally before its wall budget expired;
the 100-update target and milestones 25/50/75 were not reached. The
[training receipt](data/mtg213/gpu-v10-20261105/training/run.json) records
2,398.74 stage seconds: 1,805.37 collecting and 590.78 learning.

The [trained raw checkpoint](data/mtg213/gpu-v10-20261105/training/self-play-raw.pt)
is 9,609,496 bytes, SHA256
`235c3d9f3310047e0a1bdc1eff18a0c9dfb5eeab1a0d2c577355a2f25f104648`.
The admitted evaluation checkpoint has the same digest. EMA, Adam and
initialization exports are retained too. These are policy/learner artifacts,
not portable collector or RNG recovery.

[Paired endpoint evaluation](data/mtg213/gpu-v10-20261105/analysis.json) completed
all 64 endpoint games, with no failed or truncated games:

| Opponent | Wins / games | Score | 95% whole-pair bootstrap interval |
| --- | ---: | ---: | ---: |
| Matched first-64 heuristic | 9 / 32 | 0.28125 | [0.15625, 0.40625] |
| V10 initialization | 26 / 32 | 0.8125 | [0.6875, 0.90625] |

These are eight paired deals per comparison from the same frozen 134-deck
population. **The predeclared joint learning criterion remains unestablished.**
The full 288-game schedule retains 196 missing games: 192 at unreached milestones
and four from an operator invocation error. The first initialization callback
omitted required `ETUDE_OUTPUT_DIR` and failed before playing; its failed match
was preserved, not retried or replaced. The corrected invocation completed the
other 28 initialization games (5 wins, 23 losses). Analysis therefore reports
`complete=false`, `success=null`, with no baseline-gain estimate. No held-out-deck
or Elemental-relative learning-pace claim follows.

The [attempt ledger](data/mtg213/gpu-v10-20261105/training/game-attempts.jsonl)
records 4,201 completed games and four isolated failures out of 4,205 attempts.
Two were engine `AgentError: no legal targets`; two were resource limits:
65 continuation relations against 64 slots and 65 agent permanents against 64.
Only those games' 1,881 samples were discarded. The peak settled 100-game error
rate was 1%, below the accepted 25% threshold. All four
[retained prefixes](data/mtg213/gpu-v10-20261105/summary.json) replay with matching
pre-command frame, failure index and error. Replays explicitly admit the macOS
native binary against Linux evidence while retaining the same source/world;
they are diagnostics, not new training games or proof that the rules are correct.
The two target-selection errors remain concrete rules follow-up evidence.
The action-cutoff receipt counts 755,995 consumed decisions and zero overflows
or dropped actions; the cutoff and finite object capacities remain part of the
contract. Recovery still biases learning toward games that fit the runtime.

[Settlement](data/mtg213/gpu-v10-20261105/settlement.json) confirms this rental
ended at **01:25:19 UTC**, and fresh provider inventory is empty. Vast refused a
price change above USD0.97/hour before admission; the already-authorized L4
alternative cost USD0.59/hour with a USD0.7375 whole-rental reservation.
Whole-rental estimated cost is **USD0.438120592**; cumulative billed-plus-estimated
cost is **USD0.913464481**, with conservative reservations **USD5.839145926 of
USD10**. Provider bills are not final. Unrelated `fc6ffb-l4` rentals were left
untouched. The advertised `etude-result.json` returned HTTP 404 on both fetches;
launcher success, the training receipt, final tracking, checkpoints and paired
results are retained in the [summary](data/mtg213/gpu-v10-20261105/summary.json).
This establishes a trained, evaluated preliminary pilot endpoint, not completion
of its full 100-update scientific schedule.

## Contract and interpretation

The [v10 protocol](protocols/tla-trophy-213-v10.md) retains the damped 64×2,
four-head policy, input-v7 shared effects, 128 cards and 64 permanents per seat.
Jack Heart selected the same deterministic first-64 legal-action prefix for the
heuristic, learned candidate and initialization anchor (comment ffd29028).
Restriction precedes scoring; retained row i executes original native Command i.
The engine and Academy retain the complete legal set. Later legal choices have
zero probability on overflow; ordering neutrality is not established. Diagnostic
totals cover consumed decisions across participating policies, not actor-specific
or learner-only rates.

Jack Heart accepted game-local resource recovery (ca1660e8): discard only the
failed game's uncommitted samples from both seats, retain its replay and reason,
and replace that stream through ordinary sampling. Completed updates and other
games survive. No outcome or terminal reward is invented. The settled-game
window/warmup is 100/100; 25% failures continues, strictly above 25% stops.
Reset, malformed-buffer, provenance and infrastructure failures remain fatal.
The finite capacities are allocations, not bounds on reachable game states.
Larger capacities, grouping and variable-length inputs remain exploratory.

The [corpus](data/mtg213/corpus-v1/corpus.json) and
[original protocol](protocols/tla-trophy-213-v1.md) retain the first 1,000 observed
seven-win events from the permitted public TLA PremierDraft dump: 134 eligible
whole lists and 866 rejected, including main decks and sideboards. No card or
event was substituted. Missing source losses remain unknown. Each seat draws
independently and uniformly from those 134, including mirrors. Evaluation uses
the same population with fixed paired/swapped assignments. Live fetching and
held-out-deck generalization are deferred. A later 306-list admission audit does
not enlarge this frozen population.

All executed source, worlds, shapes, protocols and evidence stay frozen.
Prospective repairs required fresh initialization and evaluation. A common world
label, deck admission or successful CUDA gate does not establish compatibility,
rules correctness or sufficient capacity. Failed callbacks remain failed matches;
unknown games and unreached checkpoints remain in the expected schedule.
New games cannot fill historical missing slots. The companion analysis owns the
joint criterion; descriptive curves alone cannot qualify improvement. Different
worlds, populations, inputs, hardware and batch sizes also prevent a controlled
comparison with MTG-183's historical Elemental learning pace.

## Earlier execution evidence

Each completed initialization comparison below contains eight pairs × four legs.
Every listed training run stopped before exporting a trained checkpoint, leaving
256 of its 288 scheduled evaluation games missing. Scores describe initialization
only. V1 and v3 happen to share a score; they are independent executions.

| Version / Run | GPU | Learner updates / transitions | Stop | Initialization wins / games |
| --- | --- | ---: | --- | ---: |
| [V9 / 2171156493ca](data/mtg213/gpu-v9-20261104/summary.json) | RTX 5090 | 5 / 251,040 | 41 permanents > 40 | 7 / 32 |
| [V5 / e8c76bc2c06f](data/mtg213/gpu-v5-20261104/analysis.json) | RTX 5090 | 3 / 149,260 | 256 offered actions > 64 | 5 / 32 |
| [V4 / baa8671d6480](data/mtg213/gpu-v4-20261104/summary.json) | RTX A6000 | 0 / 0 | 61 card objects > 60 | 2 / 32 |
| [V3 / c216c6be7037](data/mtg213/gpu-v3-20261104/summary.json) | L4 | 0 / 0 | 270 facts > 256 | 3 / 32 |
| [V1 / 40709cd93f90](data/mtg213/gpu-corrected-20261103/analysis.json) | L4 | 0 / 0 | 140 facts > 128 | 3 / 32 |

V9 source `6da60684cb628ec9b431dbbb9d7dc91348743622` completed 983 optimizer
steps and 1,291 games. Its [371-action replay](data/mtg213/gpu-v9-20261104/capacity-replay.json)
at stream 43 / seed 32624 contains 25 Phoenix Fleet Airships, 11 lands and five
other permanents. Successive qualifying end steps can double Airships; fitting
this witness at 64 rows never established a universal bound. V9 counted 302,941
consumed decisions and no action cutoff. V10 delivered the prospective repair;
V9 remains failed, with no checkpoint inferable from its counters.

V5 source `97a2cfb963d7e3a3b2816f31c45dda72f34cdddd` completed 585 optimizer
steps and 771 settled games. Its [76-action replay](data/mtg213/gpu-v5-20261104/capacity-replay.json)
at stream 103 / seed 32172 exposed illegal X=0..255 choices for airbent Wan Shi
Tong's fixed {2} alternative cost. Those offers were not 256 payable strategies.
The later selected-cost repair fixed announcement/payment and direct rejection
of unavailable X without committed costs or state advancement. Historical
observations remain encoder-boundary evidence, not a required legal action set.

V4 completed 238 games before collection failed, none supplying an optimizer
update. Its retained stream 22 / 300-action prefix needed 40 original cards,
five generated tokens and 16 sideboard cards. Deriving card capacity from
ObservationSpace and selecting 128 repaired this witness without dropping cards.

V3's [208-action replay](data/mtg213/gpu-v3-20261104/capacity-replay.json)
at stream 251 / seed 31552 retains nine Drill activations × fifteen permanents ×
two keyword removals = 270 distinct fact rows. Its 171 completed games supplied
no update. V1's [87-action replay](data/mtg213/gpu-corrected-20261103/capacity-replay.json)
at stream 181 / seed 31482 similarly needed 7 × 10 × 2 = 140 rows; thirteen games
completed before that batch failed. Distinct timestamps/provenance ruled out
exact deduplication under frozen v5 semantics. V6/256 repaired the first witness
but failed the second. Jack Heart approved shared-effect tensors (42a12145),
delivered by Task 5da5f4 in PR #419: complete payloads encoded once and routed to
affected objects before attention. The 270 fragments become nine payloads and
135 ordered memberships without deleting rules records.

The first paid V1 attempt, `d8ff03724b8a` on A6000, failed a hardware-dependent
timing assertion after 31 CUDA checks, before training. A
[source correction](data/mtg213/gpu-corrected-20261103/source-correction.json)
changed only that fixture to injected durations; the corrected L4 attempt passed
32 CUDA checks and its optimizer probe. Neither gate predicted later capacity
failures. Local replays explicitly admit macOS native bytes against Linux evidence.

## Placement, costs and unexecuted preparations

Jack Heart authorized GPUs and enabling repairs under a cumulative USD10 cap,
including failed attempts, setup and teardown. The earlier USD1 allocation limit
became USD1.25 whole-rental / USD0.97-hour for the accepted Vast placement on
2026-11-04. Etude's job ceiling includes setup; the v9 placement copy raised
`Job.max_usd` without changing SDK job identity, source, Run or schedule.
Configured GPU alternatives were authorized, with deviations recorded before
scoring. The isolated Etude 0.2.14 control client left frozen workers on 0.2.12.
Nested Doppler `manabot/prd` then `etude/prd` supplies artifact/provider credentials.

These are dated settlements; v10's accounting above is the latest retained read.
Per-run cost can omit setup and cleanup, so allocation decisions use whole-rental
quotes plus cumulative reservations. Quotes reserve no capacity. Unrelated
rentals remain outside this Task's cleanup and budget ownership.

| Settlement | Cumulative billed plus estimated USD | Conservative reservations USD |
| --- | ---: | ---: |
| [V1, 2026-11-03 20:48 UTC](data/mtg213/gpu-corrected-20261103/settlement.json) | 0.06508817 | 1.475 |
| [V3, 2026-11-04 02:26 UTC](data/mtg213/gpu-v3-20261104/settlement.json) | 0.17051348 | 2.2125 |
| [V5, 2026-11-04 17:33 UTC](data/mtg213/gpu-v5-20261104/settlement.json) | 0.327106774 | 3.913991605 |
| [V9, 2026-11-04 22:13 UTC](data/mtg213/gpu-v9-20261104/settlement.json) | 0.475343889 | 5.101645926 |

V4's final provider bill was USD0.043283410370349884, included in subsequent
accounting. V3's final bill likewise superseded its earlier estimate. Missing
`etude-result.json` outputs and delayed cleanup during network outages remain
recorded in each attempt's receipts; launcher status alone did not prove teardown.

V2 prepared v6/256, then V3 bound the newer catalog before execution. Prepared
V6 `bcde93e9ff8c` (source `ad2c3a010f53`) and
[V7 af3661790ac5](data/mtg213/gpu-v7-preflight-20261104/source.json) selected 512
actions and were superseded by Jack Heart's first-64 decision. V7 source was
`b5406bb33abd98870af6ca07c3beed63a92b7c1c`. V8 `f47be2d3c0d8`, source
`900c207502bd28757d7cf66196bd464458e39402`, used asymmetric heuristic support;
V9 superseded it with matched support. None of V6/V7/V8 was submitted or scored.
Their preflight evidence remains under `data/mtg213/gpu-v7-preflight-20261104/`
and `data/mtg213/gpu-v8-preflight-20261104/`. V9's
[preflight](data/mtg213/gpu-v9-preflight-20261104/outcome.json) and
[recheck](data/mtg213/gpu-v9-recheck-20261104/outcome.json) record price/capacity
refusals before its later accepted execution. These refusals incurred no rental.

## Validation and retained history

The separate [CPU proof](data/mtg213/local-proof-v1/run.json) used two streams,
one update and input v5: 199 learner decisions, 38 optimizer steps, two completed
games and 12 valid evaluation games, with no paid compute. Its raw/EMA/Adam and
initialization exports remain retained. One paired deal scored 0/4 against the
heuristic both before and after training, and 3/4 against initialization. This
establishes bounded integration, not learning strength or GPU performance.

PR #416 delivered versioned fact capacity; #419 shared-effect inputs; #420 card
capacity; #423 legal-X, configurable buffers and matched first-64 support; #426
permanent64 and game-local recovery. Their implementation checks and retained
counterexamples support those contracts, not universal rules correctness or
capacity. PR #426 passed CI and 143 focused postmerge checks. This evidence/report
continuation changes no executable code.

All experiment bytes are manifest-published; restore them with:

```sh
uv run --extra artifacts python -m manabot.data fetch experiments/data/mtg213
```

Detailed historical validation, source pins, bootstrap intervals and operational
narration remain at
`07f1b2745cb43a9d774133086078dc7712c44dd2:experiments/tla-trophy-pilot.md`.
The complete report before this reduction, including v10, is retained locally as
git blob `3c1b548f0db42e42ae4b665dd7f1c770a94bf50d`. The receipts and manifest remain
unchanged; this summary replaces obsolete launch instructions with dated history.
