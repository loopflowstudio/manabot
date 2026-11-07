# Task 07c490 — closure and lambda on the ten-deck Elemental recipe (2026-11-05 UTC)

Inherited MTG-213 and fc6ffb notes no longer direct this work; they remain at
`ac539b38d:scratch/questions.md`, and main's MTG-243 naming notes at
`f6f171d34:scratch/questions.md`. The protocol
`experiments/protocols/closure-lambda-elemental-v1.md` owns the design, predictions,
budget and results pointer; this file holds working assumptions and open items.

## Direction

- Jack Heart, 2026-11-05: repeat MTG-251's two comparisons on the ten-deck Elemental
  setup, $25 total, hard cap including failures and setup. MTG-251 landed as #431
  (ac539b38d), so the Task can start.
- Steer (comment ed18e2a5): no full 10x10 grid; sample random deck pairs, fix them and
  their deals before scoring, identical across arms, exchanging decks and seats within
  each pair; still scored against the heuristic.
- Steer (comment bc3b4385): report how much wall time two-seat Elemental runs lose to
  idle streams (`idle_stream_steps`); report it, do not build two-seat exact quota.

## Assumptions taken here (reversible, recorded in the protocol)

- Pairs are drawn with replacement from the 45 distinct unordered Elemental pairs;
  mirrors are excluded because exchanging a mirror's decks changes nothing (4 legs per
  pairing: both decks × both seats). 50 pairings × 4 legs = 200 games per checkpoint,
  the same count MTG-251 scored (25 deals × 8 legs).
- Study 1 is one-seat on both arms (exact quota needs one seat), without the recipe's
  game-error recovery (exact quota rejects it; the complete arm drops it too so the arms
  differ only in closure and floor/quota). The complete arm keeps the recipe's floor of
  64; the exact quota comes from calibration of the complete arm alone.
- The frozen Elemental regime names world w13; the variants relabel it to the current
  world (w17) with new IDs, as MTG-181 did for w10 → w13. `experiments/elemental.py`'s
  own `runs()` still loads the w13 file and would fail the current-world check; out of
  scope here.
- The CUDA numerics gate (`validate_numerics`) stays off, as MTG-251's launches; it is
  job-level, not recipe.
- `learning_signal.py`'s environment digest and conditions shape changed for the Allies
  design too (design record instead of matchup; conditions carry roles). The published
  MTG-251 evaluation is complete and frozen; a fresh `etude eval` of those files would
  treat the old matches as a different environment. Nothing plans one.
- `etude launch` (0.2.12 and 0.2.14 CLIs) cannot read `runs()` signatures with tuple
  parameters, so the Elemental files take `seeds`/`arms` as comma-separated strings and
  each arm launches as its own all-or-nothing cohort.
- Idle wall-time for two-seat runs is reported as an upper bound: collection seconds
  less batch assembly, times the idle share of stream slots. The one-seat closure study
  gives the direct measurement of finishing games against pausing.

- The calibration launches from `experiments/elemental_closure_calibration.py`: Etude
  names a run's experiment after its launch file's stem (`ETUDE_EXPERIMENT`), so it
  stays out of the closure study's evaluation passes.
- Each run's dollar limit equals its reservation (window + 15 min setup at $.60); the
  launcher refused $.36 for a .6 h window. Protocol amended before any run started.

## Log

- Calibration: four attempts, ~$1.30 billed, $2.04 charged. Three died on launch errors
  of this study's making (an undersized dollar limit; a `--calibration` flag that
  renamed the experiment → `experiment_differs_from_launch`; Etude names the experiment
  after the launch file's stem). Two of those were also terminated early on the false
  belief that `setting_up` meant not booted; healthy runs show `setting_up` and "output
  not received" until done, and setup took 12.5 min. Inspect with
  `etude logs --run ID --follow`. A failed job can keep its machine rented: terminate it.
- 491fa11b9fdf (04c9cba75) trained 9 updates (~70 s each, one seat) and died on
  continuation relations 65 > 64. Study 1 stopped under its kill criterion; protocol
  amended (6d7c84db2). Lambda arms set `recover_capacity`, windows 1.5 h / $1.05.
- Lambda baseline (.5/.8) c608a0fa7d29/04cef7d8ce8e/67f108f57c2e: all reached 75
  updates, ~31–34 s median update, 19.7–19.9k games, 46–51k rows/update (~190/stream
  vs floor 64). Idle share .637–.650; idle collection bound 771–799 s of 1279–1356 s
  collection (+1087–1193 s learning). Controller died 00:43 (0.2.13 can't parse Vast
  providers); restarted from the 0.2.14 overlay (`/tmp/e07c490 controller run`).
- Lambda arms launch one seed per launch (`/tmp/07c490w/launch_seeds.sh`): L4 stock
  rarely covers three at once. Six ghost records filled the 8-machine limit ("it is
  busy") and resist `etude terminate`; `etude ondemand max 07c490-l4 14` (03:15).
  .8/.8: 6952e8b1391f (s70201, stopped at 72), ff12bf198bb3, 2e45a87e9de4.
  Eval pass 1 (elemental-lambda-v1-6860cc47ef30, 07c490-cpu32): 550 matches, 02:42.
- All 15 lambda runs succeeded (05:35). Endpoints: 75 except .8/.8 s70201/s70202 at
  72, .5/.95 s70202 at 64, .8/.95 s70202 at 73 (70-min wall on slow machines: median
  update ~52–63 s vs ~28–36 s; machine speed is bimodal, so wall-time comparisons are
  machine-confounded). Idle share .626–.670 in every run; idle bound 747–1249 s,
  about 29–34% of collection+learning. Capacity overflows: 3 games of ~290k (one each
  in .5/.8 s70201, .5/.95 s70201, .8/.95 s70203), all recovered.
- Billed (`runpodctl billing pods`, GPU pods only, hourly buckets): calibration $1.19
  (four pods); 00:00–05:59 UTC 2026-11-06 $8.94, all L4 at $.585/h (15 run pods $8.81,
  short boots $.13). Eval CPU pods never appear there, so none of it is evaluation. The
  ten ~2.1 h pods at 21–23 UTC 2026-11-05 are MTG-252's. Ghost `07c490-cpu32-*` records
  match no pod.
- Eval: a refused cpu32 create (RunPod 503s ~05:41) left a ghost session that pins
  the run id Etude derives from the pending set (`{name}-{digest(pending keys)}`);
  `terminate`/`cancel` don't clear it. Pass 2 played a different set (no u20, a
  temporary commit since reverted): elemental-lambda-v1-e486e50e04a7, 1200 matches,
  07:10. Pass 3 (elemental-lambda-v1-05e863276951, 500 matches) played the u20
  remainder; all 2,250 pairings are scored.
- Ghost `07c490-l4-*` records from refused creates show estimated dollars but no RunPod
  pod exists (`runpodctl pod list` empty); not charged.

## State (2026-11-06, after the closure study)

Done: both comparisons, critic residuals, two-seat idle report, published report
(`experiments/data/closure-lambda-elemental-v1/`, 4 entries republished). Closure:
exact 3/3 at 75 updates, complete 59/69/64 on the wall; eval elemental-closure-v1-
2ca6abbbae66, 900/900 pairings (`out_of_time` after finishing; the controller did not
close it). Spend ~$20.1 of $25: closure calibration $.54 and training $3.95 billed
(bills were still settling: the 17:00 bucket rose from $1.55 to $2.49), eval ~$.71
(Etude), ≤$1.20 cap. No machines rented; the overlay controller started here was stopped.

Assumption (decision pass, 2026-11-06): route 1 of three (capacity recovery under exact
quota; the others were a wider continuation capacity for both arms, or deciding exact
quota's future). It is reversible branch code that keeps the v1 input and stays inside
the $25; protocol amendments record it. Exact quota's future stays Jack Heart's. The new recovery path never fired in a scored run (no exact-quota overflow).

## Open

- Evaluation dollars cannot be reconciled: `runpodctl billing pods` is GPU-only, and
  neither the REST billing endpoint nor the Etude ledger ($2.07) gives a bill. Check the
  RunPod console if the exact figure matters.
- The Etude controller is not running: 0.2.13 (`~/.manabot/etude-controller`) cannot
  parse the Vast providers; the 0.2.14 overlay run hung on storage reads after ~2 h.
- `07c490-l4` on-demand max is 14 (raised from 8 for ghost records).

## Proposed Wave memory (`.lf/wave/` left unchanged; `realign` here is not its curation)

- Elemental w17, one seat, 1.9–2.3M decisions: exact quota (121) ≥ complete games per
  decision (−.035, CI spans 0) and clearly ahead per wall second (−.071), because
  complete-game targets run serially over one packed column (~7 s/update vs .06 s).
  Fixing that packing would remove most of complete games' wall cost.

- Elemental w17, two-seat, 3.1–3.8M decisions: no lambda arm beat .5/.8; .95/1 scored
  .158 below it. Early training, three seeds.
- The recipe's v1 input overflows at 65 continuation relations on w17, so runs need
  `recover_capacity`, which exact quota rejects.
- RunPod pod billing omits CPU pods; L4 host speed is bimodal (~2×), so wall and dollar
  comparisons across arms are confounded.
- Local `test_game_errors.py::test_resource_limit_receipt_replays_its_retained_encoding_boundary`
  (3 cases) returns `binding_mismatch` against the retained TLA fixture frame with this
  checkout's extension, at base f6f171d34 as well; the branch changes no Rust and main's CI
  passed at c59aeee0e. Treated as a local-extension mismatch; CI owns the check.

Check (realign 2026-11-06): `uv run --extra dev pytest tests/training/test_game_errors.py tests/training/test_batch_closure.py tests/training/test_elemental_closure_lambda.py -q` → 66 passed, 3 failed (local-extension replay cases above); `manabot.data publish experiments/data/closure-lambda-elemental-v1` → 0 manifest entries changed.

## Concept review (graph step, 2026-11-06)

Outcome: nothing to change at the concept level; no open product choice for this Task.

- `Design` (a protocol's held-out pairings, shared by every arm) and `Pairing` (deck
  pair + deal, legs exchanging decks and seats) carry the 2026-11-05 random-pairs steer
  directly; the Allies/Lessons protocol becomes one `Design` rather than a parallel
  path. `Conditions` names its design, so one `play_deal` serves both.
- Exact quota under capacity recovery is one rule in two places (regime and collector):
  game-error recovery is allowed with exact quota only when it recovers capacity, and a
  failed paused game keeps the rows an update already learned from (`learned_samples`).
  No new concept beyond that receipt field.
- Non-blocking wording, for gate/polish if touched: the collector's error "exact quota
  requires one seat, no per-game quota and capacity recovery" reads as if capacity
  recovery were mandatory; the regime's message states the rule exactly. `play_deal`
  now plays a pairing; the name is kept because Etude job commands reference it.
- Exact quota's future (keep, fix complete games' serial packing, or two-seat exact
  quota) is outside this Task: the brief asks for reports only; it stays Jack Heart's.

Sync resolution applied 2026-11-05 (merge of `423dd2fcd`): main's v8 keeps its
number and layer meaning. The branch-only `[A,2]` v8 is dropped. V9 is main's
v8 layout (4096 × 3 details, 512 default facts, fact kinds 11–14, layer
admission) plus `name_candidates[A,4]`; the checkpoint binding always records
`public-groups-v1`. V1–v8 see naming prompts anonymously. Academy fixtures,
authored-match/parity receipts and their pins were regenerated from the merged
engine (`generate_authored_match_receipt.py`, both parity `write`s,
`refresh_academy_fixtures.py`). The naming learnability witness now allows up
to eight bounded supervised steps for greedy selection, because v9's v8-based
model initializes differently; its loss-decrease and gradient checks are
unchanged.
Main's `semantic-kernel-w17-v2` corpus was recorded under state hash v15; this
branch's hash contract 16 moved every case's final hash and trace digest.
Re-recorded with `semantic_conformance -- record`; command counts, outcomes,
manifest and Phase rows are unchanged.

# Task b015d5 — training seconds per update (2026-11-05)

No design doc: the Task statement is the spec. Status and decisions:
- Done and committed: receipt/panel attribution (native clock split, stream
  bookkeeping, `TrainingRun.host`), trajectory-parallel returns, batched rollout
  packing and assembly, `scripts/bench_training_update.py`.
- Assumption: no GPU is rented for this Task (no spend authorized). GPU attribution
  uses retained receipts; fixed-hardware controlled rows use one laptop, CPU device,
  so model phases there are not GPU evidence. The first real run after merge gives
  the on-GPU after number; it is not claimed here.
- Assumption: results live in `experiments/training-update-time.md`; the rows are
  few enough that no `experiments/data/` publication is needed.
- `--max-facts` in the bench is an attribution axis; recipes and frozen observation
  versions are unchanged. `elemental-uniform` validates to one producer, so worker
  scaling cannot run on this recipe; lifting that is a separate decision.
- Open: the on-GPU after number and a CUDA digest check need a real run. 256-stream
  sequential rows showed minibatches higher after; 64-stream interleaved pairs did
  not. Report states both.
- `HostFacts.cpu_model` reads `/proc/cpuinfo`, so it is null off Linux; the GPU stays
  in run identities, not `TrainingRun.host`.
- For gate: `test_resource_limit_receipt_replays_its_retained_encoding_boundary` fails
  3/3 at `3b5dd2935` after a rebuild, and identically on base `8e5d152ab`:
  `tests/fixtures/tla-pilot-permanent-capacity.jsonl` retains a schema 7 frame and
  native emits `SEMANTIC_DECISION_VERSION` 8, so the replay returns `binding_mismatch`.
  Inherited; this Task's only Rust change is timing. `tests/training/test_game_errors.py`
  is in no CI selection, which is why main stays green. Not repaired here; it needs
  its own bug Task, because the fixture is retained evidence.
- The fixed-hardware rows measured `5d676d6df` and `8e5d152ab`. The sync to `f6f171d34`
  added engine commit `c59aeee0e` afterward. The 2026-11-06 rows at `76e71abc3`
  include it (native stepping 52.6-64.3 µs/decision), but no before/after pair
  isolates it; the engine budget harness owns that comparison.
- Proposed infrastructure Wave memory (not written; this step leaves `.lf/wave/`
  read-only): rented-host CPU class moves update time 1.6-1.8x at a fixed GPU label,
  so compare `host.cpu_model` and native stepping before attributing a slowdown
  to a change; a packed ~50,000-row column makes per-time-step device recursions
  launch-bound, so keep such recursions on the host.

## Concept review b015d5 (2026-11-05)

Walked: rent a pod, read the panel, keep or replace it; a slow host; a recovery
attempt (the "another machine" case was wrong; see the re-review below). The model stands: one phase tree in the receipt, drawn
per update; `TrainingRun.host` per attempt; the bench prints the same tree. The clock
covers every `VectorEnv` buffered path (no rayon; `encode_active`/`step_active` are
the separate slot env). No open product choice: producers, GPU spend and recipes stay
outside this Task.

All five review items are committed at `1201df60e`: host screen, parent series,
one-producer scope, stepping rename, recovery reading. A later review that read the
earlier tree called them undone; each was rechecked at that commit. Results:
- Screen values came from `r2://manabot/jobs` receipts and per-update metric blocks,
  re-read 2026-11-05: 10 runs, median updates 2-11 at 6.9-7.7 s fast, 10.6-12.4 s slow.
  Bands 8 s / 9 s are this Task's choice from that gap; they fit 256-stream two-seat
  Elemental only. Six A40 runs' early medians were not read; their stage means were.
- Recovery check: stage timers do not carry. `execution.py` zeroes seconds and
  `collection_timings` on resume while `updates` and decisions continue. Stage means
  are still the wrong reading, for that reason; `test_recovery.py` pins it.
- `time_since` lost its "timer went down" branch: every reset precedes the loop's
  first reading and the learner is never rebuilt inside it.
- Checks on the rebuilt extension: `tests/training/test_phase_timing.py`,
  `tests/infra/test_host_facts.py` (11 passed),
  `test_real_interruption_restores_complete_learning_state` (4 passed). Broader
  suites stay with gate.

## Concept re-review b015d5 (2026-11-05, at `1201df60e`)

Normal path and failure re-walked against the code. The phase tree, native clock,
panel, bench and machine recommendation stand; no open product choice. One
correction to the model, and one fact it makes redundant.

Correction: **a run never changes machines.** `settle_orphan` raises "recovery lease
belongs to a different host"; launched jobs are `max_attempts=1`; `RunContext`
refuses a launcher retry that finds checkpoints. Replacing a slow pod therefore
means a new launch and a new run, so a run has one host. Resuming does not refuse
on a changed config: Etude keeps the stored declaration, so nothing there gates.

Applied 2026-11-05 (no receipt shape change, no rerun of measurements): the report
says recovery stays on its host and "replace the pod" is a new launch from update 0;
the `host.py` docstring says the facts take no part in recovery admission; `"host"`
left `resolved_config`, so `TrainingRun.host` is the one owner and reaches
`ETUDE_OUTPUT_DIR` as the job's receipt. No test read the Etude copy.

What would invalidate 3: an Etude page or report needing the CPU without the receipt.
None exists today. If one appears, the gap belongs on Etude's `Attempt`, which already
holds per-attempt `host` and `pod`; that is an Etude change, not part of this Task.
Check: `tests/training/test_phase_timing.py`, `tests/infra/test_host_facts.py` and
`test_real_interruption_restores_complete_learning_state` pass (15 selected, 4 of them
the recovery test's parameters). Post-merge verification has not run: nothing is merged.

## Concept review b015d5 (2026-11-05, at `426317204`)

Re-walked "screen a pod, replace it, keep the evidence". The phase tree, native clock,
panel, bench and machine recommendation stand, and `TrainingRun.host` stays the one
owner. No open product choice. One correction, to the report's wording only.

Gap: the report says "replace the pod" and "Record `host.cpu_model` either way", but
not how to replace it. Removing `"host"` from `resolved_config` left no live copy of
the CPU model, and Etude's courier publishes attempt outputs only once `finished.json`
exists (`_carry_attempt`). Deleting a slow pod therefore loses its receipt. Those are
the runs a good-CPU list most needs, and the list would be built from survivors.

Applied at `524691050`, one edit to "Screen hosts" in
`experiments/training-update-time.md`: replace a slow pod with `etude cancel <run>`,
then relaunch; do not delete the pod first. The CPU model is readable from
`training/run.json` once the attempt finishes, not during the screen. The update-time
series is what the screen reads live.

Why cancel is enough: `persist()` exports `run.json`, host included, at setup, before
update 1. Cancel raises in training ("Cancellation retains failed scientific receipts")
and the worker's `_finish` then records the outputs for the courier. This comes from
reading the code; no cancel was run. No code change: a pod lost outright loses all its
outputs, which is Etude's delivery contract, not this Task's.

## engine-measure b015d5 (2026-11-06)

Jack Heart's steer moved this Task onto the engine-performance graph (training update,
not only the engine). The measured plan and active phase are in
[b015d5-update-measure.md](b015d5-update-measure.md). Assumptions (reversible):
- Active phase is host-side and exact: one recorded copy per observation row
  (assembly no longer re-stacks ~4.3 GB/update). GPU launch/sync levers are listed with
  a named CUDA check and are not implemented without GPU evidence.
- Native stepping is an unchanged control here; the engine budget harness owns it.
- Failure-capture frames and `PolicyInference.check` keep their contracts; only the
  exact single-walk check is a candidate without review.
