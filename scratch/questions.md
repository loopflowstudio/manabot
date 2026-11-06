# Current revision — 2026-11-05

Jack Heart confirmed ordinary sequential native group → name decisions, each
using the regular policy head and normal rollout; only the final name commits
the effect. Remove active neural compound decoding and its exclusive learning
paths. This supersedes proposals for internal routing or a hybrid neural head.
Read CURRENT ACCEPTED REVISION at the top of mtg-243-naming.md. Continue this same
Task/PR to interactive review; no merge or strength campaign. Compound decisions,
including mana payment, belong to a separate experimental-validation Task.

Material implementation assumptions (2026-11-05, reversible, for review):
- V1–v8 carry no name identity; they see group and name prompts as ordinary
  anonymous modal choices within capacity. Rejecting them would stop
  fresh-default v6 games that cast The Rise of Sozin. (Superseded by the sync
  resolution below: the branch-only whole-vocabulary `[A,2]` v8 was dropped.)
- Group identity is its public sorted span; the model embeds it as the exact mean
  of member name vectors. Member programs are not encoded for group prompts.
- The native compound-offer lowering (`Game::compound_offers`/`compound_commands`
  and their Python bindings) stays: it is an engine surface with Rust rule tests
  beside MTG-241's domain and a candidate input for study acc0fe. After the
  neural decoder's removal no Python code calls the bindings.
- Branching is the native constant 64, bound into v9 checkpoints; it is not a
  user configuration.

# MTG-243 assumptions — 2026-11-04

Accepted decisions, evidence and remaining work live in
[mtg-243-naming.md](mtg-243-naming.md). Jack Heart's speed direction continues the
same Task/PR against `b31998cc3`, ending at interactive review. Merge, auto-merge,
paid execution and strength experiments remain unauthorized.

Material implementation choice: keep the existing GRU, weights and serialized
buffers; pack integer tokens before embedding and execute at most 64 recurrent
steps per call. Carry hidden-state gradients across every boundary. This bounds
packed backward slice copies without changing model capacity or requiring a
checkpoint migration. It does not establish GPU throughput or training strength.

Native profiles overlap active MTG-241/242/244 ownership. Preserve those domains
and the unchanged budget reference; finish the highest-payoff measured learner
optimization here. Native measurements are unchanged-code controls. External
scheduling noise stays in raw evidence, separately from process CPU cost.

Additional production names, fresh v8 defaults and broad training remain separate
decisions. Historical detail is at `b31998cc3:scratch/mtg-243-naming.md` and the
pre-curation uncommitted plan in `.lf/logs/naming-speed/plan-before-curation.md`.

Sync resolution: main's MTG-213 pilot assumptions remain available at
`e113602a2dc1c866af513ff0953598db390db9f2:scratch/questions.md` and its delivered
results in `experiments/tla-trophy-pilot.md`. They describe a separate Task and
supply no execution or delivery authority for MTG-243. Keep this file scoped to
the naming branch; preserve the incoming pilot report and artifact manifest.

Sync resolution 2026-11-05: main `264041f99` replaced this file with MTG-245's
handoff (available at `264041f991afd3bd451cefa4f50a761577a5eb50:scratch/questions.md`);
it confers no authority on MTG-243. Main deleted docs/ARCHITECTURE.md and
docs/search-and-beliefs.md and moved policy inputs to
docs/observation-and-model.md, so the catalog naming section now lives there.

Sync blocker found at gate, 2026-11-05: main `423dd2fcd` (MTG-244, merged
22:52 UTC after reviewed head `833ef4120`) took the number **policy input v8**
for layer state (4096 × 3 effect details, fact kinds 11–14, 512 local facts).
This branch also defines v8 (`name_candidates[A,2]`, rejects groups) and v9
(v7 layouts + `name_candidates[A,4]`). PR #424 is CONFLICTING (config/bindings
version ranges, encoder errors, `__init__.pyi`, AGENTS.md, Academy fixtures,
conformance replays, parity digests). Gate did not perform the sync; it belongs
to land's sync step. Assumption for that resolution (reversible until merge):
main's v8 keeps its number and meaning; this branch's naming input becomes the
next explicit version, composing main's v8 layout with `name_candidates[A,4]`;
the branch-only `[A,2]` input was never on main and should not claim v8 (drop it
or renumber it explicitly — never reinterpret main v8 rows). Re-record fixtures
and parity digests from the merged engine; never hand-merge digests. Fresh v6
defaults stay. Re-run debug cargo tests, Engine Budget and integration on the
merged head; earlier green CI does not cover it.

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
  added engine commit `c59aeee0e` afterward; its effect on native stepping is unmeasured.
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

Next implement direction, one edit to "Screen hosts" in
`experiments/training-update-time.md`: replace a slow pod with `etude cancel <run>`,
then relaunch; do not delete the pod first. The CPU model is readable from
`training/run.json` once the attempt finishes, not during the screen. The update-time
series is what the screen reads live.

Why cancel is enough: `persist()` exports `run.json`, host included, at setup, before
update 1. Cancel raises in training ("Cancellation retains failed scientific receipts")
and the worker's `_finish` then records the outputs for the courier. This comes from
reading the code; no cancel was run. No code change: a pod lost outright loses all its
outputs, which is Etude's delivery contract, not this Task's.
