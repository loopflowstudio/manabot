# MTG-333 plan: torch forward over `Prepared`, container saves, backend rule

Policy baseline: `ef28f3a58` (PR #501). CPU/historical CUDA baseline:
`105e7e2aa` (MTG-297 mana spending). Post-#514 CUDA baseline: `de39d601fb`.
One PR. Full history of this plan, with every attempt and
intermediate receipt: `696e7e667c:scratch/mtg333-plan.md`; slice 1–6 receipts:
`1f634d89b:scratch/mtg333-plan.md`; MTG-325's plan: `65fa8c7aa:scratch/plan.md`.

**Main merged through `de39d601fb` (#514, `493127545a`) as `403996c24b`.**
#514 routes every program-table read through `SemanticCardEncoder.lookup`
(`F.embedding` on CUDA with gradients, indexing elsewhere); the merge carries
it into the moved `backends/torch/_semantic_cards.py`, `_recent_events.py`,
`_facts.py`, `_shared_effects.py` and `policy.py`. `experiments/ppo_cuda.md`
reports it CPU-bitwise and a CUDA summation reorder; its L4 PPO update fell
69.97→32.23 s, mostly backward. The merge also changes the engine. Fresh
post-merge CPU captures match all 3,410 tensors and five update cases exactly
(2026-11-13). The post-#514 comparison below now isolates this Task.
The earlier main merge `54a55684e1` (Etude SDK 0.3.2) changes only eval,
report and update-metrics readers, `pyproject.toml` and `uv.lock`; no
`prepare`, backend, model, update or engine file, so the post-#514 CPU and
CUDA receipts remained applicable at that merge. The latest main merge is
`6fff54f7d2`, incorporating `ff881e3e04` (packaged Academy and native heuristic).
Unlike the SDK-only merge, it changes native Env/VectorEnv and bindings.
The retained CPU/parity receipts below predate it; final-runtime equivalence
and installed-package validation remain open. CUDA results describe their
recorded source and runtime, not a new measurement of this merge.

## Problem

Torch's matched backend wrapped the old forward: `ActorCritic` did its integer
bookkeeping on device tensors read from a tensor dictionary, while MLX read
`manabot.policy.prepare.Prepared`. Two input representations, two admission
orchestrations, and `save_checkpoint` wrote a format only torch reads.

## What people said (Task thread, read 2026-11-13 18:05 UTC)

- Task brief, outcome 5, Jack Heart 2026-11-12: "if it is slower, and frankly
  even if it isnt. go ahead and do an /optimize path at the end of the second
  PR". Judge by CUDA cost; behaviour-preserving; USD25 without asking.
- Comment 724ea5c3 (2026-11-13), Jack Heart: "should we go so far as to say
  that most folders (anything but experiment/ ?) shouldnt ever directly import
  from backends?", then "ok thats fine" to: library code receives a backend;
  only composition roots choose one (experiments/, scripts/, recipe entry
  points and the CLI, `academy/villain.py`, backend tests); one resolver
  (`manabot.backends.resolve("torch")`); state the rule in
  `backends/README.md` and CLAUDE.md without names; boundary test with an
  allowlist that only shrinks; after the Prepared forward, sim/ and eval/ take
  a `BackendPolicy`; file the rest (belief/ torch models, executor choosing
  the backend, training/state optimizer persistence, verify/) as a follow-up
  Task.
- Comment e9d088af (2026-11-13), Jack Heart asked "should Mapping[str,
  np.ndarray] get an alias?", then "thats fine" to: `ObservationBuffers` in
  `manabot.env.observation`, `ParameterArrays` in `manabot.policy.checkpoint`,
  Python 3.12 `type` statements, other uses stay plain, no behaviour change.
- Comment 73b0776e (2026-11-13), relayed for Jack Heart: "can we widen our gpu
  serach". Any CUDA card RunPod can rent now is acceptable for speed and the
  optimize pass if base and head (ABAB) run on the same card in one rental; try
  by price L4, A5000, A6000, RTX 4090, L40S, A40, A100, H100, secure and
  community cloud, any region. Record the card as a dated line; MTG-325's L4
  numbers are a reference only. Keep the USD 10 speed cap (USD 25 total). "If
  the stuck run 4b091160edbf still holds the one-machine template slot, use a
  separate template/machine name rather than waiting for its 19:15:51Z
  deadline, but never leave two rentals running for this Task."
- Comment 688982cb (2026-11-13), same relay: at 18:05Z RunPod A6000 $0.69 and
  A40 $0.59 rentable, RunPod L4 none, Vast RTX 5090 $0.83 ("a new on-demand at
  <=$1.00 is fine"), Vast 3090 none. "Earlier RunPod A40/A6000 failures were
  HTTP 400 at pod create, not stock: read the 400 body (region/volume/image/cloud
  type) before treating it as unavailability, and try Vast in parallel rather
  than only RunPod. Still one rental at a time for this Task."

## Demo

```
uv run python scratch/equivalence/run.py --belief-input /tmp/mtg333/current-belief-input.pt
EQUIV_BELIEF_INPUT=/tmp/mtg333/current-belief-input.pt uv run python scratch/equivalence/run.py /tmp/mtg333/current-shared-head.pt
uv run python scratch/equivalence/compare.py /tmp/mtg333/current-shared-base.pt /tmp/mtg333/current-shared-head.pt
  -> tensors 3410 ... differing 0
```

```python
policy, space = load_checkpoint("self-play-raw.safetensors")  # or an old .pt
batch = policy.prepare(obs)              # NumPy rows; the Prepared MLX reads
logits, values = policy.evaluate(batch)  # NumPy out
```

and the PR body's CUDA table: base vs head medians with spread, then the
optimize pass's before/after.

## State

Slices 1–8 are implemented. The retained post-#514 CPU oracle and updates are
exact; the latest native-runtime merge still needs the refresh described above.
All CUDA jobs have ended; the successful rentals were terminated. Provider
inventory confirms the final rental absent. The host optimizations and bounded
PPO preparation pipeline are retained;
the metadata experiment was discarded. The final three-process CUDA check
clears PPO's no-slower-median condition; ranges overlap main, so no reliable
speedup over main is claimed (PPO prefetch iteration).

1. `prepare` covers every torch configuration on the host (`Prepared[A]`,
   catalog-free and hashed-record `Tables`, NumPy trimming, belief/event
   admission, shared `check`).
2. Torch encoders read device `Prepared` rows; `arrays.on_device` batches
   transfers by dtype; catalog and fact program schedules are host-built.
3. `ActorCritic` accepts only device `Prepared`; `model/admission.py`,
   `model/inference.py`, the carriers and `RecordEmbeddings` are deleted;
   encoders are private modules under `backends/torch/`.
4. `save_checkpoint` writes the container; `read_metadata` is header-only;
   `torch.load` remains only in legacy `.pt` readers.
5. `manabot.backends.resolve`; exact-allowlist boundary test; sim/eval take a
   `BackendPolicy`. Deferred imports are MTG-343
   (`task_c2ef4a5da9cb437b921cca1ef40e94a1`, unstarted).
6. `ObservationBuffers`, `ParameterArrays`; docs describe the full PR
   (`manabot/backends/README.md` examples were run as written on CPU,
   2026-11-13). No unreleased release-notes section exists; the release note
   must replace v0.2.3's `admit_observation` advice with `policy.prepare`.
7. #513 port: the shared capture recorder takes host `ObservationBuffers`,
   forwards `Prepared` and keeps raw `ActionRows.observation` for replay;
   initial policy captures are `.safetensors`; MLX reads `Prepared.is_agent`
   for trimmed ownership; `MLXBackend(precision=)` reaches file loads.
8. CUDA equivalence, speed, profile and the optimize pass are measured
   (Receipts), and repeated against main `de39d601fb` after #514
   ("Post-#514 refresh"). The initial optimize pass made `arrays.array_fields`
   and `_replace_arrays` skip scalar-only tuples.

The final PPO prefetch check clears the no-slower-median condition:
main 1.354 [1.338, 1.355], parent 1.245 [1.234, 1.246], head
1.376 [1.298, 1.433] steps/s (three process medians per arm). Head is
10.5% faster than parent and 1.7% above main; main/head ranges overlap,
so this is not a reliable speedup over main. The prior 7.2% regression and
discarded metadata attempt remain in the historical receipts below.

**Findings and remaining validation:**

- CUDA gradients meet the Task bar (≤1e-8) but exceed the head-vs-head repeat
  floor. Review 4 accepted this finding (exact reply below).
- The order-dependent 300 s timeout reproduced on main; its leaked
  manager accept thread is fixed. The full reported order now passes (below).
- Latest host inference preparation is 13.85 vs main 12.59 ms (+10.0%),
  and checkpoint load 78.15 vs 69.20 ms (+12.9%); parent is 13.41 / 79.37 ms.
  The PPO pipeline does not run on either path. Full ranges are in the speed
  table; no blanket no-regression claim.
- The untimed threaded cProfile pass has inconsistent self/cumulative times
  (`actions.unpack`: 1.293 s self, 1.016 s cumulative) and overlapping counts.
  Do not use its function breakdown quantitatively. Timed total rates and
  main-thread phase timers are the performance evidence.
- Gate's broader suites, Python 3.12 and installed-wheel checks have not run.

Captures in `/tmp/mtg333/` are temporary. If gone, or after a change to
`prepare`, `backends/torch/`, the model, the updates or the engine: rebuild the
extension when native source changed, per `AGENTS.md`, extract the
base `manabot/` (`git archive 105e7e2aa manabot`), and capture base and head
with one retained belief corpus (`EQUIV_BELIEF_INPUT`; see Receipts).

## Constraints the remaining work must keep

- Bitwise CPU hazards, each found by a failing oracle or update comparison:
  `log1p` and the known-count product stay in the graph (host values differ);
  host feature matrices handed to torch linears must be contiguous (a strided
  five-of-six-column view moved outputs 2.4e-7; strided per-seat matrices
  drifted after biases trained); compaction is chosen at preparation time, so
  bootstrap rows are prepared at `next_observation` (the forward oracle
  passed while four update cases drifted); `torch.sort` tie order supplies
  the packed-GRU order; `where`, not a float mask multiply.
- The update comparison is required beside the forward oracle; either alone
  has missed a real difference.
- `PolicyInference` imports from `manabot.backends.torch.policy`; `resolve`
  raises `ImportError` naming an absent extra and `ValueError` for an unknown
  name; a non-float32 parameter fails the save naming it; `parameter_names`
  metadata carries Adam's positional order; `AdmittedRollout.rows` returns
  device `Prepared`.
- Method names `_gather_object_embeddings`, `_attend_objects`,
  `_gather_informed_actions`, `_value_from_objects` stay
  (`scripts/bench_local_update.py` matches them).
- `docs/observation-and-model.md` and `docs/learning.md` quote encoder code;
  re-copy "Scoring legal actions" and "Value" if the optimize pass changes it.

## Forbidden outcomes

- Any forward path reading a tensor dictionary, `AdmittedObservation`,
  `CardReferences`, `PreparedObservation` or `Retained`.
- A torch-only twin of `prepare`, or a second admission orchestration.
- Deprecated aliases for moved names (steer 8332334d, MTG-325).
- A version branch in encoders or model beyond the v10 turn-state append.
- Truncating or casting container tensors; regenerating frozen evidence.
- Changed seeded games, replays, serving protocol or checkpoint meaning.
- A speed or equivalence claim taken from the CPU smoke or MTG-325's receipts.

## Delete — do not maintain

Deleted predecessors are not restored; the cutover's list is in
`1f634d89b:scratch/mtg333-plan.md`. Removed by the 2026-11-13 compress pass:
`_semantic_cards.RecordNamespace`/`NUMBER_TOKEN_KINDS`/`record_key` and
admission's private `_key`/`_word` and fact-kind tuples (one owner:
`manabot.policy.prepare`), `policy.OBJECT_VALIDITY_KEYS` (now in `prepare`),
`rollout.observation_tensors`, `FactEncoder(max_facts=)` and its
`buckets`/`bucketed` attributes, `SharedEffectEncoder.detail_rows`,
`ContinuationEncoder.relation_dim`, `ActorCritic.turn_state`/`_link_columns`/
`_numeric_permanent_columns` (read `policy.checkpoint.layout`), `pack(order=)`
(takes `order_lengths`).

Post-#514: removed unconditional `_referenced_definitions` scans and full padded-fact
integer work (the graph still gets identical padded rows). The attempted
host-only metadata markers were discarded, not shipped.

Concept cleanup removed `_slice_obs_tensors` and its optional tensor-dictionary
return branch; `_slice_obs_rows` retains host values and the action-focus cast.
Removed the grad-mode read in `prepare_admitted`, the worker's grad-mode
context, and rollout's captured inference flag/`_prepare_rows(inference)`
parameter; explicit preparation purpose and exact learning results survive.

Concept review 4 removed the update protocol name `Backend`, the public
`load_mlx_policy` entry point, parity's `backend_module` helper and all
`resolve(...).checkpoint.BACKEND` consumer access. Preserve configured MLX
precision, lazy training imports, direct module updates and loader refusals.

Direct compress, 2026-11-14, removed parity's `backend` selector in favor
of `resolve`, MLX's single-use `_load_policy` and `mlx_gradients` wrappers,
unused `Layout` capacities/`objects`, and action-gather coordinate parameters
already carried by `Prepared`. Removed NumPy→Torch→NumPy round trips in
selection collection, coverage and the MLX pilot; batch regression tensors
convert directly in one comprehension. Serialized lookup buffers, recurrent
schedules, action ordering and numerical operations remain unchanged.

Direct compress validation before `6fff54f7d2` (CPython 3.13, existing cp313 native extension;
`uv sync --locked --extra dev --extra mlx --extra artifacts --no-install-package managym`
installed the locked Python dependencies without rebuilding the unchanged engine):

- `uv run --no-sync pytest tests/model/test_prepare_rows.py tests/model/test_agent.py tests/model/test_mlx_backend.py tests/model/test_backend_capture.py -q --basetemp /tmp/mtg333/compress-direct-tests` → **56 passed (26.13 s)**.
- `uv run --no-sync pytest tests/training/test_update_backend_protocols.py tests/training/test_update_backends.py tests/model/test_torch_backend_protocol.py tests/model/test_torch_container_load.py tests/test_package_boundaries.py tests/verify/test_learning_regression.py tests/model/test_record_coverage.py::test_elemental_execution_ids_have_catalog_rows tests/training/test_automatic_game_capture.py -q --basetemp /tmp/mtg333/compress-direct-contract-tests` → **93 passed (33.17 s)**, one existing Torch nested-tensor warning.
- `EQUIV_BELIEF_INPUT=/tmp/mtg333/post514-belief.pt uv run --no-sync python scratch/equivalence/run.py /tmp/mtg333/compress-direct-head.pt`, then `compare.py post514-base.pt compress-direct-head.pt` through uv → **3,410 tensors, differing 0**. `uv run --no-sync python scratch/equivalence/updates.py /tmp/mtg333/updates-compress-direct.json` then `cmp` against `updates-post514-base.json` → **five cases byte-identical**.
- `uv run --no-sync python -m manabot.backends.parity /tmp/mtg333/compress-direct-tests/test_capture_replays_both_seat0/capture/initial.pt --capture /tmp/mtg333/compress-direct-tests/test_capture_replays_both_seat0/capture --minibatches 1 --collection 1 --json /tmp/mtg333/compress-direct-parity.json` → **parity: pass (torch, mlx)**; MLX clipped-step weight max difference 1.2e-7. The pilot's `_torch_inference` also completed on one captured batch.
- Ruff check and format check on changed code, `git diff --check` → pass.
  No CUDA rerun: values, schedules and device operations are unchanged.
  Original uncommitted Markdown is preserved at
  `/tmp/mtg333-compress-direct-before/`; historical launch detail remains at
  the git reference in "Retained measurements". Broader gate, installed-wheel
  checks and CI remain outstanding.

## Facts that still bind

- `105e7e2aa` changes seeded games (102,209 steps over the budget's 512
  games, was 101,650) and adds pooled-mana `FACT_EFFECT_PROGRAM` rows on the
  player's object row, id `token("PooledMana")`. The oracle carries a
  synthetic pooled-mana case built as
  `tests/semantic/test_pooled_mana_classes_input.py::_observation`; a
  difference there is a `prepare` defect to fix on the host, never a
  torch-side special case.
- `2c47f59aea` excludes tokens from `belief_schema_from_engine`; the baseline
  predates it and generates 932 belief rows where head generates 900 (136
  belief tensors then differ, max gradient 8.502e-4, with identical weights).
  Keep the schema correction: the oracle loads one head-generated corpus in
  both arms. Baseline's pre-TLE schema builder also rejects the current
  native definitions' duplicate card/token names.
- `docs/development.md` requires
  `uv run --extra dev pytest tests/model/test_record_coverage.py -q` after a
  rebuild for record-vocabulary changes.

## Launching again

Launch files: `scratch/equivalence/mtg333_cuda_equivalence.py`,
`scratch/speed/mtg333_speed.py` with `--rent <template>`; `run.py` and
`bench.py` run under both arms. A launch commit tracks its own generated
patches (equivalence applies `git diff BASE HEAD -- manabot` reversed; speed
applies `git diff HEAD BASE -- manabot` forward). They are not in the tree:
regenerate and commit them. Frozen launch commits: `cd7afaec86` (A40
comparison), `e6cd7f59cc` (optimization), `46e3db32d7` (final check).

After #514, a CUDA speed or gradient comparison against `105e7e2aa` mixes
this Task's change with #514's CUDA reorder. The retained runs use
`de39d601fb` as the CUDA base (the CPU oracle keeps `105e7e2aa`, since #514 is CPU-bitwise); its
`manabot/` lacks `Prepared`, so the speed patch is `git diff HEAD de39d601fb
-- manabot`, and `bench.py` runs on both arms. Any new final-tree speed
comparison must use current main `ff881e3e04` and the same current native
runtime in both arms, adapting the harness's moved imports as needed. Do not
attribute upstream native changes to this Task or replace historical tables.

Control: `ETUDE_HOME=/tmp/mtg333/launcher`, `uv run --extra artifacts etude …`
(SDK 0.3.2). It opens a new schema-specific database beside the older
`launcher.db`; the current rental template was registered through the CLI.
Earlier runs remain with their frozen 0.2.15 source. Never downgrade the
current launcher. Credentials are injected with nested
`doppler run -p etude -c prd -- doppler run -p manabot -c prd
--preserve-env=RUNPOD_API_KEY --`. Base and head run on one card in one
rental; one Task rental at a time; USD 10 per speed cohort, USD 25 for the
Task. A cohort name cannot be reused. `availability` can say rentable and the
create still fail: read the provider body. After results: `etude terminate
MACHINE`, then check the provider inventory.

Rental facts from this Task: L4 quotes were refused ("provider reports no
available machines"). RunPod A40/A6000 creates returned HTTP 400 whose body
was "There are no longer any instances available with the requested
specifications. Please refresh and try again." (stock, 19:33–19:38 UTC). Vast
RTX 5090 rented both earlier times it was asked. A cancelled, never-created request
(`4b091160edbf`) showed provisional accrual and blocked its template until its
rental deadline, then settled **lost (create_unconfirmed), $0.00**: SDK 0.2.15
`Scheduler._observe` finishes an absent machine only after the deadline and
ignores `cancel_requested_at`; `terminate` refuses unfinished allocations.
That is an Etude fix outside this Task; never edit the launcher database.

The retained PPO pipeline prepares the next minibatch on the host while the
device runs the current one. Unmeasured candidates left for later work:
reuse each admitted row's prepared integer work across epochs; vectorize
`prepare._effects`' per-record loop; int32 index arrays; `Packed.gather` for fact programs; host
compaction on CUDA inference (summation order: tolerance, not bitwise); avoid
full-capacity temporaries before trimming. Each kept change reruns the CPU
oracle, the update comparison and one same-card speed pass.

## Acceptance at gate

- CPU oracle: `compare.py mana-base.pt head.pt` → 3,410 tensors in both
  arms, `differing 0`, on the final commit.
- `uv run python scratch/equivalence/updates.py OUT` → `cmp`-identical to the
  base capture.
- `uv run --extra dev --extra mlx --extra artifacts pytest tests/test_package_boundaries.py tests/model/test_prepare_rows.py tests/model/test_torch_backend_protocol.py tests/model/test_torch_container_load.py tests/model/test_policy_container.py tests/model/test_inference.py tests/model/test_admission.py tests/model/test_bucketed_shapes.py tests/model/test_live_tokens.py tests/training/test_update_backend_protocols.py tests/training/test_update_backends.py tests/training/test_minimal_loop.py tests/sim/test_puct_compatibility.py tests/sim/test_local_update.py -q`
  → all pass, no new skips (use a dedicated `--basetemp`; shared temp cleanup
  hung a run at exit); plus the tests of every other changed file and the
  checks in `docs/development.md#validation-by-change-area`, including the
  installed-wheel smoke and an isolated Python 3.12 consumer.
- Current-world arena admission/monitoring suite from `AGENTS.md` passes.
- `uv run --extra mlx python -m manabot.backends.parity CKPT --capture /tmp/mtg325-gate-capture`
  → `parity: pass (torch, mlx)`.
- Round trip: the bounded installed-wheel training fixture in
  `scripts/release/smoke.py` writes `.safetensors` policy files and checks
  load/catalog admission; sim plays the resulting policy; a `.pt` written
  at base loads with identical logits; a belief policy and a `bc`
  `value_semantic` survive save → load.
- `rg "AdmittedObservation|CardReferences|PreparedObservation" manabot managym/academy scripts`
  → nothing.
- Slice 8's CUDA results and the optimize before/after are in the PR body.

## Exclusions

MLX support for the newly carried configurations; PPO or a categorical critic
on MLX; optimizer/update-snapshot/incident persistence; belief/ torch models,
verify/, the executor choosing a backend (MTG-343); engine, env, serving
protocol, worlds, policy-input versions; rerunning frozen experiments.

## Done when

A developer prepares NumPy observations once and runs them through either
backend; training writes `.safetensors` policies that sim, eval and Academy
load beside old `.pt` files; the CPU oracle and CUDA checks hold; the speed
table and the optimize pass's before/after are in the PR body.

## Retained measurements, 2026-11-13

Full launch attempts, costs, intermediate speed/gradient findings and prefetch
profiles: `aac1062d9a:scratch/mtg333-plan.md`, sections "Receipts",
"Post-#514 refresh" and "PPO prefetch iteration". Full speed tables and
launcher costs remain in `scratch/speed/mtg333-results.txt`. These include the
initial A40 PPO regression (−7.7%), the post-#514 −20.2% and −7.2% PPO
checks, and the discarded metadata attempt; later measurements do not erase them.

Final CUDA run `d0a1ec0dfb75`, source `fdc42e7371`, compared main
`de39d601fb`, prefetch parent `53a705735b`, and head on one RTX 5090 rental,
three processes each, USD 10 cohort cap. Outputs, losses and CPU-trained
weights exact; 52 gradients differ, max 2.3283064365386963e-9. Head repeat:
seven gradients, max 1.862645149230957e-9. The Task's ≤1e-8 bar passes;
the stricter repeat-floor finding remains (accepted in review 4).

PPO main/parent/head: 1.354 [1.338, 1.355] / 1.245 [1.234, 1.246] /
1.376 [1.298, 1.433] steps/s. Head +10.5% vs parent, +1.7% vs main;
main/head ranges overlap. All arms: 55,391 learner rows, 217 damped and
16 PPO steps. Head preparation wait + transfer: 5.551 s; forward/loss:
1.093 s. Parent forward/loss included preparation: 8.205 s. Backward:
2.111→2.112 s. Threaded cProfile timings are inconsistent (State), so only
rates and main-thread timers support quantitative claims.

Complete digest-verified outputs: `/tmp/mtg333/prefetch-results`, retained
in Etude. Rental terminated; provider inventory confirmed
`etude-f482ab7677e1` absent. Generated patches were removed from the tree.

The subsequent compress pass simplified lazy prefetch submission; preparation
order and batch contents stayed unchanged, at most one batch ahead.
`cmp updates-post514-base.json updates-compress6.json` → identical;
`pytest tests/training/test_update_backend_protocols.py` → 8 passed (55.98 s).
The CUDA numbers predate that edit: the worker now receives batch i+1 before
the caller waits on batch i, so overlap can start earlier. Early exit still
joins the worker. Earlier host/role cleanup: 3,410 CPU tensors and five updates
exact, 63 focused passes (two CUDA cases deselected); complete commands at
`aac1062d9a:scratch/mtg333-plan.md`.

## New direction, 2026-11-13

Task comments 6987a723 and 8370e331, relayed for Jack Heart:
- Policy suffix: "ok, yes to safetensors." Settled; legacy `.pt` still loads.
- Timeout: "delete or determinized flaky tests", then "and detemrinze or delete".
  Check the same order on main first; find leaked state and determinize.
  Delete only if redundant; a head-only hang is a regression to fix.

## Callback thread leak, 2026-11-13

Main `c3dfd82476`, extracted to `/tmp/mtg333/main-order-c3dfd82476`,
used this checkout's interpreter and identical native extension (no engine
diff). From that directory:
`uv run --project /Users/jack/src/manabot.mtg-333-torch-forward-over --extra dev --extra mlx --extra artifacts python -m pytest tests/eval/test_batched_eval.py tests/model/test_inference.py --basetemp /tmp/mtg333/main-order-tests -q`
→ exit 1, the restored-continuation test reached **300 s**.
`/tmp/mtg333/main-order.log` names a live
`multiprocessing/managers.py:194 accepter` thread; a native sample
(`main-order.sample`) shows repeated OSError handling. This is not head-only.

`CallbackEndpoint` hosted the process-oriented manager `serve_forever` in a
thread; its separate daemon accepter ignores the stop event and retries a
closed listener forever. Replace that entry point with one owned accept loop,
wake its blocking accept before closing the listener, then join it. Manager
authentication and request handling remain unchanged. The real spawned callback
test now checks no threads survive; no test is deleted.

`uv run --extra dev --extra artifacts pytest tests/eval/test_batched_eval.py::test_private_callback_endpoint_runs_complete_replayed_block tests/model/test_inference.py::test_prepared_collection_and_restored_continuation_match_reference --basetemp /tmp/mtg333/callback-fixed-tests -q`
→ **2 passed (86.47 s)**. Ruff check/format passes.
`uv run --extra dev --extra mlx --extra artifacts pytest tests/eval/test_batched_eval.py tests/model/test_inference.py --basetemp /tmp/mtg333/head-order-fixed-tests -q`
→ **38 passed, 1 CUDA skip (235.99 s)**, process exited 0. Existing
read-only-array warning in the meta transfer test; no thread-exception warning.
Log: `/tmp/mtg333/head-order-fixed.log`. No test was deleted.

## Publication attempt, 2026-11-13

`lf sync` merged main as `5742e44d1a`; the sync resolver's four focused MLX
precision checks passed. The publish gate exposed newly merged callers and
fixtures still using old interfaces. Commit `aac1062d9a` repairs experiment
import-time discovery (the finished BF16 probe names its original revision),
Prepared search and training fixtures, checkpoint metadata, and the new
checkpoint suffix. The main gate had 146 passes and one stale resolver failure;
the corrected case passed separately. The post-merge oracle matched all 3,410
tensors and five update cases byte for byte. Search group: 31 passed; world and
collector group: 113 passed; repaired training fixtures: 55 passed; checkpoint,
action and failed-stage cases: 45 passed. Additional MLX/release/search run:
186 passed, three stale search cases subsequently fixed and passed. Ruff and
diff checks passed.

That publication attempt left PR #519 draft with failing integration CI.
Current publication and validation status is recorded below; the CUDA
repeat-floor and performance limitations remain review findings.

## Concept review 4, 2026-11-14 (implemented)

Review page was `scratch/concept-review.html` (not retained in this tree). "Backend" names three things today:
the module `resolve` returns (updates take it), the per-backend loader object
`checkpoint.BACKEND` (`manabot.policy.protocol.Backend`: `load_policy`,
`metadata`, `seed`, `load`, layouts; sim/eval take it), and the updates'
module Protocol `training.updates.backend.Backend`. Every root that feeds
sim/eval writes `resolve("torch").checkpoint.BACKEND` (ten sites), reaching
into a backend's submodule, which the resolver was meant to spare them.

Proposed:
- A. `resolve(name, *, precision=None)` returns the configured backend object
  (today's `checkpoint.BACKEND`; MLX's `MLXBackend(precision=)`), which also
  exposes the module's `rl`, `arrays`, `learner`, `rollout`. Roots write
  `resolve("torch")`.
- B. One Protocol named `Backend` (`manabot.policy.protocol`); the updates'
  module surface becomes `UpdateBackend`, `PPOBackend(UpdateBackend)`. A
  directly imported backend module still satisfies `UpdateBackend`.
- C. `backend.load_policy(path)` is the one documented loader; `load_mlx_policy`
  goes private; torch `load_checkpoint` stays for code that needs `ActorCritic`.

Counterexample kept: the backend object carries configuration (MLX precision),
so a bare module cannot replace it; A returns the object, not the module.
No values change; seeded games, checkpoints and the oracle are unaffected.
Jack Heart, 2026-11-14, answering the page's questions 1–3: "a a a" —
1a A–C all; 2a in #519 before demo (one more implement pass and CI);
3a accept the CUDA repeat-floor finding (gradients within 1e-8; summation order).
A–C are implemented. No CUDA rerun: no values change.


## Configured backend receipt, 2026-11-14

`resolve(name, precision=...)` now returns the loader object with lazy update
modules. MLX precision stays on that object; library roots use `load_policy`.
The update-only Protocol is `UpdateBackend`; directly imported modules still
conform. The later compress pass removed MLX's single-use loader helper.
Parity's former module-identity test
skipped device transfer for the configured Torch object; it now recognizes the
Torch policy type. The corrected capture replay passes.

- `EQUIV_BELIEF_INPUT=/tmp/mtg333/post514-belief.pt uv run --extra dev --extra mlx --extra artifacts python scratch/equivalence/run.py /tmp/mtg333/backend-object-head.pt` then `compare.py post514-base.pt backend-object-head.pt` → **3,410 tensors, differing 0**.
- `updates.py /tmp/mtg333/updates-backend-object.json` then `cmp /tmp/mtg333/updates-post514-base.json /tmp/mtg333/updates-backend-object.json` → five cases byte-identical (both scripts through `uv run` with the same extras).
- Focused resolver, loading, update-interface and boundary selection → **13 passed, 28 deselected (7.80 s)**. Strengthened configured/module conformance and training-free play checks → **3 passed**. `uv run --extra dev --extra mlx --extra artifacts pytest tests/training/test_update_backends.py tests/model/test_backend_capture.py -q --basetemp /tmp/mtg333/backend-object-updates-tests` → **11 passed (61.11 s)**.

- `uv run --extra dev --extra mlx --extra artifacts python -m manabot.backends.parity /tmp/mtg333/backend-object-updates-tests/test_capture_replays_both_seat0/capture/initial.pt --capture /tmp/mtg333/backend-object-updates-tests/test_capture_replays_both_seat0/capture --minibatches 1 --collection 1 --json /tmp/mtg333/backend-object-parity.json` → **parity: pass (torch, mlx)**, MLX clipped-step weight max difference 1.2e-7.

Changed Python files: Ruff check and format check pass; `git diff --check`
passes. `lf context`: 11,756/12,000 tokens, 44,844/98,304 bytes before this line.

Broader gate and installed wheels on Python 3.12/3.13 remain outstanding;
the current CI/publication snapshot is below.

Implement sanity pass: the backend guide now opens with `resolve` and
`backend.load_policy`; MLX's package header names its damped update.
The guide's load → prepare → evaluate example ran through `uv run --extra dev
--extra mlx --extra artifacts python` on one retained capture row batch for
both backends: finite logits `(2, 512)` and values `(2,)`, exit 0. No games or
training launched. Removed-interface search found no remaining callers.

## Concept review 5, 2026-11-14

Outcome for loop-or-next: **nothing to change and no open product choice;
advance**. This headless review retains review 4's implemented design. It
neither reopens its quoted decisions nor treats pending validation as proof.
The review screen is `scratch/concept-review.html`.

The experience remains: select a configured backend once, load a policy,
prepare the viewer's NumPy observations once, and evaluate or sample through
the policy. Academy discovers the training provider through the engine-owned
entry point; the provider selects Torch and injects it into discovery. No
framework-specific input vocabulary reaches Academy or sim/eval callers.
Existing usage guidance in the backend and policy READMEs already shows this
experience; no proposed usage or skill change is needed.

Three counterexamples preserve necessary distinctions. A portable container
does not make an unsupported MLX configuration runnable: refuse it and use
Torch, converting a legacy file only when another backend needs the container.
A policy file cannot resume an optimizer or collector: recovery snapshots
remain separate, and scoped inference batches must be prepared again after
the scope closes or weights change. An incompatible world cannot be repaired
by renaming or converting the file: retain its source/bindings and use the
recorded revision. These paths keep weights, value/belief meaning, seeded
actions and recovery evidence; collapsing their records would lose behavior.

The biggest remaining mismatch is evidence coverage, not the interaction:
the retained numerical receipts predate the native-runtime merge. Next show
the final-runtime oracle plus updates/parity and the installed-wheel
train → container → load → catalog/provider proof already required below.
This review changed no code, ran no games or training, and makes no new
numerical or performance claim. MTG-343 remains the agreed follow-up.

## Main integration and remaining work, 2026-11-14

`6fff54f7d2` changes native player-policy dispatch and packages Academy in
the engine distribution. Shared preparation, Torch forward and update code
are unchanged from `59c2aaa533`; that does not establish final-runtime
equivalence. Rebuild/confirm the current cp313 extension, refresh the CPU
oracle and all five update cases using one belief corpus in both arms, and
run captured Torch/MLX parity with a compatible current-world capture.
Keep the earlier 3,410-tensor and CUDA results as historical receipts until
those final-runtime checks pass; do not rebind old captures to bypass admission.

The new Academy boundary is `managym.academy` → `managym`, with optional
training capabilities through `managym.academy.bots`.
`manabot.sim.table_bots.checkpoint_backend` selects Torch and
`TrainingBots.discover` injects it into `model.catalog.discover_checkpoints`.
`CheckpointVillain` uses the shared container/legacy loader and the existing
Prepared verification adapter. Academy imports neither manabot nor Torch;
manabot imports no Academy modules. Boundary and installed-provider checks
must cover this route, replacing the deleted `academy/villain.py` root.

The installed-wheel smoke now owns one `checkpoint_path` for both loading
`self-play-raw.safetensors` and catalog discovery. The merge had left discovery
requesting `self-play-raw.pt`, although training writes the container suffix.
The provider docstring now describes both formats. No runtime or model change.

Remaining validation follows `docs/development.md#validation-by-change-area`:
the affected model/training/sim/eval consumers and world-admission checks;
package boundaries including training-free Academy and Torch-free children;
release/CI selection and moved-module collection; installed-wheel smoke in
isolated Python 3.12 and 3.13 environments outside the checkout. The installed
smoke must exercise `managym.academy.bots` discovery and catalog admission,
not only source-tree imports. The bounded native training fixture in
`scripts/release/smoke.py` replaces the unbounded default training demo as
the automated train → container → load → catalog proof; sim consumption and
legacy `.pt`/belief/value-semantic round trips remain required.

GitHub snapshot during realign: PR #519 is draft, published head `6fff54f7d2`;
preflight passes, integration/Academy/protocol/clean-machine jobs are running,
and the four Python 3.12/3.13 Linux/macOS package jobs are queued. These are
pending checks, not failures or completed proof. This pass neither publishes
nor waits for CI; its local repairs are newer than that published head.

MTG-343 is now tracker-synced and unstarted, with no PR. The retained direction
in `scratch/README.md` quotes Jack Heart, 2026-11-14: "oh sure yeah that works"
to starting it after #519 merges; its acceptance includes Torch-free imports
of update rules and rollout code. It remains outside #519. No new product
decision is needed here; the review-4 CUDA finding retains its quoted decision.

Repair checks: `uv run --no-sync pytest tests/infra/test_release.py -q
--basetemp /tmp/mtg333/realign-release-tests` → **39 passed (0.44 s)**.
Ruff check/format on `scripts/release/smoke.py` and `manabot/sim/table_bots.py`
pass. These verify release tooling and formatting, not an installed-wheel
smoke or final-native equivalence; neither was rerun during this realign.

## QA, 2026-11-14 (current native runtime)

Report: `scratch/qa.html`, including installed-browser screenshots and transcripts.
Built Academy's static SPA, staged source archives, and built both macOS engine
wheels from the engine sdist. Empty external environments on Python **3.12.12**
and **3.13.15**, without Cargo/Node on PATH: engine-only smoke completed a
107-decision game; training smoke passed with **torch 2.15.0 / etude-sdk 0.3.2**,
including the installed Academy provider and exact safetensors catalog path.
Both installed consumers loaded a real pre-change `.pt`, resaved/reloaded
exact logits/values, retained `signed_outcome`, detected container bytes under
`.pt`, de-duplicated identical catalog files, and refused corrupt files and
empty action sampling. Terminal/bootstrap preparation may carry an empty mask.

The installed simulator completed a **129-decision game, zero timeouts** using
the written policy. Installed `belief-demo` passed its schema-bound container
round trip and hidden-world invariance checks. An installed local CLI process
was interrupted with SIGINT after its committed start snapshot (exit 130,
`interrupted`), then resumed by run ID into a sibling output and completed
both updates. The ordinary source-tree local CLI also completed one update
in 8.19 s; this is not a speed comparison.

Installed Academy on port 18933: current policy catalogued, selected by digest,
New Game → legal Play Island → Pass Turn → trained opponent's turn → reload
restored the same settled pending-decision revision **8**; no page errors.
Elemental UR/GW were correctly refused for this policy's distinct bound UR
Lessons/GW Allies lists. Read both browser screenshots. Server and browser
closed; no QA server remains. Drivers and full logs: `/tmp/mtg333qa-final/`
and `/tmp/mtg333/qa-final-*`; builds and CLI output: `.runs/qa-final/`.

Fresh root cp313 build installed atomically. `qa-final-head.pt` versus retained
`post514-base.pt`, with the same `post514-belief.pt` corpus: **3,410 tensors,
differing 0**. `qa-final-updates.json` versus `updates-post514-base.json`:
**cmp exit 0**, all five cases. Normal-admission captured Torch/MLX parity
(one learner minibatch, one collection call) passed, clipped-step weight max
MLX difference **1.2e-7** (`qa-final-parity.json`). Baseline bytes and capture
were retained, not regenerated or rebound; CUDA evidence remains historical.

No runtime implementation defect found. Corrected the training guide's stale
claim that updates cannot take configured backend objects, and updated the
repository QA skill with current loading, bounded installed smoke and settled
browser reconnect guidance. Validated existing realign catalog-suffix repair.
Workflow compiler, `lf help qa`, Ruff and diff whitespace checks pass.
Broader gate, Linux consumers, legacy package upgrade and hosted CI remain
separate; no claim that those checks or exhaustive fault scenarios ran here.
