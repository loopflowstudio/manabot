# MTG-333 plan: torch forward over `Prepared`, container saves, backend rule

Policy baseline: `ef28f3a58` (PR #501 merged). Comparison baseline for every
receipt below: `105e7e2aa` (MTG-297 mana spending), merged as `9db3f1373`.
`f4e1822fa` (#510, training precision: `manabot/training/{execution,regime}.py`,
no policy or engine file) is merged at `070da04ed`; `origin/main` had nothing
further at 2026-11-13 realign. That merge changes no `manabot/` policy code, so
no capture is repeated for it. `a289fe066` (#513, MLX half precision and shared Torch captures) is merged
as `6e20654c34`; its recorder and experiment consumers now use Prepared.
One PR. Earlier full plan with every receipt: `1f634d89b:scratch/mtg333-plan.md`;
MTG-325's plan: `65fa8c7aa:scratch/plan.md`.

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

## State (slices 1–6 at `1f634d89b`; upstream merge `9db3f1373`)

Slices 1–6 are implemented and pass their focused checks:

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
   `BackendPolicy`. Deferred imports are local Task `c2ef4a`
   (`task_c2ef4a5da9cb437b921cca1ef40e94a1`, not started; Linear sync pending).
6. `ObservationBuffers`, `ParameterArrays`; docs describe the full PR.
   Shipped docs (reviewed against the code 2026-11-13; re-check after the
   optimize pass): `manabot/backends/README.md` (example, backend rule, torch
   backend, policy checkpoints, convert, parity), `manabot/policy/README.md`,
   `manabot/backends/mlx.md`, `docs/training-loop.md`, `docs/training.md`,
   `docs/evaluation.md`, `docs/observation-and-model.md`, `manabot/README.md`,
   `manabot/model/agent.md`, `docs/academy.md`, `academy/README.md`,
   `AGENTS.md`. 2026-11-13 13:00 UTC: the `backends/README.md` examples were
   run as written on CPU (save, load, prepare, evaluate, `on_device` forward,
   `resolve`, `read_metadata`, both `TypeError`s, `NotAContainer`); one fix,
   `load_policy` returns `(policy, space)`. No unreleased release-notes section exists; the release note
   must replace v0.2.3's `admit_observation` advice with `policy.prepare`.

Receipts at the `105e7e2aa` engine are under "Receipts" below; the earlier
3,248-tensor captures belong to the previous engine. Gate's broader suites,
Python 3.12 and installed-wheel checks have not been run.

Captures in `/tmp/mtg333/` are temporary. If gone, or after any change to
`manabot/` or the engine: rebuild the extension per `AGENTS.md`, extract the
base `manabot/` (`git archive 105e7e2aa manabot`), and capture base,
base-repeat and head together.

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

## Remaining slices

6b. Complete: current native rebuilt; controlled-input CPU oracle exact.
7. Complete measurements: three-process A40 comparison and profile, plus the
   final same-card 5090 check. CUDA <=1e-8 holds; stricter repeat-floor
   condition remains open. Full tables: `scratch/speed/mtg333-results.txt`.
8. Complete optimization pass: traversal of scalar-only metadata tuples skipped; measured
   +5.6% damped throughput on 5090, CPU oracle and updates exact. PPO did not
   improve. Final original-baseline comparison still shows a 5.2% PPO loss.

**Performance acceptance remains open:** the plan's no-slower-median condition
is not met for PPO. Keep that contrary result for review; do not describe the
branch as passing that criterion. The requested speed/profile/optimization
work is measured, not pending another launch. Further PPO work would need a
revised preparation strategy; the observed limiter is large-minibatch host
preparation, not a correctness difference or a reason to restore raw forwards.

### Slice 6b facts that still bind

- `105e7e2aa` changes seeded games (102,209 steps over the budget's 512
  games, was 101,650) and adds pooled-mana `FACT_EFFECT_PROGRAM` rows on the
  player's object row, id `token("PooledMana")`. `prepare` needed no change;
  the oracle carries a synthetic pooled-mana case built as
  `tests/semantic/test_pooled_mana_classes_input.py::_observation`.
- `docs/development.md` requires
  `uv run --extra dev pytest tests/model/test_record_coverage.py -q` after a
  rebuild for record-vocabulary changes.
- A difference in the pooled-mana case is a `prepare` defect to fix on the
  host, never a torch-side special case.

### Card

Equivalence ran on RTX A6000 (`mtg333-a6000`) after L4 refusals and an A40
HTTP 400. Speed and the optimize pass take the first card in the steer's order
that rents, on a new Task template (for example `mtg333-cuda`) with a ceiling
covering that card, since `mtg333-a6000` may still be held by 4b091160edbf.
Vast (RTX 5090 at ≤$1.00/hour on-demand) is an equal option; on an HTTP 400,
read the provider body before calling it stock. The earlier 400s'
bodies were not recorded.
The optimize pass's before/after uses that same card type. One Task rental at
a time.

### Mechanics

Launch files: `scratch/equivalence/mtg333_cuda_equivalence.py`,
`scratch/speed/mtg333_speed.py` with `--rent <template>`. Each launch commit
tracks its own patches (equivalence applies `git diff BASE HEAD -- manabot`
reversed; speed applies `git diff HEAD BASE -- manabot` forward) with BASE =
`105e7e2aa`: `296c90f33` (equivalence) and `c150d5f25` (combined
equivalence/speed/profile). The compress pass after `c150d5f25` changed
`manabot/` (no numeric change; CPU oracle exact): regenerate the patches at
the next launch commit. `run.py` and `bench.py` run under both arms.

Control: `ETUDE_HOME=/tmp/mtg333/launcher`, overlay
`uv run --with 'etude-sdk[s3,eval]==0.2.15' etude …`, nested
`doppler run -p etude -c prd -- doppler run -p manabot -c prd
--preserve-env=RUNPOD_API_KEY --`. `availability` can say rentable and the
launch still be refused at creation: retry, or `etude queue` plus
`caffeinate -is etude wait COHORT`. After reading results:
`etude terminate MACHINE`, then `runpodctl pod list` → `[]`.
MTG-325 spent $0.04 (equivalence) and $2.04 over four speed passes.

### Acceptance for slice 7

- Equivalence (cohort `mtg333-equivalence`, cap USD 3): logits, values, losses
  and trained weights bitwise base vs head; gradients ≤ 1e-8 abs and no worse
  than the head-vs-head repeat. A difference is reported with tensor names
  and max absolute differences, not waived.
- Speed (cohort `mtg333-speed`, cap USD 10; ABAB, 3 processes per arm): no
  head median slower than the run-to-run spread, for learner steps/s (damped
  and PPO), `PolicyInference` obs/s, save, load and collection. The bench
  reports host preparation as its own section and implicit device reads
  (`torch.cuda.set_sync_debug_mode("warn")`) per arm as diagnostics.
  L4 reference at #501 (Etude 8a376219e430): damped 9.94 steps/s
  [9.70, 10.01], PPO 0.2534, inference 2,296 obs/s, save 0.023 s, load
  0.082 s, collection 61.9 s.
- A slower median is not a stop: it is the optimize pass's first target, and
  the PR body shows both tables.

### Optimize pass (whatever the speed check shows)

Behaviour-preserving within the stated tolerances, judged on the rented card.
Candidates, unmeasured: vectorize `prepare._effects`' per-record Python loop;
int32 index arrays; `Packed.gather` for fact programs instead of
`pack_sequence`; host compaction on CUDA inference (changes summation order:
tolerance, not bitwise); skip moving lookup buffers the forward no longer
reads; avoid full-capacity temporaries before trimming. Start from the bench's
host-preparation section and the device-read count. Each kept change reruns
the CPU oracle, the update comparison and one speed pass; a change that
measures no gain is dropped. Total rental spend for the Task stays under
USD 25 without asking.

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
- World/shape suite from CLAUDE.md passes.
- `uv run --extra mlx python -m manabot.backends.parity CKPT --capture /tmp/mtg325-gate-capture`
  → `parity: pass (torch, mlx)`.
- Round trip: `uv run manabot train --out /tmp/mtg333/run` writes
  `.safetensors` policy files that `uv run manabot sim` plays; a `.pt` written
  at base loads with identical logits; a belief policy and a `bc`
  `value_semantic` survive save → load.
- `rg "AdmittedObservation|CardReferences|PreparedObservation" manabot academy scripts`
  → nothing.
- Slice 7's CUDA results and the optimize before/after are in the PR body.

## Risks

- Main keeps moving under an engine-bound baseline: each engine merge voids
  the captures. Merge once before the CUDA launch, launch from that commit,
  and merge again only at gate; a later engine change repeats slice 6b's CPU
  capture, not the rented runs, unless it touches `manabot/` policy code.
- Rental stock can vanish at creation (L4, A40 and A6000 each did): queue
  and wait under the cohort cap.
- Host preparation on a 6–9 vCPU pod may bound collection: it is measured as
  its own section and is the optimize pass's first target.
- CUDA gradient order through `index_put`/`embedding` backward may exceed
  1e-8 on non-belief paths; the head-vs-head repeat separates that from a
  real difference.
- The optimize pass reopens the forward: every kept change is held to the
  oracle and update comparison, and seeded sim digests stay unchanged.

## Exclusions

MLX support for the newly carried configurations; PPO or a categorical critic
on MLX; optimizer/update-snapshot/incident persistence; belief/ torch models,
verify/, the executor choosing a backend (Task `c2ef4a`); engine, env, serving
protocol, worlds, policy-input versions; rerunning frozen experiments.

## Done when

A developer prepares NumPy observations once and runs them through either
backend; training writes `.safetensors` policies that sim, eval and Academy
load beside old `.pt` files; the CPU oracle and CUDA checks hold; the speed
table and the optimize pass's before/after are in the PR body.

## Receipts, 2026-11-13

Jack Heart: “retry after implement agent fix”. `105e7e2aa` merged as
`9db3f1373` (one conflict, `manabot/env/observation.py`; `ObservationBuffers`
kept). Native release rebuild 3m54s; cp313 extension replaced atomically.
Slice 6b focused checks: 142 passed, then two generic-helper boundary
failures from a runtime `ParameterArrays` import, fixed by moving it under
`TYPE_CHECKING`. Fresh CPU captures against `/tmp/mtg333-mana-base`: base vs
base-repeat and base vs head each **3,410 tensors, differing 0**;
`updates-mana-{base,head}.json` `cmp`-identical (five PPO/damped cases).

Rentals: L4 quote refused (“provider reports no available machines”); A40
admission `eb5ed2c1dabd` failed “ProviderUnavailable: POST pods: status 400”
before any command, $0.00.

Compress pass (after `c150d5f25`, dead names and duplicate tables only):
`compare.py mana-base.pt compress2-head.pt` → **3,410 tensors, differing 0**;
`cmp updates-mana-base.json updates-compress2.json` identical.

Second compress pass (`array_fields` public, `admit_rows` needs branch
flattened, plan curated): `compare.py mana-base.pt compress3-head.pt` →
**3,410 tensors, differing 0**; `updates-compress3.json` `cmp`-identical.
`tests/eval/test_batched_eval.py` → 22 passed, 1 skipped after repairing
`test_queue_boundary_excludes_cold_load_and_overlapping_requests`, which
patched the removed `serving.load_checkpoint` (now `TORCH_BACKEND.load_policy`).
The protocol, bucketed-shape, inference, update-protocol and boundary files →
55 passed.

**Open for gate:** with `test_batched_eval.py` earlier in the same pytest
process, `tests/model/test_inference.py::test_prepared_collection_and_restored_continuation_match_reference`
(line 253, the restored collection) hit the 300 s limit twice; alone it takes
16 s. Seen before and after the test repair; cause not diagnosed.

### CUDA equivalence: `047fa57e0e5e` (A6000)

Succeeded; launcher cost **$0.04**. All 3,410 tensors captured in each arm.
Logits, values, losses, inference and CPU-trained weights are bitwise exact.
95 gradients differ, max `3.725290298461914e-9` (masked-mean bucketed
`semantic_cards.encoder.bias_hh_l0` and `name_candidates.embedding.weight`).
Head-repeat: seven belief gradients differ, max `5.529727786779404e-10`.
Text receipts: `/tmp/mtg333/cuda-retry/{base_vs_head,head_vs_repeat}.txt`;
full tensors retained by Etude as `base.pt`, `head.pt`, `head_repeat.pt`.

The Task numeric CUDA bar (≤1e-8) holds; the stronger plan condition
“no worse than the head-vs-head repeat” does **not**. Keep this contrary
finding for gate review; speed measurement proceeds under the Task numeric
tolerance without claiming the stricter condition passed. No gradient
threshold or receipt was changed.

### Earlier speed attempts (before 18:20 UTC): no result

Every speed admission failed before a job command ran; nothing was measured:

| Request | Outcome |
|---|---|
| `cb8fa16ee59e`, `b3a50357784a` | refused at the one-machine limit (the finished equivalence session is not reusable); terminated `mtg333-a6000-3a2a673ba16f`, pod list `[]` |
| `e91a37b6f51b` / run `3b240dc55ea4` | RunPod HTTP 400 at create |
| `ff725204b789` / run `a1f656c6c77a` (combined) | HTTP 400 at create; cleanup requested for `mtg333-a6000-d90f6b127513` |
| queued `4b091160edbf` (combined, source `c150d5f25`, cap $10) | never reached setup; cancelled, still `renting` |

The combined launcher runs equivalence, the frozen speed benchmark (three
alternating processes per arm) and one unreported cProfile pass on a single
rental (2.5-hour job allowance), so no session reuse is needed. Its generated patch files are tracked
only in `c150d5f25`.

`4b091160edbf`: cancellation requested `2026-11-13T16:35:59.645540Z`
(`etude cancel` preserves it). `etude status` still reports `renting, ~$0.32,
machine mtg333-a6000-b2d4123a77e0`, `last loop: never`. Fresh
credential-injected waiters did not settle it (stopped with SIGINT; none
remain). `etude terminate mtg333-a6000-b2d4123a77e0` → “still has a run on
it”. RunPod SDK inventory for `etude-b2d4123a77e0` and the provider pod list
show no pod for any Task session; no other workload was changed.

Why retrying cannot help (SDK 0.2.15): `Scheduler._observe`'s missing-receipt
branch calls `service.find` and finishes an absent machine only after the
rental deadline; it ignores `cancel_requested_at` there. `terminate` refuses
unfinished allocations before its missing-receipt cleanup. SDK `rentals()`
gives the deadline **2026-11-13T19:15:51.924571Z**. An earlier settlement
needs an Etude cancellation fix, outside this Task. Never edit the launcher
database or clock, and never launch over the stale request.

18:05 UTC check: `etude status 4b091160edbf` → still `renting, ~$1.08`
(launcher accrual; no pod exists for it), `last loop: never`. The only
RunPod pod is `etude-e0f0acfca5a5`, i.e. MTG-324's `mtg324-l4-ab-e0f0acfca5a5`
in `~/.etude` ($0.69/hour, another Session waiting on it): not this Task's,
left alone.

### Upstream #513 (`a289fe066`, merged as `6e20654c34`)

The shared recorder and its callers use Prepared; initial policy capture
files carry container bytes. MLX uses Prepared.is_agent for bucket-trimmed
ownership. Both half precisions preserve float32 masters and public outputs.
The prior conflict inventory is at `9fe9eb1e7:scratch/mtg333-plan.md`.

### Next actions

1. Review the residual PPO tradeoff (5.2% slower in the final same-card check)
   and the stronger CUDA repeat-floor criterion. Neither is waived here.
2. Gate runs the remaining broader suites, installed-wheel/Python 3.12 checks
   and the previously failing full-file test order. The narrow backlog/order
   probe passed; it does not close the earlier 300-second failure.
3. Publication and landing remain the caller's steps. No Task rental or waiter
   remains active; failed allocation windows were closed after reconciliation.

### Current pass, 2026-11-13 18:20 UTC

Fresh launch source `27daef1ae5`, request `8a4dd0751651`, machine
`mtg333-speed-a40-7ccc26a44f00`: A40, $0.59/hour, cohort cap $10. Provider
inventory before launch had no Task pod. A blob-filtered history fetch
retained origin verification after the ordinary full fetch was interrupted
before admission. The original cancelled allocation still has no provider pod.

#513 integration: shared capture hooks accept host ObservationBuffers, forward
Prepared, and retain raw ActionRows.observation for replay. New containers are
discovered at their safetensors suffix. MLX uses existing Prepared.is_agent
for trimmed ownership instead of a duplicate signed array. Experiment consumers
use the surviving prepare/on_device/assemble APIs.

`pytest tests/model/test_backend_capture.py tests/model/test_mlx_backend.py -q`
→ 15 passed (62.09s); half-head public-logit check passed separately. Fresh
172-decision capture, 86 minibatches: `parity: pass (torch, mlx)`, MLX clipped
step max parameter difference 5.96e-8. Five update JSON cases cmp-identical.

Contrary oracle result: fresh baseline versus merged head differs in 136
belief tensors (max gradient 8.502e-4). Isolated probes show identical model
weights but different generated inputs: 932 baseline belief rows versus 900
at head. `2c47f59aea` excludes tokens from belief_schema_from_engine; baseline
105e7e2aa predates it. This is upstream schema behavior, not #513's no-op
float32 score cast. Keep the schema correction. The oracle now retains one
real head-generated corpus via EQUIV_BELIEF_INPUT and loads its exact bytes
in both arms. The current rental uses the earlier harness, so its generated
belief comparison is not an identical-input policy-equivalence result; speed
and profile remain usable. Identical-input CPU captures are in progress.

The first fresh A40 attempt `8a4dd0751651` failed before timing, cost $0.02:
current native definitions cause baseline's pre-TLE schema builder to reject
duplicate card/token names. The retained-input harness avoids that obsolete
generator. The rental was terminated. Local probes additionally reveal the
loaded native extension predates TLE (it allows baseline schema generation);
rebuild current native source before final CPU captures.

Retry `2f98203dd0a7`, source `cd7afaec86`, machine
`mtg333-speed-a40-029a57d11bfc`, cohort `mtg333-speed-shared`, $10 cap:
identical-input CUDA oracle captures 3,410 tensors. All outputs, losses and
CPU-trained weights are exact; 95 gradients differ ≤3.259629011154175e-9.
Head-repeat has seven differing belief gradients ≤3.026798367500305e-9.
The Task's ≤1e-8 bar holds; the stricter repeat-floor condition remains open.
Speed samples are running. Previous cohort name could not be reused; refused
request `d6e61bbf0407` created no pod or job. Current native release rebuild
completed in 3m18s and the cp313 extension was installed atomically.

Current-engine CPU oracle with the retained head corpus:
`compare.py current-shared-base.pt current-shared-head.pt` → **3,410 tensors,
differing 0**. The pre-TLE local captures remain in /tmp as earlier evidence;
new comparisons use current-shared files and current-belief-input.pt.

Current native validation: five update cases (`updates-current-{base,head}.json`)
are cmp-identical. Package boundaries, real capture replay and experiment
discovery: **20 passed, 66.98s**. Fresh current-engine capture parity passes
Torch and MLX; MLX clipped-step parameter max difference 1.2e-7. Generic MLX
learner annotations now stay under TYPE_CHECKING so importing the helper does
not eagerly load observation/checkpoint contracts.

A narrow probe of the prior test-order hang passed: serving backlog followed
by restored collection → 2 passed, 58.64s. This rules out that pairing alone,
not the previously failing full-file order; leave the broader gate check open.

First-pair speed (preliminary, one process each; final medians pending):
55391 learner rows, 217 damped steps and 16 PPO steps in both arms. Head/base:
damped 8.174/7.551 steps/s; PPO 0.1993/0.2157; inference 4136/1318 obs/s;
host preparation 0.03310/0.09876s; collection 56.56/79.60s; save
0.02930/0.03710s; load 0.1692/0.1650s. Forward sync diagnostics fall from
23 actual warnings to zero (both include one prototype-mode warning).
PPO forward/loss is 29.21s versus 17.86s; await cProfile before optimizing.

19:16 UTC: `etude wait 4b091160edbf` settled the cancelled request as
**lost (create_unconfirmed), final launcher cost $0.00**. Its earlier displayed
accrual was provisional, not a charge. No provider pod existed for it.

Optimization hypothesis (exploratory, before candidate measurement): the damped
cProfile has 493,537 recursive calls each to array_fields and _replace_arrays
for 218 transfers (1.669s and 1.755s cumulative). Both recursively visit every
integer in host-only Packed.sizes/lengths and bucket needs, constructing keys
and copied tuples that no device reads. Stop traversal of scalar-only tuples,
preserving mixed/nested tuples and every array leaf. Expect 5–10% lower PPO
wall time by eliminating metadata traversal; keep only if CUDA measurement
shows a gain and the CPU oracle/updates remain exact. Preparation's packing
and unused definition-reference collection are smaller damped costs (0.185s
and 0.077s), so they are not the first candidate.

Full A40 speed comparison `2f98203dd0a7` succeeded; final launcher cost ~$0.31.
Three process medians (head vs base, ranges in retained compare.txt): damped
7.615 vs 7.356 steps/s; PPO 0.1992 vs 0.2157 (**7.7% regression**, outside
spread); inference 4136 vs 1318 obs/s; preparation 0.03386 vs 0.09876s; save
0.02816 vs 0.03710s; load 0.1757 vs 0.1662s (within observed spread);
collection 56.56 vs 79.60s. All arms have 55391 learner rows, 217 damped and
16 PPO steps. Complete table is /tmp/mtg333/speed-live/compare.txt and Etude's
compare.txt. The candidate is scalar metadata traversal, not GRU packing or
float arithmetic. The follow-up uses one process per arm, each with the same
three timed inner repeats, on a fresh A40; repeat profiling is omitted.

PPO cProfile confirms the candidate: 1,519,556 visits each to array_fields
and _replace_arrays, 4.804s and 4.117s cumulative, versus 0.113s in GRU
packing and 0.928s in unused definition references. Scalar-only tuples now
remain untouched by traversal; mixed tuples and arbitrary leaf types retain
their behavior. Candidate sanity: packed transfer layout/value test passes;
final CPU oracle **3,410 tensors exact**, all five update JSON cases exact.
Optimization baseline is `5940609a28` (same numeric code as the speed head);
the launch patch restores it from the candidate.

Optimization run `267f945d54a0`, source `e6cd7f59cc`, on Vast instance
60539549 (`mtg333-opt-vast5090-295daf64df5d`): RTX 5090, 12 vCPUs, 64 GB,
$0.7881481481481482/hour; job cap $2, cohort cap $10. A40 was unavailable at
create (after its quote said available), as was A6000; the 400 bodies were read,
all unconfirmed requests reconciled absent, and no second Task rental was left
running. Before/after both use this one 5090; the first A40 table stays separate.

Optimization `267f945d54a0` succeeded, final launcher cost ~$0.18; its window
was terminated and provider inventory confirms the Vast instance and every
Task RunPod request absent. Same-5090 before/after: damped 18.2057→19.2337
steps/s (+5.6%, beyond the three inner-repeat ranges), inference 9667→10170
obs/s (+5.2%), collection 31.46→21.86s (-30.5%). PPO 0.33794→0.3379 is
unchanged; the prediction of a 5–10% PPO wall reduction was not met. Keep the
measured damped/collection improvement, without claiming PPO improved.
The initial A40 PPO regression therefore needs a direct optimized-head versus
original-baseline check; regenerate the speed patch to 105e7e2aa for that pass.

Final direct check `ce884945fab8` completed from `46e3db32d7`: original
105e7e2aa versus optimized Torch, one process/arm with three inner repeats,
on one Vast 5090 rental (`mtg333-opt-vast5090-1ea3bc7d1921`, instance
60541658). $2 job/$10 cohort caps; $0.7481481481481482/hour. A40 admission
4c4ca07ee2a0 was refused before any machine acquisition.

#513 loader consistency: explicit MLXBackend precision now reaches file-based
load_policy as well as in-memory load. The two existing half-policy loading
cases pass (18.03s), including assertions through the backend file-loader API.
Torch numerics and the frozen final comparison are unaffected.

### Final receipt, 2026-11-13 20:17 UTC

`ce884945fab8` succeeded; final launcher cost ~$0.15. Original baseline versus
optimized head, on one 5090: damped 17.3248→18.1048 (+4.5%), PPO
0.360453→0.341814 (-5.2%), inference 9742.44→10725.6 obs/s, save
0.02085→0.01762s. This last pass has one process/arm with three timed inner
repeats; the initial A40 comparison has three processes/arm. Full ranges and
other sections are in `scratch/speed/mtg333-results.txt`, with frozen source
commits and Etude run IDs. No cross-card ratios are used.

All final CUDA outputs/losses and CPU-trained weights are exact; 55 gradients
differ ≤2.561137080192566e-9, versus six repeat gradients ≤2.0954757928848267e-9.
Task numeric tolerance passes; stronger repeat-floor condition stays open.
Final CPU oracle: 3,410 tensors exact; five update cases exact. Fresh parity
passes Torch and MLX. Focused checks: 20 current-engine boundary/capture/discovery
tests, transfer-layout case, mixed/nested/scalar-leaf checks, and two half-loader
cases pass. Ruff checks and formatting pass on the changed production pieces.

Generated launch patches are removed from the final tree, retained in frozen
launch commits `cd7afaec86`, `e6cd7f59cc` and `46e3db32d7`. Regenerate and commit
patches for a new launch. All Task rentals were terminated; provider inventory
was checked independently. The old cancelled request settled at $0.00.
