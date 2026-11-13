# MTG-333 plan: torch forward over `Prepared`, container saves, backend rule

Policy baseline: `ef28f3a58` (PR #501 merged). Comparison baseline for every
receipt below: `105e7e2aa` (MTG-297 mana spending), merged as `9db3f1373`.
`f4e1822fa` (#510, training precision: `manabot/training/{execution,regime}.py`,
no policy or engine file) is merged at `070da04ed`; `origin/main` had nothing
further at 2026-11-13 realign. That merge changes no `manabot/` policy code, so
no capture is repeated for it. `a289fe066` (#513, MLX half precision and shared
Torch captures, 2026-11-13 18:03 UTC) is on main and **not merged here**; see
"Upstream #513".
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
uv run python scratch/equivalence/run.py /tmp/mtg333/head.pt
uv run python scratch/equivalence/compare.py /tmp/mtg333/integrated-base.pt /tmp/mtg333/head.pt
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

6b. **Complete** — `105e7e2aa` merged, rebuilt; fresh CPU oracle and updates exact.
7. **CUDA equivalence captured** at `296c90f33` on A6000: numeric bar holds,
   stricter repeat-floor condition unresolved. **Speed has no result**: see
   "Speed and profile" and "Next actions".
8. **Optimize pass pending** speed/profile output; no optimization claim yet.

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

### Speed and profile: no result; launcher request stuck

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

### Upstream #513 (`a289fe066`, not merged)

Trial `git merge-tree HEAD origin/main` conflicts in `manabot/policy/{prepare,protocol}.py`,
`manabot/policy/README.md`, `manabot/backends/{parity.py,mlx/checkpoint.py,mlx/learner.py,mlx/policy.py}`,
`tests/model/test_mlx_backend.py` and modify/delete of `scratch/questions.md`.
Breakage beyond conflicts: new `manabot/backends/torch/capture.py` (and
`tests/model/test_backend_capture.py`) imports the deleted
`manabot.model.admission.AdmittedObservation` and
`manabot.model.inference.PreparedObservation` and hooks
`ActorCritic.forward(AdmittedObservation)`; it must move to `Prepared`. The new
`test_half_precision_head_keeps_finite_public_logits` calls
`admit_observation`; port it to `policy.prepare`. `Prepared` gains
`ownership` [N] (bucket-trimmed coordinates), used by MLX; torch's `.float()` on
`policy_head` scores is a no-op at float32, so the CPU oracle against the
`105e7e2aa` base should stay exact. This needs a recapture to confirm.

### Next actions

1. Launch speed/profile now, without waiting for 19:15:52 UTC: commit a launch
   source with patches regenerated against `105e7e2aa` (the `manabot/` code
   changed after `c150d5f25`), on a new Task template per the steer, cohort
   `mtg333-speed`, `--max-usd 10`. Before launching, confirm the provider pod
   list has no Task pod. Later, `etude terminate` the
   `4b091160edbf` machine once Etude settles it, then confirm `runpodctl pod list`
   shows no Task pod. The speed question predates #513, so the launch does
   not wait for that merge.
2. While it runs, merge #513 (above), port `torch/capture.py` and its tests to
   `Prepared`, then rerun the CPU oracle and update comparison against the
   `105e7e2aa` base, and run the parity CLI.
3. Slice 8 runs from the speed/profile output on the merged code, with the
   same card. Then gate: review the repeat-floor finding and run the broader
   suites, Python 3.12 and the wheel smoke.
