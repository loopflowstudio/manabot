# MTG-333 plan: torch forward over `Prepared`, container saves, backend rule

Policy baseline: `ef28f3a58` (PR #501). Comparison baseline for every receipt:
`105e7e2aa` (MTG-297 mana spending). One PR. Full history of this plan, with every attempt and
intermediate receipt: `696e7e667c:scratch/mtg333-plan.md`; slice 1–6 receipts:
`1f634d89b:scratch/mtg333-plan.md`; MTG-325's plan: `65fa8c7aa:scratch/plan.md`.

**Main merged through `de39d601fb` (#514, `493127545a`) as `403996c24b`.**
#514 routes every program-table read through `SemanticCardEncoder.lookup`
(`F.embedding` on CUDA with gradients, indexing elsewhere); the merge carries
it into the moved `backends/torch/_semantic_cards.py`, `_recent_events.py`,
`_facts.py`, `_shared_effects.py` and `policy.py`. `experiments/ppo_cuda.md`
reports it CPU-bitwise and a CUDA summation reorder; its L4 PPO update fell
69.97→32.23 s, mostly backward. The merge also changes the engine. After it,
the CPU oracle and update comparison have not run, and every PPO ratio below
predates it.

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

Slices 1–8 are implemented; nothing is pending a launch.

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
6. `ObservationBuffers`, `ParameterArrays`; docs describe the full PR
   (`manabot/backends/README.md` examples were run as written on CPU,
   2026-11-13). No unreleased release-notes section exists; the release note
   must replace v0.2.3's `admit_observation` advice with `policy.prepare`.
7. #513 port: the shared capture recorder takes host `ObservationBuffers`,
   forwards `Prepared` and keeps raw `ActionRows.observation` for replay;
   initial policy captures are `.safetensors`; MLX reads `Prepared.is_agent`
   for trimmed ownership; `MLXBackend(precision=)` reaches file loads.
8. CUDA equivalence, speed, profile and the optimize pass are measured
   (Receipts) on the pre-#514 tree. The optimize pass kept one change: `arrays.array_fields` and
   `_replace_arrays` skip scalar-only tuples.

**Open for review, not waived:**

- PPO is slower: 7.7% on A40 (three processes per arm, outside spread), 5.2%
  in the final same-5090 check. The plan's no-slower-median condition is not
  met. In the final check PPO "Forward and loss" is 9.612 s vs 5.725 s while
  backward is 34.672 s vs 36.041 s; the limiter is large-minibatch host
  preparation in the forward, not a correctness difference or a reason to
  restore raw forwards. #514 halves PPO backward on both arms, so head's
  extra forward time is now a larger share; the regression must be
  remeasured against a base that contains #514 (Launching again).
- CUDA gradients meet the Task bar (≤1e-8) but exceed the head-vs-head repeat
  floor the plan also asked for.
- With `tests/eval/test_batched_eval.py` earlier in the same pytest process,
  `tests/model/test_inference.py::test_prepared_collection_and_restored_continuation_match_reference`
  hit the 300 s limit twice (16 s alone). Serving backlog followed by restored
  collection alone passed (2 passed, 58.64 s); the full-file order is
  undiagnosed.
- Gate's broader suites, Python 3.12 and installed-wheel checks have not run.

Captures in `/tmp/mtg333/` are temporary. If gone, or after any change to
`manabot/` or the engine: rebuild the extension per `AGENTS.md`, extract the
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
this Task's change with #514's CUDA reorder. Use main's `de39d601fb` as the
CUDA base (the CPU oracle keeps `105e7e2aa`, since #514 is CPU-bitwise); its
`manabot/` lacks `Prepared`, so the speed patch is `git diff HEAD de39d601fb
-- manabot`, and `bench.py` must still run on both arms.

Control: `ETUDE_HOME=/tmp/mtg333/launcher`, overlay
`uv run --with 'etude-sdk[s3,eval]==0.2.15' etude …`, nested
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
RTX 5090 rented both times it was asked. A cancelled, never-created request
(`4b091160edbf`) showed provisional accrual and blocked its template until its
rental deadline, then settled **lost (create_unconfirmed), $0.00**: SDK 0.2.15
`Scheduler._observe` finishes an absent machine only after the deadline and
ignores `cancel_requested_at`; `terminate` refuses unfinished allocations.
That is an Etude fix outside this Task; never edit the launcher database.

Optimize candidates not tried (unmeasured): vectorize `prepare._effects`'
per-record loop; int32 index arrays; `Packed.gather` for fact programs; host
compaction on CUDA inference (summation order: tolerance, not bitwise); skip
unused definition-reference collection (0.928 s of the PPO profile); avoid
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

Launcher cost as recorded once per run; no cross-card ratios. Full speed
tables with ranges and section timings: `scratch/speed/mtg333-results.txt`.

| Run | Card | Source | Result | Cost |
|---|---|---|---|---|
| `047fa57e0e5e` equivalence | A6000 | `296c90f33` | outputs, losses, CPU-trained weights exact; 95 gradients ≤3.725290298461914e-9; repeat: 7 belief gradients ≤5.529727786779404e-10 | $0.04 |
| `8a4dd0751651` | A40 | `27daef1ae5` | failed before timing (baseline schema builder, see Facts) | $0.02 |
| `2f98203dd0a7` equivalence + speed + profile | A40 | `cd7afaec86` | identical-input oracle 3,410 tensors: outputs exact, 95 gradients ≤3.259629011154175e-9, repeat 7 belief gradients ≤3.026798367500305e-9; speed below | ~$0.31 |
| `267f945d54a0` optimization before/after | Vast 5090 | `e6cd7f59cc` | damped 18.2057→19.2337 steps/s (+5.6%), inference 9667→10170 obs/s, collection 31.46→21.86 s, PPO 0.33794→0.3379 (unchanged) | ~$0.18 |
| `ce884945fab8` original baseline vs optimized head | Vast 5090 | `46e3db32d7` | damped 17.3248→18.1048 (+4.5%), PPO 0.360453→0.341814 (−5.2%), inference 9742.44→10725.6 obs/s, save 0.02085→0.01762 s; 55 gradients ≤2.561137080192566e-9, repeat 6 ≤2.0954757928848267e-9 | ~$0.15 |

A40 comparison (`2f98203dd0a7`, three processes per arm, head vs base
medians): damped 7.615 vs 7.356 steps/s; PPO 0.1992 vs 0.2157 (**7.7%
regression**, outside spread); inference 4136 vs 1318 obs/s; host preparation
0.03386 vs 0.09876 s; save 0.02816 vs 0.03710 s; load 0.1757 vs 0.1662 s
(within spread); collection 56.56 vs 79.60 s; forward sync warnings 1 vs 24.
All arms: 55,391 learner rows, 217 damped and 16 PPO steps. The 5090 passes
have one process per arm with three timed inner repeats.

Profile (same run): PPO made 1,519,556 visits each to `array_fields` and
`_replace_arrays` (4.804 s and 4.117 s cumulative) over host-only
`Packed.sizes`/`lengths` and bucket needs, against 0.113 s in GRU packing.
The prediction that skipping them would cut PPO wall time 5–10% was not met;
damped and collection improved instead.

CPU, current engine (native release rebuild 3m18s, cp313 extension replaced
atomically): `compare.py current-shared-base.pt current-shared-head.pt` →
**3,410 tensors, differing 0**; `updates-current-{base,head}.json`
`cmp`-identical (five PPO/damped cases); `parity: pass (torch, mlx)` on a
fresh 172-decision capture, MLX clipped-step parameter max difference 1.2e-7.
Boundaries, capture replay and experiment discovery: 20 passed (66.98 s);
`tests/model/test_backend_capture.py tests/model/test_mlx_backend.py` → 15
passed.

Compress pass, 20:40 UTC (`array_fields` skips scalar-only tuples for every
leaf type; `MLXBackend.__init__` first): `compare.py current-shared-base.pt
compress4-head.pt` → **3,410 tensors, differing 0**; `updates-compress4.json`
`cmp`-identical; transfer-layout and two half-loader tests → 3 passed.

Rentals at 20:38 UTC: RunPod holds two pods, both other Tasks'
(`etude-e0f0acfca5a5` = `mtg324-l4-ab`, `etude-b0805de6f0f4` = `mtg347-l4`),
left alone. Every MTG-333 launcher machine is a never-created request at
$0.00 whose window has ended or ends by 20:51 UTC (`4b091160edbf` settled
lost, $0.00). No Task rental is live.

## Next actions

1. Repeat the CPU oracle (fresh base and head captures, one belief corpus)
   and the update comparison on the merged tree. The extension was rebuilt
   for it (20:41 UTC, release, cp313 replaced atomically);
   `tests/model/test_semantic_cards.py` → 10 passed, 8 CUDA-only skipped.
2. One same-card CUDA speed pass, base `de39d601fb` vs head, within the
   remaining USD 25 budget (spent so far ≈ $0.70): PPO, damped, inference,
   collection, with the PPO section split. If PPO is still slower, the
   untried candidates above target host preparation in the forward.
3. Review the PPO tradeoff and the CUDA repeat-floor finding (State).
4. Gate: broader suites, installed wheel, Python 3.12, the full-file test
   order. Publication and landing are the caller's steps.
