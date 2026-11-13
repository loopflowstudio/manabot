# MTG-333 plan: torch forward over `Prepared`, container saves, backend rule

Policy baseline: `ef28f3a58` (PR #501 merged). Comparison baseline for every
receipt below: `105e7e2aa` (MTG-297 mana spending), merged as `9db3f1373`.
`f4e1822fa` (#510, training precision: `manabot/training/{execution,regime}.py`,
no policy or engine file) is merged at `070da04ed`; `origin/main` had nothing
further at 2026-11-13 realign. That merge changes no `manabot/` policy code, so
no capture is repeated for it.
One PR. Earlier full plan with every receipt: `1f634d89b:scratch/mtg333-plan.md`;
MTG-325's plan: `65fa8c7aa:scratch/plan.md`.

## Problem

Torch's matched backend wrapped the old forward: `ActorCritic` did its integer
bookkeeping on device tensors read from a tensor dictionary, while MLX read
`manabot.policy.prepare.Prepared`. Two input representations, two admission
orchestrations, and `save_checkpoint` wrote a format only torch reads.

## What people said (Task thread; no new comments at 2026-11-13 12:19 UTC)

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
   stricter repeat-floor condition unresolved. **Speed pending** combined
   queued run `4b091160edbf` at `c150d5f25`, blocked so far by rental API refusals.
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

The brief names L4. L4 was not rentable at any launch attempt (probes
2026-11-13 12:19 and 16:00–16:10 UTC; quotes $0.69/hour refused for stock);
A40 creation failed HTTP 400; the equivalence ran on **RTX A6000**
(`mtg333-a6000`, $0.69/hour, ceiling $0.75, same image/storage). Base and
head run on one machine, so the card does not change the question; it is a
dated line in the PR body and MTG-325's L4 numbers stay a reference. Speed
and the optimize pass use the same card type.

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
threshold or receipt was changed. Speed launch uses the same A6000 template
and frozen source `296c90f33`.

### Speed admission

Two admissions (`cb8fa16ee59e`, then `b3a50357784a` with a shorter job
allowance) refused at the one-machine limit: the completed equivalence
session was not reusable. Neither job started. Terminated
`mtg333-a6000-3a2a673ba16f`; provider pod list returned `[]`. Fresh speed
rental uses the original 2.5-hour allowance and $10 cohort cap, same A6000
card type, both comparison arms on that single new machine.

The fresh speed create (`e91a37b6f51b` / requested run `3b240dc55ea4`)
failed with RunPod HTTP 400 before any command. Combined the speed launcher
with the existing equivalence capture and one separate, unreported cProfile
pass after speed summaries. Benchmark code, workload and repeats stay frozen;
all checks can now finish on one rental without needing session reuse.

Combined direct admission `ff725204b789` / requested run `a1f656c6c77a`
also failed HTTP 400 before job start. Cleanup requested for
`mtg333-a6000-d90f6b127513`. Queued combined run `4b091160edbf`, source
`c150d5f25`, same $10 cohort cap; one bounded queue attempt before stopping
on external rental unavailability. No speed or optimization result yet.

### Stop state, 2026-11-13

**Implementation remains incomplete:** speed and the required optimization pass
have no result. Do not advance on an assumed performance pass. The Task numeric
CUDA tolerance passed, but the stricter repeat-floor condition still needs gate
review. Gate's broader suites and consumer/wheel checks remain unrun.

Queued run `4b091160edbf` never reached setup during the bounded wait. Cancellation
is recorded (`cancel_requested_at` present); CLI still reports `renting` on
`mtg333-a6000-b2d4123a77e0`. `etude terminate` refuses while the launcher considers
that run active. Interrupted both local waiters after cancellation; process
inventory confirms zero waiters. A fresh coordinator did not reconcile it, so
no further coordinator was left running. Provider inventory has no pod matching
any Task session ID; no other workload was changed. Launcher reconciliation is
pending; do not launch over that stale request or edit its database directly.

Resume by inspecting `etude status 4b091160edbf` and its retained logs under
`ETUDE_HOME=/tmp/mtg333/launcher`, then resolving cancellation/cleanup through
Etude. Frozen combined measurement source is `c150d5f25`; benchmark code and
its three alternating processes per arm were not changed. The generated patch
files were removed from the final diff; their bytes remain in that source
commit. Retry the combined job only when rental admission works.

Realign, 2026-11-13 (after `070da04ed`): `etude status 4b091160edbf` still
reports `renting, ~$0.32, machine mtg333-a6000-b2d4123a77e0`, source
`c150d5f25`, `last loop: never`. Reconciling that request through Etude is the
next action before any speed launch; slices 7 (speed) and 8 (optimize) and the
gate remain.
