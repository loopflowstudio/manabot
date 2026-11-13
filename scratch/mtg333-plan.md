# MTG-333 plan: torch forward over `Prepared`, container saves, backend rule

Policy baseline: `ef28f3a58` (PR #501 merged). Upstream `6ce618b8a` is merged
in (`0c43cc16e`). `origin/main` moved to `105e7e2aa` (MTG-297 mana spending) at
2026-11-13 12:21 UTC and is **not merged yet**: slice 6b.
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
  -> tensors N ... differing 0   (3,248 before the 105e7e2aa engine)
```

```python
policy, space = load_checkpoint("self-play-raw.safetensors")  # or an old .pt
batch = policy.prepare(obs)              # NumPy rows; the Prepared MLX reads
logits, values = policy.evaluate(batch)  # NumPy out
```

and the PR body's CUDA table: base vs head medians with spread, then the
optimize pass's before/after.

## State (HEAD `1f634d89b`)

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
   `AGENTS.md`. No unreleased release-notes section exists; the release note
   must replace v0.2.3's `admit_observation` advice with `policy.prepare`.

Last receipts (laptop, one native build from the integrated source in both
arms; base `manabot/` extracted to `/tmp/mtg333-integrated-base`):
- CPU oracle after the compress pass: `compress.pt` vs `integrated-base.pt`
  → **3,248 tensors, 0 differing**; `updates-compress.json` byte-identical to
  `updates-base.json` (five PPO/damped cases).
- Torch/MLX parity on the frozen capture passes (`injected-parity.json`):
  post-Adam weights 7.5e-9 / 1.2e-7, MLX logits 5.4e-9, values 3.9e-7.
- Focused suites: inference/container/prepare/protocol/value 63 passed; MLX 8;
  loader/injection 32; Academy catalog/admission/play 6.
- No file under `manabot/` changed after those captures except annotations
  and the compress edits they cover. Gate's broader suites, Python 3.12 and
  installed-wheel checks have not been run.

Those receipts are at the pre-`105e7e2aa` engine; slice 6b replaces them.
Captures in `/tmp/mtg333/` are temporary. If gone, or after any change to
`manabot/` or the engine: rebuild the extension per `AGENTS.md`, extract the
base `manabot/` (`git archive 105e7e2aa manabot` once merged), and capture
base, base-repeat and head together.

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

Deleted predecessors are not restored; the list is in
`1f634d89b:scratch/mtg333-plan.md` ("Delete — do not maintain").

## Remaining slices

6b. **This slice — merge `105e7e2aa` and re-establish the CPU baseline.**
7. **CUDA equivalence and speed**, launched from the merged commit. Blocked
   since 12:12 UTC on L4 stock; the card route below unblocks it.
8. **Optimize pass**, on the card the speed check used.

### Slice 6b: upstream `105e7e2aa` (read 2026-11-13 12:23–12:27 UTC)

What it changes for this Task:
- Engine rules: native unit/class mana spending adds decisions (step digest
  moved, 102,209 steps instead of 101,650 over the budget's 512 games). The
  oracle's seeded games therefore differ; every capture in `/tmp/mtg333/` and
  the count 3,248 belong to the old engine.
- Policy input: a player's pooled mana classes become `FACT_EFFECT_PROGRAM`
  rows on the player's object row, id `token("PooledMana")`
  (`manabot/env/observation.py`, native `observation_encoder::POOLED_MANA`,
  listed in `agent/record_vocabulary.rs`). No model file changed upstream.
  `prepare._facts`/`_fact_programs` route programs by row, not by object
  kind, and resolve the id through `Tables.names`, so no code change is
  expected; this is unproven until the checks below run.
- Under `manabot/`, `ef28f3a58..105e7e2aa` differs only in
  `env/observation.py` and `semantic/text_observation.py`; neither upstream
  file nor its changed tests name anything this branch deleted.
- `git merge-tree HEAD origin/main` → one conflict, `manabot/env/observation.py`:
  both sides moved the `managym.enums` import below the local imports. Keep
  upstream's order plus this branch's `Mapping` import and
  `ObservationBuffers` alias.
- `docs/development.md` now requires
  `uv run --extra dev pytest tests/model/test_record_coverage.py -q` after a
  rebuild for record-vocabulary changes.

Work: merge, rebuild the extension per `AGENTS.md`, extract the base
`manabot/` from `105e7e2aa`, and add one case to `scratch/equivalence/run.py`
built as `tests/semantic/test_pooled_mana_classes_input.py::_observation`
builds it (a pooled class with a three-token program on each seat), since the
seeded games may never pool restricted mana. The case must import only names
present in both arms.

Acceptance:
- `compare.py integrated-base.pt head.pt` → equal tensor counts in both arms
  (the new count replaces 3,248 everywhere in this plan), `differing 0`,
  including the pooled-mana case's logits, values and gradients.
- `updates.py` output `cmp`-identical between arms.
- `uv run --extra dev pytest tests/model/test_record_coverage.py tests/semantic/test_pooled_mana_classes_input.py tests/semantic/test_text_observation.py tests/belief/test_state.py tests/model/test_prepare_rows.py tests/model/test_inference.py tests/test_package_boundaries.py -q`
  → all pass.
- A difference in the pooled-mana case is a `prepare` defect to fix on the
  host (row routing or id lookup), never a torch-side special case.

### Card: L4, else A40 (probed 2026-11-13 12:19–12:20 UTC)

- `etude availability mtg333-l4` → "none available … not rentable now"
  (quote $0.69/hour, ceiling $0.75); `mtg333-vast-l4` the same.
  `runpodctl gpu list` did not list an L4 at all.
- `mtg333-a40` (added to the isolated launcher; RunPod "NVIDIA A40", ≥6 vCPUs,
  48 GB, same image digest and storage, ceiling $0.75) → "$0.59/hour …
  rentable now". Also in low stock: RTX A6000 $0.69, RTX PRO 4500 $0.72.
- `runpodctl pod list` → `[]`; total Task rental spend **$0.00**. The launcher
  still shows phantom `mtg333-l4-4b3f1e50f0a1` "rented until 13:26 UTC, $0.00"
  from refused run `4d6117d0b975`; no pod exists.

Decision: ask L4 availability once at launch; if it is not rentable, run on
`mtg333-a40`. Both checks compare base and head on the same machine, so the
card does not change the question; it is a dated line in the PR body, and
MTG-325's L4 numbers stay a reference, not the baseline. Same cohort caps.
Recorded in `questions.md`. Equivalence, speed and the optimize pass all use
one card.

### Mechanics

Launch files: `scratch/equivalence/mtg333_cuda_equivalence.py`,
`scratch/speed/mtg333_speed.py` with `--rent <template>` (their docstrings
say L4; make them name the template's card). Frozen comparison source
`a222dc9df` holds both patches (equivalence applies `base.patch` reversed,
`git diff BASE HEAD -- manabot`; speed applies `git diff HEAD BASE -- manabot`
forward). The compress commit `8c65474c6` and the `105e7e2aa` merge changed
`manabot/` after that freeze: regenerate both patches at the launch commit
with BASE = `105e7e2aa`, track them only in that commit, and launch from it. `run.py` and `bench.py` run under both arms.

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

- CPU oracle: `compare.py integrated-base.pt head.pt` → slice 6b's tensor
  count in both arms, `differing 0`, on the final commit.
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
- A40 stock is "Low" and can vanish at creation as L4 did: queue and wait
  under the cohort cap; A6000 is the next template at the same ceiling.
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
