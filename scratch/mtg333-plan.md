# MTG-333 plan: torch forward over `Prepared`, container saves, backend rule

Policy baseline: `ef28f3a58` (PR #501 merged). Integrated upstream:
`b531853a5`, merged by `0c43cc16e`; its tree matches main `6ce618b8a`.
One PR. MTG-325's plan is in git (`65fa8c7aa:scratch/plan.md`;
"Pass of 21:30Z" sized this Task).

## Problem

Torch's matched backend still wraps the old forward: `ActorCritic` does its
integer bookkeeping on device tensors read from a tensor dictionary (`nonzero`,
`searchsorted`, `unique`, host round trips for effect routes), while MLX reads
`manabot.policy.prepare.Prepared`. Two input representations, two admission
orchestrations, and `save_checkpoint` writes a format only torch reads.

## What people said (Task thread)

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
uv run python scratch/equivalence/compare.py /tmp/mtg333/base.pt /tmp/mtg333/head.pt
  -> tensors 3248 ... differing 0
```

and, in a Python session, the one policy input every backend reads:

```python
policy, space = load_checkpoint("self-play-raw.safetensors")  # or an old .pt
batch = policy.prepare(obs)              # NumPy rows; the Prepared MLX reads
logits, values = policy.evaluate(batch)  # NumPy out
```

## Probes run at kickoff (2026-11-12, laptop, torch 2.15.0, NumPy 2.4.2)

- The oracle runs at base: `scratch/equivalence/run.py` in MTG-325's checkout
  (tree `145cb111`, identical to this base): 3,248 tensors, 54 s wall, repeat
  0 differ. Captures: `/tmp/mtg333/base.pt`, `base_repeat.pt` (temporary;
  re-capture from `git archive ef28f3a58 manabot` on `PYTHONPATH` if gone or
  after a rebase that changes `manabot/` or the engine).
- `scratch/equivalence/probe.py`, float32, one thread:
  - `x / c` (60, 128, 256, 20, 10): NumPy and torch bitwise equal on 2.14M
    values. Host division is safe.
  - `sign(x)·log1p(|x|)/10`: 2,101 of 2.14M values differ (max 2.4e-7); 66 of
    the integers in ±70,000 differ in `log1p` alone. `program_number` stays in
    the graph; `Prepared` carries the signed integers.
  - `(h·c)/60` against `h·(c/60)`: 7,467 of 16,000 values differ. Known-hand
    counts stay raw in `Prepared`; the backend divides after the product.
  - Embedding gradient, select + `index_put` (padded learning today) against
    a full-shape masked lookup (bucketed today; what host rows give): weight
    gradients bit-identical. Forward: `torch.where` bit-identical;
    multiplying by a float mask gives -0.0 at 13,000 slots (value-equal).
    Use `where` in torch.

## Current implementation and cutover points

Slice 1 and the arithmetic part of slice 2 are implemented. Torch's action,
object/card/belief, link, fact, effect, continuation and history encoders now
read device `Prepared` rows. Admission uses shared `Spec/check`; host trimming,
compaction and live-token routes are consumed. `arrays.on_device` batches
transfers by dtype. The final arithmetic CPU oracle passed at `prepared-forward.pt` (below).

Commit `775e5d823` contains the verified arithmetic, learner-rollout and container
cut (receipts below). The following caller cut is in progress: `ActorCritic`
now accepts only device `Prepared`; `model/admission.py` and
`model/inference.py` are deleted. `PolicyInference` lives in the Torch policy
module, uses its catalog cache and installs a lifetime guard on the batch.
Prepared retains bucket needs as host tuples for collection. Direct callers and
tests are migrated; the fresh carrier-free CPU and update oracles pass (receipt below).

Encoders live in private modules under `backends/torch/`. `RecordEmbeddings`, old record lookup/admission wrappers and their exclusive
tests are deleted. The current slice carries catalog and fact program schedules on Prepared;
its oracle and focused results are below. Static naming joins use nonpersistent
precomputed routes. No catalog lengths are read back from the device in forward.

General container saves and metadata readers, policy fixture mutation and new
export suffixes are implemented. The docs describe the full PR; backend
injection, aliases, L4 and optimize work are still incomplete.

The backend-selection rule is written in `AGENTS.md` (`CLAUDE.md` links to
it) and `backends/README.md`, and its lazy resolver is implemented. The import guard and sim/eval injection
remain. `tests/test_package_boundaries.py` currently guards framework
imports and update injection, not the new composition-root rule.
`ParameterArrays`, header-only metadata reads and general container support are
implemented. Parameters come from `named_parameters` with float32 required;
all other state entries remain buffers. MLX checks support at its own load.
Metadata readers, policy fixture mutations and new export suffixes are migrated;
targeted consumer checks are in progress.

### Upstream reconciliation (2026-11-13 UTC)

`lf task status MTG-333 --json` returned no Wave and no new direction beyond
the quoted backend/alias additions. `git ls-remote origin refs/heads/main`
returned `6ce618b8a`; that commit is already in this checkout. Since
`ef28f3a58`, upstream changed native mana-payment handling, its documentation
and the repository QA skill, with no changes under `manabot/` or the backend
tests. The policy baseline therefore stays `ef28f3a58`, but the next base/head
comparison must use one native build from the integrated source in both arms.
The historical captures and speed numbers below remain evidence of their
original runs, not measurements of the integrated engine. No new runtime or
L4 check was run during realign; the local extension's rebuild provenance was
not established. Rebuild as required by `AGENTS.md` before the next runtime
comparison and recapture base, base-repeat and head together. Existing frozen
parity captures remain usable only while their world binding matches.

The base versions at `ef28f3a58` are the authority for old call sites and
admission messages. Remaining consumers include sim/search/serving, verify,
belief, eval, training, checkpoint discovery and metadata readers,
`experiments/` and `scripts/bench_*`. Find them when making each cut rather
than maintaining a line-number inventory. The full kickoff inventory and
pre-compress notes are preserved in `/tmp/mtg333-compress-before/`.

## Chosen approach

**One input.** `Prepared` is the only policy input in both backends. Integer
bookkeeping, masks and exactly-rounded elementwise float32 (division by a
constant, casts) run on the host in `prepare`; every other float operation
stays in the backend graph in base's order. The torch forward keeps base's op
sequence at each site (`where`/`masked_fill`/`scatter_add`/`index_put`,
`pack_sequence` for fact programs: `torch.sort` is not stable, so its tie
order is not the host's), reading host indexes instead of computing them.

**`Prepared` grows to every torch configuration** (unused groups are `None`):
- `Tables` without a catalog (definition rows are transport ids; no naming
  program rows) and with hashed records (`% buckets` rows, 5 numeric columns).
- v10 turn-state columns concatenated onto player and card rows.
- Belief rows (ids, probabilities, validity, globals); object count in every
  flat index is the batch's actual object axis (trimmed objects + belief rows
  + 1), never `layout.objects`.
- Recent events: kind/owner/zone/status ids, amounts, ages (integers; `log1p`
  in graph), definition rows, focus indexes and masks.
- `CatalogPrograms.rows` (sorted unique catalog rows + 1; `np.unique` =
  `torch.unique`'s order) and packed tokens/schedule for non-bucketed learning.
- Raw `known_counts`; signed program/detail integers. `FactRoutes.programs`
  is one optional `FactPrograms`; `Packed.lengths` owns sequence boundaries.
- Legal coordinates at full action capacity (logits return at capacity).
- Ownership sign/`is_agent` at the batch's object coordinates.
- Live-token `index`/`offsets` when the policy packs live tokens (replaces
  the learner's per-forward `nonzero` read and `live_token_count`).
- `binding` (model + observation configuration JSON); the forward refuses a
  batch prepared for another configuration, as `_require_admitted` does.
- Compaction flags follow base: attention columns only for CPU no-grad
  without live tokens; fact columns only for CPU no-grad non-bucketed.

**Host trimming.** NumPy `trim_observation` (object-reference remap included)
in `policy/buckets.py`; `prepare` runs on trimmed rows. Torch
`trim_observation`/`trim_admitted` are deleted.

**One admission.** `policy/admission.py` gains belief and recent-event checks
(same messages) and one `check(obs, spec, legal)` both backends call.
Those checks now exist. Legal-mask validation now runs through host `admission.legal_mask`; backend
wrappers create device evidence only on rejection, retaining numerical-error
invariant names and invalid-mask payloads. Empty support keeps the original
action-sampling and loss refusals.
This is admission work, separate from the later CUDA optimization pass.

**Device rows.** `Prepared[A]` is generic over its array type (3.12 syntax,
no default): `prepare` returns `Prepared[np.ndarray]`;
`backends/torch/arrays.on_device(batch, device)` returns
`Prepared[torch.Tensor]` with one pinned copy per dtype (generalizing
`model.inference.transfer`). `ActorCritic.forward*` takes the device form and
raises `TypeError` for anything else.

**Policy surface.** `ActorCritic` satisfies `BackendPolicy` itself:
`prepare(obs)` (check, buckets, trim, index), `prepare_admitted(obs, needs=)`,
`evaluate`, `forward_distribution`, `cache_programs`, `clear_cache`.
`PolicyInference(agent)` keeps its name and `prepare`; it owns the version
check and drives the policy's catalog cache (used only without autograd,
validated against parameter versions, never in `state_dict`).
`AdmittedRollout` keeps its Protocol; `evidence` returns the batch's arrays
by field name, as MLX's does.

**Torch layers live in the backend.** The encoders become parameter holders
with `Prepared`-driven forwards under `manabot/backends/torch/` (private
modules); `manabot/model/` keeps `architecture.py`, `catalog.py`, `world.py`.
Method names `_gather_object_embeddings`, `_attend_objects`,
`_gather_informed_actions`, `_value_from_objects` stay
(`scripts/bench_local_update.py` matches them).

**Container saves.** The container carries every configuration:
parameters = `named_parameters` (float32 required, never cast), buffers =
every other state entry present; `Tables` built from what exists.
`require_supported` moves to the MLX backend's load. New metadata keys beside
MTG-321's (`hypers`, `world_binding`, `architecture`, `source`): `bc`,
`global_step`, `belief_schema_identity`, `belief_content_manifest_identity`;
`converted_from` only on conversions. `save_checkpoint` writes it;
`load_checkpoint` restores `value_semantic` and the belief binding from
either format. A header-only `read_metadata(path)` (NumPy) and a torch-side
legacy reader replace the bare `torch.load` call sites. New policy files are
named `.safetensors`; readers that find checkpoints by suffix accept both;
`load_checkpoint` keeps choosing by bytes. Optimizer, update-snapshot and
incident files stay torch `.pt`.

**Backend rule.** `manabot.backends.resolve(name)` (lazy; replaces
`parity.backend_module`); rule text in `backends/README.md` and CLAUDE.md;
`test_package_boundaries.py` refuses `manabot.backends.*` imports outside the
roots with an exact-equality allowlist. sim/ and eval/ modules whose torch
use was tensor conversion plus a forward take a `BackendPolicy`; sampling
that reads torch's RNG stays behind a backend method so seeded games are
unchanged. Follow-up Task filed for what remains on the allowlist.

**Aliases.** `type ObservationBuffers = Mapping[str, np.ndarray]`
(`manabot.env.observation`), `type ParameterArrays` (`manabot.policy.checkpoint`);
`training/updates/backend.py`'s NumPy observation side points at the former.

## Alternatives rejected

- Forward takes NumPy `Prepared` and converts per field inside: dozens of
  small device copies per forward on CUDA.
- Keep the dictionary on device and move only lookups to the host: ships two
  input representations (forbidden) and keeps the full-capacity transfer.
- Pre-divided counts / host `log1p` with a tolerance instead of bitwise: the
  probes show these differ, and a bitwise oracle is what makes a
  seven-encoder rewrite checkable per commit.
- Container bytes under `.pt` names: fewer renames, but a suffix that lies and
  `torch.load` failures in outside tools. Recorded in `questions.md`.

## Forbidden outcomes

- Any forward path reading a tensor dictionary, `AdmittedObservation`,
  `CardReferences`, `PreparedObservation` or `Retained` at the PR's end
  (intermediate commits may carry `Prepared` on the admitted object).
- A torch-only twin of `prepare`, or a second admission orchestration.
- Deprecated aliases for moved names (steer 8332334d, MTG-325).
- A version branch in encoders or model beyond the v10 turn-state append.
- Truncating or casting container tensors; regenerating frozen evidence.
- Changed seeded games, replays, serving protocol or checkpoint meaning.

## Delete — do not maintain

Removed in slice 1: the MLX-specific admission body and host `program_number`.
Removed in compress: the six separately optional fact-program fields, duplicate
sequence-length fields, the fact-route kwargs dictionary and redundant masks.
`FactPrograms | None` owns its indexes together; `Packed` owns lengths. Existing
program-boundary and effect-padding tests follow these surviving types.

Remaining for the Torch cutover (do not polish these predecessors):

`manabot/model/admission.py`, `manabot/model/inference.py` (scope and
transfer move to the backend), `AdmittedObservation`,
`PreparedObservation`, `transfer_admitted`, `admit_on_device`, `SemanticCardEncoder.record_names*`/`record_tokens*`/
`require_record_*`, `RecordEmbeddings`, `records.masked_record_rows`/
`join_record_rows`/`admit_record_rows`/`record_indexes`, the torch
`admit_*` wrappers,
`parity`'s torch-against-itself
reference path, `convert`'s refusal of unsupported configurations.
Survives: every admission message tests assert; `PolicyInference` scope
semantics (`tests/model/test_inference.py`); legacy `.pt` loading; registered
lookup buffers and state-dict names.

## Slices

Each commit that touches the forward ends with the CPU oracle at 0 differ
against a base capture from the same native build. `run.py` needs an adapter
for the old admitted dictionary and new host/device `Prepared` APIs, including
the moved `PolicyInference` import and legal-mask access. The adapter must
leave fixture generation, seeds, losses and output keys unchanged.

1. **Implemented — `prepare` covers every torch configuration, on the host.**
   `Prepared[A]`, the new fields, `Tables` without catalog / with hashed
   records, NumPy trimming, belief and event admission, shared `check`. MLX
   forward takes raw counts and integers. Torch forward untouched.
   `tests/model/test_prepare_rows.py` covers ten configurations ×
   padded/bucketed on three early real observations, plus belief coordinates,
   effect memberships/padding, program boundaries and v10 trimming. It compares
   selected joins, action features, live indexes, ownership and fact widths;
   it does not establish equivalence of every field or a Torch forward.
   Completed checks and MLX parity are recorded below.
2. **Arithmetic migrated; finish the input boundary.** The admitted carrier now
   holds `Prepared`; actions/focus, links, relations/events, facts, shared
   effects, cards/known/decklists and belief consume it. `on_device` exists.
   Remaining: drop the dictionary/carriers, move `PolicyInference` to the
   backend (using the policy cache), and remove catalog length reads from the
   device forward.
   Cover nonempty event, relation, naming and fact-program routes as their
   consumers move, retain malformed-input refusal tests, and compare CPU
   outputs/gradients rather than treating host row tests as forward evidence.
3. Callers: backend rollout, collection, sim, verify, belief, eval, academy,
   scripts, experiments, tests; encoders move under `backends/torch/`;
   deletions above; `_TORCH_NOT_YET_MOVED` shrinks.
4. **Implemented; consumer verification in progress.** General container saves,
   metadata readers, export suffixes and editable container fixtures. Legacy
   `.pt` readers remain and MLX support checks belong to its load.
5. Backend rule: implement the resolver and boundary test for the already
   documented rule; move sim/eval over `BackendPolicy`; file the explicitly
   deferred belief/executor/optimizer-persistence/verify imports as a follow-up
   Task. Verify the rule text against the final allowlist.
6. Aliases sweep; docs (`policy/README.md`, `backends/README.md`,
   `backends/mlx.md`, `docs/training-loop.md`, `manabot/README.md`,
   `docs/training.md`, `manabot/model/agent.md`).
7. L4 equivalence and speed (below).
8. Optimize pass (below); delivery clears added scratch through `lf`.

## Implementation receipt — host preparation (2026-11-13)

Slice 1 is implemented; slices 2–8 remain. Torch's forward, its admission
wrappers and checkpoint writes are still the baseline. The end-state docs
remain the specification for those slices, not a claim that they are shipped.
`ObservationBuffers` is introduced at its owner and used by host preparation;
the wider alias sweep remains in slice 6. MLX uses the shared `Spec`/`check`
and refuses a batch bound to another model/observation configuration.

Checks (all through the locked CPython 3.13 environment, no skips):
- `uv run --no-sync pytest tests/model/test_prepare_rows.py -q -x` → 21 passed;
  added effect/program cases → 3 passed; v10 trimming and MLX binding cases
  → 2 passed. Host coverage is now 25 cases, plus the MLX binding case.
- `uv run --no-sync pytest tests/model/test_policy_container.py tests/model/test_mlx_backend.py tests/model/test_bucketed_shapes.py -q`
  → 20 passed. Shared-policy import and backend isolation checks → 2 passed.
- `uv run --no-sync python -m manabot.backends.parity /tmp/mtg325-gate-capture/initial.pt --capture /tmp/mtg325-gate-capture --json /tmp/mtg333/prepare-parity.json`
  → `parity: pass (torch, mlx)`. Torch/MLX post-clipped-Adam max absolute
  weight differences: 7.5e-9 / 1.2e-7 (bar 1e-6); MLX max logits 6.0e-9,
  values 3.9e-7, relative loss 0.0008%. This is tolerance evidence for the
  changed MLX arithmetic, not a new Torch forward-equivalence receipt.
- Changed Python files pass Ruff check/format; `git diff --check` is clean.

The host trim also slices v10 `card_turn_state` with its card axis and counts
its nonzero rows when choosing buckets. The old tensor trim lacks that slice;
the v10 object-bucket case is a focused regression on the surviving host path.
When migrating Torch, verify padded/bucketed v10 with masked-mean as well as
this plan's existing historical-mean v10 oracle configuration.

## Acceptance at gate

- CPU oracle: `compare.py base.pt head.pt` → `tensors 3248`, `differing 0`.
  A difference is an unresolved failure with tensor names and max absolute
  differences, not an exception granted by documenting it. Keep the original
  3,248-key cohort; v10 masked-mean coverage supplements it separately.
- Updates: `uv run python scratch/equivalence/updates.py` JSON `cmp`-identical
  to a base capture taken the same way.
- `uv run --extra dev --extra mlx --extra artifacts pytest tests/test_package_boundaries.py tests/model/test_prepare_rows.py tests/model/test_torch_backend_protocol.py tests/model/test_torch_container_load.py tests/model/test_policy_container.py tests/model/test_inference.py tests/model/test_admission.py tests/model/test_bucketed_shapes.py tests/model/test_live_tokens.py tests/training/test_update_backend_protocols.py tests/training/test_update_backends.py tests/training/test_minimal_loop.py tests/sim/test_puct_compatibility.py tests/sim/test_local_update.py -q`
  → all pass, no new skips; plus the tests of every other file changed and
  the package-move/initializer/CLI checks required by
  `docs/development.md#validation-by-change-area` (including installed-wheel
  smoke where the CLI changes). Isolated consumer checks cover Python 3.12
  as well as the development interpreter; local 3.13 receipts alone do not.
- World/shape suite from CLAUDE.md (elemental world, evaluation models,
  mirror admission, monitor evaluation, choice separability) passes.
- `uv run --extra mlx python -m manabot.backends.parity CKPT --capture DIR`
  → `parity: pass (torch, mlx)` (capture: `/tmp/mtg325-gate-capture`, or a
  fresh `experiments/mlx_pilot.py capture`).
- Round trip: `uv run manabot train --out /tmp/mtg333/run` writes
  `.safetensors` policy files that `uv run manabot sim` plays; a `.pt` written
  at base loads with identical logits; a belief policy and a `bc`
  `value_semantic` survive save → load.
- `rg "AdmittedObservation|CardReferences|PreparedObservation|Retained" manabot academy scripts`
  → nothing.
- L4 equivalence (Etude, cohort cap USD 3): logits, values, losses and
  trained weights bitwise base vs head; gradients ≤ 1e-8 abs, and no worse
  than the head-vs-head repeat.
- L4 speed (ABAB, 3 processes per arm; cohort cap USD 10): no median slower
  than the run-to-run spread. Base on L4 at #501 (Etude 8a376219e430): damped
  9.94 steps/s [9.70, 10.01], PPO 0.2534, `PolicyInference` 2,296 obs/s, save
  0.023 s, load 0.082 s, collection 61.9 s. The bench also reports host
  preparation as its own section and the count of implicit device reads
  (`torch.cuda.set_sync_debug_mode("warn")`) per arm, as diagnostics.

## L4 mechanics

Launch files are `scratch/equivalence/mtg325_cuda_equivalence.py` and
`scratch/speed/mtg325_speed.py` (rename the stems to `mtg333_*`: Etude names
the experiment after the stem). Regenerate patches at the launch commit and
track them only there: equivalence applies `base.patch` reversed
(`git diff BASE HEAD -- manabot`); speed applies it forward
(`git diff HEAD BASE -- manabot`). `run.py` and `bench.py` must run under
both arms' `manabot/`. Launch notes: memory `etude-launch-client`,
`rented-launch-traps` (overlay client version, nested Doppler, queue + wait,
terminate the machine after success and check `runpodctl pod list`).
MTG-325 spent USD 0.04 (equivalence) and USD 2.04 over four speed passes.

## Optimize pass (after the speed check, whatever it shows)

Behaviour-preserving within the stated tolerances, judged on L4. Candidates
found while reading, unmeasured: vectorize `prepare._effects`' per-record
Python loop; int32 index arrays; reuse `Packed.gather` for fact programs
instead of `pack_sequence`; host compaction on CUDA inference (changes
summation order: tolerance, not bitwise); skip moving lookup buffers the
forward no longer reads; avoid full-capacity temporaries before trimming.
Each kept change reruns the oracle and one speed pass; total L4 spend for the
Task stays under USD 25 without asking.

## Risks

- Size: ~45 test files and six internals-patching experiments. The experiment
  files are finished studies; port what is mechanical (`scope.prepare`,
  `agent(batch)` keep their names) and make the rest fail at call time naming
  the commit that ran them, keeping `tests/eval/test_experiment_files.py`
  green. Recorded in `questions.md`.
- Host preparation on few-vCPU pods becomes the collection bottleneck:
  measured as its own bench section; the optimize pass targets it first.
- CUDA gradient order through `index_put`/`embedding` backward may exceed the
  recorded floor on non-belief paths; the head-vs-head repeat arm separates
  that from a real difference.
- Sim over `BackendPolicy` could change seeded action streams: move a module
  only when its existing digest/replay tests pass unchanged. A mismatch is
  unfinished migration work; it does not authorize silently moving sim/eval
  into the follow-up reserved for the other named dependencies.

## Exclusions

MLX support for the newly carried configurations (it keeps refusing them);
PPO or a categorical critic on MLX; optimizer/update-snapshot/incident
persistence; belief/ torch models, verify/, the executor choosing a backend
(follow-up Task); engine, env, serving protocol, worlds, policy-input
versions; rerunning frozen experiments.

## Done when

A developer prepares NumPy observations once and runs them through either
backend; training writes `.safetensors` policies that sim, eval and Academy
load beside old `.pt` files; the oracle and L4 checks above hold; the speed
table and the optimize pass's before/after are in the PR body.

## Documentation contracts still to implement

Draft docs in `manabot/{backends,policy}/README.md`, `manabot/model/agent.md`
and the training/evaluation/Academy guides describe the completed PR. They
settle these API details:
- `PolicyInference` imports from `manabot.backends.torch.policy`.
- `resolve`: absent extra → `ImportError` naming it; unknown name →
  `ValueError` listing backends.
- `save_checkpoint` writes container bytes regardless of suffix; a non-float32
  parameter fails naming it. The `.safetensors` naming assumption is in
  `questions.md`.
- `read_metadata` decodes JSON values (`metadata["hypers"]["agent_hypers"]`).
- `AdmittedRollout.rows` returns device `Prepared`; the minimal loop uses
  `admitted.legal(prepared)`. Keep `docs/training-loop.md` and
  `training/loops/minimal.py` aligned.
- `on_device(batch, device)` accepts a device string.

After moving encoders, replace the directory links in
`docs/observation-and-model.md` with exact files. Re-copy its “Scoring legal
actions” and `docs/learning.md`'s “Value” excerpts from the final forward.

## Compress (2026-11-13)

Scope is the implemented host/MLX slice. Torch's old admission, forward and
inference objects remain deletion targets for slices 2–3; none was repaired.
`FactPrograms | None` replaces mutually dependent optional arrays and allows
one typed `FactRoutes` construction. `Packed.lengths` is the sole owner of
sequence boundaries needed by the future Torch `pack_sequence` path.
Referenced-definition collection has its own helper; no-world preparation
returns before building that subset. Belief and history constructors name
fields. History detection reuses `buckets.has_history_events`. Effect routes
select live headers before numeric casts, avoiding a full padded-table copy
and leaving arbitrary padding untouched. `Prepared.is_agent` is required:
every configuration supplies ownership.

Initial focused check: `uv run --no-sync pytest tests/model/test_prepare_rows.py tests/model/test_mlx_backend.py -q` → 31 passed in 2.91s.
`uv run --no-sync pytest tests/training/test_update_backend_protocols.py tests/training/test_update_backends.py tests/model/test_torch_backend_protocol.py tests/model/test_torch_container_load.py tests/model/test_prepare_rows.py tests/model/test_mlx_backend.py tests/test_package_boundaries.py -q`
→ **63 passed**, one existing Torch nested-tensor warning, 12.82s; no skips.
Ruff check/format on the four touched Python files and `git diff --check` pass.
`uv run --no-sync python -m manabot.backends.parity /tmp/mtg325-gate-capture/initial.pt --capture /tmp/mtg325-gate-capture --json /tmp/mtg333/compress-parity.json`
→ **parity: pass (torch, mlx)**. Eight learner minibatches and eight collection
calls per backend; post-clipped-Adam max absolute weight differences 7.5e-9 /
1.2e-7 (bar 1e-6). MLX logits 5.9e-9, values 3.9e-7, relative loss 0.0009%
(bar 0.1%); gradient cosines pass 0.999. The -1 update-direction cosine is
still a diagnostic, not the gate. This tests the changed host/MLX path; it is
not a completed Torch cutover or a fresh L4 receipt.

Original uncommitted notes were copied to `/tmp/mtg333-compress-before/`
before trimming. The curated notes and receipts before realign are committed
at `b015041bd:scratch/`.


## Torch arithmetic receipt (2026-11-13 UTC)

The integrated native release build completed in 2m44s; the cp313 macOS
extension was extracted from its wheel and replaced atomically. Baseline
`ef28f3a58`'s `manabot/` was extracted to `/tmp/mtg333-integrated-base`, using
the checkout's current content and that same rebuilt extension in both arms.
`lf task status MTG-333 --json` returned no Wave and no new direction.

`run.py` produced `/tmp/mtg333/integrated-base.pt` and
`integrated-base-repeat.pt`; comparison: **3,248 tensors, 0 differing**.
`forward-head5.pt` also compares **3,248 tensors, 0 differing** against that
base, including all gradients, serving/live-token outputs and trained weights.
The final `prepared-forward.pt` capture after direct ActorCritic API changes also
matched all 3,248 tensors. After the learner-rollout cut, container saves and encoder moves,
`rollout-container.pt` again matched all 3,248 tensors exactly. `updates.py` base/head comparisons of all five
PPO/damped cases are byte-identical after the bootstrap correction below.

Two measured layout hazards, beyond the kickoff arithmetic probes:
- Hashed fact columns were a five-column view of a six-column NumPy allocation.
  Equal input values chose a different Torch linear path: numeric output differed
  by 2.4e-7, producing two belief fact-output gradient differences (2.9e-11).
  Making the host feature matrix contiguous removed them.
- Per-seat permanent matrices must be contiguous. The initial update matched,
  but after biases trained, the strided projection drifted. Contiguous seat
  inputs restored the one-iteration training probe and then the full oracle.
  Inference preparation also must select compaction at preparation time, even
  when a collector enters `no_grad` only around its later forward.

Deleted: `CardReferences`, the Torch trimming path, `_bucketed`, `_ownership`,
`_referenced_definitions`, `live_token_count`, `LiveTokens.pack`, and the old
fact/shared-effect route-building bodies and record-routing helpers. The
admitted carrier is temporary, explicitly not the completed one-input API.
Input-buffer gradients are no longer a contract: route tests check model and
object gradients; the timestamp-capacity test differentiates prepared numeric
features. The saved parameter names and all state buffers are unchanged.

Focused checks so far: nonempty facts/effects/programs, naming and history
→ 21 passed; agent, attachment and value-readout files → 65 passed;
host preparation (including new v10 masked-mean forward/backward cases),
record vocabulary, shared effects and layer inputs → 61 passed, one existing
CUDA-only skip. Final checks and remaining work follow in the next receipt.


Container/input follow-through (2026-11-13 UTC): bootstrap observations originally
prepared outside `no_grad` missed compaction; four update cases drifted despite
the forward oracle passing. Preparing the already-admitted host bootstrap rows
at `next_observation` restored byte-identical update JSON. This is retained as a
counterexample to relying on the forward oracle alone.

`pytest` over the general container cases, file format, catalog, capacity and
policy-input refusals: **45 passed**. Parameters retain exact dtype; scalar
buffers retain shape. Metadata includes belief/value/source/architecture records
and ordered `parameter_names` for Adam warm starts. Legacy policies still load;
optimizer/recovery/incident files remain Torch. Targeted warm-start, export,
belief and Academy checks: **38 passed in 258.89s**. `ruff check --fix` and `ruff format`
ran over changed Python files; finish with the final check and oracle.

After the private encoder moves, the 3,248-tensor CPU oracle remains exact.
Header-only metadata reading, real legacy conversion, update backend protocols
and the affected framework-import guards: **17 passed in 20.25s**.
The action-only estimator fixture now supplies its own rollout protocol instead
of patching deleted player admission functions; 18 estimator checks passed,
with its final materialization assertion updated to the fixture's two reads
and passing separately (1 passed). Changed-file Ruff checks and formatting pass.


Caller-cut check status (after `775e5d823`): inference lifetime/cache and agent
checks passed, **31 tests** before an admission-count spy leaked into the next
fixture's collection. Resetting that spy fixed both rule cases (2 passed in the
next run). Bucketed equivalence then reached an obsolete assertion on the raw
carrier's full-capacity facts; it now checks the trimmed Prepared fact shape.
The rerun of bucketed/live/host/backend/belief/guidance tests passed: 53 tests.
Frozen profiling functions that patch removed dictionary internals now raise
at call time naming `ef28f3a58`; mechanical consumers use Prepared. The oracle
and speed harnesses contain only API adapters, preserving fixtures and seeds.


Carrier-free receipt (2026-11-13 UTC): `/tmp/mtg333/no-carrier.pt` compared with
`integrated-base.pt`: **3,248 tensors, 0 differences**; `updates-no-carrier.json`
is byte-identical to `updates-base.json` for all five cases. This includes removal
of RecordEmbeddings and the old admission wrappers. The first oracle attempt
found one remaining belief caller indexing Prepared as a dictionary; fixed.
Record/effect/layer/capacity checks: **53 passed, one existing CUDA skip**;
empty-record CPU/Adam and shared-effect backward cases: **5 passed**.
Naming dense-reference checks: **9 passed**. Serving residency found an old
`scope.catalog()` call; callers now use guarded `scope.cache_programs()`.
Serving transfer/residency and numerical refusal checks: **6 passed**.
Resolver, host invalid-mask evidence and update protocol checks: **9 passed**.
A broader MPS file exposed a stale default-execution assertion omitting the two
existing layout options; corrected to include their `auto` defaults, focused
recheck passed (1 test). Static catalog scheduling, sim/eval backend injection/import
boundary, L4 and optimize are still outstanding.


Program-schedule receipt (2026-11-13 UTC, after `f9b5a3e7`): host catalog
packing first matched **3,248 tensors, zero differences** (`catalog-schedule.pt`).
After fact packing and static naming routes, `all-schedules.pt` again matched
all **3,248 tensors exactly**. Torch supplies only its CPU length-sort order to
the shared host builder, preserving tie order; it has no second preparation.
The full schedule is cached with the immutable admission tables, invalidated by
buffer identity/version or configuration. Scoped inference transfers no catalog
program rows because the scope owns that weight-only cache. No new persistent
buffers or checkpoint keys are added.
Program-gradient, empty/reordered catalog, naming and inference cases: **23 passed**.
Record/layer/MLX/container checks reached **37 passed**, then the container test
incorrectly counted new nonpersistent routes as saved state. It now compares
state-dict buffers; all **7 general policy configurations passed**. The five update cases are byte-identical to baseline (`updates-schedules.json`);
changed-file Ruff check/format pass.
