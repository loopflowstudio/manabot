# Resin assessment: defer for L4

Decision accepted by Jack Heart on 2026-10-26 for Task b4295a: **defer Resin
for Qwen3.5-9B BF16 on NVIDIA L4**. The pinned deployment preset selects a
different checkpoint and GPU; the model runner independently rejects L4.
Obtaining a comparable performance result would first require kernel and
correctness work. Reuse the useful execution ideas through the existing
inference Tasks, prioritizing ordinary manabot policies.

This is a source and retained-evidence assessment. It changes no runtime and
selects no engine port, model download, rental or backend migration. The
compatibility conclusion does not depend on a new GPU benchmark. Alternative
runtime latency, full-context fit and concurrent capacity remain unmeasured.

## Support at the inspected revision

Resin source is pinned to
[`08a67226b42701ae2a8244abec250b71f204beaf`](https://github.com/ekzhang/resin/tree/08a67226b42701ae2a8244abec250b71f204beaf).
All Resin links below refer to that revision, not a moving support promise.

| Route | Exact Qwen3.5-9B BF16/L4 evidence | Assessment |
| --- | --- | --- |
| Transformers 5.3.0 | One completed, audited game at the pinned model/tokenizer revision below | Retain as the measured baseline; capacity limits remain open. |
| Resin deployment preset `qwen3_5` | Selects `RadixArk/Qwen3.8-27B-NVFP4` on B200, checkpoint revision `554ebba9b5f1b79dc11246341960360e6ef05ef4` | Different model, precision and hardware; its registry rejects L4. |
| Resin Qwen runner with a custom checkpoint | BF16 loader path exists, but runner requires compute-capability major >=9 | L4 is 8.9 and is rejected before weight resolution. Exact pretrained 9B execution on admitted hardware was not tested here. |
| vLLM | Official Qwen3.5-9B recipe lists single-GPU BF16, L4 and version >=0.17.0 | A supported comparator candidate; our exact checkpoint, capacity and latency have not been validated on it. |
| Resin with another GPU/model | A separate research treatment | Cannot establish an improvement over the unchanged L4 baseline. |

The [registry][registry] also enables two speculative tokens, FP8 KV and the
`qwen3_8` recipe. Changing only its model ID is insufficient. The
[Qwen runner][runner] defaults to FP8 KV, requests vision weights and rejects
pre-Hopper devices. [NVIDIA's capability table](https://developer.nvidia.com/cuda/gpus)
identifies L4 as 8.9. The [loader][loader] constructs unquantized models in
BF16 and recognizes a particular ModelOpt mixed FP8-projection/NVFP4-MLP
contract; loader support is narrower evidence than a working device path.
Removing the hardware guard would not validate paged attention, DeltaNet or
graph execution on Ada.

The [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.5-9B), inspected
2026-10-26, is support documentation, not our capacity receipt. A future
comparison must pin an actual release/image and match text-only BF16, KV
precision, checkpoint, token IDs and decoding semantics. Its large context
example and optional MTP configuration are not settings to inherit for this
comparison.

## What the Transformers baseline establishes

MTG-185's Etude run `4c2a54fdfaac` used Qwen/Qwen3.5-9B BF16 on an L4,
Transformers 5.3.0, and model/tokenizer revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Its source is
[`e747f0445e31ac08d674f855ad0128a4a722b100`](https://github.com/loopflowstudio/manabot/tree/e747f0445e31ac08d674f855ad0128a4a722b100).
The worker kept one model resident, allowed one outstanding decision and
retained no prefix state across requests. Its rental-local weight cache ended
with the rental. Source and receipt identities remain historical; this report
does not re-admit them under a different world or current code.

The retained summary and independent audit agree on 160 valid initial model
selections, zero repairs/fallbacks, and 308 Commands replayed exactly. The other
148 choices came from the seeded-uniform opponent. This was one completed
fixture game, not a protocol-edge suite or evidence of playing strength. The
assessment inspected those retained results; it did not run inference or
repeat native replay.

All equivalent Resin and vLLM measurements remain unmeasured.

| Measurement | Retained Transformers evidence |
| --- | --- |
| Cold acquisition plus verification | 87.5167 s |
| Resident model load | 10.0418 s |
| Complete decision latency | Median 1.95578 s; maximum 2.51822 s |
| Complete fixture game | 280.5549 s |
| Prompt processing / prefill | Not instrumented separately |
| Short-answer decode / time to first token | Not instrumented separately |
| Maximum observed prompt | 4,329 tokens; configured total budget 8,192 |
| Total device memory | Sampled peak 22,494 / 23,034 MiB, at 1 s intervals |
| Torch peak allocated | 19,063,865,344 bytes; not total device usage |
| Concurrency | Sequential worker; concurrent fit untested |
| Prefix reuse | None across requests |

Only **540 MiB** of sampled device headroom remained. One-second sampling can
miss spikes; the 4,329-token maximum does not prove the full 8,192-token budget
fits. There is no supported speedup, cost reduction or concurrent-fit claim.

Resin's own [benchmark protocol][benchmarks] excludes startup, loading, graph
capture and admission. Its decode window also excludes prefill. Those timings
answer useful execution questions, but cannot substitute for complete
short-answer decision latency or complete-game cost.

## How Resin works, and what adoption would own

The useful boundary is between scheduling/resources and the model forward
pass. Resin's Rust Engine owns request admission, compatible batching, prefill
chunking, progress, allocation, page leases and commit. The Python runner owns
weights, device allocations, kernels, sampling, graphs and synchronization.
Conversation recipes sit above the token Engine. These responsibilities are
documented in the pinned [architecture][architecture] and
[model forward-pass contract][forward].

The lifecycle makes that division concrete:

1. The runner declares a frozen `RunnerConfig`, including cache groups. The
   Engine builds a layout and binds resources once.
2. Submission records exact input tokens, limits and sampling policy. An
   optional prefix hit attaches a previously retained aligned prefix.
3. The scheduler chooses compatible requests within token and memory limits.
   It acquires page ownership and emits an immutable `BatchPlan`.
4. The runner executes against those resources. An asynchronous completion
   retains its requests, buffers and lease until completion.
5. The Engine validates the raw result and write span, applies stop conditions,
   publishes eligible prefixes and commits progress before rescheduling.

The [cache pool][cache-pool] accommodates attention history and recurrent
state. Prefix reuse restores all state families at a common aligned token
boundary; a recurrent checkpoint alone or KV pages alone are insufficient for
this hybrid model. Active reader/writer leases are distinct from optional
prefix retention. Evicting a cached prefix cannot release storage still owned
by live execution.

The failure distinction is equally important: rejection before persistent
effects can restore the selected cohort; failure after state may have changed
poisons the Engine and quarantines resources. Synchronization cannot undo an
in-place recurrent update. These are model execution rules, not permission to
retry a native game Command. Resin can overlap device continuations within a
token-generation request; manabot must still apply a game's preceding action
before asking for that game's next decision.

A Qwen/L4 port would own Ada-compatible attention and recurrent kernels,
checkpoint mapping, resource geometry and the following correctness proof:
full versus chunked prefill; recurrent copy-on-write; prefix hit/miss and
cross-request isolation; EOS and length boundaries; cancellation and uncertain
execution; eager versus captured execution numerics. Merely returning a legal
action reference does not show equivalent numerical computation.

The pinned [model tests][model-tests] compare tiny native models with
Transformers and cover recurrence, tensor names and rotary positions. The
[GPU tests][gpu-tests] skip below Hopper. The inspected
[CI workflow][ci] runs Clippy and lint/types, not a numerical GPU matrix.
These are useful tests, but neither their existence nor a skip certifies our
pretrained 9B/L4 configuration.

## Maintenance and cache boundaries

Resin 0.1.0 is marked Alpha, and its [README][readme] recommends vLLM/SGLang
for production inference. The inspected revision's LFM2 residual-mutation fix
shows active maintenance, not an API stability guarantee. Its
[CUDA dependencies][dependencies] pin Torch 2.13.0 and require Transformers
>=5.14.1 with a specific FlashInfer/FA4/FLA/Triton stack. This checkout pins
Torch 2.10.0; the retained baseline used Transformers 5.3.0. Adoption would
therefore include dependency qualification as well as model implementation.

Resin declares Apache-2.0. Its [NOTICE][notice] records adapted components and
separately licensed weights. No upstream code is copied by this assessment;
future reuse needs the exact dependency/license review and applicable notices.

Three stores have different identities and lifetimes:

| Store | What establishes reuse | Limit |
| --- | --- | --- |
| Downloaded model checkpoint | Pinned source plus verified weight bytes in our baseline | Resin's [checkpoint marker][checkpoint-cache] checks inventory filenames/sizes; it cannot replace publisher-SHA verification. |
| Compiled artifact | ABI, logical identity and content validation for a covered compiler integration | A persistent directory is not sufficient proof of compatibility or no recompilation. |
| Live attention/recurrent state | Exact tokens and model/template identity, compatible state boundaries and active ownership | It is neither a weight cache nor a portable compiled artifact. |

The [kernel-artifact cache][kernel-cache] has no distributed compiler lock.
Atomic publication protects readers from partial artifacts, but concurrent
cold writers may duplicate compilation. Valid hits do not remove framework
import, device initialization, model loading, graph capture or warmup. Claims
of no compilation need compilation-event or artifact evidence, not similar
time-to-first-token values.

## Reuse first for ordinary manabot policies

The accepted direction is to improve existing owners. At assessment base
[`a0dead2b5195853e0931bf244d84529c08944acb`](https://github.com/loopflowstudio/manabot/tree/a0dead2b5195853e0931bf244d84529c08944acb),
[`manabot/sim/serving.py`](https://github.com/loopflowstudio/manabot/blob/a0dead2b5195853e0931bf244d84529c08944acb/manabot/sim/serving.py)
already has `PolicyStore` residency/admission, `PolicyKey` model/input/runtime
identity, compatible request grouping and `DecisionReply` game/revision
validation. `PolicyInference` caches semantic definitions within unchanged
weights, and training's `SeatRoutedCollector` batches under frozen behavior
scopes. Residency, batching and catalog caching are existing foundations.
Shared serving currently rejects compound and belief policies; broader model
coverage needs its own admission decision.

The current `_infer` path assembles arrays, transfers inputs, forwards, copies
logits to CPU and samples. A useful next experiment is fixed input/output
storage and measured shape buckets within that path, followed by overlapping
CPU preparation for independent games with a running batch. Preserve scalar
sampling semantics, legal masks, deadlines and per-game identity. Measure
transfer/dispatch costs first: small policies may still favor CPU.

For feed-forward policies, cache immutable semantic encodings within the
existing policy scope. An LLM KV allocator adds no corresponding benefit.
A future recurrent policy would need game/seat/version-bound state and resets;
that does not justify adding a general state manager now. Queued work and
cached encodings must respect training's weight-update boundary.

The existing Tasks captured from the Modal/Resin discussion already cover
most of these ideas. The following are refinements, not new Tasks or an
implementation commitment in this assessment:

| Existing owner | Existing scope | Resin-derived refinement |
| --- | --- | --- |
| MTG-188 | Cold/warm and stage timing, throughput, memory, caches | Completion-based timing; graph-hit and fallback-reason counts; metrics reads that do not synchronize every time. |
| MTG-189 | Resident models, compatible batches, queue delay, bounded memory, per-game RNG, cancellation | Immutable execution plans; explicit ownership through asynchronous completion; overlap preparation for independent games. |
| MTG-190 | `torch.compile`, graphs, buckets, transfers, precision, eager baseline | Prepare and validate shapes before readiness; reusable fixed-address buffers; capture profitable regions with measured padding and explicit eager execution elsewhere. |
| MTG-191 | Supported optimized LLM backend comparison | Preserve prefill/decode, EOS and hybrid-state semantics at the backend boundary. |
| MTG-192 | Exact prefix reuse and identity, KV/recurrent state, safe eviction | Restore all state families at the same prefix boundary; lease live pages independently of cache retention. |
| Etude eb2edc | Prepared runtime and weight reuse across rentals | Distinguish downloaded weights, compiled artifacts and live device state. |
| Etude 2ccd7c | Worker health and recovery without duplicate execution | Separate rejection before device work from uncertain execution after possible state mutation. |

Jack Heart accepted this direction on 2026-10-26. The most explicit additional
design work is MTG-189's in-flight ownership/commit contract, coordinated with
MTG-190 and worker recovery. Sequence the work as MTG-188 evidence, MTG-189's
existing serving path, then MTG-190's measured buffer/capture/overlap changes.
Continue LLM comparisons and prefix work through MTG-191/192 on supported
backends. Rust scheduling or custom kernels require profile evidence; the
implementation language alone is not a performance argument.

## Conditions for revisiting

Revisit when a pinned runner admits Ada/L4 with the exact checkpoint, BF16
weights and matched cache precision, backed by credible numerical and
cache-isolation evidence; or when Jack Heart separately selects a changed
hardware/model treatment.

A future proof must retain the original template/tokenizer bytes and exact
IDs, thinking disabled, greedy answers capped at 32 tokens and EOS IDs
`[248044, 248046]`. Preserve complete legal references, viewer-safe input,
bounded protocol repair, marked seeded-uniform fallback and native
revision/Command/replay authority. Runtime failure, deadline exhaustion,
context overflow and stale bindings remain failures. Do not trim actions or
prompts, strip other special tokens, silently change KV precision, enable MTP
or substitute another recipe under the baseline label.

First compare retained prompt IDs and initial/repair boundaries, then
eager/graph and prefix-hit/miss correctness. Only after admission measure
complete short-answer decisions and one complete Command/replay game. Keep
acquisition, load, readiness/warmup, queueing, prefill, decode, total latency
and device memory distinct. Concurrent checks should use bounded actual game
work, never artificial resource pressure. Any paid proof needs declared
hardware, runtime/image, time/cost caps, attempt limits, cleanup and stop
conditions; MTG-185's spending authority does not transfer.

## Evidence identity and bounded reproduction

Private evidence was copied unchanged into this Task's checkout on 2026-10-26
under `.runs/resin/baseline/l4-result/`. Raw observations, decisions and game
archives are intentionally not published with this document. The following
SHA-256 identities from `.runs/resin/inspection.json` make the retained inputs
identifiable even if that local cache is lost. Hashes do not make private bytes
available; a reader without them can inspect the public source conclusion but
cannot independently reproduce the baseline audit.

Paths below are relative to `.runs/resin/`:

| Input | SHA-256 |
| --- | --- |
| `upstream/deploy/models.py` | `6b61b928770709ee0a40198ce80a416e70069d5eacb9cec29123f305b6c0640d` |
| `upstream/models/cuda/resin_models_cuda/qwen3_5/cuda.py` | `5582dd8a35a6efdec13a3806597a2e7c2dfa1a48cfd30701c49866cf97d4f0b5` |
| `baseline/l4-result/summary.json` | `01585c6c8fe9dc6a6ab4aff8490e2a2036e5b9d8fd26f93dc2ca6d8d6012aabb` |
| `baseline/l4-result/verified.json` | `08a83d7db826720b011178a2f92e00ea1dca592aa43e28c3b786205825cd276a` |
| `baseline/l4-result/game.tar.gz` | `72d6962dc823893d618aeb01d9f04d37836503c7db5d19fbd34ba98ef8a3a161` |

The bounded check below uses the retained upstream checkout and extracted
baseline files. It executes only the stdlib-only deployment registry and
parses the runner source; it never imports the CUDA runner, loads weights or
initializes inference. It checks retained counts and legal references, not
fresh native replay or numerical equivalence. Missing inputs make the check
unavailable, not successful; do not launch inference to fill that gap.

```sh
uv run --no-project --python 3.12 python - <<'PY'
import ast
import json
import subprocess
import sys
from pathlib import Path

root = Path('.runs/resin')
upstream = root / 'upstream'
revision = subprocess.check_output(
    ['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True,
).strip()
assert revision == '08a67226b42701ae2a8244abec250b71f204beaf'
sys.path.insert(0, str(upstream.resolve()))
from deploy.models import get_model, validate_gpu

spec = get_model('qwen3_5')
assert spec.repo_id == 'RadixArk/Qwen3.8-27B-NVFP4'
try:
    validate_gpu(spec, 'L4')
except ValueError as error:
    print(error)
else:
    raise AssertionError('L4 unexpectedly admitted')
tree = ast.parse((upstream / 'models/cuda/resin_models_cuda/qwen3_5/cuda.py').read_text())
assert any(isinstance(n, ast.If) and
           ast.unparse(n.test) == 'torch.cuda.get_device_capability(self.device)[0] < 9'
           for n in ast.walk(tree))
base = root / 'baseline/l4-result'
summary = json.loads((base / 'summary.json').read_text())
audit = json.loads((base / 'verified.json').read_text())
rows = [json.loads(p.read_text()) for p in (base / 'game').glob('decision-*/selection.json')]
assert len(rows) == summary['model_decisions'] == audit['model_selections_validated'] == 160
assert all(r['origin'] == 'initial' and r['failure'] is None and
           r['selected_reference'] in r['decision']['legal_references'] for r in rows)
assert summary['decisions'] == audit['commands_replayed_locally'] == 308
assert len(list((base / 'game').glob('decision-*/command.json'))) == 308
assert max(r['attempts'][0]['prompt_tokens'] for r in rows) == 4329
print('PASS: L4 rejected; 160 selections / 308 Commands agree; no inference or replay run')
PY
```

[registry]: https://github.com/ekzhang/resin/blob/08a67226b42701ae2a8244abec250b71f204beaf/deploy/models.py
[runner]: https://github.com/ekzhang/resin/blob/08a67226b42701ae2a8244abec250b71f204beaf/models/cuda/resin_models_cuda/qwen3_5/cuda.py#L85
[loader]: https://github.com/ekzhang/resin/blob/08a67226b42701ae2a8244abec250b71f204beaf/models/cuda/resin_models_cuda/qwen3_5/checkpoint.py
[architecture]: https://github.com/ekzhang/resin/blob/08a67226b42701ae2a8244abec250b71f204beaf/docs/architecture.md
[forward]: https://github.com/ekzhang/resin/blob/08a67226b42701ae2a8244abec250b71f204beaf/docs/model-forward-pass.md
[cache-pool]: https://github.com/ekzhang/resin/blob/08a67226b42701ae2a8244abec250b71f204beaf/docs/cache-pool.md
[model-tests]: https://github.com/ekzhang/resin/blob/08a67226b42701ae2a8244abec250b71f204beaf/models/cuda/tests/test_qwen3_5_model.py
[gpu-tests]: https://github.com/ekzhang/resin/blob/08a67226b42701ae2a8244abec250b71f204beaf/models/cuda/tests/test_qwen3_5_kernels_gpu.py
[ci]: https://github.com/ekzhang/resin/blob/08a67226b42701ae2a8244abec250b71f204beaf/.github/workflows/python-checks.yml
[readme]: https://github.com/ekzhang/resin/blob/08a67226b42701ae2a8244abec250b71f204beaf/README.md
[dependencies]: https://github.com/ekzhang/resin/blob/08a67226b42701ae2a8244abec250b71f204beaf/models/cuda/pyproject.toml
[notice]: https://github.com/ekzhang/resin/blob/08a67226b42701ae2a8244abec250b71f204beaf/NOTICE
[checkpoint-cache]: https://github.com/ekzhang/resin/blob/08a67226b42701ae2a8244abec250b71f204beaf/deploy/checkpoint_cache.py
[kernel-cache]: https://github.com/ekzhang/resin/blob/08a67226b42701ae2a8244abec250b71f204beaf/docs/kernel-cache.md
[benchmarks]: https://github.com/ekzhang/resin/blob/08a67226b42701ae2a8244abec250b71f204beaf/docs/benchmarks.md
