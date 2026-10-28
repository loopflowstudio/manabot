# One text decision through an untouched LLM

For hosted models, see the [OpenRouter adapter, cost estimates and cache layout](openrouter.md).

`manabot.llm` consumes observation text and its complete legal-reference list,
then returns one inspectable selection record. It does not render observations,
resolve Commands, advance a game or train a model. The caller remains responsible
for viewer-safe input, completeness, source identity and native stale/legality
checks when submitting the selected reference.

Run with explicit input and configuration files:

```sh
uv run --extra llm python -m manabot.llm \
  --input decision.json --config inference.json --out .runs/llm/decision.json
```

The output path must be new. Exit status is zero for any selection, including a
marked fallback, and one for a runtime, context or deadline failure. Invalid
input/configuration fails before starting inference. Records can contain private
observations; retain them with the same access as their source.

An initial handwritten fixture can use this shape; it is not a renderer or a
captured native observation:

```json
{
  "text": "Synthetic protocol fixture. A0 means wait. A1 means continue.",
  "legal_references": ["A0", "A1"],
  "representation_id": "handwritten-protocol-fixture-v1",
  "source_identity": {"fixture": "synthetic; no native world or Command"}
}
```

The first playing-model integration is [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B),
using its unchanged text weights at BF16. Run its pinned configuration on CUDA:

```sh
uv run --locked --extra llm python -m manabot.llm \
  --input decision.json --config configs/llm/qwen3.5-9b.json \
  --out .runs/llm/qwen-decision.json
```

The configuration pins model and tokenizer to
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, explicitly disables thinking through
the native chat template, and uses a provisional 4,096-token total context,
32-token response allowance and 120-second decision deadline. Each full prompt,
including repair, must fit alongside the response allowance. These are starting
engineering limits, not a measured game corpus budget or a single-L4 fit claim.
Pre-cache the pinned checkpoint before a timed run; the deadline includes loading.
Full-weight inference and largest-input/repair memory still need an L4 proof.

The optional runtime pins Transformers 5.3.0. Its `AutoModelForCausalLM` loads
Qwen3.5's text decoder from the composite checkpoint without instantiating the
vision encoder. The retained `runtime.model_class` identifies the loaded class.
`enable_thinking: false` passes the native template option on both attempts;
omitting it retains a tokenizer's default. Reasoning-output parsing is not
supported, and enabling thinking is rejected by configuration validation. The
same generic loader and configuration still support ordinary causal models.

This configuration pins the small model used for the CPU plumbing smoke. Model
selection and limits are explicit run inputs; this example does not establish a
scientific baseline or an admitted game-context budget. Both revisions must be
full, immutable 40-character Hub commit hashes.

```json
{
  "model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
  "model_revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
  "tokenizer_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
  "tokenizer_revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
  "device": "cpu",
  "precision": "float32",
  "context_tokens": 2048,
  "max_new_tokens": 32,
  "deadline_seconds": 120,
  "fallback_seed": 37,
  "decoding": "greedy"
}
```

CUDA accepts `float32`, `float16` or `bfloat16` as explicitly configured. Loading
uses safetensors and the installed Transformers implementations, without remote
Python code, adapters or model modifications. The `llm` extra is optional;
ordinary manabot imports do not load Transformers. Each command loads one model
in a fresh worker. The one decision deadline includes downloads, loading,
tokenization, generation and the optional repair. Pre-cache pinned model bytes
when measuring inference latency; retained wall time still includes loading.

The model's chat template formats a user message containing the supplied text,
all legal references and the output instruction. Both attempts tokenize the full
chat without truncation. Prompt length plus the fixed generation allowance must
fit `context_tokens`; the repair's longer conversation is checked separately.
The configured context must also fit the model's declared position limit.
Greedy decoding uses a fresh explicit generation configuration, recorded in full,
without inheriting sampling or repetition penalties from the model repository.

An answer must finish with EOS and contain exactly one listed reference after
whitespace trimming. Generation recognizes both the model's declared EOS tokens
and the tokenizer's EOS, which can differ for chat models such as Qwen3.5.
A generation-limit cutoff remains incomplete even if its
text looks legal. An invalid answer receives exactly one repair, preserving the
original message and the failed answer and appending:

> Invalid response. Return exactly one action reference from the legal-actions list, with no other text.

Two invalid answers produce a seeded uniform draw over the original reference
list. The record distinguishes `initial`, `retry` and `fallback`, and includes
raw decoded responses, token IDs/counts, template, model/tokenizer revisions,
precision, generation configuration, per-attempt and total timings, and fallback
seed/RNG state/index. Only the final EOS terminator is removed from decoded
response text; other special tokens remain visible to protocol validation.

Runtime failures and exhausted deadlines stop with evidence, without fallback.
The supervisor terminates a stalled worker and retains its last completed
snapshot, including a pending fully tokenized prompt when available. Unfinished
inference cannot supply partial tokens or final memory measurements. CUDA memory
is PyTorch's peak allocated tensor bytes since loading began, not total device
usage or reserved memory; CPU records use null. This measurement alone does not
certify fit on a 24 GB L4. Measure the largest admitted prompt and repair on the
actual GPU before making that claim.

`attempts[0].validation_error` and the selection origin expose first-response
validity, repair success and fallback usage. Treat failed attempts separately;
future playing scores must include fallback games and describe the model plus
recovery policy. No playing-strength or fine-tuning claim follows from this
single-decision instrument.

The 2026-10-24 CPU compatibility check used the pinned SmolLM2 example above
through the CLI on Transformers 5.3.0. Its first response, `A0\nA1`, was rejected;
the repair returned `A0`. The record retained 88/124 prompt tokens, 6/3 output
tokens and 18.72 seconds including loading, with null GPU memory. This checks
pretrained inference and repair on a synthetic input, not Qwen3.5 or L4 capacity.
The check used Hub access; an offline-only invocation failed during metadata
lookup and retained a runtime failure without fallback.

The first full-weight Qwen3.5-9B CLI proof on 2026-10-24 used the pinned revision
above at BF16 on CPU, with one OpenMP thread and a 900-second deadline. On a
synthetic Magic decision offering cast Lightning Bolt (`A0`) or pass (`A1`), it
returned exactly `A0` on the initial attempt: 184 prompt tokens, three output
tokens including the chat EOS, and 195.98 seconds including loading. No repair
or fallback was used. This verifies model action selection; it does not execute
a native game action or establish L4 capacity. An earlier invocation exposed
the distinct model/tokenizer EOS tokens and retained a fallback receipt; the
runtime now recognizes both declared terminators without relaxing validation.

Implementation references: [Transformers chat templates](https://huggingface.co/docs/transformers/v5.3.0/chat_templating)
and [Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py).

## Cached full-game GPU proof

Prepare the pinned checkpoint separately from inference:

```sh
uv run --extra llm python -m manabot.llm.cache \
  --config configs/llm/qwen3.5-9b-game.json \
  --manifest configs/llm/qwen3.5-9b-cache.json --out cache.json
```

The manifest pins publisher LFS sizes and SHA-256 hashes for every shard. Add
`--offline` to verify an existing cache without Hub access. Set `HF_HOME` to a
reusable disk location before both preparation and execution. A new remote
machine still needs its first transfer; a local cache is not a remote cache,
and ephemeral rental disks do not survive deletion.

`InferenceSession` loads one worker once, then applies the existing selection
protocol to successive decisions. Loading has its own deadline; each decision
has a hard parent-owned deadline. A timeout retains progress and closes the
session without selecting an action or automatically restarting. Generation
does not retain conversational or KV state across game decisions.

Admission binds the pinned inference configuration to the runtime and chat
template in a `behavior_version`. Each decision has a unique request ID; the
parent checks it and the returned version, configuration, runtime and decision
before forwarding progress. A mismatch closes the session without selecting an
action. Selection receipts retain the version; historical receipts without it
remain readable and keep an unknown version.

`manabot.verify.llm_game` connects the native text renderer and revision-bound
Commands to that session. Its functional fixture is a 60-card mirror of 24
Mountains, 24 Gray Ogres and 12 Lightning Bolts, Qwen in seat zero against a
seeded uniform random opponent. This is a small engine test deck, not a
constructed-format deck or a playing-strength benchmark. Native trivial-decision
skipping is disabled; public history v1/32 is declared, and there is no writable
model memory. Every executed Command is checked against a second native replay.

The separate game configuration admits 8,192 tokens because a local 426-decision
random control reached 5,118 prompt tokens. Inputs and repairs are never clipped;
overflow stops the game explicitly. This budget is not a capacity certificate.
Game limits, model failures and replay divergence retain incomplete summaries.
The random control can run without loading a model:

```sh
uv run python -m manabot.verify.llm_game --random-control --out .runs/llm-control
```

The L4 entry point runs offline after setup, records loading separately from
decision latency, samples total device memory with `nvidia-smi`, and packs full
decision evidence into `game.tar.gz`. `scripts/llm-game-setup.sh` expects an
Etude-verified Linux CPython 3.12 engine binary at `$ETUDE_INPUTS_DIR/native.so`;
its source must match the checkout's native code and content. The initial test
uses the passing PR #376 release wheel, with that source comparison retained.
Etude's rental/job limits bound paid execution, including setup and cleanup.

The reusable `experiments/llm_game.py` recipe takes `native_artifact` and
`native_source` receipt paths. It checks the artifact identity and requires an
empty native-source/content diff from the CI build before making a job. Setup
installs dependencies once; `UV_NO_SYNC=1` prevents the typed run command from
rebuilding the project after the verified engine has been copied. The first
remote run predated that environment setting and did one extra package build.

Measured L4 result, 2026-10-25 (Etude run `4c2a54fdfaac`): Qwen won the seed-23
functional game in 308 decisions, including 160 model selections, all valid on
the initial response. No repairs or fallbacks occurred. All 308 Commands replayed
to the same terminal state on the worker and again locally. Game time was 280.55
seconds; median model decision 1.96 seconds, maximum 2.52 seconds, loading 10.04
seconds. The largest actual prompt was 4,329 tokens. Cold model acquisition and
checksum verification took 87.52 seconds, outside game timing.

That run used source `e747f0445e31ac08d674f855ad0128a4a722b100`, before
MTG-184's final capture-v2 renderer landed. Its retained prompts and replay
belong to that original runtime; they are not a GPU measurement of later
renderer versions. Reusing the launch recipe requires a native binary built
from the current source, as enforced by its provenance check.

Peak allocated Torch tensors were 19,063,865,344 bytes. One-second `nvidia-smi`
samples reached 22,494 of 23,034 MiB total device memory (about 22.0 of 22.5 GiB).
The game fits this L4, with limited observed headroom; this is not proof of every
8,192-token initial/repair prompt or stronger models fitting. The rental ended
after one successful attempt, with Etude reporting approximately $0.08. Its
ephemeral remote cache ended with it; local verified bytes remain cached.
