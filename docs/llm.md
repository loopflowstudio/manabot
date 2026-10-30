# LLM players and inference

`manabot.llm` asks an unmodified language model to choose one action from a
viewer-safe text observation and its complete legal-reference list, and retains
an inspectable receipt of that choice. It runs locally through Transformers,
behind a loopback vLLM server, or hosted through OpenRouter. It does not render
observations, resolve Commands or advance games: the native game runner
(`manabot.verify.llm_game`) owns capture, stale rejection, submission and replay.
Importing `manabot.llm` loads no model runtime.

Contents:

- [Selection protocol](#selection-protocol)
- [Local models: Transformers and Qwen3.5](#local-models-transformers-and-qwen35)
- [Sessions and vLLM serving](#sessions-and-vllm-serving)
- [Hosted models: OpenRouter](#hosted-models-openrouter)
- [Games and campaigns](#games-and-campaigns)
- [Offline inference reports](#offline-inference-reports)
- [Results](#results)

History: [llm.md](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/llm-harness.md),
[llm.md#hosted-models-openrouter](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/openrouter.md)
(including the 2026-10-26 cost planning and prompt-cache layout analysis).

## Selection protocol

One decision from a saved `DecisionInput` JSON (`text`, `legal_references`,
`representation_id`, `source_identity`, optional `action_descriptions`):

```sh
uv run --extra llm python -m manabot.llm \
  --input decision.json --config configs/llm/qwen3.5-9b.json --out .runs/llm/decision.json
uv run --extra openrouter python -m manabot.llm --backend openrouter \
  --input decision.json --config configs/llm/openrouter-sol.json --out .runs/openrouter/decision.json
```

The output path must be new. Exit status is 0 for any selection, including a
marked fallback, and 1 for a runtime, context or deadline failure. Invalid input
or configuration fails before inference. Receipts can contain private
observations; retain them with the same access as their source.

The chat protocol (Transformers, vLLM and OpenRouter `chat` transport):

- The prompt carries the text, all legal references and an output instruction.
  It is never truncated; a prompt that does not fit with its response allowance
  is a failure, checked separately for the repair.
- An answer must finish with EOS (local) and be exactly one listed reference
  after trimming. A generation-limit cutoff is incomplete even if it looks legal.
- One invalid answer gets exactly one repair (original message, failed answer,
  and "Invalid response. Return exactly one action reference from the
  legal-actions list, with no other text."). A second invalid answer produces a
  seeded uniform draw over the legal references.
- The receipt's origin is `initial`, `retry` or `fallback`;
  `attempts[0].validation_error` exposes first-response validity. Playing scores
  must include fallback games and describe the model plus recovery policy.
- Runtime failures, provider errors and exhausted deadlines stop with evidence
  and **never** fall back.

## Local models: Transformers and Qwen3.5

`InferenceConfig` (`manabot/llm/schema.py`) pins `model_id`/`model_revision` and
`tokenizer_id`/`tokenizer_revision` as full 40-character Hub commits, `device`,
`precision` (`float32`, `float16`, `bfloat16`), `context_tokens`,
`max_new_tokens`, `deadline_seconds`, `fallback_seed`, and `decoding`
(`greedy`, or `sampled` with an explicit `sampling` block that never inherits
repository defaults). The `llm` extra pins Transformers 5.3.0; loading uses
safetensors without remote code or adapters. Each CLI call loads one model in a
fresh worker, and its deadline includes download and loading.

The playing model is [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) at
BF16, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`; `AutoModelForCausalLM`
loads only its text decoder. Configs in `configs/llm/`:

| Config | Purpose |
|---|---|
| `qwen3.5-9b.json` | single decisions, thinking disabled, 4,096-token context |
| `qwen3.5-9b-game.json` | functional game, 8,192-token context |
| `qwen3.5-9b-reasoning*.json`, `qwen3.5-9b-instruct-sampled.json` | reasoning and sampled treatments |
| `qwen3.5-9b-cache.json` | shard size/SHA-256 manifest for the cache command |

`enable_thinking` passes the native chat-template option; with `true`, a reply
needs exactly one closing `</think>` followed by one legal reference.
`thinking_budget` forces a closing delimiter after that many tokens, leaving the
rest of `max_new_tokens` for the answer, and records the forced close. Limits in
these configs are engineering starting points, not measured budgets.

CUDA memory in receipts is PyTorch peak allocated tensor bytes, not device usage;
CPU records null. Prove fit on the actual GPU with the largest admitted prompt
and repair before claiming it.

Prepare and verify the pinned checkpoint separately from inference (set
`HF_HOME` to reusable disk first; add `--offline` to verify without Hub access):

```sh
uv run --extra llm python -m manabot.llm.cache \
  --config configs/llm/qwen3.5-9b-game.json \
  --manifest configs/llm/qwen3.5-9b-cache.json --out cache.json
```

## Sessions and vLLM serving

`InferenceSession` (`manabot/llm/session.py`) loads one worker once and applies
the selection protocol to successive decisions, with a separate load deadline
and a hard parent-owned deadline per decision. A timeout retains progress and
closes the session without selecting or restarting. No conversation or KV state
carries across decisions.

Admission binds configuration, runtime and chat template into a
`behavior_version` hash. Each decision has a unique request ID; a mismatched ID,
version, configuration, runtime or decision closes the session. Receipts retain
the version; historical receipts without it read as unknown.

vLLM runs in its own Linux environment, `.venv-vllm`, synced from
`configs/llm/vllm-0.17.1-linux.lock` by `scripts/qwen-mtg183-vllm-setup.sh`.
`manabot/llm/vllm_server.py` owns the loopback server and serves the model under
the behavior hash as its only name; startup and client admission check it via
`/v1/models`, and replies naming another model are rejected.
`vllm_runtime.py` sends exact locally encoded token IDs and never asks the
server to apply a template. `vllm-command.json` retains the full server command.
This checks the service's declared identity, not the loaded GPU weights.

## Hosted models: OpenRouter

`OpenRouterSession` (`manabot/llm/openrouter.py`) reads `OPENROUTER_API_KEY`
from the process environment and fails if it is absent. Credentials are never
part of configuration or receipts. Hosted calls are paid; slugs are not
immutable weight revisions, and provider tokenization happens remotely.

`OpenRouterConfig` fields: `model_id`, `transport` (`chat` default, or
`decisions`), `objective` (`legal-action` default, or `win-game`),
`reasoning_effort`, `max_output_tokens` (covers reasoning and visible text),
`deadline_seconds`, `fallback_seed`, `cache_prompt` (default false; true marks
the initial user message as an ephemeral cache breakpoint), `provider`,
`max_prompt_bytes` (bounds the whole JSON request, not tokens) and
`connect_retries`. Supplied configs: `openrouter-opus.json`,
`openrouter-sol.json`, `openrouter-haiku-high.json` (high reasoning, 8,192
tokens, `win-game`) and `openrouter-jev.json`.

- **Chat** follows the selection protocol above. HTTP requests are not retried.
- **Decisions** (`typesafe/jev-1.14`) sends the observation as `state` and one
  [Choice](https://docs.typesafe.ai/primitives/choice) question whose criteria
  are the captured references and their renderer descriptions
  (`action_descriptions` must cover exactly the legal references). It rejects
  more than 255 choices, never shortlists, and has no repair, fallback or
  caching. `connect_retries` resends only when no request bytes reached
  OpenRouter, within the decision deadline. Runtime `openrouter-decisions-v1`.

Receipts retain native prompt/completion (or input/output) tokens, optional
reasoning and cache counts, reported cost, generation ID, returned model and
provider, raw final content, finish reason and timing; hidden reasoning text is
excluded. Missing usage stays null and is counted as `unpriced_attempts`.
`reported_cost_usd` covers every reported attempt but is not a bill or a cap.

`BudgetLedger` (`manabot/llm/budget.py`) reserves a justified maximum inclusive
charge in SQLite before every dispatch (ceiling at most $10) and settles it
with the reported cost times the billing fee factor, rounded up to nanodollars.
Failed or unpriced requests keep their full bound; process death leaves a
pending reservation that blocks restart.

## Games and campaigns

`manabot.verify.llm_game` connects the native text renderer and
revision-bound Commands to a session: one model seat against a seeded random
opponent, defaulting to a 24 Mountain / 24 Gray Ogre / 12 Lightning Bolt mirror
(an engine test deck, not a benchmark). Every executed Command is replayed
against a clone. `--config` takes a `GameConfig` JSON (seed, deck, seat, history
capacity, limits); `GameConfig.match` takes a full `MatchConfig`, and
`opponent_checkpoint` substitutes a frozen flat policy sampled on CPU, with its
digest retained in `opponent.json`. `max_reported_cost_usd` stops before the next
model decision once crossed. Inputs are never clipped; overflow, model failures
and replay divergence retain incomplete summaries.

```sh
uv run python -m manabot.verify.llm_game --random-control --out .runs/llm-control
uv run --extra openrouter python -m manabot.verify.llm_game \
  --openrouter configs/llm/openrouter-opus.json --out .runs/openrouter/game
```

Without `--random-control` or `--openrouter` it runs the GPU entry point used by
`experiments/llm_game.py`: offline after `scripts/llm-game-setup.sh`, which
expects an Etude-verified CPython 3.12 engine at `$ETUDE_INPUTS_DIR/native.so`
whose source must match the checkout (`UV_NO_SYNC=1` prevents a rebuild).
Decision evidence is packed into `game.tar.gz` by `scripts/llm-game-collect.sh`.

Campaigns:

- `experiments/qwen_mtg183.py` and `experiments/qwen_mtg183_vllm.py`: Qwen
  against MTG-183's final checkpoint on a fixed schedule; the vLLM cohort freezes
  the acknowledged model identity before admitting games and stops admission on
  a failed game or a changed identity.
- `experiments/jev_mtg183.py`: sequential OpenRouter Decisions campaign from a
  frozen protocol (digest-bound checkpoint, billing-bound evidence, budget at
  most $10). A complete pilot projects cost; the driver plays the affordable
  prefix without replacement, and three consecutive failed games stop it. Its
  output directory, including `spend.sqlite`, is the sole accounting authority;
  rerunning reconstructs results without resetting spend.

```sh
uv run --extra openrouter --extra artifacts python -m experiments.jev_mtg183 \
  --protocol path/to/frozen-protocol.json --out experiments/data/jev-mtg183
```

Publish campaign data with `uv run --extra artifacts python -m manabot.data publish <directory>`.

## Offline inference reports

`manabot/reports/inference.py` reads historical receipts without importing the
executable model, simulation or native runtime. It preserves sampled decoding,
thinking budgets and behavior versions, and rejects mixed treatments within one
game or sequence. `generation_tokens` separates received outputs from missing
responses, excludes harness-inserted closing tokens and groups reasoning lengths
by native continuation stage; repairs and failed attempts stay in each
population. Historical `output_tokens` still counts inserted delimiters.

`experiments/report_prototype/build.py` (`--llm-receipt`, `--eval-receipt`,
`--baseline-manifest`) renders the Perf report; `experiments/inference_baseline.py`
(`freeze`, `validate`, `run`, `replay`) freezes and runs bounded local recipes.
Load the evaluation projection only for evaluation inputs. Check the LLM-only
boundary in a fresh process with those imports blocked, and exercise mixed
reports through the builder CLI in the headless Perf proof
(`scripts/verify-eval-performance-report`).

## Results

- **Qwen3.5-9B on an L4, 2026-10-25** (Etude run `4c2a54fdfaac`, source
  `e747f0445e31`): won the seed-23 functional game in 308 decisions, 160 model
  selections all valid on the first response, every Command replayed. Median
  decision 1.96 s, largest prompt 4,329 tokens, peak 22,494 of 23,034 MiB device
  memory; about $0.08. It predates the capture-v2 renderer and does not measure
  later renderers.
- **Claude Haiku 5.5 versus MTG-183, completed 2026-10-27**: 51–49 over 100
  games, $37.08 reported, no clear strength difference. See
  [haiku-mtg183.md](../experiments/haiku-mtg183.md).
