# Hosted game decisions through OpenRouter

`OpenRouterSession` sends the same complete `DecisionInput` as the local
Transformers harness. Its default chat transport selects exactly one legal
reference, allows one protocol-only repair, then records a seeded uniform
fallback. HTTP failures, provider errors/refusals, malformed envelopes, byte
overflow and deadline expiry stop the decision without fallback. It does not
execute Commands; the native game runner still owns capture, stale rejection,
submission and replay.

Set `OPENROUTER_API_KEY` in the process environment. Credentials are never part
of inference configuration or saved receipts. Run one saved observation:

```sh
uv run --extra openrouter python -m manabot.llm --backend openrouter \
  --input decision.json --config configs/llm/openrouter-sol.json \
  --out .runs/openrouter/decision.json
```

Or run the existing bounded functional game (one model seat against seeded
random, Mountain/Gray Ogre/Lightning Bolt mirror):

```sh
uv run --extra openrouter python -m manabot.verify.llm_game \
  --openrouter configs/llm/openrouter-opus.json --out .runs/openrouter/game
```

These commands make paid API calls. Output paths must be new. `--config` on the
game command accepts the existing `GameConfig` JSON for seed, deck, seat,
history capacity and game limits. The OpenRouter config is independently retained
in `runtime.json` and every selection. No Transformers weights or GPU are needed.
This runner is a functional proof, not a 10,000-game campaign scheduler.

The supplied configs select `anthropic/claude-opus-5.5`,
`anthropic/claude-haiku-5.5` or `openai/gpt-6.1-sol`. Their base configs use low
reasoning effort, a 4,096-token output allowance and a 120-second total decision
deadline; the Haiku high-effort config below allows 8,192 output tokens. Reasoning
is billed inside completion tokens; a short final action does not imply a cheap
completion. The allowance
covers reasoning and visible text. It is an engineering starting point, not
a measured reasoning budget. No temperature is forced on models that do not
support it. Hosted slugs are not immutable weight revisions.

`max_prompt_bytes` bounds the complete JSON request, not a token count.
Provider-native tokenization, templates and context admission happen remotely;
the adapter never estimates these as exact or silently truncates. Usage receipts
retain native prompt/completion tokens, optional reasoning/cache counts, cost,
generation ID, returned model/provider, raw final content, finish reason, and
attempt timing. Hidden reasoning text is excluded. Missing usage/cost stays null;
the game summary exposes `unpriced_attempts`. `reported_cost_usd` includes all
reported attempts, even unsuccessful protocol answers, and is not a hard budget
or a complete bill when any attempt is unpriced. A cancelled connection cannot
guarantee remote cancellation. HTTP requests and provider failures are not retried.

## Typed Decisions and the bounded Jev campaign

`configs/llm/openrouter-jev.json` selects `transport: "decisions"` and
`typesafe/jev-1.14`. `OpenRouterSession` sends the full observation as `state`
and one Choice question asking which action maximizes the chance of winning.
Each criterion contains the captured reference and its renderer description.
`DecisionInput.action_descriptions` must cover exactly the legal references;
older chat inputs can omit it. The native game runner supplies these descriptions
from the same capture that binds the eventual Command.

The adapter rejects more than 255 choices or an oversized complete request
before dispatch. It never shortlists actions, invents an extra option or replaces
the returned choice with a probability threshold. Typed requests have no repair,
random fallback, chat history or chat caching. `connect_retries` (default 0; the
Jev config uses 3) resends only after connection establishment failed, when no
request bytes reached OpenRouter, within the same decision deadline. The chat-only reasoning, output
allowance and fallback-seed fields do not affect Decisions.
See the [Choice contract](https://docs.typesafe.ai/primitives/choice)
and [OpenRouter Decisions examples](https://openrouter.ai/blog/insights/what-is-jev/).

Receipts retain the typed request, `answers`, returned model/provider and native
`input_tokens` / `output_tokens` / cost. They do not manufacture chat completions.
The runtime is `openrouter-decisions-v1`; game summaries use
`openrouter-decisions-v-checkpoint` or `openrouter-decisions-v-random`.
Existing chat defaults and historical receipts remain readable.

`experiments.jev_mtg183` runs one sequential campaign through the same native
runner. Its `Protocol` names exactly one opponent — a digest-bound checkpoint
with its attributed selection rule, or the `heuristic` anchor pinned by
`heuristic_sha256` over its source files — plus billing-bound evidence,
source/native/driver identities and an inclusive budget of at most $10. Historical checkpoints require their matching
runtime; these fields never override native world admission. The supplied Jev
inference config alone does not authorize or admit a campaign.

```sh
uv run --extra openrouter --extra artifacts python -m experiments.jev_mtg183 \
  --protocol path/to/frozen-protocol.json --out experiments/data/jev-mtg183
```

The frozen treatment has 1,000 uniquely seeded games: five per ordered deck
pair and Jev seat, shuffled with a retained seed. The first 10–20 games are the
included pilot under the same treatment. History capacity, forced-choice
skipping, policy RNG and game limits are retained. `first_game` and
`planned_games` select a fixed slice of that shuffle instead of all 1,000; a
continuation after an outage starts past retained positions. Only a complete,
valid pilot produces a 1,000-game cost projection.
The driver continues the affordable prefix, with no replacement games. A failed
or replay-divergent game is retained unscored; three consecutive failed games
stop the campaign. Reported
Wilson intervals describe wins among completed, replay-verified games and do not
claim fixed-sample coverage under cost-based stopping.

`BudgetLedger` uses SQLite transactions to reserve a justified maximum inclusive
request charge **before every dispatch**, including chat repairs when used by a
chat caller. The provider's reported cost is multiplied by the declared billing
fee factor, rounded up to nanodollars, and settles that reservation. Missing
usage and failed requests keep their full bound: the live session abandons the
reservation, which stays counted but no longer blocks later requests. Process
death leaves it pending and blocks restart. Charges exceeding the bound are retained and stop the campaign even
on restart. Rental, publication and shutdown allowances are reserved up front;
the caller must establish that those allowances and the request bound are valid.
A payload-byte limit or an observed pilot mean is not that evidence.

The output keeps the immutable protocol/schedule, an admission record for every
attempt, native game evidence, `spend.sqlite`, a pilot summary and `report.json`.
The report includes actual seat counts, wins/losses/draws, failures and interrupted
attempts, replay failures, unpriced reservations, connection resends, latency, tokens, per-game API
cost and inclusive totals. Rerunning the directory reconstructs retained results
without continuing play or resetting spend. Keep this directory as the campaign's
sole accounting authority. Publish its bytes with
`uv run --extra artifacts python -m manabot.data publish <directory>` before
committing an artifact manifest; publication is separate from executing games.

## Frozen policy opponent

`GameConfig.match` accepts a complete `MatchConfig`, including authored decks,
sideboards and content pack; when set it replaces the tiny `deck` mirror.
`opponent_checkpoint` selects an ordinary flat checkpoint instead of the random
opponent. Admission verifies its frozen world, roster and observation contract;
the runner retains its SHA-256, size, parameter count and input versions in
`opponent.json`. The policy samples on CPU using `opponent_seed + decision_index`
in an isolated Torch RNG. Both seats submit captured Commands and every command
is replayed against a clone. Native trivial-decision skipping remains disabled.

Set `history_capacity` explicitly and use the checkpoint's original observation
configuration. Text continuation details and saved policy inputs can differ;
record that limitation instead of claiming observation parity. Optional
`max_reported_cost_usd` stops before another model decision after the reported
threshold, or when any attempt has missing billing data. An in-flight decision
and its repair may exceed it; it is not a hard billing cap.

## Completed evaluation

The [Haiku versus MTG-183 report](../experiments/haiku-mtg183.md) retains the
100-game protocol, 51–49 result, measured $37.08 cost, limitations and cache
retrieval commands. The experiment cache contains every prompt, model response,
Command and replay transition, plus the checkpoint and frozen source snapshot.

`configs/llm/openrouter-haiku-high.json` selects high reasoning, an 8,192-token
completion allowance, and `objective: "win-game"`. That objective asks the
model to maximize its chance of winning as the player labeled You. The default
`legal-action` objective preserves the earlier protocol-only instruction.
Neither treatment changes observations or legal actions; both are retained in
configuration and exact request messages. The report's initial pilot changed
both objective and reasoning before the 100-game batch, so it cannot isolate
either change's effect.

## Cost planning, 2026-10-26

OpenRouter's [live model catalog](https://openrouter.ai/api/v1/models) reports
these USD rates per million tokens. Model-specific cache rates take precedence
over generic examples in the caching guide.

| Model | Ordinary input | Cache read | Cache write | Output, including reasoning |
|---|---:|---:|---:|---:|
| Claude Opus 5.5 | $4 | $0.20 | $5 (5 minutes), $8 (1 hour) | $20 |
| GPT-6.1 Sol | $2 | $0.10 | $2.50 | $10 |
| GPT-6 Sol | $2 | $0.20 | $2.50 | $10 |

The estimates below assume **one LLM seat, 200 model decisions per game,
3,500 input tokens and 512 completion tokens per decision**, no repair and no
cache savings. These are planning assumptions, not measurements of Opus/Sol
playing Magic. Both models currently require reasoning; 512 is a cost scenario,
not a promise that low effort will use that many tokens.

| Model | One game | 1,000 games | 5,000 games | 10,000 games |
|---|---:|---:|---:|---:|
| GPT-6.1 Sol | $2.42 | $2,424 | $12,120 | $24,240 |
| Claude Opus 5.5 | $4.85 | $4,848 | $24,240 | $48,480 |

At 32 / 512 / 2,048 completion tokens per decision, 1,000 games cost
$1,464 / $2,424 / $5,496 for Sol and $2,928 / $4,848 / $10,992 for Opus.
The 32-token case is a theoretical low-output floor, especially for mandatory
reasoning models. 10,000 games multiply each number by ten. Two similar LLM
seats roughly double the spend; mixed Opus/Sol games add their respective seats.
A 5% repair rate adds approximately 5% with comparable token sizes; every
repair resends the original prompt plus the rejected answer and correction.
Longer games, larger decks/observations, and high reasoning effort can dominate.

Formula, with token rates per million:

```text
games × model_decisions_per_game ×
  (uncached_input × input_rate + cache_reads × read_rate +
   cache_writes × write_rate + completion_tokens × output_rate) / 1,000,000
```

Input categories are disjoint: do not charge cached tokens again as ordinary
input, or reasoning again on top of completion tokens. Add repairs separately.
[OpenRouter Standard](https://openrouter.ai/pricing) adds a 5.5% credit-purchase
fee (Business 8%); taxes and local orchestration costs are excluded above.
At Standard rates the base 1,000-game budgets become about $2,557 and $5,115.

For latency planning, 200 requests/game at an assumed 2–5 seconds/request means
6.7–16.7 minutes/game, or 111–278 serial hours per 1,000 games. This is arithmetic,
not measured hosted latency. Independent games can overlap requests; decisions
within one game remain sequential. Respect provider limits and retain failed
attempts. The linked Haiku report supplies measured latency for its distinct
workload.

## Request layout and caching

Related work is [MTG-188](https://linear.app/loopflow/issue/MTG-188) (comparable
timings), [MTG-191](https://linear.app/loopflow/issue/MTG-191) (vLLM/SGLang),
and [MTG-192](https://linear.app/loopflow/issue/MTG-192) (exact-prefix reuse).
Their local engine work and this hosted transport share the same need for exact
prefixes, but hosted cache storage, scheduling and memory are provider-owned.

A fresh seed-23 random control on the current `manabot-text-2` renderer completed
426 Commands with exact native replay. Seat zero took 218 decisions and seat one
208. Using **o200k_base as a proxy**, complete text plus protocol instructions
averaged 3,509 / 3,476 tokens per decision, with maxima 4,839 / 4,937. This excludes
provider chat templates and is neither model's authoritative billed tokenizer.
The earlier Qwen game had 160 model decisions under the prior renderer. These
two plumbing runs motivate the rounded planning assumptions, not a population
estimate of game lengths or playing strength.

For the current control, consecutive same-seat prompts share a median of only
**63 proxy tokens**. Only 3.2% / 0.5% of consecutive pairs reached 512 tokens.
A hypothetical definitions-first reorder raises the median to **174**; the tiny
fixture's definitions average only 105 tokens. This is a text-layout probe,
not an implemented renderer or measured provider cache hit rate. The current
[OpenRouter caching guide](https://openrouter.ai/docs/guides/best-practices/prompt-caching)
lists minima of 512 tokens for Opus 5.5 and 1,024 for OpenAI caching.

The adapter therefore defaults `cache_prompt` to false. On OpenAI it explicitly
disables automatic cache writes. Setting it true marks the complete initial
user message with an ephemeral cache breakpoint; a repair reuses that block
unchanged. OpenRouter translates the marker for supporting OpenAI endpoints.
A stable opaque `session_id` pins routing best-effort, and optional `provider`
restricts the provider. Exact requests, replies and cache usage remain auditable.
There is no cross-decision conversation memory. Cache writes can increase cost
when the changed observation has no reusable prefix.

The next representation experiment should put unchanged instructions and the
**currently admitted visible definitions** before changing state, then bounded
public history, continuation and the entire legal action list. Keep per-decision
digests, revisions and request IDs out of that stable prefix. Preserve viewer
boundaries and exact definition aliases: a newly visible definition can change
aliases and legitimately break reuse. Do not fill a shared prefix with unseen
hands, whole decks, additional rules, old decisions or invented memory merely
to reach a cache threshold. This needs a separately versioned renderer treatment
and parity proofs; the adapter does not reorder the existing representation.

For a hypothetical 75% reusable prefix at the base workload, the warm-read
cost would be about $1.43/game Sol and $2.85/game Opus, before writes/expiry.
That is a sensitivity calculation, not a forecast: the current fixture does
not achieve it. Before choosing a larger run, measure actual prompt/cache/
reasoning usage, repair rates and p50/p95 decision latency on a small admitted
pilot. Group independent games by model, exact prefix and viewer-safe treatment
to keep prefixes warm; do not conflate inference batching with a discounted
asynchronous batch API, since each game waits on its previous action.
