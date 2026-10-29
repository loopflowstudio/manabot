# Qwen sampled reasoning versus MTG-183 in historical w13 — bb4160

Jack Heart requested this evaluation on 2026-10-28 with a fresh **$10 GPU
budget** (cumulative, including setup, failures, transfer and cleanup), and on
the same day accepted running the exact MTG-183 checkpoint in a separately
pinned **historical w13** experiment. Results are w13 evidence only. Earlier
MTG-188 treatments and their budgets stay closed and are not pooled here.

## Frozen identities

- **Opponent:** MTG-183 run `f3748bcabac8`, final raw update 5950, checkpoint
  SHA-256 `4f2ad656ba036a33d527212a7ac2badf2c5b9db136916f730846b07eef2fbea4`,
  1,071,166 bytes, 196,482 parameters; CPU, one Torch thread, stochastic
  categorical selection seeded per game. Bytes re-verified from R2 before rental.
- **World:** w13, policy-input v1. Campaign source is branch
  `jack-heart/bb4160-w13-campaign`: this Task's code plus a revert of the w14
  commit `e742e8bff`. Its engine/content source equals `f5e7289b1a6d`, which
  built the Linux native input (release workflow 40446963764, SHA-256
  `1d1b674f314bf72cf8de2e66d999b01d71cb946fd0488c7bad5748adb8192edb`,
  6,555,520 bytes). `runs()` re-checks that equality. Main keeps w14 admission;
  the campaign branch is never merged.
- **Model:** Qwen/Qwen3.5-9B BF16, model/tokenizer revision
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, vLLM 0.17.1 from
  `configs/llm/vllm-0.17.1-linux.lock`; no quantization, prefix caching or
  prompt truncation. Thinking enabled. The served name is the behavior version,
  which hashes the full inference configuration and runtime identity; every
  game summary names it and the opponent digest, and a departure stops admission.
- **Schedule:** the MTG-188 100-game ordered-pair Elemental schedule (deals
  `188183000+i`, policy seeds `188595000+i`, seat `(x+y)%2`, shuffled with seed
  188183), unchanged. A cohort of N games is the first N entries.

## Treatment: sampled reasoning (declared before any game)

`configs/llm/qwen3.5-9b-reasoning-sampled.json`: Qwen's published thinking-mode
setting for general tasks (model card at the pinned revision): temperature 1.0,
top-p 0.95, top-k 20, min-p 0, presence penalty 1.5, repetition penalty 1.0.
Each request's seed is derived from base seed 4160 and the exact prompt token
IDs. 16,384 generated tokens per attempt, 65,536-token context and 1,800 s per
selection including the single protocol repair — the limits Jack Heart accepted
on 2026-10-27. Only the exact legal reference after `</think>` and EOS is
executable; two invalid answers use the declared seeded random fallback, which
is reported as fallback and never counted as Qwen play.

Rationale: the greedy 2,048-token treatment (`b1d7ed75492f`) completed 18 games
with 519 random fallbacks among 793 selections, all after generation-limit
cutoffs; the greedy 16,384-token treatment (`7b8c16de1b60`) completed no game in
about two hours at eight concurrent games (roughly 70 Qwen selections). The model
card warns that greedy decoding in thinking mode causes endless repetition and
recommends these sampling values. This changes the treatment: sampled results
are not comparable with, or pooled into, either greedy result.

### Sampled pilot result and accepted treatment change

Pilot `dddaf3001dfa` (sampled, one game, concurrency 1) was cancelled at Jack
Heart's direction after about 93 minutes, mid-game, before its $2 cap could be
spent without a result. Its 41 retained Qwen selections: 38 valid initially,
2 after repair, none by fallback, and no generation-limit cutoff (every attempt
ended with EOS). Reasoning per attempt: median 4,616 tokens, p90 7,154, max
8,223; prompts median 5,706 tokens. Generation ran at about 33 tokens/s
single-stream, the A6000's bandwidth ceiling for 18 GB BF16 weights, so the
slowness was output length at concurrency 1, not a serving fault. 36 of the
41 decisions offered two legal references. Sampling removed the greedy
repetition failure. The game had native `skip_trivial` enabled, as every game in
this schedule does.

On 2026-10-28 Jack Heart accepted two new, never-pooled arms (`runs(arm=...)`
in `experiments/qwen_mtg183_vllm.py`):

- **`budget` (main arm, nearly all remaining budget):** the same sampling with a
  3,072-token thinking budget. A response with no closing delimiter by then gets
  `\n</think>\n\n` inserted by the harness (s1-style budget forcing) and
  answers within the 4,096-token per-attempt total; 16,384-token context. Each
  response records `forced_close_at` and the inserted token IDs; game receipts
  count `forced_closes` separately from repairs and fallbacks.
- **`nothink` (small labeled floor, about 10 games, well under $1, after the
  main cohort is under way):** thinking disabled with Qwen's non-thinking
  sampling (T 0.7, top-p 0.8, top-k 20, presence 1.5), 512 output tokens.

Serving for these arms: 32 concurrent games (`--max-num-seqs 32`) and the
checkpoint's own MTP head as a one-token speculative draft (vLLM 0.17.1
`{"method":"mtp","num_speculative_tokens":1}`; its rejection sampler applies the
declared top-k/top-p and presence penalty). The draft is part of the hashed
server argv. Every Qwen attempt prints prompt, reasoning and answer tokens,
forced close, seconds and tokens/s to job stdout, with vLLM's throughput and
speculative-acceptance lines once a minute. Receipts also record single-choice
Qwen decisions (not auto-resolved) and the native skip count per game.

## Stages and stopping

1. **Short pilot per arm:** a bounded run cancelled once a few minutes of
   decisions measure aggregate tokens/s, acceptance, reasoning per decision and
   forced-close rate. Pilot games are never scored.
2. **Scored cohort:** a new run reserving the first N schedule entries at a
   declared concurrency, fixed from pilot evidence before launch and recorded
   below. A failed game or identity departure stops admissions; admitted games
   settle. No replacements or retries inside a cohort.

Each attempt is one Etude job on one RTX A6000 48 GB allocation at no more than
$0.60/hour, one vLLM service. Each job's cap is the $10 budget minus all earlier
attempts' charges minus a $0.50 cleanup/ancillary reserve. Allocation absence
and charges are reconciled after every attempt.

### Budget pilot result and frozen scored cohort

Pilot `139099571cbb` (budget arm, 32 games at concurrency 32, MTP draft 1) was
cancelled after about 10 minutes of decisions. Of 86 Qwen attempts, 61 (71%)
reached the 3,072-token budget and were force-closed; 58 of those answered in
exactly three tokens. 80 decisions were valid initially and 6 after repair, with
no fallback. Prompts median 2,898 tokens (early-game). Each stream generated
about 17–19 tokens/s; vLLM reported 650–710 generated tokens/s aggregate with 32
running requests, MTP acceptance 63–71% (mean acceptance length about 1.65) and
37–47% KV-cache use. Projected at roughly 50 Qwen decisions and 165k generated
tokens per game, 100 games need about 7 hours of serving.

**Frozen scored cohort (declared 2026-10-29 before launch):** arm `budget`, the
first 100 schedule entries (all of them), 32 concurrent games, MTP draft 1, job
cap $7.00 (11-hour rental, 10-hour game watchdog), w13 campaign source as
launched. The behavior version is the one the server acknowledges at startup;
every game must name it and the MTG-183 digest. A failed game or identity
departure stops admissions; admitted games settle; unattempted entries remain
unattempted. Results are reported as this arm only.

### MTP crash and re-declared cohort

Cohort attempt `8f68c0bc488b` admitted 32 games; after about 13 minutes vLLM's
engine died: the 0.17.1 gated-delta-net attention backend asserts that a batch
never mixes plain and speculative decodes (`AssertionError: num_decodes: 1,
num_spec_decodes: 31`, `gdn_attn.py:310`). Nine games received HTTP 500; the
other 23 then failed while closing their inference worker (`Cannot close a
process while it is still running`), a harness defect that masked the runtime
failure. No game completed, 68 entries were unattempted, and its receipts stay
under its run ID. MTP is therefore unusable with this model in the pinned vLLM;
upgrading vLLM is outside this budget. The session now waits for its killed
worker before closing.

**Re-declared scored cohort (2026-10-29, before launch):** identical to the
frozen cohort above except no speculative draft (`--speculative-tokens 0`) and
the session-close repair; new behavior version from the new server argv. Job
cap $7.00. The MTP attempt is not pooled with it.

## Attempts

| Run | Stage | Cap | Outcome | Charge |
| --- | --- | --- | --- | --- |
| `dddaf3001dfa` | sampled pilot, 1 game, c=1 | $2.00 | cancelled mid-game; 41 Qwen selections, 0 fallbacks; pod absent | $0.566 billed through 02:00 UTC |
| `139099571cbb` | budget pilot, 32 games, c=32, MTP 1 | $1.20 | cancelled after ~10 min of decisions (86 attempts); pod absent | ~$0.16 (estimate) |
| `8f68c0bc488b` | scored cohort, budget, 100 games, c=32, MTP 1 | $7.00 | engine crash (MTP); 32 attempted, 0 completed, 68 unattempted; pod absent | ~$0.18 (estimate) |
