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

## Stages and stopping

1. **Development pilot:** one game (first schedule entry), concurrency 1.
   Inspect completion, repairs/fallbacks, reasoning length, replay, memory,
   latency and inclusive cost. It is not part of the scored cohort.
2. **Scored cohort:** a new run reserving the first N schedule entries at a
   declared concurrency, both fixed from pilot evidence before launch and
   recorded below. A failed game or identity departure stops admissions;
   admitted games settle. No replacements or retries inside a cohort.

Each attempt is one Etude job on one RTX A6000 48 GB allocation at no more than
$0.60/hour, one vLLM service. Each job's cap is the $10 budget minus all earlier
attempts' charges minus a $0.50 cleanup/ancillary reserve. Allocation absence
and charges are reconciled after every attempt.

## Attempts

| Run | Stage | Cap | Outcome | Charge |
| --- | --- | --- | --- | --- |
| `dddaf3001dfa` | pilot, 1 game, c=1 | $2.00 | running | pending |
