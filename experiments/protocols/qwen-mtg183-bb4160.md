# Qwen versus MTG-183 in historical w13 — bb4160

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
  prompt truncation. Thinking is enabled except in the declared `nothink` floor.
  The served name is the behavior version,
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
spent without a result. Its 41 retained Qwen selection files contain 38 valid
initial selections, 2 after repair, and 1 interrupted selection, with no fallback.
There are 43 attempts, 42 with responses; received responses ended with EOS.
Reasoning per received attempt: median 4,616 tokens, p90 7,154, max
8,223; prompts median 5,706 tokens. Generation ran at about 33 tokens/s
single-stream. This is consistent with bandwidth-limited decoding of 18 GB
BF16 weights and costly long output at concurrency 1; it does not independently
rule out a serving fault. 36 of the 41 decisions offered two legal references.
No greedy-style repetition failure
was observed in these received responses; this incomplete pilot does not
establish full-game reasoning quality. The game had native `skip_trivial`
enabled, as every game in this schedule does.

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

The later accepted settlement ordering supersedes the original no-thinking
timing above: final cohort receipts, all 100 scheduled IDs, inclusive charges
and allocation deletion precede another commitment. The bounded pilot below
is declared but unlaunched; an affordable scored floor is not yet frozen.

Initial serving declaration: 32 concurrent games (`--max-num-seqs 32`) and the
checkpoint's own MTP head as a one-token speculative draft (vLLM 0.17.1
`{"method":"mtp","num_speculative_tokens":1}`; its rejection sampler applies the
declared top-k/top-p and presence penalty). The draft is part of the hashed
server argv. Every Qwen attempt prints prompt, reasoning and answer tokens,
forced close, seconds and tokens/s to job stdout, with vLLM's throughput and
speculative-acceptance lines once a minute. Receipts also record single-choice
Qwen decisions (not auto-resolved) and the native skip count per game. The MTP
failure below supersedes this serving choice for the replacement cohort;
the current job constructor defaults to no speculative draft.

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
cancelled after about 10 minutes of decisions. The early log sample used to
declare the cohort is retained here; final archive totals below supersede it
for reporting. Of 86 Qwen attempts in that sample, 61 (71%)
reached the 3,072-token budget and were force-closed; 58 of those answered in
exactly three tokens. 80 decisions were valid initially and 6 after repair, with
no fallback. Prompts median 2,898 tokens (early-game). Each stream generated
about 17–19 tokens/s; vLLM reported 650–710 generated tokens/s aggregate with 32
running requests, MTP acceptance 63–71% (mean acceptance length about 1.65) and
37–47% KV-cache use. Projected at roughly 50 Qwen decisions and 165k generated
tokens per game, 100 games need about 7 hours of serving. This was a prelaunch
projection, not demonstrated affordability or complete-game qualification.

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
under its run ID. MTP is therefore unqualified for this campaign in pinned vLLM;
upgrading vLLM is outside this budget. The session now waits for its killed
worker before closing.

**Re-declared scored cohort (2026-10-29, before launch):** identical to the
frozen cohort above except no speculative draft (`--speculative-tokens 0`) and
the session-close repair; new behavior version from the new server argv. Job
cap $7.00. The MTP attempt is not pooled with it.

## Attempts

| Run | Stage | Cap | Outcome | Charge |
| --- | --- | --- | --- | --- |
| `dddaf3001dfa` | sampled pilot, 1 game, c=1 | $2.00 | 1 admitted/canceled, 0 completed; pod absent | $1.11978240 provider billed |
| `139099571cbb` | budget pilot, 32 games, c=32, MTP 1 | $1.20 | 32 admitted/canceled, 0 completed; pod absent | $0.16946718 provider billed |
| `8f68c0bc488b` | scored cohort, budget, 100 games, c=32, MTP 1 | $7.00 | 32 failed, 0 completed, 68 unattempted; pod absent | $0.18681270 provider billed |
| `cdb0c09f90c6` | **scored cohort**, budget, 100 games, c=32, no MTP | $7.00 | running at 07:27; admissions stopped; 42 complete/3 failed/4 pending in 07:25 log | $2.01204356 provider billed at 07:14 snapshot; not refreshed or final |

### Retained evidence and reporting (2026-10-29)

Digest-verified Etude fetches are published in `experiments/data.manifest.json`
under `experiments/data/bb4160/`: the sampled pilot's game archive, schedule,
configuration, runtime identity, GPU samples and campaign receipts; and the MTP
cohort's campaign/results receipts and engine log. Restore them with
`uv run --extra artifacts python -m manabot.data fetch experiments/data/bb4160`.
These are separate attempts; the manifest does not declare a successful cohort.

The offline reader now accepts sampled/budget-forced receipts, preserves their
sampling settings and behavior identities, and excludes inserted delimiters from
generated-token counts. In the pilot, retained native continuation stages account
for 40 priority responses (41 attempts), one declare-blockers response and one
targets response. Generated reasoning floor-index p50/max: priority 4,606/8,223,
declare-blockers 5,853/5,853, targets 3,903/3,903. These distributions include
repairs; the interrupted attempt stays in the denominator with an unknown split.

The latest retained log, collected at 07:25 UTC, shows the replacement cohort: 49
admitted games, 42 completed, 3 failed and 4 unsettled; 51 schedule entries
remain unattempted. No admission follows the first failure. These are log
observations, not final game/replay receipts. The 07:27 scheduler read still reports
running and zero output files. The bounded read returned retained log bytes, then
reported `RUNPOD_API_KEY is not set`; direct provider refresh is unavailable.
The key is absent in this environment and the configured `loopflow/prd` and
documented `manabot/prd` Doppler environments. No auth configuration was changed.
`accounting-20261029T072713Z.json` and its linked log are manifest-published;
provider fields are explicitly unknown and refer to the earlier observation.
Games 41, 60, 69, 76 and 89 completed since 07:14; pending IDs are 13, 70, 77
and 98. The admitted prefix and disjoint partition of all 100 IDs still hold.
Final W/L/D, replay qualification
and failure classification remain unavailable. The frozen stopping rule allows
admitted games to settle under the watchdog. These dated observations do not
establish the run's later status or terminal evidence coverage.

The last direct RunPod reads, for the 07:14:23 UTC snapshot, confirmed the first three pods absent
and their bills totaling **$1.47606228455**. The current cohort pod was then
present; its whole-hour provider bills totaled **$2.01204355841**, excluding the
current partial hour and unknown ancillary charges. The rental deadline remains
2026-10-29 14:04 UTC. Provider totals supersede understated launcher estimates;
frozen launcher receipts remain unchanged. The manifest-published
`experiments/data/bb4160/accounting-20261029T071423Z.json` retains these readings,
output counts and transfer-coverage gaps. Its `cohort_log` links the 07:14 UTC
log snapshot and partitions all 100 IDs into 37 complete,
3 failed, 9 pending and 51 unattempted entries. Games 6, 92, 85 and 25 completed
since the retained 06:57 snapshot; admissions and failures are unchanged. The admitted prefix matches
the retained unchanged schedule. The provider revised its earlier 05:00-hour
bill upward; whole-hour snapshots alone are not final inclusive accounting.
Inclusive cost and useful games/dollar remain uncertified until final settlement
and cleanup.

The earlier accounting snapshot remains at
`experiments/data/bb4160/accounting-20261029T0620.json`. Keeping the current $7
commitment and $0.50 reserve leaves **$1.02393771545** uncommitted. The previous
$1.05 constructor minimum was a software constraint, not proof of final
budget-limited infeasibility. `budget-64k` remains unlaunched; its declaration
at `8f2711df3` provides a candidate treatment for late-game context pressure
without changing this cohort. The retained log identifies the three failed games
but does not establish their causes; final receipts must support diagnosis before
this candidate can be described as a repair.

### Bounded no-thinking pilot declaration (unlaunched)

Declared 2026-10-29 before any no-thinking execution: one pilot game, concurrency
one, no MTP, `arm="nothink"`, `attempt_usd="0.80"`, `decision_seconds=300`.
The pinned w13 matchup, non-thinking sampling, 512-token output and 16k context
remain as above. The five-minute deadline includes the single protocol repair,
reaches the saved inference configuration and changes the behavior hash. Pilot
outcomes are not scored or pooled with any cohort. This is a prospective bound,
not measured full-game feasibility.

The constructor now requires room for one full decision plus the 120-second
admission reserve instead of a fixed 1.5-hour minimum. At the $0.60/hour ceiling,
$0.80 buys 80 minutes: 15 minutes of machine acquisition, 45 minutes reserved for
setup/startup/delivery, and 20 minutes of gameplay. It refuses a $0.80 run with
the original 1,800-second decision deadline. Controlled checks verify both the
rental bound and the effective saved deadline; the current scored run is unchanged.

This pilot fits the remaining uncommitted balance, but the requested final-cohort
reconciliation must precede another commitment. No pilot has been launched,
no ten-game floor is frozen, and final budget-limited infeasibility has not been
established. After settlement, the pilot must qualify answer completion, replay,
context fit and observed cost before an affordable floor cohort is declared.

### Receipt recovery and process integration

All 32 game archives from each of the budget pilot and MTP cohort were fetched
with digest verification and published in the data manifest, alongside their
schedule, inference/runtime and GPU receipts. The pilot's `vllm.log` fetch failed
Etude's range/identity validation (HTTP 206); it is not treated as verified.
Frozen game bytes were not rewritten. Expanded local copies live under
`.runs/bb4160/`; `receipt-accounting.json` beside each retained attempt summarizes
selection files and generated-token splits.

The pilot's final archives supersede its earlier 86-attempt log sample: 130
selection files, 95 initially valid, 3 repaired, 0 fallback and 32 interrupted;
133 generation attempts, 101 responses, 76 forced closes, 286,422 generated
output tokens. Priority reasoning has 99 received splits among 124 attempts,
p50/max 3,072/3,072; declare-attackers 2,776 (one), declare-blockers 3,072 (one).
Learn and targets have no received split and remain unknown. There are 265
committed Command receipts, no final game summaries and no complete-game replay
qualification. Whole-device sampled peak was 43,141,562,368 bytes.

The MTP failure archives contain 143 selections: 106 initially valid, 5 repaired,
0 fallback and 32 failed; 149 attempts, 117 responses, 84 forced closes and
329,862 generated output tokens. All 32 games failed; 68 were unattempted.
Priority reasoning has 113 received splits among 134 attempts, p50/max 3,072;
attackers 1,428 (one), blockers/learn/targets 3,072 (one each); private inspection
has no received split. Complete-game replay coverage is 0/32 admitted games.
The historical close exception left some summaries one selection short: the
failed terminal selection was journaled before the counter advanced. The offline
reader now admits exactly that uncommitted failed terminal decision, while
rejecting excess successful or unrelated files. The resulting projection retains
all 143 selections; no success or replay qualification is inferred.

The review branch now launches game, resident-session and one-shot inference
children through `ChildProcess`, with bounded supervisor threads for games.
Admission, identity/deadline checks, IPC failure and reap behavior remain covered.
Games inherit the existing outer watchdog process group, so forced shutdown also
covers orphaned inference descendants. New campaign receipts distinguish failed,
canceled and unattempted games; watchdog interruption retains canceled admission
IDs. These repairs do not retroactively describe source `cfcd947042c4` running
on the current rental. Broader mixed-report/Perf and review checks remain with gate.
