# Qwen reasoning versus MTG-183 final checkpoint — MTG-188

Jack Heart requested 100 Qwen games with reasoning enabled against MTG-183's
final checkpoint, with comparable inference and simulation timing, on 2026-10-26.
He authorized **$10 total on RunPod as necessary**. This supersedes the older
no-new-paid-execution direction for this campaign only. It does not authorize
training, a different opponent, another campaign or spending beyond $10.

## Frozen treatment before launch

- Qwen/Qwen3.5-9B BF16, model/tokenizer revision
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, Transformers from `uv.lock`.
  Explicit `enable_thinking=true`; greedy decoding, 16,384-token context,
  2,048 generated tokens per attempt, 120 seconds per selection including one
  protocol repair. No hidden prompt truncation. Only the exact legal reference
  after the closing thinking delimiter is executable. Raw text and token IDs,
  cutoffs, repairs and seeded fallbacks remain evidence.
- MTG-183 run `f3748bcabac8`, final raw update 5950, 331,180,341 learner
  transitions, 196,482 parameters. Checkpoint SHA-256
  `4f2ad656ba036a33d527212a7ac2badf2c5b9db136916f730846b07eef2fbea4`,
  1,071,166 bytes. CPU, one Torch thread, stochastic categorical selection,
  no training. Artifact and current-world/deck admission required.
- Exactly one game for each ordered pair in `elemental_grid.DECKS`, mirrors
  included. Native seat `(x+y)%2` gives Qwen 50 games per seat; fixed deal seed
  `188183000 + 10*x+y`, policy seed `188595000 + 10*x+y`. Schedule shuffled once
  with Python RNG seed 188183 and retained before any game. No replacements.
- Elemental world v2; viewer-safe text, complete legal choice set, bound native
  Commands, immediate clone replay. Native forced-action skipping enabled for
  both seats (existing arena behavior); this differs from the historical L4
  random-opponent receipt. It does not conceal multi-choice decisions.
- One resident Qwen GPU worker and one resident CPU policy; sequential games,
  no prefix reuse between requests and no separately excluded warmup. First
  observed decisions remain in latency populations. 2,000 decision and one-hour
  bounds per game. Stop on a failed game/worker; do not silently restart it.

## Resource and evidence boundary

One RunPod RTX A6000 48GB rental, at most $0.60/hour quoted compute price,
minimum 6 vCPU and 48GB RAM, 80GB ephemeral disk. One attempt; 11-hour job
ceiling and 10-hour executable campaign watchdog, with final delivery and cleanup
allowances. Etude job/cohort ceiling $9.50 leaves $0.50 of the authorized $10
for unmodeled ancillary storage/publication costs. Reject a quote exceeding the
reserved ceiling. No automatic retry. Stop early when complete. A timeout keeps
partial games and unattempted schedule entries, without fabricated results.

The Linux native binary is from successful release workflow 40446963764 at
`f5e7289b1a6d96b1d85fe9d58876adf24900bd69`, SHA-256
`1d1b674f314bf72cf8de2e66d999b01d71cb946fd0488c7bad5748adb8192edb`.
Launch checks engine/content source equality. Etude pins the final committed
source, inputs, image, hardware and budget before renting. Retain the quote,
launch receipt, finalized provider charges and rental-ended evidence.

## What the report measures

Report valid replay-verified wins/losses/draws and all failed/unattempted games.
This small heterogeneous grid is descriptive; one game per cell does not
establish matchup strength or Elo. Report fallback use beside outcomes.

Partition **game host wall** into exclusive Qwen wait/dispatch (subtract nested
synchronous evidence writes), CPU policy encoding and inference, native reset
and command execution (includes capture validation), rendering, replay, evidence
writes, and explicitly unattributed remainder. Report allocation/setup/load and
archive/publication outside that denominator. Never call the remainder simulation.

Under Qwen, show worker tokenization, generation host (including transfers and
decoding), validation, decision p50/p95/max and tokens. These nested intervals
are not added again to game wall. Policy inference includes tensor preparation,
forward and action sampling, separately from observation encoding. Compare the
actual CPU policy and GPU Qwen workloads with labels; do not call their ratio a
backend speedup. Prefill/decode device timing remains unavailable. Token split
coverage is explicit. Retain sampled whole-device memory separately from Torch
allocator peak. Billing is not inferred from game seconds.

## Accepted batched continuation — 2026-10-26

The first attempt `b6a4716bb321` completed zero games: the second Qwen selection
hit the 120-second deadline. Its original receipts remain unchanged. Provider
absence was verified on 2026-10-27 at 05:12 UTC; the retained estimated charge is
$0.06238391746666666459, with final inclusive billing unavailable.

Jack Heart selected a faster batched backend. The next attempt is a separate
treatment using vLLM 0.17.1, pinned by the hashed Linux lock at
`configs/llm/vllm-0.17.1-linux.lock`. The client keeps the root `uv.lock` and
exact pinned tokenizer. The model remains BF16, with the same reasoning,
context, output-token and selection-time limits. No quantization or prefix
reuse. Exact prompt/output token IDs cross a loopback completion endpoint;
the ordinary selection validator, repair and native Command binding still apply.

Four isolated game processes share one GPU server. Each game loads the same
CPU checkpoint and seeds its own policy RNG. The original 100-game schedule,
decks, deals and seats remain fixed. All 100 games are reserved before execution;
admissions are recorded before submission. A failed game stops new admissions;
already admitted games settle under their original bounds. No replacement games
or automatic retry. Startup and per-game client/policy preparation are retained
separately; overlapping game windows are never summed into campaign throughput.

The next allocation retains the RTX A6000 48GB hardware and $0.60/hour quote
limit, 11-hour job ceiling, 10-hour game watchdog, independent rental guardian
and cleanup/absence verification. vLLM startup has a 900-second ceiling within
the job lifetime. One new allocation/attempt is authorized by the backend choice.
Its job/cohort cap is **$9.40**, plus the original **$0.50 ancillary reserve** and
the first attempt's estimated charge: $9.96238391746666666459 total maximum
commitment against the original **$10**, not a new allowance. Stop early on
completion or failure and retain setup failures as attempts.

The supported model and exact-token completion contracts were checked against
the [official Qwen recipe](https://recipes.vllm.ai/Qwen/Qwen3.5-9B) and the
[v0.17.1 completion protocol](https://github.com/vllm-project/vllm/blob/v0.17.1/vllm/entrypoints/openai/completion/protocol.py).
Local controlled tests establish binding and accounting; actual GPU startup,
fit, latency and complete-game outcomes remain to be observed.

## Retry authorization — 2026-10-26

After the first batched attempt failed before model startup, Jack Heart said
“yes retry”, then clarified “no cap on retries” and “just cap on money”. This
supersedes every retry-count or single-allocation restriction above. The original
**$10 cumulative ceiling** remains; it is not replenished by failures or repairs.
Keep the $0.50 ancillary reserve and deduct every previous rental before quoting
the next one. Only one allocation runs at a time. Each repaired source/treatment
gets a new run identity, its own declared job/cohort cap and retained failure
evidence. The SDK remains at one automatic attempt per job so retries can inspect
the failure and record what changed. No further retry-count approval is needed.

Run `594e466c66c1` failed because its offline snapshot lookup requested three
documentation files excluded by cache preparation. Source `fea08def9` repairs
that mismatch with shared file patterns and a real offline-resolver regression
test. All 100 games were unattempted. Its pod was verified absent; its estimated
charge was $0.02901644665277777951. Together with `b6a4716bb321`, retained rental
estimates total $0.09140036411944444410. The next job/cohort cap of $9.40 plus
that sunk amount and the $0.50 reserve totals $9.99140036411944444410. Later
attempts must lower their cap as needed; final inclusive billing remains unknown.

The cache-fix attempt `25ce352f9533` passed acquisition but failed its server
environment probe: resolving `.venv-vllm/bin/python` followed its symlink to
system Python. The repair preserves the absolute virtual-environment path and
retains probe stderr. A local uv-created environment test verifies the actual
command selects the server environment even with an active client environment.
No games were admitted. Provider absence was verified at 2026-10-27 05:55:25 UTC;
its estimated charge is $0.02914788226388888835. The three prior estimates total
$0.12054824638333333245. The next job/cohort cap is **$9.37**, preserving the
same $0.50 reserve under the original $10. Hardware and scientific treatment
remain unchanged; each failed attempt remains separate evidence.
