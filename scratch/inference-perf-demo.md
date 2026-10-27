# Inference Perf demo — 2026-10-26

Task: [MTG-188](loopflow://manabot/task/MTG-188). Review experience: retained inference receipts in the existing Perf page. PR #381 base is `558e0186b585228dd2130cf14d9cb40de217e66f`, head is `f5e7289b1a6d96b1d85fe9d58876adf24900bd69`.

The user requested `$pr-review`, then `$demo the perf report`. The local walkthrough is [pr-review.html](pr-review.html); the interactive runbook is [demo.html](demo.html). Accepted design remains [mtg188-context.md](mtg188-context.md).

## Observed by the agent

The actual builder CLI rebuilt the historical manifest against retained local bytes into `.runs/pr-review/historical-report/report.html`. It reported 160 selections, 308 committed Commands and one replay-verified game, with warmup and inclusive billing unavailable. No model or new performance workload ran.

Controlled `_shared_fixture` data, generated from the published tests, demonstrated a failed LLM receipt, two policy cells, one shared charge and an unattempted GPU cell. The real CLI rejected completed/failed contradiction with exit 2. After correcting only the fixture declaration, missing source directories preserved declared attempts with unknown observations; exact restoration recovered links without changing declaration bytes. A one-byte change to shared execution evidence failed digest validation; restoring those bytes allowed rebuilding. These observations establish report interpretation, not empirical performance.

Headless browser checks loaded the actual generated reports without page errors. The policy measurement link reached `#perf-run-0`. The historical Perf view at width 390 had document width 390 (no horizontal page overflow). Screenshots and exact captured CLI output are retained in `.runs/pr-review/`; `browser-evidence.json` and `transcripts.json` record observations. `base-ui.png` renders the base's HTML/JS assets against the same current controlled report payload for UI comparison; it is not a base execution or historical performance result.

## Feedback and remaining judgment

Jack Heart pasted the historical report, including its full status/metric/evidence inventory, and said: “this is not really evaluable.” This is negative demo feedback, not acceptance. The agent interprets the issue as both poor prioritization in the presentation and missing comparative evidence. After the agent described the revised conclusion-first demo and the remaining comparison gap, Jack Heart replied “ok.” This acknowledges the revision; it does not select product implementation, a merge, or paid execution.

Agent-observed wording issue: the historical `llm-many-game` declaration at `experiments/inference_baselines/mtg185-l4.json:8750` says upstream integration remains pending, despite MTG-184/185 now being integrated. The page faithfully prints that frozen reason. Proposed clarification: distinguish declaration-time reason from current dependency context without changing historical evidence. This is a proposal, not accepted direction or an implementation change.

The revised `scratch/demo.html` proposes a conclusion-first summary: game completion and latency, which decisions cannot be made, grouped startup/memory observations, then detailed evidence in disclosures. It explicitly says no CPU/GPU, many-game or cost-effectiveness comparison is supported. The product report remains unchanged. Recommended next action: review this proposed hierarchy and choose whether to implement it or shape the missing measurement plan first. The implementation walkthrough explains source ownership. Task acceptance still needs checkpoint access/current-world admission, trained-policy CPU/GPU and LLM single/many-game observations, and raw-artifact publication. No new paid execution is authorized. This demo makes no merge decision and starts no Graph.

## Proposed implementation direction — not yet accepted

If the revised demo is accepted, change the existing Perf renderer to lead with the run-level conclusion, a few measured outcomes and the comparisons they support. Group missing observations by the decision they prevent. Put raw field inventories, precise timing contracts, identity and corpus links under progressive disclosure; preserve all existing validation, source links and attempt/observation semantics. Show the frozen declaration reason as historical context rather than a current dependency claim. Keep this presentation work separate from obtaining empirical comparison data.

The local walkthrough and demo were checked at 1280px and 390px, in light and dark themes; no page errors or document overflow were found. Keyboard-operable navigation/disclosures and local note persistence were exercised. All code excerpts were rechecked against their pinned Git revisions.

## Accepted empirical direction — 2026-10-26

Jack Heart subsequently requested 100 reasoning-enabled Qwen games against
MTG-183's final checkpoint, measuring LLM performance and breaking down both
models' inference time versus simulation. He explicitly allowed $10 total on
RunPod as necessary. Current protocol is `experiments/protocols/qwen-mtg183-188.md`;
it supersedes the earlier no-paid-run condition for this campaign. Exact final
u5950 checkpoint bytes were fetched and verified; all 100 authored deck pairs
passed checkpoint admission locally. A two-decision native fixture passed
binding/replay with that checkpoint; it is not an empirical campaign result.

### Campaign launched

Etude run `b6a4716bb321`, one RunPod machine `mtg188-a6000-af187e461b9c`,
source `6c7ca1ca71a0`, started 2026-10-27 05:01 UTC. Quote maximum $6.64 at
$0.59/hour, with job/cohort cap $9.50 and $0.50 ancillary reserve under the
user's $10 cap. The installed launcher's storage-session limit reduced the
prelaunch protocol to 11 hours total job / 10 hours executable campaign.
Independent machine guardian confirmed active. No rerun or second rental.
Receipts and status are under `.runs/qwen-mtg183/`; the scoped observer advances
only `mtg188-a6000`, because this SDK's general `status()` also schedules unrelated
queued jobs. Existing other rentals and jobs were left alone.

Runtime validation: 48 LLM tests and 76 receipt/execution/native tests passed.
The exact pinned tokenizer was checked offline for thinking prefix and the
single closing delimiter token. The two-decision checkpoint fixture is separate
from the paid result. Report validation: 54 receipt tests and the actual
mixed CLI/headless proof passed; the new matchup card passed desktop/mobile
controlled-fixture checks without overflow or page errors. These checks are
software evidence, not the 100 requested games.

### First paid attempt stopped; execution choice pending

Run `b6a4716bb321` delivered its receipts and ended. The execution command
succeeded in retaining the failure; **the evaluation did not succeed**:
0/100 completed, one failed game (schedule index 96), 99 unattempted. It made
three committed, clone-replayed Commands, two CPU policy decisions and two Qwen
selection requests. The first Qwen request used its full 2,048-token reasoning
allowance, then repaired successfully: 101.953110 seconds total. The second used
2,048 tokens and then 942 more during repair before the 120-second selection
supervisor deadline. There are 5,500 retained reasoning tokens, three final-answer
tokens and 5,505 total output tokens across four generation attempts. Never
promote this incomplete game to a win, a completed replay or a throughput result.

Game wall 222.307068 s: Qwen parent wait/dispatch 222.221002 s; CPU policy
inference total 0.051464 s (two calls); native reset plus commands 0.003819 s;
rendering 0.012003 s, replay 0.002508 s, evidence writes 0.010619 s,
policy encoding 0.000581 s, unattributed 0.005072 s. Nested Qwen generation-host
time 221.899093 s is not added again. Qwen readiness 9.828495 s and CPU policy
load 0.047354 s are outside games. The optional optimized Qwen kernels were
unavailable; the runtime explicitly reported its Torch fallback.

Provider absence was verified at 2026-10-27 05:12:05 UTC for pod
`0hyjle6zz0nfhn`. Etude confirms `rental_ended=true`, estimated charge
$0.06238391746666666459; final inclusive billing remains unavailable. Raw
outputs were fetched with digest verification and are retained at
`.runs/qwen-mtg183/result/`, including the full first-game archive and all
selection receipts. The actual builder produced
`.runs/qwen-mtg183/report/report.html`; the report passed desktop/mobile
browser checks with no page errors or overflow. It accurately shows one failed
and 99 unsummarized/unattempted games, and zero completed outcomes.

The agent asked in the current conversation: use a faster batched inference
backend (recommended), or keep Transformers with a smaller explicit reasoning
budget? This is a pending execution-direction choice, not another request for
the already-authorized $10. No additional rental is launched while it is pending.
MTG-191 is the existing unstarted optimized-backend Task (no checkout), rather
than new duplicate work. Any next allocation must deduct the first attempt and
preserve an ancillary reserve from the same original $10; $9.40 plus a $0.50
reserve is below the remaining allowance. Do not reuse this attempt's result
identity for a changed backend or reasoning budget.

Jack Heart selected **“Use a faster batched backend (recommended)”** in this
conversation. Proceed with a supported batched backend for the same pinned Qwen
BF16 model and final MTG-183 opponent; reasoning remains enabled. The $10 is the
original cumulative ceiling, not a fresh allowance. The previous pending choice
is resolved. Retain the Transformers failure and keep new backend results separate.

### Batched continuation preparation — 2026-10-26

Source `4db6f6f30` freezes the vLLM 0.17.1 server, exact-token client, four-game
coordinator and revised protocol. It retains admissions before submitting games
and archives only settled game owners, preserving late startup receipts. The
sequential Etude entry point keeps its required one-config signature. Controlled
validation: 113 LLM/report tests, six package-boundary tests, eight final focused
backend tests, 679 tests collected in the exact main integration group, mixed
CLI/headless Perf proof, Ruff and diff checks passed. No GPU fit or speed claim
follows from these checks.

The current shell lacks provider/storage credentials; the existing documented
Doppler `manabot/prd` and `etude/prd` configuration supplies them without account
or policy changes. The successful read-only quote is $0.59/hour, at most $6.64,
under the $9.40 continuation job/cohort cap. A single launch command is in progress
for cohort `mtg188-qwen-vllm`; inspect `.runs/qwen-mtg183/vllm-launch.txt` and the
existing process before any other launch. Never repeat a pending launch.

All associated Loopflow Graphs were observed completed/stopped; only this
conversation is active in the checkout. Automatic Task checks are disabled and
the CI watcher is not running. The Task remains open; PR #381 is unmerged.

The launch succeeded: Etude run **`594e466c66c1`**, machine
**`mtg188-a6000-8417b7bc9bda`**, source **`4db6f6f3073d`**. The first scoped
observation reports phase `running`, with dependency installation in retained
logs and no completed games yet. Do not launch again. The observer at
`.runs/qwen-mtg183/vllm/observe.py` advances only the existing `mtg188-a6000`
supply and retains status, machine and logs under that directory. Invoke it
through the two existing Doppler projects. The SDK's global status call can
schedule unrelated work and must not be used for this campaign.

### Batched attempt failed before model startup; cache repair verified

Run `594e466c66c1` ended failed before admitting any games: 0 attempted,
0 completed, all 100 unattempted. Model acquisition/shard verification and
dependency installation succeeded. The new server's unfiltered offline
`snapshot_download` requested `.gitattributes`, `LICENSE` and `README.md`, which
the existing cache preparation deliberately excludes. Hugging Face Hub 1.33's
completeness check rejected that mismatch. Neither vLLM model startup nor GPU
fit was tested by this attempt.

Cleanup finished and provider absence for pod `mk7yh37t6pr0sy` was verified at
2026-10-27 05:46:21 UTC. Etude reports `rental_ended=true`, estimated charge
$0.02901644665277777951. Both attempts total approximately **$0.09140036**;
final inclusive billing is unavailable. All eight output artifacts were fetched
with SDK verification into `.runs/qwen-mtg183/vllm/result/`; SDK records, logs
and provider-absence receipt are retained beside them. No rental remains active.

The repair gives preparation and both runtime backends one owner for their
model/tokenizer file patterns. Regression tests call the real Hub offline
snapshot resolver with a controlled cached repository listing: missing docs
are allowed, missing required weights/tokenizer files still fail. LLM and
package-boundary suites passed **66 tests**; Ruff/format and diff checks passed.
No second batched allocation has been launched. The frozen continuation protocol
allowed one allocation/attempt; another requires deciding a revised attempt
limit while preserving the original cumulative $10 ceiling and ancillary reserve.

### Retry count restriction removed by Jack Heart

Jack Heart replied “yes retry”, “no cap on retries”, and “just cap on money”.
Continue repaired attempts without asking again about count. The original $10
cumulative cap includes all failures; reserve $0.50 and deduct earlier charges
before each quote. Preserve one active allocation and a separate identity for
each attempt. This supersedes the immediately preceding pending-choice note and
all historical single-attempt restrictions, not the money ceiling. The committed
protocol records the same direction. Current retry begins from cache fix
`fea08def9`; no new GPU result is implied by the local tests.

Retry launched once as Etude **`25ce352f9533`**, cohort
`mtg188-qwen-vllm-cache-fix`, machine **`mtg188-a6000-c33cbe696bbb`**, source
**`075f2a0e7b41`**. Both earlier pods were freshly verified absent before launch.
Quote remains $0.59/hour, at most $6.64; job/cohort cap $9.40 with cumulative
prior estimates plus $0.50 reserve below $10. Receipts and scoped observer are
under `.runs/qwen-mtg183/vllm-retry/`. This run is active; inspect it before
any new launch. Retry count is unrestricted by Jack Heart's latest direction,
but each allocation must fit the remaining original money ceiling.

Run `25ce352f9533` ended failed before any game. It passed cache acquisition but
the server command resolved its venv interpreter symlink into system Python.
Preserving the absolute venv path passes a real uv-created-environment regression
test. Probe stderr is now retained, and server readiness/game admissions are
printed to the ordinary job log. All outputs were fetched; pod `ixv39jp7gfsisd`
was confirmed absent at 2026-10-27 05:55:25 UTC. Estimated charge
$0.02914788226388888835; all three attempts total $0.12054824638333333245.
Next cap is $9.37 plus the unchanged $0.50 reserve. Jack Heart authorized
unlimited repaired retries within the original money cap; no new permission
is required for this repair.

The interpreter repair passed all 11 backend tests and was committed as
`b25cc1fc279c`. New run **`dcd5044dd2e0`**, cohort `mtg188-qwen-vllm-env-fix`,
machine **`mtg188-a6000-85ecd963eb6f`** launched once with job/cohort cap $9.37.
Its quote was $0.59/hour, at most $6.64. Receipts/observer are under
`.runs/qwen-mtg183/vllm-env-fix/`. Inspect this active allocation before retrying.

Run `dcd5044dd2e0` ended: vLLM loaded and reached health in 79.136809 s,
but all four admitted games failed before play due to a missing optional
`generation_config.json`. The client now follows config-derived EOS defaults;
the exact cached Qwen metadata admits offline with EOS IDs 248044 and 248046.
All 12 backend tests pass. The remaining 96 games were unattempted. SDK outcome
`succeeded` refers to retaining these failures, not successful evaluation.
Pod `pu8kpfq85fgp4n` was verified absent at 2026-10-27 06:03:11 UTC. All outputs
and archives are retained under `vllm-env-fix/result`. Estimated cost
$0.03861355726111111192; cumulative four-attempt estimates $0.15916180364444444437.
Next cap $9.34 plus the original $0.50 reserve stays below $10.

The config fallback repair is committed as `7c8ee50deacf`. All 12 backend
tests passed, and a real spawned `InferenceSession` using the exact cached Qwen
metadata and retained vLLM identity became ready locally (no model execution).
New run **`12f050b52057`**, cohort `mtg188-qwen-vllm-generation-fix`, machine
**`mtg188-a6000-3288b2bebad2`** launched once with job/cohort cap $9.34. Quote:
$0.59/hour, maximum $6.64. Receipts/observer live in
`.runs/qwen-mtg183/vllm-generation-fix/`. Inspect this allocation before any
further retry; the original cumulative $10 ceiling remains authoritative.

Run `12f050b52057` ended with four failed games and 96 unattempted. Game 11
reached 23 native decisions and six successful Qwen selections; its seventh
timed out. The other three timed out on their first selection. Retained raw
selections show warm full 2,048-token generations take about 60.7 seconds,
which leaves too little of the 120-second deadline for a full repair. Server
logs show approximately 129 aggregate tokens/s at four requests and 34 at one.
The first requests include additional startup work. No completed-game throughput
or backend speedup follows. All outputs/archives are fetched and extracted under
`vllm-generation-fix/result`; provider absence was confirmed at
2026-10-27 06:23:40 UTC. Cost estimate $0.13343769908333333976; cumulative five
attempt estimates $0.29259950272777778413.

Next treatment: 16 actual scheduled games concurrently, 300-second selection
deadline including repair, 7,200-second between-decision game limit. Original
model/tokenizer/opponent/schedule/decoding and 2,048 tokens per generation stay
fixed; effective inference settings are retained before admission. No inference
workload was added merely to induce load. This uses the accepted batched direction
and money-only retry limit. Next job/cohort ceiling $9.20 plus $0.50 reserve.

The revised configuration passed 13 backend tests, including controlled 16-game
failure/admission accounting. Source **`9cd68fff2fbc`** launched once as run
**`b1d7ed75492f`**, cohort `mtg188-qwen-vllm-batch16`, machine
**`mtg188-a6000-7edf30fa9e1d`**, job/cohort cap $9.20. Quote $0.59/hour,
maximum $6.64. All previous pods were confirmed absent. Current receipts and
scoped observer: `.runs/qwen-mtg183/vllm-batch16/`. This is the active allocation;
inspect it before any retry. No game result is implied by its launch.

Latest observation: `b1d7ed75492f` is running, guardian active, server healthy
after 88.267 seconds, all 16 scheduled game owners admitted, no failed/completed
game yet in the observed log. Actual progress includes `decisions=10 model=3
seconds=268.9`. The current run estimate is $0.09977119201388888680; cumulative
estimate with the five ended attempts is approximately $0.39237069. The derived
budget ledger at `.runs/qwen-mtg183/budget.json` references every prior absence/
charge receipt and reserves at most $9.99259950 including this job's $9.20 cap
and the $0.50 ancillary reserve. `budget.py` only projects those receipts and
never allocates. Final billing is unavailable.

The remaining dependency is this active measurement. Leave it running; the
remote worker owns its game/campaign limits and rental cleanup. This conversation
has no scheduled follow-up: on return, run the scoped observer, inspect new results
and cleanup, then repair/retry as needed within the original cumulative money
cap. Jack Heart removed retry-count limits; do not ask again about count. Task
automation is disabled and the CI watcher is not running. MTG-188 remains open.

### Batch-16 outcome and accepted 16k continuation — 2026-10-27

Run `b1d7ed75492f` has ended. All 26 admitted games settled: 18 completed
and replay-verified (Qwen 0 wins, 18 losses), eight hit the two-hour game wall
limit, 74 unattempted. All recorded outputs were fetched with SDK digest
verification, and 26 archives extracted under `vllm-batch16/result`. The actual
report CLI rebuilt `vllm-batch16/report/report.html`: 793 Qwen selections,
2,556 committed Commands, 18 replay-verified games, failed overall campaign.

The critical finding is output truncation: 519/793 Qwen selections used seeded
random fallback after both attempts exhausted the 2,048-token allowance. Another
241 succeeded after repair and 33 initially. Of 1,553 generations, 1,279 ended
at the token limit; none of the fallback pairs was a malformed completed answer.
Median complete selection time was 195.609 s, versus 0.026546 s for 1,763 CPU
policy decisions. Across overlapping game windows, Qwen wait/dispatch totaled
144,187.217 s, policy inference 49.030 s, native reset/commands 12.597 s. These
worker sums are not campaign wall: server preparation through cleanup took
13,664.353 s. Do not interpret the Qwen losses as unrestricted model strength.

Fresh SDK settlement reports all six rentals with `estimated=false`, totaling
$2.253210135812151860498486614. `charges-current.json` retains each charge; the
latest pod `nb4r2gk2o7uqd2` was confirmed absent in `vllm-batch16/rental-ended.json`.
Earlier estimates remain historical. Final ancillary billing remains unavailable.

Jack Heart selected increasing reasoning allowance, then said "lets do 16k
reasoning tokens allowed". The next generation allowance is 16,384 tokens,
including reasoning/final answer/EOS under the existing runtime. The committed
protocol describes 65,536 context tokens, eight concurrent games, 1,800-second
selection deadline including one repair, and game wall bounded by the existing
10-hour campaign watchdog rather than a separate two-hour cutoff. Start the
original schedule under a new treatment identity; do not pool previous results.
No changes to model/opponent/world/decks/deals/greedy decoding or legal semantics.
The proposed next job/cohort cap is $7.24, preserving the $0.50 reserve under
the original $10: maximum cumulative commitment $9.99321013581215186050.
No allocation has yet been launched for this treatment. Retry count remains
unlimited within the original money cap.

Fresh Loopflow task/graph reads fail with `missing field interactive`; Sessions
and `lf ps` succeed and show only this conversation in the checkout, no live
Graph. No new Graph is launched. The installed CLI no longer recognizes
`lf task automation` or `lf ci watch --status`, so their current installation
status is unknown (prior receipts said disabled/not running). Direct experiment
work continues under the existing accepted workaround.

The 16k treatment passed 15 backend tests, Ruff and diff checks; source
`6ea6e0bef45d` was committed through `lf commit`. All six preceding pods were
freshly verified absent; quote maximum $6.64 at $0.59/hour. Launched once as
Etude run **`7b8c16de1b60`**, cohort `mtg188-qwen-vllm-16k`, machine
**`mtg188-a6000-ad69ea4f896c`**, job/cohort cap $7.24. Initial phase is
`setting_up`. Receipt, exact run ID and scoped observer are retained under
`.runs/qwen-mtg183/vllm-16k/`. Do not launch another allocation while this exists.
The observer now reads the exact run ID, so terminal runs cannot disappear due
to the SDK's default unfinished-only cohort list.

Latest scoped observation: run `7b8c16de1b60` is `running`, guardian
confirmed ready with its independent deadline. Locked client dependencies
installed; no shared server health or game admission is present in observed logs
yet. Current run estimate $0.03017908098611110929709866469; cumulative exact prior rental charges
plus this estimate are retained in `.runs/qwen-mtg183/budget.json`. The serialized
Etude job was inspected: 16,384 generated tokens, 65,536 context, concurrency 8,
1,800-second selection limit, 36,000-second campaign/game limits, cap $7.24.
This remote execution is the remaining measurement dependency. Leave it alone
while active; no automatic turn is scheduled here. Task/Graph reads remain
unavailable due to the Loopflow serialization error; no new Graph was started.

Status return at 2026-10-27T12:21:03.659193Z: scoped read confirms `7b8c16de1b60`
is still running. Shared vLLM health arrived after 79.908 s; eight games
(96, 11, 15, 53, 65, 27, 8, 28) are admitted, with no completed/failed game
in the refreshed job log. Logs were received without retrieval errors. No
recorded outputs are yet available via the SDK, so current decision counts,
latency and fallback rate remain unavailable; do not extrapolate improvement
from the larger configured allowance. Current run estimate $0.2432067483888888735539997772; total
prior finalized rental charges plus this estimate $2.496416884201040734052486391. Original $10 cap
and independent guardian remain. Leave the active measurement alone. Fresh
Task status still fails `missing field interactive`; session/process reads show
only this conversation, no live Graph.

Status return at 2026-10-27T13:51:39.909975Z: run `7b8c16de1b60` remains running.
Refreshed logs now show continued in-game progress, including `decisions=30
model=9 seconds=5628.2` and `decisions=20 model=11 seconds=6291.3`. These
anonymous log lines are not independent game totals and must not be summed.
No completed/failed game is reported; all eight original admissions remain
in progress by the observed log. The 16k treatment is advancing slowly, with
one reported game window exceeding 104 minutes. No fallback-rate conclusion
is supported by these counters. Current rental estimate $0.5853927391106656653802783003; cumulative
prior settled charges plus this estimate $2.838602874922817525878764914. Leave
the active run alone within the original $10 cap; no retry is needed now.

The latest `lf task status MTG-188 --json` read succeeded again. It retained
the Task identity and existing PR #381, and still describes the historical
implementation Session settlement failure (`database is locked`). Existing
implementation and paid experiment effects remain preserved; this is not a
reason to replay the Graph or compete with the active measurement.

### Throughput diagnosis discussion — 2026-10-27

Jack Heart called the progress problematic, asked why Qwen is slow, whether
inference code needs updating, and whether a dedicated LLM setup should batch
across many games. These are diagnosis/architecture questions, not acceptance
of a new sampling treatment or permission to change the live run's source.
The current setup already has a resident shared vLLM process: eight concurrent
games in the 16k treatment, sixteen in the earlier 2k treatment. Keep the
active allocation; do not create a competing server or another rental.

Retained 2k receipts show 144,128.093 generation-host seconds out of
144,305.160 summed overlapping game seconds (99.8773%); those are worker
intervals, not additive GPU time or campaign throughput. Total output was
2,872,732 tokens. The earlier four-game run measured ~60.7 s per full 2,048-token
generation; extrapolating that rate gives ~8 min for 16k and potentially twice
that with repair, not a measured current latency. Larger output budgets do
not address poor termination or unnecessary repairs.

The client hardcodes greedy sampling (temperature 0, top_p 1, top_k -1,
presence penalty 0). Qwen's current official model card recommends general
thinking temperature 1, top_p .95, top_k 20, presence penalty 1.5:
https://huggingface.co/Qwen/Qwen3.5-9B#best-practices
This mismatch is a concrete configuration issue to investigate, not proof that
sampling alone fixes runtime. Read-only repetition diagnostic on all 1,553
retained generations: duplicate 16-token-window fraction median .0226, p95
.4530; 65 responses exceeded .5. This does not establish that every cutoff is
a repetition loop. No raw reasoning or hidden game observations were printed.
The old server log shows FLASH_ATTN, compilation and CUDA graphs, with no
preemption or fallback warning found. Current server's detailed log is not yet
published. No claim of optimal GPU utilization follows.

Recommended architecture retains one GPU inference owner and many game
clients. Needed measurements are decode throughput, request latency, useful
legal decisions, repair/fallback fractions, queue/cache occupancy and per-game
progress with identities, published while work runs. Tune actual game concurrency
against these signals; increasing it improves potential throughput, not single-
generation latency. Changing decoding is a separately declared treatment; keep
model/opponent/world pins, 16k allowance, every attempt and the original $10 cap.

### Accepted split and Qwen cancellation — 2026-10-27

Jack Heart requested a Qwen follow-up and refocused MTG-188 on non-LLM model
evaluation time. The existing unstarted MTG-191 was refined rather than filing
a duplicate; both Task title/description updates appeared in fresh status reads
after initially pending planning-provider confirmation. Self-contained briefs
and source/evidence handoff are committed at 3d9bfedc6. No MTG-191 worker
was launched. The Wave remains intelligence in loopflowstudio/manabot.

Run 7b8c16de1b60 was cancelled, SDK outcome cancelled/rental_ended true.
Pod crijtcbtuulx7w was confirmed absent at 2026-10-27T14:03:49.843896Z.
SDK finalized rental charge $0.66449568431326774258; all seven rentals total
$2.91770582012541960308, with ancillary final billing still unavailable.
All eight partial game archives and other recorded outputs were fetched with
SDK verification except vllm.log: repeated terminal fetch fails with
StorageUnavailable, invalid download range or identity (status 206). Retain
this publication gap rather than bypass verification. All eight games remain
interrupted, with no terminal summaries. Last committed decisions show 27
initial selections, 16 repairs, 35 fallbacks, 122 CPU policy moves. These 78
Qwen selections and 200 committed native decisions are partial observations,
not eight finished games or an unbiased comparison to the previous treatment.

Non-LLM CPU protocol .runs/policy-baseline/protocol.json freezes admitted
MTG-183 u5950, Elemental UR/GW, heuristic opponent, seeds 18801-18804, scalar/
resident/shared-four compute and three rotated repetitions. Single workload
uses one leg/deal; many uses four deals and both seats. Five cells/repeat, 78
full-game opportunities, 120s/game, 300s/cell, 1800s total, one Torch thread
per worker and at most four workers. Development host is not isolated; no
artificial pressure is induced. The existing bounded runner is now executing
into .runs/policy-baseline/games; inspect run.log/ledger before another run.
No new paid resource was started. CUDA comparison remains unavailable locally.
