# Inherited notes

Current latency scope and remaining work are in `mirror-rollout-evidence-plan.md`.
The dated choices below are historical; earlier launch/review boundaries do not
reopen the approved study or override the current no-games/no-mini slice.

## MTG-269 analysis choices — 2026-11-09

- Keep the owning v1 report/summary paths and add a distinct extension section.
  V2 snapshot layout uses `games/` for runner output and `reconciliation/` for
  dated receipts. Missing paths without receipts are errors, not zero games.
- Use each contrast's complete whole-deal vectors for its nominal and stratified
  interval, retaining all planned slots in missing-evidence sensitivity. This
  applies the existing uncertainty method to four-cell interactions; it does
  not change the schedule, decision rule or launch authority.

## MTG-269 execution choices — 2026-11-09

- Latest schedule: fixed beta 0.01, depth 20, oracle/uniform × 4/8/16 worlds,
  raw sampled policy opponent only, 500 matched deals per cell (3,000 total).
  Earlier 5,000- and 2,500-game schedules were superseded before any new games.
- Keep the existing SDK/game environment frozen. Select the already installed
  schema-13 Etude source only for launcher reads; no DB migration or reset.
- An unresolved allocation for other work is not free capacity even when SSH
  shows no game process. Preserve it and stop execution pending its owner's
  reconciliation. Do not create a second pool/machine name or launch around it.
- For the inclusive seven-day ceiling, use this step's 18:16:29 UTC start on
  November 9, ending November 16 at 18:16:29 UTC (1794852989). Share this absolute
  deadline between smoke and campaign. Delay does not replenish the allowance.
- Store full private search/Command evidence in per-game compressed attempt
  streams. Preserve incomplete attempts and require explicit reconciliation;
  do not call them losses or automatically replay them. Keep v1 evidence intact.

MTG-174/MTG-176 preflight, payment-repair and recovery assumptions are retained at
`003644bda6d3ed2e1e6754186960b135ff815e76:scratch/{questions,elemental-baseline-preflight,elemental-mana-repair}.md`.
They do not select current work or authorize paid execution.

# MTG-269 design assumptions (2026-11-08)

Execution reconciliation (2026-11-09): use the established analysis environment
at `/Users/jack/src/manabot.search` through `uv run --project ... --no-sync`,
importing source from this frozen checkout. Publish final game-file pointers in
the existing store and preserve prior content hashes in a reconciliation receipt.
Leave the 4,899-game HTML/summary intact until the following analysis Graph step
can make the necessary report corrections together. No environment upgrade,
engine rebuild, launcher DB migration or new execution is needed.

- The existing runner is the retained campaign's authority for settings and
  schedule; `experiments/protocols/search-mirror-v1.md` reconstructs it without
  retroactive preregistration. The original source copy lacks a complete launch
  inventory, so current disk hashes cannot prove what every worker loaded.
- Simpler proposed follow-up: reuse the 500 existing deals per opponent and
  existing controls, label selection exploratory, add only beta 0.01 at 8/16
  oracle shuffles and uniform comparisons at those settings and the four-world
  reference. The 5,000-game / 96-hour / 672-worker-hour / $0 ceilings are design
  choices for review, not numbers supplied or approved by Jack Heart.
- No new training seed or fitting: this is one frozen trained policy. New
  shuffle samples and noise resamples cannot substitute for independent learners.
- The current directive's interactive-review boundary takes precedence over
  interpreting the broad continuation request as authority to launch follow-ups.
  Leave the six live mini workers alone; worker 2 has completed its assigned keys.
- MTG-268 is idle, not still implementing. Its recorded unmet latency target
  and unresolved w13 admission remain dependency limits; this step does not
  restart it or choose a relaxed performance acceptance threshold.

Inherited MTG-183 training assumptions are retained at `7fd6c202f163bc7dd0d507c7eab121086af84a68:scratch/questions.md`.

# MTG-269 analysis assumptions (2026-11-09)

For the live v2 analysis, take a bounded dated snapshot of committed game-file
prefixes and their sealed attempts. Leave the existing Etude job running; do not
wait for the multi-day campaign or recopy private in-flight streams into completed
evidence. Per-shard cutoffs are explicit rather than claiming an atomic snapshot.

Use seed 26901 for whole-source-deal noise bootstrap as well as the protocol’s
stratified game bootstrap; only the game bootstrap seed was prescribed. Keep
noise position weighting unchanged, resampling source deals jointly across
opponents and retaining all their positions/draws. PUCT comparison families use
their own common-complete subsets across all eight mirror settings so missing
PUCT evidence does not discard complete sampled-policy comparisons.

## Mirror mini execution — 2026-11-09

Direction: “ok so lets use it” and “clear the curriculum stuff”. The canceled
`curriculum-held-out-707e62cad5fe-r3` had no remote boot claim, receipt, or worker.
Etude's SSH bootstrap returned no end receipt for that absent startup, while
its scheduler waited indefinitely for a creation receipt. A process-local
transport correction lets the existing DirectoryJournal close a session without
both receipt and boot claim; sessions with a boot claim retain uncertainty.
Under Etude's mini lock, its `_end_session` recorded shutdown and cancellation.
No database rows were deleted, environment upgraded, or other pool operated.
The exact repair and retained diagnostics are in the v2 reconciliation receipts.

Reversible operational choices: stage committed source `0063d781` in mini
`~/src/manabot.w13-mirror-v2`, preserve the original source directory, reuse its
frozen environment and exact native bytes. Smoke means six separate smoke-key
games (one per factorial cell), three one-thread workers, oracle then uniform,
at most two hours inside the unchanged absolute campaign deadline. No strength
claim follows from that tiny admission. Campaign stays seven workers/3,000 games.

## W13 batching admission — 2026-11-09

Jack Heart: “thats pretty slow” and “you might want to adopt the work from mtg-268”.
The explicit continuation is bounded by the same 3,000 slots and deadline.
Reversible admission choice: compare three roots (steps 0/100/286) reconstructed
from the already sealed oracle-four-world smoke, both physical samplers and all
4/8/16 allocations, with MTG-268's unchanged atol 1e-6 / rtol 1e-5. These are
fixed-position checks on the laptop, not new games or a campaign speed claim.
Retain all diagnostics, including harness errors, in the existing artifact store.

## Fixed-root latency baseline — 2026-11-10

The current request is a measurement of the existing batched checkout, not a
new sequential-versus-batched comparison or implementation. Use the retained
sealed smoke f38ccc9074d39356… and seeds at steps 0/100/286, depth 20, alpha
0.02, beta 0.01, 16 worlds, CPU with one torch intra-op thread as the runner
uses. Warm each cell once, alternate oracle/uniform for five measured calls,
and profile separate calls. Keep the original profiling script intact.
Store local diagnostic receipts under scratch; no campaign source, game,
allocation, or scientific report changes. The heavily loaded laptop limits
absolute latency and any projection to mini. Next work is a plan for flags,
not implementing them during this baseline.

## Compact rollout implementation choices — 2026-11-10

Use a single opt-in `rollout_evidence="compact"` flag for both approved omissions;
keep `full` as the default and emit v4 only for compact. The retained alternating
single-Command audit measurement already isolates the audit switch, so there is
no additional publicly exposed or mislabeled audit-only receipt mode. Full v3
receipts omit the new config field to preserve prior bytes; registration accepts
an omitted field as historical full evidence without rewriting the stored spec.
The fixed-position comparison captures actual native Commands and both final
viewer observations externally during untimed admission. No trained-policy game,
campaign transition, remote mutation, engine rebuild or model change is included.

## Inference thread measurement — 2026-11-10

Choose threads only at fixed-root process startup; default remains one, inter-op
unchanged. Leave the campaign runner's one-thread/full-evidence defaults and
remote source untouched. Compact-only measurement requires a completed one-thread
control with matching runtime/source/config and actual Command/final-observation
captures. Alternate separate 1/2/4/8-thread processes across five rounds; each
round supplies fresh one-thread timings. Historical control timings alone cannot
establish a gain on this contended host. The sweep is prepared for measure; no
new games or sweep execution is part of this implementation check.

## Thread measurement choices — 2026-11-10

Use the prepared compact harness without edits and the specified retained
one-thread behavior; compare fresh startup-thread settings on that same source.
After the frozen five-round sweep, profile all four counts sequentially with
one companion timing each. Those 24 extra timings stay diagnostic and are not
added to the five-repeat sweep medians. Preserve every load spike and use
nonoverlapping ranges to call improvement. Leave existing source changes and
production defaults intact; the current request ends with a measured decision,
not implementation or deployment of the next encoding hypothesis.

## Encoding iteration 3 choices — 2026-11-10

Use a small golden-tensor/Command fixture extracted from every sealed smoke actor
view instead of retaining a second Python encoder in tests. Capture before edits,
check each pending frame/observation against the original attempt, and include the
terminal view. Measurement selects archived control/candidate encoders only in
isolated diagnostic processes; the production tree has one encoding path. Keep
all existing compact and thread changes as the control, rather than reverting
them to HEAD's full-evidence search. This isolates the requested encoder change.

## Native encoding iteration 4 — 2026-11-11

Use an explicit teacher constructor opt-in and process-start extension selector;
keep campaign defaults and binary unchanged. Replay terminal state projections through an explicit viewer; the sealed terminal
Command snapshot additionally uses the existing native snapshot encoder to retain
its transition events. Live search always uses the native acting player.
The initial implementation used the VectorEnv buffer layout including unused
step-state arrays; compression removed those arrays from observation-only calls.
Root .venv was absent: build with UV_NO_SYNC=1 after installing maturin into a
fresh Python 3.12 venv; execute checks in the retained search environment.

## Native thread sweep iteration 5 — 2026-11-11

Reuse the unchanged compact measurement harness and iteration-4 native binary.
For all 2/3/4-thread processes, admit against the original one-thread compact
Commands/final observations with explicit runtime-identity projection; fresh
two-thread samples, not historical timings, are the latency controls. Alternate
2/3/4 and 4/3/2 over five rounds; each count gets a separate startup process.
Use the retained uv search environment, CPU, inter-op 16 and warm-cache policy.
Run one separate profile process per count across all six roots; its companion
timings remain excluded. Require both step-100 samplers' five samples below 1 s
to advance the fixed-root target; report median-only success separately if any
sample crosses it. Keep every host-load spike. No campaign changes or games.
Expected claim: "Three or four startup intra-op threads bring both step-100
samplers below one second without changing Commands or final observations."
This is a configuration sweep of the last kept uncommitted implementation,
not a source change against HEAD's older full-evidence implementation. No rebuild
or production thread-default change is needed or included.

## Final local measurement — 2026-11-11

The requested fresh full/Python controls and compact/native candidates use the
same separately selected post-compression binary (`3f086259…`) to isolate the
flags. Historical full and compact receipts remain behavioral references, not
timing controls. This simpler reversible measurement choice leaves the frozen
binary untouched. Five rounds alternate both mode order and 1/3-thread process
order. Equal-root latency sums describe this fixed workload only; campaign root
frequencies and seven-worker M1 contention are not inferred.

## Uniform implementation transition — 2026-11-11

Jack Heart selected “Switch at oracle barrier (Recommended)”: finish oracle,
then use compact evidence and the native batch encoder with one thread per
worker, a separate implementation identity, and the existing deadline.

Use the six retained fixed roots and the 304-view golden tensor replay as the
short native smoke on the mini. These use the actual checkpoint and observed
Commands without adding scientific game slots. Run this before stopping the
original Etude job so a failed admission leaves its uniform path available.
The original shell has no barrier pause: if uniform starts before cancellation
settles, preserve completed outcomes and explicit interrupted attempts. Resume
only missing keys in the new segment; never replay a scored failure. A truncated
or unsealed attempt blocks the transition rather than being called a failure.

## Artifact reconciliation — 2026-11-11

Hydrate the published 17.9 GB snapshot with four disjoint digest batches through
`manabot.data.fetch_data`; this changes local transfer concurrency, not mini game
workers or experimental bounds. A later collector may reuse a prior published
seal only after verifying its inventory hash, identical ledger prefix and exact
trace bytes on both mini paths. Newly completed attempts still receive the full
start/outcome scan. Keep the previous inventory digest in the new receipt.

Near the oracle barrier, shorten only the foreground observer's delay from
30 seconds to one second (plus each read-only SSH probe). Stop and reap the old
observer first. This reduces the original automatic uniform stage's cancellation
window; it changes no game source, schedule, worker assignment or deadline.

## Confirmed cancellation with missing callbacks — 2026-11-11

The original Etude attempt stopped after all 1,500 oracle outcomes. Seven uniform
attempts began; three wrote interruption footers and four ended at readable
pending decisions. No uniform outcome or failure event exists. Worker 0's log
shows repeated SIGTERM interrupting traceback formatting and gzip cleanup; the
other missing-callback logs are partial. Preserve all seven raw streams unchanged.

Treat the four missing callbacks as externally reconciled interruptions only
through an explicit receipt binding the actual Etude cancellation, exact trace
hashes and all seven worker logs. Reject corrupt/truncated data, failures,
uncommitted outcomes, changed hashes, traces outside the attempt's time window
and unused recovery entries. This uses the authorized missing-key continuation;
it does not synthesize a footer/score or replay a scored game. Make the runner's
stop handler idempotent to prevent the reproduced cleanup defect from recurring.

## Retained-snapshot analysis — 2026-11-12

Use MTG-269's existing mirror report and the November 11 sealed snapshots;
the MTG-245-specific reader instruction does not select a different experiment.
No new collection, execution or replay is needed for this report clarification.
The previously named search environment no longer contains its dependencies;
use an ephemeral uv overlay with retained numpy 2.4.2, pydantic 2.12.5,
torch 2.10.0 and etude-sdk 0.2.6 for offline analysis. Do not sync or rebuild
the frozen checkout or change the remote execution environment.
