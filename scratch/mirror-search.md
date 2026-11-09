# Mirror search continuation

## Extension analysis handoff — 2026-11-09

The same report builder now takes `--extension experiments/data/search-mirror-v2`
and updates the existing v1 `report.html` / `summary.json`; no v2 summary store.
All v1 numerical results reproduce unchanged. Six extension cells and all 15
planned contrasts retain their denominators and unavailable estimates. The
dated execution receipts still show no launch; analysis started no games or
smoke and did not alter the allocation or deadline. Scientific acceptance remains
unresolved. Use the existing report for results and the v2 protocol for execution.

18 focused tests and Ruff pass. Broader CI collection finds 75 tests but keeps
the same six Etude/duckdb import errors; no dependency/environment changes.
Updated HTML and summary were published at their existing artifact paths through
Doppler (two manifest entries changed). Desktop/mobile light/dark browser checks
pass with expanded tables and no page overflow; the temporary browser is closed.
All 37 v1/v2 artifacts verify against the manifest; a second rebuild reproduces
both published files byte-for-byte. The post-publication untracked-data check passes.
Capacity reconciliation and native smoke are still execution prerequisites,
not a request for another design review. The absolute deadline is unchanged.

## Execution preparation and capacity block — 2026-11-09

Latest direction supersedes the earlier schedules below: “to be clear the games
shoudl be against raw policy” and “right, that and unerstanding performance across
multiple shuffles (should be more more meaning ful for uniform random)”. V2 is
now 3,000 games: oracle and uniform at beta 0.01, depth 20, worlds 4/8/16, each
against raw sampled policy on the same 500 deals. Oracle stage first. No new games
had started, so no evidence was relabeled. Beta 0.03/four worlds is context only.

Runner changes and focused checks are in this checkout. Fifteen tests pass;
broader CI collection retains the known six Etude/duckdb import errors. Native
smoke has NOT run. New source has NOT been copied to the mini. V1 HTML/summary
reproduce exactly and all seven remote game-file hashes remain unchanged.

Etude 0.2.6 cannot read launcher schema 13. A compatible existing clean checkout,
`/Users/jack/src/etude.ad319b-show-live-progress`, can read it via
`PYTHONPATH=/Users/jack/src/etude.ad319b-show-live-progress/src uv run --project
/Users/jack/src/etude --no-sync etude ...`; do not upgrade the w13 environment or
move/delete the launcher DB. Current inventory retains an unresolved other-work
allocation on mini: `curriculum-held-out-707e62cad5fe-r3`, attempt `ab69a427b4a1`,
session `c24c6c598b42`, phase `renting`. A bounded log refresh leaves its SSH-create
outcome unknown. Read-only capacity quote: “cannot launch on mini: no free
machine; etude queue would wait”. No allocation was bypassed or canceled.

Execution is blocked on Etude reconciliation/free capacity, not design approval
or MTG-268. No runs were queued, no campaign/smoke processes started, no rental.
Read the protocol's execution status and published v2 reconciliation receipts.
Once capacity is available, admit the source-bound mini smoke before the six
fixed cells; retain private attempt streams in the existing store. Use shared
absolute deadline 1794852989 (2026-11-16 18:16:29 UTC), counting setup from this
Graph step. Do not reset it between smoke, stages or retries. The report's
factorial analysis still belongs to the next analysis step; do not feed v2 rows
into the v1-only report CLI without extending its arm validation/comparisons.

## Active extension — 2026-11-09

Current direction: “i want to try 8 and 16 shuffles, and then i want to combine
magic mirror with uniform random vs oracle”. The prospective bounded plan is
`experiments/protocols/search-mirror-v2.md`: first beta 0.01 at depth 20 with
8/16 oracle worlds; then uniform versions of those and the beta 0.03/four-world
reference. Five thousand new games, same deals/opponents, owned mini, no rental.
Prior statements below that the extension is unapproved are superseded.
Keep v1 evidence unchanged. MTG-268 is still idle with no w13 admission; use
the existing sequential implementation rather than blocking this extension
on batching. Implement the minimal runner changes, validate, launch and analyze
without another design-review loop. The protocol records bounds and controls.

## Current direction — continue the retained experiment

The conversation supersedes the design-review stop: “i dont think i need to
review the deisgn, we were kind of half way through an existing experient when
i transferred this to a task” and “make sure to not throw away that work or do
unnecessary . kjust start from what we have”. Resume execution/reconciliation
and analysis of the existing 8,000-game run. Do not redesign or rerun completed
work, block on MTG-268, or require prospective capture upgrades for historical
results. The proposed 5,000-game extension is separate, not approved or required
to finish the original sweep. Preserve all evidence and report its limitations.

## Analysis handoff — 2026-11-09

Final analysis now updates the existing `report.html` / `summary.json` through
`experiments/search_mirror_report.py`; consult those for results, not the older
interim numbers below. All 8,000 mirror outcomes are used. PUCT-1024 policy
controls have 479/500 valid outcomes; the report retains the missing denominator,
common-complete subsets and failure sensitivity. Whole-deal stratified and noise
bootstrap intervals use 10,000 resamples. Training exposure and native/replay
admission remain absent; no new games, training, resources or follow-up launched.

Final HTML and summary were published to their existing artifact paths (two
manifest entries changed); all 33 mirror-study artifacts verify. A second rebuild
reproduced both files byte-for-byte, and every prior noise point estimate remained
unchanged. Browser checks found no page overflow at desktop or mobile widths.

Focused accounting/noise tests and the untracked-data check passed (5 tests).
Full integration collection found 62 tests but stopped on six import errors
(`etude.eval.match.agent` unavailable and `duckdb` absent). Do not upgrade the
frozen environment to repair that unrelated collection. Run pytest through
`uv run --project /Users/jack/src/manabot.search --no-sync --with pytest python -m pytest`
so the current checkout, rather than the external environment's experiment
package, is imported. Next Graph step is evidence review, retaining the report's
unresolved acceptance and keeping the proposed extension separate.

## Execution handoff — 2026-11-09

The original run is complete: 8,000/8,000 unique valid planned outcomes, 500 per
arm, with no missing/duplicate/unexpected keys. Last mini game-file write:
01:14:12 UTC. No mirror/noise workers remained at read-only inspection; no new
execution or rental was launched. All 771,862 decision records reconcile with
game counts/indices, with no orphans, duplicates or malformed tails. Noise:
300 positions, 41 completed opponent/deal entries, 21 distinct deals.

Final files are published at `experiments/data/search-mirror-v1/games/`; reconciliation
receipts and the audit script are in its sibling `reconciliation/`. Seven worker
logs are retained in `games/logs/`. The owning protocol now records full execution
status and accounting limits. Publication changed 28 manifest entries; all 33
current mirror-study files verify against the manifest. The untracked-data check
passed (1 test); pytest ran through an ephemeral `--with pytest` overlay because
the established analysis environment does not include it. The 4,899-game report reproduced exactly before
the copy, and all original evidence bytes remain prefixes of the final files.
The existing HTML/summary are still the interim report: the next Graph step is
`experiment-analysis`, using the final snapshot and required report corrections
already listed in the protocol. No additional execution or design review is needed.

Keep these observed limitations in the report: 104 invalid-focus warnings, seven
event truncations to 32, rounded base probabilities (882 chosen actions round
to zero), and unconstrained scalar Q (maximum |Q| 1.2144; all retained values
finite). Historical restart preserved 89 completed games; in-flight attempts lack
receipts. The removed smoke runs retain only printed results/timing. Hero search
time is 244.6554 hours, not inclusive worker-hours. Etude inventory reads fail
because the retained SDK rejects the current local launcher DB schema; leave it
untouched and do not infer remaining budget. MTG-268 remains idle/no PR.

## Design handoff — 2026-11-08 23:40 UTC

The reconstructed owning protocol is
`experiments/protocols/search-mirror-v1.md`. It separates the retained executable
sweep from proposed beta-0.01 / 8–16-shuffle and uniform-hand follow-ups. This
experiment-design step ends at interactive design review; it launches nothing.

Read-only SSH inventory at 23:40:23 UTC: 7,657 unique games, no duplicate keys,
zero invalid, 477–481 games per arm. Worker 2 has completed all assigned keys
and exited; six workers remain live. Missing keys by original shard 0–6:
76, 42, 0, 23, 47, 79, 76. No restart or extra worker is needed. Noise-v1 has
300 positions on 21 distinct deals, all at grid index 5. These live inventory
counts do not replace the published 4,899-game snapshot or its strength estimates.

`lf task status MTG-268 --json` now reports idle, no PR; its code graph stopped
at loop-or-next, pass 6, updated 12:29:26 UTC. Its current `scratch/questions.md`
says the latency target remains unmet and w13 admission unresolved. Latest
benchmarks use a different, initialized merged-world checkpoint. Leave ownership
there; neither those timings nor a generic sync admits batching to this study.

The mini source is a copy without `.git`; its `REVISION` predates mirror code.
Current runner/local-update/lock hashes match this checkout; the protocol records
them and the mini native hash. They are observations of disk bytes, not launch
receipts. Complete source/runtime admission remains necessary before new games.

Review proposal: reuse existing deals and controls for 2,000 oracle candidate
games plus 3,000 matched uniform games, at most 96 elapsed hours / 672 worker-hours
on the mini, $0 rental spend. These are proposed ceilings, not an ETA or authority.
Reuse makes the follow-up exploratory; fresh confirmation would need a separately
frozen untouched cohort and controls. Report fixes and follow-up capture/admission
requirements are listed in the owning protocol, with no duplicate batching effort.

## Task and workflow

This study is now MTG-269 (`task_ed311b47c8704f18965f6a0ae8cbb370`), with
this existing checkout and branch adopted. The current conversation is its
primary Session, `session_c73fe3a8e9e94b56b1ffe915b575a9c0`.
The current conversation requests: “lajnch the task, using the experiment
workflow” and “make sure thati t knows about existing work”.

Continue from the existing study, not a blank design. Reconcile current remote
completion and MTG-268 status before proposing execution: the snapshots below
are historical. The experiment graph and its domain skills were copied from
the main checkout into this frozen checkout; both graphs compile and the
repository checker's experiment boundary checks pass. No engine code was synced.
Do not let a generic software sync step merge current-world main into w13.
The workflow launch stops at its first interactive review; it supplies no new
rental, merge or auto-ship authority.

## Recovered study

Recovered 2026-11-08 from Claude session
`/Users/jack/.claude/projects/-Users-jack-src-manabot-search/5537a213-b123-46cf-a71e-dc3ac90c37f2.jsonl`.
The conversation lived in this checkout; its study code and evidence live in
`/Users/jack/src/manabot.w13-search`, branch `jack-heart/w13-search-scaling`.
Do not move the study onto the current engine: it uses MTG-183's frozen w13
checkpoint. The log records “We can use w13 here” and “you can use the mini if
it helps”. The 2026-11-08 continuation request is “and continue the work on
magic mirror search”.

At 11:43 UTC the mini had seven existing mirror workers running. A subsequent
local snapshot contains 4,899 unique games of 8,000 planned, zero invalid.
Remote evidence: `mini-heart.local:~/search-scaling/mirror-v1/`.
The noise study at `~/search-scaling/noise-v1/` finished all 300 positions.
No new games or rentals were launched during recovery.

The refreshed report is
`/Users/jack/src/manabot.w13-search/experiments/data/search-mirror-v1/report.html`.
Depth 20, beta 0.03, four shuffles leads this interim snapshot. Paired gains
over sampled policy: +10.13 points against heuristic (306 deals), +14.59
against policy (305 deals). Against PUCT-1024 oracle: +4.25 and +8.52 points.
These compare performance against common opponents, not mirror versus PUCT
head-to-head. Oracle hands, multiple comparisons and interim sampling limit
the interpretation; host-mixed PUCT timings are not a controlled speed comparison.

MTG-268 (`task_fae941fd7346493998abc55737dd2564`) already owns rollout batching
in `/Users/jack/src/manabot.mtg-268-batch-local-update`. At recovery its code
Graph was running implement, loop 1 pass 3. Leave that live work running.
The proposed next game settings are depth 20, beta 0.01, 8 and 16 shuffles,
after batching is validated and adapted to the frozen study. The prior assistant
proposed these; they have not run. Continue checking the existing study and
batching work before starting additional execution. No merge was requested here.

Report rebuild (analysis only; no games on the laptop):

```sh
cd /Users/jack/src/manabot.w13-search
uv run --project /Users/jack/src/manabot.search --no-sync python -m experiments.search_mirror_report --mirror experiments/data/search-mirror-v1/games --baseline /Users/jack/.manabot/search-scaling/hosts --baseline-summary experiments/data/search-scaling-v1/summary.json --noise experiments/data/search-mirror-v1/noise --extension experiments/data/search-mirror-v2 --out experiments/data/search-mirror-v1
```

Results belong in the artifact store; publish changed evidence before committing
the manifest. Never commit raw `experiments/data/` files.

Recovery publication: the 4,899-game snapshot, HTML and summary were published
to the existing artifact-store paths (16 manifest entries changed). Light and
dark report renders inspected; untracked-data check passed (1 test).
