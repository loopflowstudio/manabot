# Mirror search: more shuffles and uniform versus oracle hands

## Execution status — 2026-11-09: campaign running

Etude run `mtg269-mirror-v2`, attempt `3a2b62144b9a`, session
`04058802ff84`, started on the owned mini at `2026-11-09T20:40:21.931237Z`.
The launch snapshot at `2026-11-09T20:41:21.920412+00:00` verifies seven one-thread oracle
workers, seven in-flight attempts, and **0/3,000 completed campaign outcomes**.
This is a launch receipt, not a completion or strength claim. The Etude-owned
job advances to uniform only after all 1,500 oracle outcomes reconcile. Its
maximum attempt count is one; it never retries failed games. Both stages keep
`--deadline 1794852989` (November 16, 18:16:29 UTC), counting setup from the
original November 9, 18:16:29 UTC start. No rental, training, new export or
callback is expected; observed rental spend is $0. Training exposure is still
unmeasured. Campaign evidence remains on the mini under
`~/search-scaling/mirror-v2`; do not launch a duplicate while this job runs.

The separate Etude smoke `mtg269-mirror-v2-smoke` succeeded and released its
allocation: six valid outcomes on one original deal, one per cell, no failures
or worker warnings. Its 1,784 Commands match retained legal offers/revisions;
771 search receipts retain complete configured policy-support allocation,
finite values and matching update algebra. Uniform receipts validate through
the existing physical-sampler reader. Oracle is the explicit custom point-mass
sampler. Maximum retained |Q| was 1.04313; this scalar estimator is not clipped.
This proves only the observed smoke positions, not full-grid validity or replay
equivalence. No deterministic replay was performed.

Smoke wall time per game at 4/8/16 worlds: oracle 273/430/812 seconds;
uniform 223/525/1,375 seconds. Summed game wall time is 1.011 hours. Extrapolating
those six games to 500 per cell and seven workers gives about **72.2 hours**,
excluding setup and differences in seven-worker contention. One reused deal
per cell is not a reliable full-grid ETA. Different game lengths confound raw
wall-time scaling. Inclusive CPU/worker hours remain unavailable; the shared
elapsed ceiling has not been reset or expanded.

Source is committed `0063d781710928df996402957a0b30123d332875`, staged separately
at mini `~/src/manabot.w13-mirror-v2`. The original w13 checkout and v1 evidence
were preserved. Archive SHA-256:
`fb50c92e66f2ceffb2f57fef915cf1af919d836984d0eddebfbdff3f20851240`.
All 349 source bindings match that archive; smoke and campaign agree on source,
checkpoint, native build, packages and deadline. They use the existing frozen
mini environment and native build. The execution copy of this protocol is the
prelaunch committed text; this status amendment changes no scientific setting.

The explicitly canceled curriculum allocation is now `cancelled` in Etude.
Its absent SSH startup exposed the existing EXP-13 defect: the scheduler waited
for a creation receipt, while SSH cleanup returned no end receipt even without
a boot claim. A process-local transport correction allowed Etude's existing
DirectoryJournal and scheduler to retain shutdown and cancellation under the
mini lock. No database rows or prior evidence were deleted. Reproduction and
the immutable recovery receipt were attached to EXP-13; no new Etude graph was
started. The frozen game environment was not upgraded.

Sealed smoke match blobs, identity, decision rows, worker logs, audit source and
receipts are published under `experiments/data/search-mirror-v2/smoke/` and
`reconciliation/`. In-flight campaign traces stay private on the mini until
closed; the launch snapshot publishes its identity, counts and worker inventory.
The existing v1 HTML/summary own scientific reporting and retain all six full
500-slot denominators. A successful smoke does not fill any study slot.

Prospective continuation of MTG-269, 2026-11-09. The conversation directs:
“i want to try 8 and 16 shuffles, and then i want to combine magic mirror with
uniform random vs oracle”. This authorizes implementation, validation and
execution of this continuation on the already authorized owned mini. It
supersedes the earlier pending-follow-up status for these comparisons.

The subsequent clarification is “to be clear the games shoudl be against raw
policy” and “right, that and unerstanding performance across multiple shuffles
(should be more more meaning ful for uniform random)”. The primary comparison
is therefore a fixed-beta factorial, with all six cells against the raw sampled
MTG-183 policy. No v2 games or smoke had started when this schedule superseded
the earlier 5,000- and 2,500-game proposals; no evidence is relabeled.

Reuse `search-mirror-v1.md`, its checkpoint/world, 500 paired deals,
search implementation, completed controls and report machinery.
Keep the original 8,000 outcomes and noise evidence unchanged. No training,
noise collection, PUCT rerun, main merge, or rental is required.

## Fixed comparisons

Depth 20, alpha 0.02 and beta 0.01 throughout. The opponent samples the raw
checkpoint policy, with no search, mirror update, greedy choice or EMA replacement.

| Stage | Hand sampler | Beta | Worlds/shuffles | New games |
| --- | --- | --- | --- | --- |
| 1 | Oracle | 0.01 | 4 | 500 |
| 1 | Oracle | 0.01 | 8 | 500 |
| 1 | Oracle | 0.01 | 16 | 500 |
| 2 | Uniform physical | 0.01 | 4 | 500 |
| 2 | Uniform physical | 0.01 | 8 | 500 |
| 2 | Uniform physical | 0.01 | 16 | 500 |

Each row is 500 games against raw sampled policy, on
the original five deals per ordered deck cell: 3,000 new games total. Run the oracle stage first,
then the uniform stage without selecting settings based on stage-1 winners.
Within each stage, visit four worlds before eight and sixteen. Reuse the
policy-versus-policy baseline. The existing oracle beta-0.03/four-world result
is a separate contextual reference, never a control for shuffle effects.

Use `PhysicalHandSampler` for uniform physical copies from the unseen pool,
conditioned on publicly known cards and hand size. Oracle keeps the true hand.
Both reshuffle libraries and sample rollout actions. Uniform does not mean
random root actions, uniformly weighted card names, or a trained belief model.
Verify that the uniform path does not call the oracle private-hand reader.

## Minimal preparation and execution

Extend the existing runner with explicitly selected sampler/settings and
versioned keys; keep its default v1 schedule and keys unchanged. Retain source,
checkpoint, runtime and plan hashes with the new run; keep outputs under
`search-mirror-v2` / mini `~/search-scaling/mirror-v2`, separate from v1.
Record worlds and sampler in rows and use the same admitted implementation
for both stages. Do not copy an unvalidated current-world wheel into w13.

MTG-268 is a separate optimization. Its current results are not admitted for
the trained w13 policy. Use the existing sequential search here; do not wait
for its aspirational latency target or start another optimization campaign.
Test plan counts, distinct identities, sampler selection/privacy, preservation
of v1 defaults and resume behavior. Run a small bounded real smoke on the mini
under separate smoke keys, then proceed with the fixed schedule when correct.
Capture actual timing and report a measured ETA; do not promise batching speed.

Inspect current remote work before starting. At 2026-11-09 18:14:57 UTC no
mirror/noise or local-update benchmark processes were found on the mini; only
the existing v1 directories were present. Recheck immediately before launch.
Use at most seven one-thread workers; no laptop games or artificial load.
An operational ceiling of seven elapsed days (168 hours, at most 1,176
worker-hours), including smoke/setup, and zero rental spend bounds this run.
Retain existing 600-second decision and 10,000-command game bounds; enforce
the campaign stop and preserve partial outcomes if reached. Do not silently
increase the schedule or spend to obtain a favorable result. These ceilings
are operational choices, not measured durations or a new paid budget.

Resume missing keys only after reconciling live ownership, retaining failures
and partial logs. Never replace failed outcomes with retries. Publish all
evidence bytes through the existing artifact store before manifest commits.

## Analysis and delivery

Keep the v1 report accessible and extend its comparison machinery for v2.
Report paired gains over sampled policy, oracle-minus-uniform at each shuffle
count, 8-minus-4 and 16-minus-8 within each sampler, and paired
difference-in-differences: (uniform higher minus uniform lower) minus (oracle
higher minus oracle lower). Do not assume the interaction is positive. Retain
game coverage/failures, uncertainty, and actual decision/game cost.
Reuse the v1 uncertainty methods and describe
all results as exploratory because the policy, deals and candidate selection
are already inspected. These are against the same raw policy opponent, never
oracle-versus-uniform or mirror-PUCT head-to-head matches. No additional controls need rerunning merely to add
these settings. Preserve historical evidence limits rather than demanding
new instruments retroactively. Publish progress and the final charts.

There is no new design approval needed for these requested comparisons.
Finish necessary preparation, then execute and analyze in the same Task.
Report genuine technical failures or resource limits; do not reinterpret
older notes saying the follow-up was unapproved as a current blocker.

## Execution status — 2026-11-09 18:27 UTC

**Not launched: Etude reports no available mini allocation.** The frozen SDK
expects launcher schema 10 while the existing database is schema 13. The clean
installed checkout `/Users/jack/src/etude.ad319b-show-live-progress` (SDK 0.2.14,
schema 13) can read it through the existing Etude environment with `PYTHONPATH`
selecting that source. Neither database nor frozen game environment was changed.

Etude retains `curriculum-held-out-707e62cad5fe-r3` on machine `mini`, phase
`renting` (an owned host, not a newly rented machine), attempt `ab69a427b4a1`,
session `c24c6c598b42`. Its evidence reports an unknown SSH create reply and
missing worker/job output. A bounded log refresh did not resolve that state.
A read-only zero-rental capacity quote returned:
`cannot launch on mini: no free machine; etude queue would wait`.
That allocation belongs to other work. It was not canceled, replaced or bypassed.
Etude must reconcile it before smoke or campaign admission; an empty process
listing alone does not release the allocation.

Read-only SSH at 18:26:41 UTC found no search workers and no `mirror-v2`
directory. All 8,000 v1 game keys and the seven game-file hashes still match the
published evidence. The v1 HTML and summary rebuild byte-for-byte unchanged.
No new games, smoke, training, paid resources or queued runs were created.
The requested comparisons remain unmeasured; there is no measured v2 ETA.

Runner preparation now selects the six factorial cells explicitly, leaves v1
defaults/keys intact, and distinguishes smoke keys. Uniform uses the existing
physical sampler; opponents use the existing sampled checkpoint player. V2
binds source/runtime/package/plan identities, workers and an absolute deadline,
locks each shard, requires oracle outcomes before uniform, and rejects changed
identities, duplicate outcomes, or unreconciled partial attempts. Failed outcomes
remain completed keys. Private attempt streams retain pending observations,
Commands, original errors, RNG seeds/state, configured capacities and full
search receipts; campaign interruption is separate from a failed game.

Fifteen focused runner/report/data checks pass, as does Ruff. The required
broader CI collection found 72 tests but still has the six previously observed
import errors (`etude.eval.match.agent` unavailable and `duckdb` absent).
No environment upgrade was made to hide that limitation. These are fixture and
accounting checks, not native runtime admission: the bounded mini smoke remains
required before the campaign. No source or native binary was copied to the mini.

Reconciliation evidence is under
`experiments/data/search-mirror-v2/reconciliation/` (`launcher-inventory.json`,
`remote-inventory.json`, and `validation.json`). Bytes are published through the
existing content-addressed artifact store before their manifest is committed.
There are no game blobs to publish yet. The next execution step needs a free
Etude-owned mini allocation, the source-bound smoke, then the fixed stages.
The following analysis step must extend the existing report machinery to the
factorial contrasts above; the v1 report remains at its original path.

For conservative inclusive accounting, the setup window began with this Graph
step at 2026-11-09 18:16:29 UTC. Its seven-day ceiling is
**2026-11-16 18:16:29 UTC** (`--deadline 1794852989`), shared by smoke and campaign.
Do not reset that deadline on retry, stage transition or allocation delay.

## Analysis handoff — 2026-11-09

The existing `search_mirror_report.py` now accepts `--extension
experiments/data/search-mirror-v2` alongside its v1 inputs and updates the same
`experiments/data/search-mirror-v1/report.html` and `summary.json`. The v1
results remain separate and unchanged; no second result store is introduced.
Extension snapshots put runner outputs in `games/` and dated receipts in
`reconciliation/`. No game evidence exists in the retained prelaunch snapshot.

The report preserves all six 500-slot cells, the sampled-policy control, and
all 15 declared comparisons: six gains over sampled policy, three hand-sampler
contrasts, four adjacent shuffle contrasts, and two four-cell interactions.
Each estimate pairs complete whole-deal vectors for its contributing arms.
Bootstrap uncertainty uses 10,000 within-cell resamples, seed 26901, equal cell
weights; a wholly missing cell prevents a full-grid interval. Sensitivity keeps
all 500 slots and ranges missing scores over [0, 1]. Interactions therefore have
possible missing-evidence bounds as wide as [-2, 2], not [-1, 1]. These are
exploratory procedures, not a new acceptance threshold.

At the retained 18:26 UTC execution snapshot all 3,000 new slots are missing,
with zero attempted-game failures; estimates, intervals and measured cost remain
unavailable. This cannot answer whether uniform benefits more from shuffles.
Health, native admission, replay proof and training exposure remain unresolved
for evidence review. Capacity reconciliation and the bounded native smoke belong
to execution; analysis launched nothing and did not refresh or reset the deadline.

Validation: 18 focused runner/report/data checks pass, as does Ruff. The broader
CI invocation collects 75 tests but retains the six known import errors involving
`etude.eval.match.agent` and absent `duckdb`; the frozen environment was not
changed. Every pre-existing v1 summary field except the analysis source digest
reproduces exactly. The report carries these results and the dated execution
receipts; this handoff is not a second scientific summary.
