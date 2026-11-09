# Mirror search: more shuffles and uniform versus oracle hands

Prospective continuation of MTG-269, 2026-11-09. The conversation directs:
“i want to try 8 and 16 shuffles, and then i want to combine magic mirror with
uniform random vs oracle”. This authorizes implementation, validation and
execution of this continuation on the already authorized owned mini. It
supersedes the earlier pending-follow-up status for these comparisons.

Reuse `search-mirror-v1.md`, its checkpoint/world, opponents, 500 deals per
opponent, search implementation, completed controls and report machinery.
Keep the original 8,000 outcomes and noise evidence unchanged. No training,
noise collection, PUCT rerun, main merge, or rental is required.

## Fixed comparisons

Depth 20 and alpha 0.02 throughout. The beta choices below are implementation
choices from the previously proposed continuation and noise study, not additional
verbatim user specifications.

| Stage | Hand sampler | Beta | Worlds/shuffles | New games |
| --- | --- | --- | --- | --- |
| 1 | Oracle | 0.01 | 8 | 1,000 |
| 1 | Oracle | 0.01 | 16 | 1,000 |
| 2 | Uniform physical | 0.03 | 4 | 1,000 |
| 2 | Uniform physical | 0.01 | 8 | 1,000 |
| 2 | Uniform physical | 0.01 | 16 | 1,000 |

Each row is 500 games against sampled policy plus 500 against heuristic, on
the original five deals per ordered deck cell. Run the oracle stage first,
then the uniform stage without selecting settings based on stage-1 winners.
The oracle beta-0.03/four-world control already exists in v1 and is reused.
The direct 8-versus-16 comparison holds beta fixed; comparing either to the
v1 four-world reference changes beta as well as worlds and must say so.

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
Report paired gains over sampled policy and uniform-minus-oracle at each
matched setting, 8-versus-16 at beta 0.01, game coverage/failures, uncertainty,
and actual decision/game cost. Reuse the v1 uncertainty methods and describe
all results as exploratory because the policy, deals and candidate selection
are already inspected. These are against common opponents, not mirror-PUCT
head-to-head matches. No additional controls need rerunning merely to add
these settings. Preserve historical evidence limits rather than demanding
new instruments retroactively. Publish progress and the final charts.

There is no new design approval needed for these requested comparisons.
Finish necessary preparation, then execute and analyze in the same Task.
Report genuine technical failures or resource limits; do not reinterpret
older notes saying the follow-up was unapproved as a current blocker.
