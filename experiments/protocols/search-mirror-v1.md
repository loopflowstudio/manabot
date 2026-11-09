# Magnetic-mirror search on MTG-183

Status: continuation of the existing experiment in MTG-269, 2026-11-08.
Finish and analyze the existing oracle campaign under its original authority;
Task adoption does not reopen its design or require another approval. Preserve
all existing work and resume only what remains. Follow-up settings and bounds
below remain separate proposals and do not block this campaign's completion.
This document owns the study protocol; `scratch/mirror-search.md` is its handoff.

## Question and scope

How much playing strength does a magnetic-mirror local policy update add to
MTG-183's frozen endpoint, and what test-time compute does it require? First
characterize the retained oracle-hand sweep; then decide whether lower beta with
more shuffles improves it and how much survives a playable uniform-hand sampler.

The intervention is inference-time search. Architecture, checkpoint, training
regime, deck pool, opponents, reward and scheduled deals stay fixed. No training,
optimizer updates, learned belief fitting, curriculum, or pool expansion is
needed. This measures one trained lineage on its familiar Elemental deck grid,
not architecture quality or generalization. Held-out cards, held-out deck
combinations and unseen mechanics are three distinct claims; none is tested.
The growing-pool protocol and its Task 04/08 campaigns do not apply here.

## Evidence and identity

Read alongside `experiments/search_mirror.py`, `search_mirror_noise.py`,
`search_mirror_report.py`, `search_scaling.py`, `elemental_grid.py`, and
`manabot/sim/search/local_update.py`. The parent training protocol and frozen
deck/ABI definitions are `experiments/protocols/elemental-183.md` and
`elemental-183-identity.json`. Do not edit those historical identities.

| Identity | Binding |
| --- | --- |
| Local reconstruction source | `2dfb89ca51f57f9ceb8d7211c98bf1d60af342b4` |
| Mirror runner introduced / four-shuffle extension | `75c4dc885b5fb45d73dab9932624fcaf86ae3119` / `2c2e65b6bd76bcf989043a72cb8c0ff4adce29a9` |
| Noise runner | `aece1000db20b184529a15e1031f9f723677a13b` |
| Published snapshot manifest | `a856e267c1a01777e9d26583e55fe95f13e70fd2` |
| Policy role | MTG-183 final **raw**, update 5,950, training seed 18301; no EMA substitution |
| Policy artifact | `jobs/f3748bcabac8/checkpoints/learner_transitions-331180341.pt` |
| Policy SHA-256 | `4f2ad656ba036a33d527212a7ac2badf2c5b9db136916f730846b07eef2fbea4` |
| World / pack | w13 / `elemental-world-v2` |
| Frozen full grid SHA-256 | `ecc487e4708a8565b9c787068ff676f62b0c35420b179204fbbf72056f436264` |
| Mirror runner SHA-256, local and mini | `a88d0cd190967cee4f263ce29c32b8614cd8140b53c4e766ba4acb96541528d9` |
| Local-update SHA-256, local and mini | `3df30ba204fade75e0974049cd054ada271176883f1eff2cf9376f287f16be4e` |
| Mini native extension SHA-256, observed on disk | `7c22a1e101874ae4818ba8250326277b2fb3b0a55260edcb9da0c162080baf25` |
| Dependency lock SHA-256, local and mini | `9f903a8560b3faf6da184a9f7b84dcb76110b4cba462555f56851bbc462b1aae` |

The mini is a copied source directory, not a git checkout. Its `REVISION` says
`f0c87d65efaa2bf92c08ef4e9f1e016329dd1b6f`, predating the mirror additions;
that marker alone does not identify the running source. The hashes above were
read on 2026-11-08, not captured at launch. Before any new execution, bind a
complete source inventory, loaded native build, checkpoint input/config and
world/observation/action/content identities in an admission receipt. Reconcile
them with w13; never infer byte identity from the world label. The original
MTG-183 Darwin build has a different hash, so it is not this mini build's receipt.

Published evidence remains at `experiments/data/search-mirror-v1/{games,noise,
report.html,summary.json}` and `experiments/data/search-scaling-v1/`; baseline
game rows are retained under `/Users/jack/.manabot/search-scaling/hosts`.
The published mirror snapshot has 4,899 games. Its selected leader, depth 20 /
beta 0.03 / four shuffles, gains 10.13 points against heuristic (306 paired
deals) and 14.59 against policy (305) over sampled policy. Gains over PUCT-1024
oracle are 4.25 and 8.52 points against those same opponents, not head-to-head.
Selection and repeated interim inspection make these exploratory results.

Live reconciliation at **2026-11-08 23:40:23 UTC** found **7,657 unique game
rows, zero duplicates, zero invalid**, with 477–481 rows per arm. Worker 2 has
no missing assigned keys and has exited; workers 0, 1, 3, 4, 5, 6 remain live.
Missing assigned keys are respectively 76, 42, 0, 23, 47, 79, 76 (343 total).
Do not restart worker 2 or create another worker set. These are live inventory
counts, not a refreshed published strength report or a final immutable snapshot.

Noise-v1 is complete: 300 positions, 21 distinct deals, all at grid index 5.
The prior summary reports mean single-rollout SD 0.183, between-action mean-Q
SD 0.066, and 41.5 seconds per measured position. Its 20 subsampling draws per
position are resamples, not 20 independent positions or training seeds.

## Retained executable protocol

Eight settings, each against both the sampled endpoint and heuristic v2:

| Depth (canonical decisions, including root action) | Beta | Shuffles |
| --- | --- | --- |
| 2 | 0.1 | 1 |
| 10 | 0.1 | 1 |
| 20 | 0.1 | 1 |
| 40 | 0.1 | 1 |
| 20 | 1.0 | 1 |
| 20 | 0.03 | 1 |
| 20 | 0.1 | 4 |
| 20 | 0.03 | 4 |

Alpha is 0.02, reference uniform, full-search probability 1. Every action with
positive base probability receives one rollout per world. Worlds retain the
true opponent hand, shuffle libraries independently, and share a world's root
deal across actions. Rollouts sample the frozen policy from each acting player's
view; terminal outcomes or signed leaf values estimate Q. The hero samples
`softmax((Q + alpha log reference + beta log base)/(alpha + beta))`. A local
distribution update is not a model weight update. Oracle access is an information
advantage benchmark, not a proven mathematical upper bound on this approximate
algorithm's uniform-hand strength. Determinization and value error remain confounds.

Each arm plays indices 0–4 of all 100 ordered cells from `elemental_grid.schedule`:
500 games, 16 opponent/setting arms, **8,000 games total**. Deck names, actual
lists/lesson pools and random seats come from the existing pack and schedule.
Grid seat seed is 18,190,001; deal = 1,819,000,000 + 1,000 × cell + index.
Each deal seeds the environment, Torch and hero search; opponent seed is deal+1.
These are distinct per-game seeds deliberately shared across arms for pairing,
not independent policy-training replicates. Different action trajectories consume
different RNG streams; pairing fixes the starting deal, not every subsequent draw.

Controls reuse the existing sampled-policy, greedy and PUCT rows. Sampled policy
is the primary control for sampled mirror; greedy is separately labeled. PUCT
uses 16/64/256/1,024 simulations, four worlds, c_puct 1.5 and most-visited action.
Uniform and oracle PUCT must remain separate. PUCT is contextual evidence against
common opponents; this design adds no mirror-versus-PUCT matches.

The completed noise screen uses only sampled-policy play on indices >=5,
10% sampling of eligible hero decisions, depth 20, 64 oracle rollouts per move.
Subsamples 1/2/4/8/16/32 are evaluated against their disjoint remainder, at beta
0.01/0.03/0.1/0.3/1.0, alpha 0.02, analysis RNG seed 0. Finite remainder Q is a
noisy reference, not ground truth; the reference's own optimized value is optimistic.
Positions cluster within games. This screen motivates candidates; it does not
measure their win-rate gains. Its draws/actions are evaluation, never training
exposure. No additional noise positions are needed for this continuation.

## Analysis and uncertainty

For each opponent and comparison, join valid rows by exact deal identity and
report common-deal count, per-cell and seat coverage, wins/draws/losses, invalid
and missing counts. Score is 1/0.5/0. Pairwise score difference is the estimand.
Retain the current mean ±1.96 sample-SD/sqrt(n) intervals as explicitly nominal
95% descriptive intervals. At incomplete coverage also report a common-complete
deal subset across settings; do not compare rankings on different subsets alone.
At complete coverage each cell has equal weight. Never treat a missing row as loss
or redraw a failed deal. Show a failure sensitivity range assigning unresolved
scores to 0 and 1, separately from the valid-only estimate.

The eight settings against two opponents were inspected repeatedly; no selected
winner is a confirmatory finding. Show all settings and both opponents. A final
report should add cell-stratified paired bootstrap intervals (10,000 resamples,
seed 26901, resample deal vectors jointly across conditions within each cell),
and label them exploratory too. A fresh untouched deal cohort and a prospectively
fixed comparison family would be required for confirmation; none is authorized
or silently added here. For the noise screen, uncertainty resamples whole source
deals, not rollout draws; 21 deals limits precision. Do not translate one-decision
value gains into game-win percentages.

Report hero search wall seconds per decision and per game, median and p95,
supported root actions, worlds, depth, actual rollout commands, early terminals,
network batch calls and evaluated rows where captured. The historical
`moves × depth × worlds` count is only an approximate unbatched work proxy.
It cannot measure batched calls or equal compute. Historical mixed-host PUCT
timings cannot establish a speedup. Any controlled comparison uses identical
retained positions, checkpoint, machine, thread/load regime, warmup and repeated
timings, and reports distributions. No artificial load. Strength and timing
remain separate measurements; this is not currently an equal-wall-budget contest.

Inconclusive means the interval spans no gain, coverage/failures can change the
conclusion, required identity/admission is missing, or timing is host-confounded.
An oracle gain alone is inconclusive about playable strength. One training seed
cannot establish robustness across learned policies. No significance-based early
stop or extra games selected to make an interval clear zero.

## Proposed follow-up, pending judgment

The simplest continuation reuses the same 500 deals per opponent and existing
controls, explicitly remaining exploratory. It adds only the unresolved axes:

1. Depth 20, beta 0.01, 8 and 16 oracle shuffles: **2,000 additional games**.
2. Uniform physical hidden-hand sampling at the reference depth 20 / beta 0.03 /
   4 worlds and both candidates above: **3,000 additional games**. Each uniform
   arm pairs with its identically configured oracle arm, holding other factors
   fixed. The reference oracle's 1,000 games already exist in mirror-v1.

Thus the proposal is **at most 5,000 new games**, zero training updates, no new
noise collection, no learned sampler, no new opponents and no PUCT rerun.
Use separate versioned arm identities including sampler and software identity;
never append a new sampler or batched implementation under a mirror-v1 key.
Uniform means sampling physical unseen card copies subject to public constraints,
not uniformly weighting card names or peeking at the true hand. Preserve known
cards, root viewer hash, complete legal offers and observed history. The existing
`PhysicalHandSampler` is the implementation owner; validate its distribution and
privacy before calling the resulting arm playable.

Review must decide whether to take these candidates and uniform comparisons,
and accept or revise the proposed resource cap. Reusing inspected deals buys
comparability, not independent confirmation. If confirmation is chosen instead,
freeze new deals excluding both prior games and noise-screen deals and budget
their new controls before execution; do not silently expand this design.

## Bounds, failure capture and recovery

Retained run: seven original one-thread workers on the owned mini, zero new
rental cost; finish only the existing 8,000 planned keys. Each decision has a
600 s deadline and each game a 10,000-step bound. There is **no campaign wall
deadline or per-game wall watchdog in this runner**. Those limits are not a
useful campaign ETA; do not retrospectively claim a wall budget was enforced.
Observe the existing workers without relaunching them in this design step.

Proposed follow-up ceiling: **96 elapsed hours, at most seven one-thread workers
(672 worker-hours), $0 rental spend**, inclusive of admission checks, fixed-position
timing and all games. This is a proposed ceiling, not a forecast or permission.
Reserve the first two hours (at most 14 worker-hours within that ceiling) for
admission/timing; proceed only if measured conservative throughput projects the
complete schedule inside the remainder. Keep the 600 s decision deadline and
10,000-step bound; add a 3,600 s game watchdog and a campaign watchdog. On cap,
stop with partial evidence and call it inconclusive, without automatic extension.
No laptop games/evaluations. Fixed-position timing checks are allowed there,
but a mini throughput forecast must be measured on the mini without adding load
while mirror-v1 still occupies it. No timing or gameplay was started by this design.

The runner retains exceptions as invalid rows and fsyncs game rows after writing
compressed decision logs. `_done` skips retained keys, including failed games.
Keep invalid rows, traceback/class, host and partial decision logs; never replace
them with successes. On interrupted writes, retain original bytes, identify partial
JSON/gzip tails and orphan decisions, and reconcile planned keys before resuming.
Deduplicate decisions by game key and decision index, with conflicting records
reported rather than silently selected. Resume only absent keys under the original
identity after the operator verifies no worker owns them. Do not change active
shard assignment. The noise runner marks failed games done after logging errors;
its 300 positions alone do not prove zero attempted-game failures.

Historical rows retain scores, deck/seat/deal, steps, turns, host, hero time, and
per-decision base/Q/target/chosen action/terminal count. They do not retain full
Commands, replay proof, all per-rollout seeds/commands, actual network calls or
training card opportunities. Do not invent those receipts retrospectively.
New execution should retain plan/source/sampler/checkpoint identities, actual
rollout and inference counters, commands/root identities and failure/deadline
receipts sufficient for replay. Parent training exposure belongs to MTG-183's
training records; no claim about card-level training opportunity/resolution is
made from this evaluation's draws or action counts.

Snapshot and publish evidence bytes through `manabot.data publish` before
committing `experiments/data.manifest.json`; verify the untracked-data test.
Keep immutable prior snapshot hashes when publishing refreshed report pointers.
Raw `experiments/data/` stays untracked.

## Required software and admission, without duplicating MTG-268

MTG-268 owns batching. At 2026-11-08 reconciliation its code graph was stopped
at loop-or-next, pass 6 (updated 12:29:26 UTC); Task idle, no PR. Its retained
notes report the latency target unmet and w13 admission unresolved. Recent
measurements use a new merged-world initialized policy, not MTG-183; eight
worlds were not measured. They do not admit the adaptation here. Do not restart
that Task, duplicate its optimization work or merge current-world main into w13.

Existing code suffices to finish the retained sweep. No fitting or placeholder
software PR is needed for that. Before final reporting or a follow-up freeze:

- Report owner: replace hard-coded zero-invalid/running claims and static depth
  conclusions with observed status; expose missing/invalid/duplicate evidence,
  comparison coverage and the declared uncertainty analysis. Remove the claim
  that oracle is a guaranteed upper bound. Derive denominators from the frozen
  plan. Check fixtures covering complete, partial, failed, duplicated and empty
  comparison data; reconcile the published 4,899-row snapshot exactly before
  analyzing a new snapshot. Inspect final light/dark rendering.
- Follow-up runner: parameterize sampler/settings with distinct identities,
  capture the missing counters/receipts and enforce prospective watchdogs.
  Keep one runner and existing sampling/search owners; do not fork a second
  permanent search implementation. Check deterministic plan counts/uniqueness,
  sampler isolation, interrupted-write recovery and failed-key retention.
- Batching dependency: adapt only what w13 needs after reviewing MTG-268's
  delivery. Compare sequential and batched behavior on the same trained w13
  checkpoint and retained representative positions, including terminal, signed
  value, stochastic sampling, legal support, Command/hash/audit and deadline
  behavior. Exact discrete parity; numerical tolerances must be declared before
  comparing. Measure 1/4/8/16-world depth-20 cost and memory under one host/thread
  regime. Do not carry over timings from a changed world/checkpoint. Retain git
  as the sequential reference rather than maintaining two production paths.
  Native changes require repository debug Rust/Clippy checks and an atomic wheel
  install. All checks here are acceptance requirements, not results already run.

Scientific choices belong to design review. Software routing happens afterward;
missing acceptance is not a reason to invoke a generic sync or delivery graph now.

## Authority and review boundary

Original Claude session:
`/Users/jack/.claude/projects/-Users-jack-src-manabot-search/5537a213-b123-46cf-a71e-dc3ac90c37f2.jsonl`.
Jack Heart wrote “We can use w13 here” (2026-11-06 23:40 UTC), “you can use the
mini if it helps” (23:58 UTC), “can we start on just one world, he oracle world”
(2026-11-07 12:39 UTC), and “oh ok yeah you can randomize the shuffles” (13:03 UTC).
The retained conversation summary mentions an earlier $25 rental allowance;
remaining spend has not been reconciled, so it supplies no usable new rental
budget here. The beta-0.01 follow-up was an assistant proposal, not a launch receipt.

The earlier assistant launch directive stopped at design review. The subsequent
conversation direction supersedes that stop for the retained experiment:
“i dont think i need to review the deisgn, we were kind of half way through an
existing experient when i transferred this to a task” and “make sure to not
throw away that work or do unnecessary . kjust start from what we have”.

Resume the experiment workflow at execution/reconciliation and analysis for the
existing 8,000-game campaign. Preserve source, checkpoints, logs and completed
keys; inspect current workers before any recovery. Do not rerun design, training,
completed games or noise collection, wait for batching to finish the existing
sweep, or make new receipt instruments a prerequisite for analyzing retained
data. Report historical evidence limits honestly. Make only reporting fixes
needed for an accurate final result and reuse checks that still apply.

The proposed 5,000-game extension, new resource ceiling and batching adaptation
remain future decisions; they are not approved by this continuation and must not
become prerequisites for finishing and reporting the original experiment. No new
rental, merge or auto-ship authority is supplied here.
