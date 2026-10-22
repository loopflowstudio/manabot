# Elemental $10 baseline: research and local preflight

Research step, 2026-10-21, at checkout HEAD
`6b34e7e0119b97ad231ab0f64262bc96a95f1fe7`. Scope comes from
`scratch/elemental-ten-dollar.md` and the current Task's steering. This note
records existing capabilities and gaps; it freezes no new recipe, launches no
job, and makes no training-completion claim. No paid resources were requested
or scientific games played during this step.

## Delivery dependency

`lf task status f72151 --json` confirms the selected Task and this checkout;
the current live Flow is `ten-dollar-baseline`, executing research. Its latest
steering requests local preflight now, followed by a wait for MTG-164.
`lf task status MTG-164 --json` returned `state: unstarted`,
`completed: false`, with the accepted ten-deck integration still required.
These are dated observations, not a prediction of the next step's state.
The Task's Linear creation export is pending because its Wave has no Linear
Initiative; local Task identity and steering were available. No repair attempted.

This branch documents development world w9, not a delivered ten-deck world
(`WORLDS.md`). Its Elemental validation packs are predecessor content; their
presence does not establish MTG-164's accepted full roster. Do not pick a future
world ID or relabel the current source compatible. The later launch decision
must read MTG-164 fresh and verify its delivered source, roster, Lesson pools,
checkpoint binding and runnable native extension.

## Retained MTG-159 baseline

Primary sources: `experiments/scale-ladder.md`,
`experiments/regimes/scale-ladder-allies-lessons-v1.json`,
`experiments/data/scale-ladder/v1-evaluation.json`, and
`experiments/data/scale-ladder/allocation-allies-lessons-v1.json`.
The original files also exist under `/Users/jack/manabot-runs/mtg159/`.

The retained `main/pull/out/run/run.json` identifies run
`a9fba80adaa3436abd051eb46ff70f00`, seed 15902, world w6, source
`e6a2aab1e1f60ce3d7e7bad1fdbc1016c377f1ff`, and regime digest
`db2280d3589c2bd32735e0e344e9939422ba834b352f4335c2555f284fc8f6c0`.
It records Torch `2.10.0+cu128`, CUDA 12.8 and an NVIDIA L4. The retained
toolchain receipt records uv 0.8.22, Rust 1.98.1, driver 595.91.07 and EPYC
9254. The source commit and native extension digest are retained in that run.

The small-tier artifact is `main/pull/out/run/monitor-00000036.pt`, update
9,250: 151,552,000 learner decisions and 308,858,770 environment decisions.
The report dates it to 2026-10-21 14:40 UTC and $9.88 of cumulative pod time,
including startup. This step independently hashed the bytes:

| Artifact | SHA-256 |
|---|---|
| Small raw checkpoint | `ef453b11c7cb8529128bc016951e024c18f04986755069738286e1e8cdc3cd29` |
| `self-play-initial-raw.pt` | `1a7dbd96e0a727fc402f81ecdffed54f1cc8f529dd6e731a30347c62a22a4e26` |
| `eval/results/monitor-results.tgz` | `cf4ce4ba98fb199a14039e50790fdef26cab7a3c9870ddf7afb79249d35c71c0` |
| `eval/results/h2h-results.tgz` | `a6916d3796b8f0f86874387f099c9c8236d509960ca52c7b839773df7e8333a3` |

All four agree with the published evaluation receipt. Hash verification proves
retained-byte identity, not current-world load compatibility. No old checkpoint
was loaded into w9.

Baseline recipe: semantic input for `ur-lessons-vs-gw-allies`, 64 hidden units,
two attention layers, four attention heads, scalar value, Adam, current-self,
256 streams × 64 learner transitions, one learning seat per stream. Ataraxos
move learning uses policy/value lambdas 0.5/0.8, actor-critic quantile filter
0.75 with minimum advantage 0.01, action-type-uniform reference, clip 0.2,
collection KL 0.1, LR scale/power/min/max 0.5/1.1/5e-6/1e-4, tau scale/power
0.05/0.3, gradient norm 0.267, EMA 0.999. There is no iteration schedule;
`schedule_clock` is `iteration_fraction`. The JSON omits `policy_film`, whose
historical default is false (`manabot/infra/hypers.py`); using today's new-agent
default would silently change the policy. The baseline's `cross-balanced`
training has no mirrors; evaluation includes mirrors.

Published evaluation: 25 deals, both seats and four deck cells, 200 terminal
games per heuristic checkpoint. At 9,250, heuristic mean is 0.69 [0.64, 0.735],
Lessons vs Allies 0.70 [0.58, 0.82], versus initialization mean 0.11
[0.065, 0.16]. Frozen Mini comparison is a different 400-game cross-deck
protocol: 0.81 [0.775, 0.842], zero invalid games. Intervals describe deals for
one trained seed, not variation between trained agents.

Cost definitions differ materially. The old $10 point uses pod rate $0.59/h;
its allocation also reserved $0.02/h storage, and calibration/evaluation were
separate. The new $10 ceiling includes all paid startup, probes, failed attempts
and evaluation. Do not call both points $10 inclusive spend or use the old
151.6M decisions as an enforceable projection for the new world. The old run
was closed at update 12,000, $13.34 pod time; $50 was never reached. Its JSON
still says `running`, consistent with a killed process lacking a final export.
The report/stop receipt describes operator termination and deletion. That stale
snapshot is neither proof of a live competing job nor a completed-target export.

## Training and checkpoint data flow

`TrainingRegime` / `TrainSelfPlay` (`manabot/training/models.py`) feed
`execute_regime` (`execution.py`), which builds `SeatRoutedCollector`
(`manabot/sim/net_opponent.py`). The collector obtains native setups from
`stream_matches` (`training/matchups.py`), routes policy actions by acting seat,
and records only the designated learner seat's transitions. The designated seat
is stream index modulo two. Fixed strata balance learner transitions, not
completed games. Monitoring exports initial and milestone raw policies;
`model/world.py` binds actual decks, outside Lesson pools, native world and
content/input identities before consumers load weights.

At this source, `stream_matches` accepts one two-player Match. `fixed` repeats
it; `cross-balanced` repeats the pair and reversal; `mirrors-balanced` requires
the exact Allies/Lessons roster and uses 12-stream strata. `TrainSelfPlay`
exposes only those three choices. `checkpoint_world` requires exactly two
setups, and repetition admission checks membership in that checkpoint's setup
list. A pack's larger catalog alone does not expand checkpoint admission.
These paths cannot currently express the accepted ten-deck uniform run.
MTG-164 owns the roster/world binding delivery; reassess its final APIs first.

Even after binding is available, uniform ten-deck scheduling needs an explicit
supported path: 100 ordered pairs including ten mirrors, retaining learner-seat
identity. 256 streams do not divide evenly into 100 cells (or 200 pair/learner
seat strata). Changing stream count alters decisions per iteration and hence
schedule exposure; rotation would require declared balancing semantics. No
choice is silently made here. This is a compatibility requirement for the
baseline, not authorization for adaptive curriculum or two-seat learning.

Existing `tests/training/test_matchups.py` checks real deck/sideboard strata,
seat exposure, terminal-reset recovery, mirror manifests and rejection of
undeclared roster changes. It covers the existing two-deck implementation,
not ten-deck uniformity. `tests/model/test_world.py` and arena admission tests
cover identity rejection; the new full-roster proof must come from the delivered
implementation. No code changed or tests rerun in this research step.

## Budgeted remote execution

`manabot/remote/plan.py::JobSpec` bounds projected machine plus storage spend
and includes setup, checkpoint, upload and cleanup reserves. Allocation derives
absolute pause/export/deletion deadlines. `jobs.py`, the supervisor and
`guardian.sh` retain lifecycle/shutdown ownership; `docs/remote-jobs.md`
documents the supported boundaries. `tests/remote/test_job_spec.py` tests
cost/reserve rejection, immutable deadlines and long-job refusal.

Construction and compilation are not execution admission. `JobSpec.admit`
refuses lifetimes above 43,140 seconds (12 hours minus 60 seconds), because
renewable credentials and portable complete-state CUDA recovery are not
implemented. Temporary-role credentials can impose a shorter 3,540-second
bound. The old approximately 17-hour L4 $10 point cannot simply be submitted
through this path. Do not bypass the refusal, extend the guardian or reset
accounting through another job.

The brief explicitly allows a safe cutoff below $10. A shorter admitted
single run may therefore be sufficient if its accepted world and schedule
are runnable; its actual spend and smaller exposure must be reported. At the
historical ceiling $0.59+$0.02/h, the maximal single allocation would project
about $7.31 before separate probes/prior charges. This is arithmetic on old
rates, not a live offer or an admitted budget. Any probe, prior paid consumption
and cleanup allowance must fit the same cumulative $10 ceiling. This step
did not inspect credentials, enumerate current rentals or establish live price
availability. Duplicate-job and remaining-budget checks are mandatory fresh
launch checks, not assumed clear because the historical baseline was deleted.

## Evaluation reuse and limits

`training/monitor_evaluation.py::MonitorProtocol` already supports fixed seeds,
command/time caps, mirrors and authored matchups. `arena/matchup.py` represents
two roles as four matrix cells/eight legs; multiple declared role pairs can
cover a larger roster, but duplicate mirror cells must not distort weighting.
`arena/match.py` records traces and replay receipts. Admission against the
complete new roster remains prerequisite. CPU evaluation can compare final
raw policy versus initialization in the same new world; old-world weights
cannot serve as Elemental opponents.

The `heuristic` anchor is not demonstrated to understand the Elemental roster.
`arena/heuristic.py` derives targeted-spell knowledge from
`content/semantic/v1/complexity.source.json`, falls back to scripted greedy for
unknown prompts, and approximates mana by untapped lands without colors.
Legal fallback actions are not evidence of a strong or equivalent anchor.
Report unknown mechanics/coverage rather than extending the heuristic or
claiming cross-world Elo. The meaningful comparison is declared within-world
gain over initialization, legal/terminal/replay coverage, learner/environment
decisions, throughput and actual inclusive dollars, with world difficulty and
anchor limitations stated.

## Prior simple-to-complex study and remaining questions

The requested earlier study is `experiments/curriculum.md` (MTG-122 follow-up),
with `curriculum-design.md`, `curriculum-plan.json` and the campaign/analysis
runners. It tested L1 creatures/lands for the first 32 of 128 iterations,
then Allies/Lessons L3, against split scratch L3, six paired seeds. Both arms
carried weights, EMA, Adam and absolute schedule; total transitions were
matched. Primary iteration-64 effect: greedy −0.8 points [−4.0, +2.5], classified
no useful effect; heuristic −2.3 [−5.6, +0.9], inconclusive leaning negative.
The report's broad “no useful curriculum effect” headline should retain that
anchor-specific distinction. This does not test adaptive sampling, Elemental,
or a longer-horizon curriculum. Preserve the supplied order: uniform baseline,
then separately scoped adaptive sampling, then any hard-coded world switch.

Open launch questions are delivery facts rather than reasons to start another
experiment: Does final MTG-164 provide a full-roster collector/binding and
uniform schedule? Can an admitted short allocation export successfully on
the chosen runtime? What prior charges and live rentals exist then? Which
Elemental anchors are supported? These require fresh evidence after delivery.
Research is complete for this step; paid execution and Task completion remain
unachieved. The next authored Flow step owns waiting for MTG-164.

## Router decision after MTG-164 delivery

Fresh `lf task status MTG-164 --json` reports completed delivery, with PR 337
merge commit `6d7fee869d8210669ebbe152a87d33944603f635`. Inspection of that
commit's `manabot/training/matchups.py` and `training/models.py` confirms a
remaining executable-capability blocker: `stream_matches` still takes one
Match and supports only fixed, cross-balanced and the exact Allies/Lessons
mirrors-balanced schedule. `TrainSelfPlay.matchup_curriculum` exposes those
same choices. Full checkpoint roster admission does not supply a uniform
collector schedule across all 100 ordered Elemental pairs and mirrors.

Decision: blocked. A compatible executable ten-deck uniform scheduling path
and frozen balanced recipe are required before this run can proceed. Do not
substitute pair-only training, waive coverage, or launch paid work. Fresh
live-job exclusion and cumulative inclusive budget admission also remain
unverified; they must pass before any later launch. No paid work was requested
by this routing step, and no actual training completion is claimed.

## MTG-174 execution admission stopped — 2026-10-22 00:31 UTC

The sampler implementation was verified (9 focused Python tests passed in
29.52 seconds) and preserved in commit `3406a548`; no scientific run or
prospective scientific protocol freeze occurred. Existing exact 256×64 proof
and preceding native debug/recovery tests remain implementation evidence only.

Fresh supported provider/S3 reads found two rentals: unrelated CPU `loo-554-qa`
and Manabot GPU `dsist8j859mj94`, whose exact resource receipt belongs to
`manabot/jobs/mtg165-two-seat-002/training.json`. Neither was modified. The
complete one-page `manabot/jobs/` and `manabot/cohorts/` prefix listings contained
no MTG-174/Elemental job. This does not establish all historical charges as zero.
No MTG-174 provider create, paid probe, calibration or training was requested.
The live L4 catalog quote was $0.59/hour. Existing `manabot-issuer` credentials
and existing worker role admitted a scoped 43,140-second session without account
changes. The inclusive $10 cap and all lifetime guards remain unchanged.

**Blocking evidence:** while inspecting overlapping evaluation work, found an
independently retained native legality counterexample in the adaptive-curriculum
checkout. Replayed its exact authored root and 34-action prefix in THIS native
extension: w10, seed 2165384, eight offered actions, offered action 4 (Gnarled
Professor in the original frame) raises `PyAgentError: not enough producible
mana`. This is a scalar native Env reproduction, independent of either sampler.
The mana source SHA-256 is
`9bf74653518bfdaefb88ba69a1611d698c2b8a0d187c736835ded3993ce8a4f7`,
identical to the original failure. Current native SHA-256 is
`ddee3f41deeb5b561bccd0930698c802ac6e40af2499fc340fd67b04d371d5a8`.
The exact isolated root SHA-256 is
`2fd006c251f3064018303bd5c124e9bb00fb82b6c4261a790ebbbc033f16a4e1`.
Source at reproduction: `44406a97481f1718141777450bfca90fe19d4fcf`.

Receipts and runnable reproduction are retained under
`experiments/data/mtg174/admission-native-legality-v1/`, including original
attribution/source receipt. Reproduce from this repository root with:
`uv run python experiments/data/mtg174/admission-native-legality-v1/reproduce.py`.
An offered action failing to execute blocks full-roster world admission; do not
select a seed/opponent to avoid it, waive it, or spend before its engine-owner
resolution and affected validation. This step stops at that dependency failure;
it does not expand into a rules/world redesign or change the accepted recipe.

The Etude bridge attempt used exact source
`3929f25b023aaa3289e1cb4a0914a584a5c7873f` in an isolated overlay and isolated
launcher DB `.runs/mtg174-etude/launcher.db`; shared launch state was untouched.
Current Etude cannot directly import Manabot because its `etude.track` module
was removed. A process-boundary adapter was attempted, but actual local typed
evaluation job `fbaf4f12b6ec` / attempt `830fbace4202` failed before its command
started: the overlay's temporary interpreter was unavailable to the worker.
Status is terminal `failed`, $0.00 rental cost; cleanup confirmed at
2026-10-22T00:31:28Z. Cancellation/status confirmed it terminal after stopping
the launcher. Logs, source attempt and finished receipt are retained with the
failure data. No games or first successful Etude evaluation/live report were
established. Unvalidated adapter scaffolding was removed from runtime source;
its historical bytes remain in git and the failure artifact.

No scientific checkpoint, within-world gain, exposure/cost result or actual
baseline report exists. Task MTG-174 remains incomplete. No PR, merge, Task
completion, top-up, replacement run or paid evaluation was performed. Resume only
after the native legality dependency is resolved; then finish the tested Etude
bridge/owner, freeze protocol/source/seed, and repeat fresh cumulative-cost and
live-job admission before the same single authorized run.
