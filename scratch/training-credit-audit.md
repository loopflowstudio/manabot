# MTG-179 training credit audit — 2026-10-22

Status: accepted consolidation implemented; reconciled with main
`8625df97ebff0f7fed383322de7bf01faa283ba9` on 2026-10-22. MTG-179's original
independent audit is completed; consolidation is published for review as
https://github.com/loopflowstudio/manabot/pull/356 at
`6e33e0f4f7a8d371d5200cfaad3c92953aa7dee1`.
No product decision is outstanding. The second pursue pass changed only
`docs/game-crash-recovery.md`, completing the narrow correction from concept
review; the subsequent sync succeeded without further implementation.
Current-world diagnostic evidence is
recorded below; broader affected-suite acceptance remains with gate.

Jack Heart requested independent counterexamples and reviewable repairs,
assessing the frozen w6 study separately, then accepted training consolidation.
The original audit and full learnability receipts bind base
`95ca2059ea443351cabd4e8c322991c36c0d8212`, world w11 and the working diff at
that time. The integrated checkout is w13; those receipts are historical proof,
not w13 qualification.

## Confirmed findings, ordered by consequence

### P1 — legacy PPO cuts credit at the wrong transition

`Trainer.train` records `_rollout_step`'s post-action `done` beside that action's
reward. `_compute_gae` instead used `dones[t+1]` for interior rows. It therefore
cut the trace before terminal actions and let terminal actions bootstrap from
the next game's value. The final row used the correct flag, obscuring this in
short smoke tests.

Independent example: two two-decision games, win then loss. Rewards
`[0,1,0,-1]`, values `[.2,.4,-.3,.6]`, ends `[F,T,F,T]`, gamma=1, lambda=.5.
Correct returns are `[.7,1,-.2,-1]`: the first return is
`.5*.4 + .5*1`, and neither terminal return may see the reset value. Executing
the original HEAD function extracted from Git gives `[0,.85,0,-1]`.

Surviving repair: `manabot/model/returns.py::transition_gae` owns the estimator;
its direct regressions in `tests/model/test_returns.py` preserve this example and
a nonterminal tail with gamma=.9 and expected returns `[1,.189,.72]`. The later
consolidation removes both old trainer loops and the old collector-module import.

Scope: ordinary `Trainer` PPO. The regime PPO and Ataraxos paths already used
transition-end GAE and do not have this indexing defect.

### P1 — checkpoint admission leaves the training owner in eval mode

`load_checkpoint_agent` returns an eval-mode model. `Trainer.__init__` previously
moved it to the device without enabling training; `NetOpponentTrainer` inherits
that constructor. Fresh models work, so fresh-model CPU smoke tests miss this.
Semantic encoders include a GRU, whose cuDNN backward requires a training-mode
forward. The Task brief records the private CUDA failure; this audit did not
reproduce that GPU failure or operate that GPU.

Repair: the optimizer owner calls `.train()` on admission. A real save/load
regression verifies that the admitted checkpoint starts in eval mode and that
all its modules enter training mode when owned by the trainer. Production
checkpoint warm starts therefore inherit the same correction. PolicyInference
continues to own only preparation/caching, without changing the model's mode;
evaluation restores the prior training mode.

Scope: loaded-policy training through these trainers. The retained w6 study
driver already applies `agent.train()` explicitly.

### P1 — plain Trainer accepted mixed-player streams

`VectorEnv(opponent_policy="none")` returns successive acting players. Plain
Trainer packs each environment column as one trajectory, without seat routing.
An A/B/A sequence can therefore borrow B's value for A's return. Changing signs
on alternating rows cannot fix consecutive decisions or unequal decision counts.

The audit first rejected this configuration before training. Consolidation
then deleted the plain Trainer and its exclusive rejection test; maintained RL
uses SeatRoutedCollector, with same-seat regressions below. The independent
VectorEnv still traverses fixed-seat passive/random opponents natively and
negates an opponent terminal reward before returning it to the fixed hero.

Scope: this formerly admitted legacy configuration; ordinary seat-routed
self-play is not changed.

### P2 — shaped terminal draws became losses

`VectorEnv._apply_reward_policy` selected win reward for positive native reward
and loss reward for everything else, mapping a terminal draw's zero to a loss.
Repair: distinguish positive/negative/zero before adding the terminal potential
correction. With win=7, loss=-3 and previous potential=.5, a draw/win pair must
be `[-.5,6.5]`; the original draw was `-3.5`. The new potential-shaping regression
checks the exact result. Legacy progress shaping already masks terminal rows.

Scope: non-native reward shaping in VectorEnv. SeatRoutedCollector derives
per-seat terminal outcomes directly from winner_index and already keeps draws
at zero.

## Production ownership proof

`tests/training/test_credit_assignment.py` uses actual collection, packing,
target construction, selection and optimizer code. Only the game sequence and
table policy are controlled. Two streams use A,A,B,A,B and B,A,B with asymmetric
predictions; B wins. Visible markers identify observations. Deliberately bogus
native rewards (+37) demonstrate that seat-routed labels use winner_index.

For A's three actions, values are `.2,-.2,.4`; with policy lambda=.5,
advantages are `-.45,-.1,-1.4`. B's two actions have values `-.4,.6` and
advantages `1.2,.4`. Scalar value-lambda=.8 returns are independently tabulated
as `[-.616,-.72,-1,.92,1]`. Separate categorical loss/draw/win mixtures verify
the full distribution, not merely its expectation. Hooks compare actual loss
gradients to `A*(p-one_hot)/batch_size` and the scalar MSE/categorical
cross-entropy derivatives before clipping and Adam.

Coverage includes one/both learning seats, native seat reversal, both winners,
consecutive actions, unequal stream lengths, exact-quota bootstrap at the same
player's paused observation, changed weights at the next collection, reset into
a draw game, truncation rejection, forced actions, padded action logits with
zero gradient, negative-advantage selection, actor-only versus actor-and-critic
filtering, frozen/random opponent exclusion, and independent learning policies.
Complete-game packing includes no padded trajectory rows. No defect was found
in those tested production ownership boundaries.

Native source inspection connects winner_index to the pre-step actor and
preserves terminal reward/info while returning auto-reset observations.
Existing real-native collector, two-seat, inference and recovery suites verify
paused buffers, stream continuity, model ownership and exact restoration. This
does not exhaustively qualify all deck/mirror configurations or native rules.

## Recovery, clocks and intentional semantics

The first broader run found a stale test expecting exactly 192 rows after its
recipe default changed to both-seat complete games (actual 668). Before that
assertion, learner/optimizer/EMA and all tested RNG/collector state already
matched exactly between uninterrupted and resumed execution. The test now
explicitly covers one and both seats for both PPO and Ataraxos. One-seat retains
the exact 192-row assertion; both-seat requires at least the quota, equality
with native decision accounting, zero excluded opponent rows, and equality
with the resumed count. All four cases pass.

Discount and lambda units are retained same-player decisions. Skipped native
trivial decisions and opponent decisions do not add discount steps. Ataraxos
uses gamma=1, separate policy/value lambdas, absolute-advantage selection and
no advantage normalization. Actor-only filtering preserves critic coverage.
Complete-game batches freeze behavior weights through the game, while one-seat
exact-quota games can span updates. Minibatch updates change weights inside a
batch, so stored action likelihoods remain necessary. These are documented
learning choices, not new algorithm changes.

Private recovery restores optimizer, EMA, sampler, minibatch, Python, NumPy and
Torch RNG state; policy exports alone are not exact training recovery. Current
model recurrence encodes card programs within an observation, rather than
carrying an episode RNN state between players. The audit did not certify CUDA
recovery or arbitrary external trainer integrations.

The deleted legacy Trainer warned and cut bootstrap on truncation without
recovering the missing final observation. Strict seat-routed collection rejects
truncation. Integrated opt-in game-error recovery quarantines failed/truncated
games and regathers under the same policy; one-seat recovery therefore also
finishes games before optimizing. Strict one-seat quotas remain exact unless
complete-game collection is explicitly selected. Generic time-limit bootstrap
remains unsupported; quarantine is not a continuing-task estimator.

## Frozen w6 study applicability

Read-only inspection used the retained local
`.runs/two-seat-scaling-25/proof-source` under the both-seats checkout and its
driver/protocol. Frozen base is `fabb96db2bb88f98891aba66cc9d12e65d700545`.
The retained collector and objective dispatcher exactly match that Git source.
The Ataraxos module differs from the audited source only by the declared minibatch
width override (`learning.minibatch_rows or num_envs`), not target construction.
The driver uses 32 streams, cross-balanced matches, complete games in both
arms, scalar Ataraxos, 32,768-row collection floors and 256-row minibatches.
It concatenates terminal-ended batches and calls `agent.train()` after loading.

Consequently none of the four confirmed defects establishes corrupt credit in
the retained study: it bypasses plain Trainer's GAE and VectorEnv shaping,
routes seats explicitly, and already corrects training mode. This is a source
applicability assessment, not a replay of the exact w6/CUDA execution. The audit's
w11 test passes do not certify the frozen world's rules or equal learning value.
The reported throughput advantage remains a timing observation; strength and
decision-efficiency conclusions still need the declared endpoint evaluations.
No live job status was inferred from old progress and no experiment was altered.

Retained inspected SHA256 identities:

| File in proof-source | SHA256 |
|---|---|
| experiment.py | 730c50f6df1d67bfed90685ad73f0d722ddde7f11902e7eca63d160855bc9530 |
| manabot/sim/net_opponent.py | 7459a86b1bd5deae787063b954a8a5376b65be6c46a2f71354c79a1af827579d |
| manabot/training/ataraxos.py | 72bba48eafd2d909c3fa2f3fafb436fe5fa4e52cb29022f2d7171a1463b3454c |
| manabot/training/objectives.py | 542b18e4af26d5d09e7cb3acc1f67096b1d629c4f1ae0439f5a0277e726a8bbe |

## Maintained training contract

Jack Heart's 2026-10-22 direction is one RL execution path: TrainingRegime +
execute_regime + SeatRoutedCollector + Learner. Local smoke, named experiments
and saved regimes use it; bounded diagnostics may reuse its collection/update
primitives. Distinct learning mathematics, one/both-seat controls and
supervised/search/belief contracts remain required. Policy warm start is a new run; exact
recovery requires admitted complete state and retains previous attempts/costs.

Counterexamples that reject consolidation: cross-seat returns, a different
local optimizer path, eval-mode warm starts, silently translated reward or
schedule semantics, a changed next action/update on recovery, or removal of a
current required consumer without preserving its behavior. The checks below
cover these boundaries; CUDA and generic time-limit bootstrap remain unproved.
Superseded proposal prose and uncommitted incoming notes were preserved at
`.runs/training-compress-input/` before curation. The original audit source is
`95ca2059ea443351cabd4e8c322991c36c0d8212`; the frozen study source is above.

### Accepted implementation direction — 2026-10-22

Jack Heart accepted the consolidation proposal with “great, pursue”. The
2026-10-22 boundary preserves the audit repairs and frozen scientific evidence;
this reconciliation makes no publication, landing or scientific-run decision.

Additional accepted boundary: no Ataraxos-keyed code in the trainer itself.
Ataraxos-named experiments or builders are appropriate. Reusable learning types,
dispatch, objective APIs and execution should describe their actual algorithms
and behavior, not a source experiment's name. Existing mathematics and
semantics remain unchanged; method provenance belongs in documentation/citations and experiment
definitions. Historical configuration decoding belongs explicitly at the input
boundary, without a second trainer path or rewritten frozen recipes/results.
This decision superseded the proposed AtaraxosMoveLearning shared type.

### Resolved implementation choices — 2026-10-22

The exact local defaults and retired flag handling were engineering choices
within the accepted consolidation, not separately selected numbers from Jack
Heart. `local-smoke-v1` uses CPU, two updates, two streams, 16 combined decisions
per stream as a floor, complete games, both seats, the authored baseline model,
terminal outcome rewards, horizon-scaled damped learning and a 120-second failure
deadline. `--learning-seats one` retains exact quotas. `--preset` and training
`--set` fail with migration guidance; historical shaping/schedules cannot be
translated faithfully. Baseline keeps its separate evaluation contract.

### Implementation outcome — 2026-10-22

The accepted boundary is implemented. The audit findings above retain the
original counterexamples; their surviving repairs use the shared owners below.

- `training.learner.Learner` owns train-mode admission, device, Adam state
  (unchanged epsilon 1e-5/default initial rate) and native-stream minibatch width.
  It has no environment, tracking dependency, inheritance or train loop.
  `execute_regime` remains the lifecycle owner. Paired learners take their shared
  collector explicitly; snapshots/incidents receive collector state separately.
- Deleted the two standalone loops, fake environment shim, executable legacy
  training presets, step1–4 verification runners, first-light execution and exp11
  launcher. Retained first-light store/report readers, all historical evidence,
  supervised/search/belief stages and the production PPO learnability control.
  Migrated CI, the Elemental training/reload smoke and direct objective consumers.
  `infra.TrainHypers` and `Hypers` remain serialized historical/diagnostic shapes;
  their unused model re-export and training launcher are removed. Simulation
  presets construct ExperimentHypers directly.
- `DampedPolicyLearning`, `damped_policy.py` and `update_damped_iteration` replace
  experiment-keyed reusable APIs. No trainer dispatch/type is Ataraxos-keyed.
  Provenance stays in docs and experiment/builders. The input compatibility
  owner maps historical `ataraxos_move` to the same runtime algorithm and retains
  its serialized tag. The frozen schema-1 preset remains byte-unmodified and its
  canonical round-trip digest remains
  `43e7ece151480504a795f3d355c792080d8088215151bf9fe24d64e6c22166cb`.
- Bare CLI now runs the bounded recipe specified above through
  the existing executor; completed runs print their admitted raw checkpoint path.
  Baseline retains its evaluation path and accepts the explicit learning-seat
  control. No automatic translation of retired PPO hyperparameters occurs.
- Preserved the draw-shaping repair and hand-calculated credit/gradient evidence.
  The old GAE regressions now test `model.returns.transition_gae`; checkpoint
  train-mode admission now tests the shared learner. The removed mixed-player
  Trainer path cannot be entered; collector routing/truncation regressions remain.
- A broader check exposed two additional stale exact-512-row runtime assertions
  under the both-seat default. They now explicitly cover one/both seats: exact
  quota for one, floor plus native-decision equality for both. The existing
  one/both-seat recovery correction remains intact.
- Removing the diagnostic's tracking construction also removed its RNG reset.
  Review caught the changed game count; retained that setup independently of
  the learner. The final full diagnostic again records 1,541 PPO games.

### Simplified surviving ownership

- Learner APIs use the `learner` name, including LearningPlayer. Learning rules
  alone select update rates; the optimizer retains its historical initial Adam
  defaults for empty-filter snapshots. No unused rate override remains.
- Local smoke validates its seat choice through baseline construction and returns
  the fresh typed regime directly, without another serialize/parse cycle.
  Damped-rule construction has one branch; checkpoint cadence checks its one
  admitted stage directly rather than repeating list traversals.
- The PPO control calls shared `seed_random(197)` at the original post-evaluation
  boundary, without constructing an unused Etude tracking run. The production
  Experiment uses the same helper. Collector cleanup remains explicit.
- Compatibility review found two necessary, different contracts: the learning
  tag reader preserves `ataraxos_move` serialization, while schema-1 regime
  decoding supplies its historical one-seat default. Both remain necessary; merging them
  would lose standalone learning-input round trips or recipe identity. The
  frozen `ataraxos-mtg-v1.json`, result readers and source evidence stay intact.
- The CLI has one explicit-regime branch after recipe dispatch; empty inputs
  fail rather than silently returning success. The inference dependency guard
  now checks the surviving executor/Learner instead of a deleted module.
- No remaining executable Trainer/NetOpponentTrainer/preset consumer was found.
  Historical experiment/protocol references describe pinned source, not current
  APIs. The current LearningRule documentation now names DampedPolicyLearning.

### Concept review outcome — 2026-10-22

Final rereview outcome: **nothing-to-change**, with no open product choice.
The focused recovery-guide correction closes the prior review's only finding:
output scope, coordinates and standalone consumers now match the accepted
executor/collector/Learner ownership and neutral DampedPolicyLearning model.
This rereview inspected the correction and its capture owners; it repeated no
broad audit or tests and leaves all frozen-study proof limits below unchanged.

The prior outcome was **design-change**, limited to the operator-facing recovery guide.
Jack Heart's accepted one-executor contract remains unchanged; no product choice
is open and no runtime redesign is needed. Review against main `8625df97e` plus
the combined tracked and untracked implementation confirms this interaction:

| Action | Concept and state-changing API | Failure/recovery contract |
| --- | --- | --- |
| Run `uv run manabot train`, or select a saved regime and learning settings | Recipe resolves a `TrainingRegime`; `execute_regime` owns the run, `SeatRoutedCollector` owns same-seat experience, and `Learner` owns train-mode policy/Adam updates | Local smoke exports an admitted policy on completion; its deadline is failure, not a shorter successful schedule. One/both seats and PPO/damped learning remain explicit controls. |
| Inspect a failed game | Strict capture retains private inputs and run/stage coordinates; bounded replay diagnoses the retained failure | Capture alone never enables retries. A replay after repair cannot rewrite the original outcome or certify a training restart. |
| Explicitly enable game-error tolerance and, separately, admitted process recovery | Complete-game collection quarantines failed-game rows; the durable attempt ledger owns allowance; `execute_regime(..., resume_from=...)` restores committed learning/collector state | One-seat tolerance also uses complete-game decision floors. Both seats' failed-game samples are discarded before optimization; earlier complete games survive. Restoring a snapshot cannot erase later failures, prior costs or the original applicable deadline. |

The required distinctions have concrete consequences: exported policy weights
are not complete training state, automatic capture is not tolerance, and an
immutable historical recipe tag is not a runtime algorithm dispatch. The
implementation keeps these distinctions without retaining a second RL executor.
The existing counterexamples remain decisive: cross-seat credit, failed-game
samples reaching an optimizer, restored error allowance, changed continuation,
or changed frozen recipe identity would invalidate the consolidation. Inspection
found no new violation; the retained credit/recovery/compatibility proofs cover
these boundaries within their stated CPU limits.

Review found three stale references in `docs/game-crash-recovery.md`: the
deleted Trainer's experiment directory, its coordinates and an executable
"Ordinary Trainer/preset" route. The correction below preserves the guide's
one-seat complete-game exception, fatal-error boundaries, bounded replay,
ledger allowance and source-repair admission. It changes only current operator
guidance; historical plans, recipes, receipts and frozen study evidence remain
unchanged.

### Recovery-guide implementation outcome — 2026-10-22

Corrected only the stale current-guide passages in `docs/game-crash-recovery.md`:
strict training uses the TrainingRun output directory and failed-stage rejected
artifacts; regime coordinates replace the deleted Trainer coordinates; standalone
`sim.rollout`/vector environment capture remains a separate consumer. Verified
against `execute_regime`'s capture scope and artifact registration,
`manabot.env.failures`' unscoped fallback, and the existing strict-regime capture
test. No runtime, historical evidence, publication or Task status changed.

## Integrated behavior and evidence — 2026-10-22

Main `8625df97e` adds w13 rules, automatic game-failure capture, bounded recovery
and source-repair admission. The pre-sync patch remains in stash
`11ba27a584772adabfdbd4259a0e36135c01a2a0`. All six stash conflicts and subsequent
integration leftovers are resolved; no merge/sync or unmerged paths remain.
Legacy trainer loops remain deleted. The current native extension was rebuilt
and installed atomically before the post-sync proof.

Reconciliation compared the surviving math and ownership with main. PPO changes
are owner/import substitutions. Normalizing names in the old damped module
leaves documentation/import changes and the equivalent explicit native-stream
minibatch width. The collector's upstream game-error logic is unchanged. The
executor now passes the collector separately to snapshots/incidents and applies
the durable game-error ledger to that owner after restoring committed state.
Strict capture retains failure coordinates through the existing one/two-stage
executor tests. Source-repair options remain restricted to explicit regime
recovery; frozen recipe bytes and canonical identity remain unchanged.

The training guide now distinguishes strict one-seat exact quotas from opt-in
recovery's complete-game decision floors and distinguishes truncation rejection
from quarantine. Both seats of a failed game are discarded before optimization;
restoring a snapshot cannot erase later attempts or replenish allowance. This
upstream exception changes collection semantics only when explicitly selected,
so comparisons must declare it. Automatic capture itself does not enable it.

Retained CPU evidence:

- Original audit: 42 focused passes; a broader run had 113 passes and two stale
  both-seat exact-count assertions. Their explicit one/both-seat replacements
  passed, including exact optimizer/EMA/RNG/collector continuation for PPO and
  damped learning. Later runtime checks corrected the analogous exact-512-row
  assumptions. These failures were test expectations, not evidence of equal
  decision counts across seat modes.
- Before sync: consolidated affected suites passed, including the 149-test
  compress suite (`.runs/training-compress-tests.log`) and six final CLI/import
  checks. Real local both-seat execution exported after 441 rows; one-seat after
  64 learner/111 native decisions. Default-dependency isolated execution exported
  after 32 learner/49 native decisions. All admitted ordinary policy reloads.
- Full pre-sync learnability is retained at
  `.runs/learnability/training-consolidation-seed-preserved/result.json` and its
  report: source `95ca2059e`, w11, three capacity seeds and production PPO,
  65,536 transitions/1,541 games, scores .59375 before / 1.0 after / .640625
  no-update. The compress PPO-only repeat retained the same counts/scores at
  `.runs/training-compress-ppo-result/result.json`. The initial harness failed
  only while serializing the completed result; the corrected dataclass writer
  produced that retained result. These are not w13 full-diagnostic receipts.
- After sync and native rebuild: `.runs/training-post-sync.log` records 118
  passes in 167.99 seconds across credit, game errors/recovery, strict automatic
  capture/replay, recipe compatibility/CLI and CI selection. This is the current
  integration proof; it does not replace the full world-change diagnostic.
- Caller-reported current w13 arena models, mirror admission and monitor
  evaluation: 31 passed in 61 seconds. Concept review did not rerun these tests.
- Caller-completed full w13 diagnostic, inspected at
  `.runs/learnability/training-consolidation-w13/report.md`: passed in 66.8 seconds,
  65,536 PPO transitions/1,541 games, scores .59375 before / 1.0 after / .640625
  no-update. No tests or diagnostic were repeated for the guide correction.

## Remaining acceptance and limits

No substantial consolidation mechanism, known deletion target or recovery-guide
correction remains unimplemented. Gate owns broader affected-suite acceptance;
the full w13 diagnostic and current arena admission/monitoring have the passes
above. Debug Rust validation remains part of the repository's landing checks.
The existing w11 diagnostic remains valid for its recorded source/world only.
No repository-wide type-safety or CUDA recovery claim is made. Generic time-limit
bootstrap remains unsupported; retired executions require pinned source.

Frozen w6 source applicability above remains separate: no finding establishes
corrupt credit there, and no current-world proof certifies that frozen execution.
Timing observations do not establish equal learning value, decision efficiency
or full-game strength; declared endpoint evaluation is still required. No
frozen study operation, paid compute or evidence rewrite is part of this change.
No additional product decision is needed for the accepted consolidation.

Check: `git diff --check` — passed; guide/code and retained-report inspection reused the 118 post-sync passes, 31 caller-reported admission/monitor passes and full w13 diagnostic; no tests repeated for prose. Gate owns broader acceptance.

Publication inspected the committed range against main `8625df97e`, all incoming
staged/unstaged/untracked work and the retained logs above. `lf pr publish`
created ready PR #356; the returned title/body, exact head and proof limits were
checked. The PR is open, auto-merge is unset and MTG-179 remains completed.
Scratch notes were restored with their staged/unstaged state; private `.runs`
evidence and scratch edits are absent from the published range. No new test,
merge, browser, Task settlement or experiment execution was performed.

## Accepted review simplification — 2026-10-22

Jack Heart explicitly dropped the historical learning-tag compatibility layer,
asked to keep simplifying, selected Config as the single configuration naming
convention, and required experiment/runtime and simulation/training ownership
to agree with module names. This supersedes the earlier tag-roundtrip promise.
Delete LearningInput, the frozen preset executable loader, unused tracker
arguments/methods, legacy Hypers/TrainHypers and their exclusive tests. Preserve
frozen files as evidence, not executable current recipes. New recipes use
damped_policy; old tagged recipes require pinned source and cannot resume
through the current parser. Runtime setup/tracking becomes RunContext and
RuntimeConfig; experiment definitions remain Experiment. Built-in recipes move
out of core training into packaged recipes. Supervised search fitting/evaluation
moves from sim into training; search data generation stays in sim. All live
configuration types use Config, while serialized checkpoint keys stay intact.
No scientific run or evidence rewrite is authorized. Refresh PR356 and the
requested HTML walkthrough after focused validation.

Concept mapping: ActorCritic is the nn.Module with policy/value heads; ModelConfig
constructs it. Learner holds one trainable model and Adam state, with the learning
rule functions owning updates. SeatRoutedCollector gathers same-player experience.
execute_regime is orchestration, not a new Executor object. ModelOverride is an
experiment configuration component, not another network. No runtime aliases.

## Training/sim concept review requested by Jack Heart — 2026-10-22

After the accepted cleanup, Jack Heart requested an HTML guide to the current
training/sim architecture, identifying misplaced training responsibilities and
grouping by how training runs today rather than by generations of experiments.
The guide is `scratch/training-sim-guide.html`; the separate PR implementation
walkthrough is `scratch/pr-review.html`. Both are local review artifacts.

Proposed, not yet accepted or implemented: move SeatRoutedCollector to training;
put portable checkpoint admission/export in model; separate dataset contracts,
search target generation and supervised fitting; extract shared runtime identity
from teacher1_evidence. Group experiment scheduling, checkpoint evaluation and
read-only reports outside core learning. Keep sim/local_update as frozen-policy
search, which owns no trainable optimizer state. Avoid a universal Trainer or
Executor abstraction that hides real algorithm/data differences.

Open scope choice: the only non-test callers found for standalone train_bc and
train_value are scripts/exp10_value_gate.py. Recommend retiring those executable
paths if that workflow is no longer supported, retaining its evidence and pinned
source. If retained, preserve scalar BCE/frozen-encoder and freshest-round
validation contracts explicitly before consolidating through supervised training.
The local experiment runner remains used by the baseline recipe; its existence
is not evidence that it can be deleted in favor of Etude without another design.

Proposed next proof after layout acceptance: local recipe, asymmetric credit,
one/both-seat boundaries, model roundtrip, exact recovery, supervised partition
and target contracts, wheel contents and package import boundaries. Do not
silently relax old source fingerprints to accommodate a file move.

The earlier Wave memory paragraph retaining ataraxos_move compatibility is
superseded by Jack Heart's explicit instruction to drop that decoder. Frozen
bytes remain unchanged and require pinned source; no Wave write is performed
by this concept-review step.
