# Ordinary baseline learning regression — 2026-10-15

Status: software execution qualified; outcome learning remains inconclusive. Jack Heart requested a
nightly focused on what ordinary training learns, retaining the existing
handcrafted capacity checks as sanity tests. Bare `manabot train` remains the
small smoke run. `manabot train --recipe baseline --steps N` selects the higher
level recipe shared by the nightly and interactive CLI.

## Frozen question and prediction

Can a bounded current-world Allies/Lessons self-play run show paired outcome
improvement, while preserving action information and checkpoint invariants?
The baseline uses the existing Ataraxos move update, scalar masked-mean value,
width 64/depth 2/four heads, four streams of 64 transitions, current-self play,
and terminal game outcomes. There are no probe labels, modified rewards,
per-choice curricula, or optimization steps in evaluation.

The new recipe maps the declared horizon onto reference iterations 1–10,000
of the existing learning-rate/tau schedule. Endpoints and normalized-progress
rates match across run lengths. A one-step run uses the initial rates. This is
a deliberately horizon-dependent schedule, not a prefix guarantee or a claim
of reproducing the historical absolute schedule. Existing regimes omit this
mapping and keep their behavior and serialized identities.

Before execution: predict all software invariants pass; at 256 updates predict
positive outcome gain in at least two of seeds 197/281/419, but evidence may be
inconclusive. A 16-update preliminary run calibrates throughput and the complete
reporting path only; it is not selected as a learning qualification. Preserve
all attempts. Do not change thresholds after inspecting their scores.

## Measurements and acceptance

- Three independently initialized seeds, identical recipe/horizon, and 24
  reserved regression deals (1915100000–1915100023). Each checkpoint plays all
  four deck/seat legs against uniform legal random through the ordinary arena.
  Initial and final evaluation are paired by deal/leg and action-sampling scheme.
  Both deck assignments are retained in the raw rows and paired leg differences.
- The primary statistic is mean final-minus-initial game score, draws worth
  one half. A fixed-seed hierarchical bootstrap resamples training seeds and
  complete deal blocks; the shared deal indexes are resampled jointly across
  seeds. Never treat individual decisions or four legs as independent games.
  Require mean gain >= .05 and a positive 95% lower bound for `learning`.
  Symmetric negative evidence is `regression`; otherwise `inconclusive`.
  Missing seeds, incomplete games, non-replaying traces or runtime violations
  are `error`. Only `learning` is a passing learning gate. Per-seed reports are
  conditional on that trained instance, not claims about the method.
- Save initialization, midpoint and endpoint checkpoints. Reuse their frozen
  weights against the same model-independent, seed/trace-retained corpus of
  native public choices. Report per-family pair log-odds and value changes.
  No direction or minimum drift is required: random weights can distinguish,
  sensible choices can tie, and useful learning need not change every family.
- Audit native/scalar encoders, legal masks and focus joins during collection.
  On ordinary loaded checkpoints check batching, offer/object permutation,
  value invariance, zero padding probability and a second ordinary reload.
  Existing exact setup/world admission stays mandatory. Probe labels never
  reach the learner; the frozen corpus is outside training's seed namespaces.
- Summarize the existing learner selection diagnostics by chosen action type:
  collected and retained rows. These counts are not decision-family exposure
  and do not prove a nonzero actor gradient for every retained sample.
- Keep the full short learnability workflow as a separate sanity job and its
  existing relevant-PR triggers. Its artificial preferences remain labeled as
  capacity tests. The 30-minute generated capacity search remains available
  manually, without consuming most of the ordinary-learning nightly budget.

## Reliability and cost

Common evaluation deals and fixed initial/final pairing reduce avoidable noise.
Independent seeds and complete-case requirements prevent one lucky run or a
partial cohort from producing green. No-update, positive, negative and
seed-disagreement analysis controls are tested without training. Initial and
endpoint are chosen before scoring; there is no best-checkpoint selection or
requirement that every intermediate checkpoint improve.

Strongest confounds: random is weak, and the two decks differ in strength;
outcome improvement against it is early learning evidence, not tactical mastery.
The fixed choice corpus is generated independently, not necessarily visited by
the trained policy. Logit/value movement is descriptive and cannot establish
strategic correctness. With only three seeds and 24 deals, small improvements
can remain inconclusive; a longer run does not justify weakening the criterion.
Repeated nightly deals are regression/development evidence, not a perpetually
unseen scientific test set.

Use three concurrent standard GitHub-hosted runners in this public repository:
$0 purchased compute, no launcher credentials, no laptop/mini training. Each
baseline has a 1,680-second coordinator limit plus a 1,800-second external
watchdog for its reporting; setup is additional and jobs have a 35-minute cap.
The cohort uses up to about 90 runner-minutes plus setup/evidence/sanity. The
nightly cron only becomes active after merge. Remote calibration uses a
temporary branch/commit-marker trigger, removed from the final workflow.

## Qualification attempts

- Run 38791569833, source 793d4dd1: all three 16-update attempts stopped
  before training because the minimal CPU environment lacked `nbformat`, used
  by the existing experiment coordinator's notebook writer. Original artifacts
  are retained. Add the direct runtime dependency and a report-creation preflight;
  retry only after that repaired cause. No learning score exists for this attempt.
- Run 38792239992, source e1d2bfb1: all three 16-update learners completed,
  but arena admission rejected the setup identity. The new recipe used default
  player labels rather than the selected arena's labels; decks/sideboards were
  identical. Use `selected_match()` as the shared owner and preflight the real
  monitor manifest with an ordinary checkpoint. Seed 281 recorded 66.30 seconds
  of collection+learning, 4,096 transitions and 36 completed games. This is
  throughput evidence only; no outcome comparison was admitted.
- Frozen probes now use three independently seeded eight-game prefixes (up to
  96 selected states), scored in batches of at most 32 to bound memory. Cheap
  native-only local verification collected all 96 states across ten families in
  7.10 seconds, with zero audit findings and no training. Record this prospective
  corpus expansion before the next remote run; it does not change learning gates.

### Prospective nightly horizon after throughput calibration

The first 16-update training receipts measured 100.12, 66.30 and 91.75 seconds
for seeds 197/281/419 (collection plus learning). Outcome evaluation remains in
progress in run 38792948721 and no outcome scores have been inspected. At the
slowest measured pace, 256 updates alone project to 26.7 minutes, leaving too
little of a 30-minute run for final evaluation. Select **128 updates** for the
first complete nightly qualification and the recipe's bounded default. Keep
all learning parameters, horizon mapping, paired deals and acceptance thresholds
unchanged. This is a cost-driven prospective horizon choice, not selection on
outcomes. Nightly omits `--steps` unless manually overridden, so it inherits the
single public recipe default. Longer runs use the same `--steps N` function.

Also reuse the existing all-offer input collision audit in the frozen-policy
suite. All 96 prospective corpus states had zero clean input collisions in a
native-only check. The declaration/definition erasure controls remain in the
cheap sanity tests; this adds no training labels or fitting.
- Run 38792948721, source 58532d7e: seed 197 completed the 16-update
  end-to-end qualification. Initial/midpoint/endpoint all passed 96-state
  invariants across ten families (maximum error 2.54e-7). Its initial/final
  scores were 47/96 and 50/96; paired gain .03125, interval [0, .0625], correctly
  `inconclusive`. Coordinator elapsed 751.11s, evaluator charge 745.98s;
  initial and final evaluations took 372.52s and 368.85s. The other two seeds
  are still running. The 128-update horizon decision was committed at 9ae00d62
  before these scores were read. Queue the first full 128-update cohort with
  unchanged criteria; all attempts remain retained.


### Complete 16-update calibration

All three software checks passed in run 38792948721 at 58532d7e. Each exported
initial, 8-update and 16-update checkpoints and passed the same 96 native states
across ten choice families. The aggregate learning gate correctly returned
`inconclusive` (exit 2), so the workflow is red despite successful execution.

| Seed | Initial score / 96 | Final score / 96 | Paired gain | Coordinator seconds |
|---|---:|---:|---:|---:|
| 197 | 47 | 50 | +.03125 | 751.11 |
| 281 | 50 | 49 | −.01042 | 1270.27 |
| 419 | 49 | 47 | −.02083 | 1208.46 |

Combined gain is 0, with hierarchical 95% interval [−.04167, +.04167].
This is no evidence of improvement at 16 updates. Frozen-policy movement and
training exposure did not override that result. Evaluation dominates runtime;
those measurements include overlapping learning and evaluation and must not be
summed as sequential wall time.

Full qualification run 38795378947 uses source 49316156 and the previously
selected 128-update horizon. No acceptance threshold changed after calibration.


### Full 128-update result

[Run 38795378947](https://github.com/loopflowstudio/manabot/actions/runs/38795378947)
at `49316156e9ac47a4dc2b8e2e6baa67526a3c69ef` completed all three training
and inspection jobs, plus the retained sanity suite. Every learner collected
32,768 transitions and exported initial/64/128-update checkpoints. Each frozen
checkpoint passed the same 96 native states across ten families; the largest
invariance error across the cohort was 2.98e-7 (limit 1e-4). No clean input
collision was found. These are software and information-preservation checks.

| Seed | Initial score / 96 | Final score / 96 | Paired gain | Coordinator seconds | Learner process seconds |
|---|---:|---:|---:|---:|---:|
| 197 | 47 | 45 | −.02083 | 1329.64 | 746.92 |
| 281 | 50 | 49 | −.01042 | 1397.90 | 760.49 |
| 419 | 49 | 52 | +.03125 | 1319.58 | 707.90 |

The complete cohort's paired gain is **0**, with hierarchical 95% interval
**[−.09375, +.08333]**. The evidence job returned `inconclusive` / exit 2;
GitHub therefore marks the workflow failed. This is not an analysis error or
an established regression. Only one seed improved at the cost-adjusted horizon;
the original prediction for 256 updates was not directly tested. No threshold,
seed selection or checkpoint selection changed to obtain a passing result.

This qualification establishes the run/evaluation path within the elapsed-time
budget, not reliable learning at that budget. The nightly can measure this
uncertainty and detect software failures, but cannot yet be described as a
reliably green learning regression. Repeating it unchanged does not make an
inconclusive result proof of learning. Further recipe or evaluation-power work
needs a prospectively declared comparison; it must retain this cohort and must
not turn arbitrary policy movement into an outcome claim.

The [compact receipt](../docs/evidence/learnability/baseline-qualification.json)
retains all four remote attempts, source revisions, artifact digests, paired
outcomes, checkpoint hashes, family changes, exposure and costs. Original
models/traces/reports are in the linked Actions artifacts (14-day retention;
aggregate 30 days) and local `.runs/baseline-qualified-{16,128}` directories.
All compute used standard public GitHub runners; purchased compute was $0.
The final workflow has only nightly/manual triggers; branch calibration triggers
were removed. Scheduled execution starts after merge. No automatic merge is
armed, and Discord activation remains unconfigured.
