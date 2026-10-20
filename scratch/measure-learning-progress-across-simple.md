# Curriculum experiment design

Draft for review, revised 2026-10-19 after the complexity contrast completed.
Design only; no implementation, rental or experiment launch. Direction: “Jack
Heart asked for curriculum experiments; design step only” and “Rerun design step
after the account limit reset.” The 2026-10-18 draft is superseded: its seeds
12261–12263 belong to the corrected-world baseline, and its recipe, budget and
evaluation placement were unresolved. Those are now settled by evidence.

## What to build

Test whether spending the first quarter of a fixed training budget on simple
L1 full games makes subsequent Allies versus Lessons (L3) learning faster or
stronger than spending it on L3, at equal learner transitions.

## What the complexity result changes

From [environment-complexity.md](../experiments/environment-complexity.md)
(all nine fits, 36,000 games, 2026-10-19):

- L3 already learns from scratch: +31.7 [+25.8, +37.7] points over
  initialization against greedy at 128 iterations, +27.5 at 64. The question is
  no longer “can a curriculum unlock learning” but “does it buy speed or
  ceiling on a task that already learns.” Expect a small effect.
- Most gain arrives by iteration 64, so a curriculum benefit should appear
  first as speed. The primary milestone is 64, not the endpoint.
- The tested 64x2 recipe costs about 400 s per L3 fit on one L4 (~$0.02).
  Training is nearly free; evaluation is the binding cost. Spend it on
  more seeds, not longer runs.
- Seed spread at the L3 endpoint is a few points; three seeds cannot resolve
  a +5-point effect. Use six.
- Initialization tensors are identical across pools for a seed (checked by
  the GPU job), so per-seed pairing across arms is already proven.
- Frozen controls are effectively one near-uniform baseline per pool.
- `active_training_seconds` is null at initialization checkpoints, so the
  frozen seconds clock was unavailable. This must be fixed before it can be a
  reported clock here.

## Arms

Every arm uses the unchanged complexity recipe: 64x2, 64 streams x 64
transitions, cross-balanced matchups, w6, shared `mtg122-complexity-v1`
research catalog, `iteration_schedule {steps: 128, reference_steps: 2048}`,
128 iterations = 524,288 learner transitions = 8,192 optimizer updates total.

| Arm | Iterations 1–32 | Iterations 33–128 | Boundary at 32 |
| --- | --- | --- | --- |
| A Curriculum | L1 `l1_swift`/`l1_heavy` | L3 `ur_lessons`/`gw_allies` | Learning-state continuation |
| B Split scratch | L3 | L3 | Same continuation |
| C Continuous scratch | L3 | L3 | None (existing fits, seeds 12251–12253 only) |

The boundary carries raw weights, EMA, Adam moments and the schedule clock
(`LearningStateImport` semantics), so iteration 33 uses the 33rd schedule value
in both A and B. Only game streams and RNGs restart, identically derived in A
and B. This is the smallest intervention: A and B differ only in which decks
were played during the first quarter.

- **Primary contrast A − B** isolates the source pool under one identical
  boundary.
- **B − C** (seeds 12251–12253) checks that the boundary itself is harmless,
  at no new training cost; C's L3 scores at 64/128 already exist on the same
  deal families.

Weights-only transfer with fresh optimizer and schedule (the 10-18 draft's
choice) is deliberately not the primary design: it confounds the source pool
with an optimizer/schedule restart. It stays a follow-up, alongside L2→L3,
L1→L2→L3, other switch points and adaptive switching. None is launched by an
attractive result.

## Seeds and pairing

Six seeds: 12251–12253 (reusing their proven shared initialization and the
existing continuous L3 fits as arm C) and fresh 12271–12273. Check the ledger
for collisions before freezing; 1227x appears in no MTG-122/137/135 plan.
No calibration seed is needed: the recipe and its GPU timing are measured.
Every seed runs both A and B; no seed is replaced or dropped.

## Evaluation

Rented CPU (Jack Heart, 2026-10-19), through the ordinary checkpoint loader
and arena, from hash-checked exports, with no inline GPU diagnostics. One
held-out L3 matrix for every arm: Lessons–Lessons, Lessons–Allies,
Allies–Lessons, Allies–Allies, both play orders, 50 paired deals x 8 legs =
400 games per anchor per checkpoint, against fixed heuristic, greedy and
random anchors.

| Milestone | Transitions | Scored | Deal family |
| --- | --- | --- | --- |
| 0 | 0 | Shared initialization (one per seed) | Complexity L3 initialization family |
| 32 | 131,072 | A and B | New, disjoint family |
| 64 | 262,144 | A and B (**primary**) | Complexity L3 midpoint family |
| 128 | 524,288 | A and B | Complexity L3 endpoint family |

A at 32 is an L1-trained policy playing L3 decks it has never seen: zero-shot
transfer, reported descriptively. Reusing the complexity families at 0/64/128
makes arm C's existing cells directly comparable. The frozen no-update
control is scored on every later milestone family for every seed (sharing the
reused seeds' existing cells by digest is a possible later saving). Missing
games stay incomplete.

### Anchor validation, before any curriculum game

Jack Heart expects `heuristic` (MTG-152) to be the opponent that matters most,
but that is a hypothesis to test. MTG-152's exploratory check (600 games, one
baseline checkpoint) found lower scores against it in every cell. No heuristic
cells exist yet for the complexity fits. Validate it on the retained L3
complexity checkpoints (seeds 12251–12253: initialization, 64, 128 and frozen
controls, on their existing deal families), with no new training:

1. **Stronger:** heuristic beats greedy head-to-head, whole-deal interval
   above 50%. Evidence: MTG-153's greedy-vs-heuristic pairing, 25 deals × 8
   legs = 200 games on the product Allies vs Lessons decks (same cards and
   programs as the research L3 decks). Accepted instead of a separate
   400-game run by Jack Heart, 2026-10-19, before any curriculum scoring.
2. **Headroom:** every checkpoint scores lower against heuristic than against
   greedy on the same deals, and no endpoint exceeds 80%.
3. **Resolves learning:** all three seed gains over initialization at 128 are
   positive against heuristic, and the ordering 0 < 64 ≤ 128 matches greedy's.

If all three pass, heuristic is the curriculum's decision anchor; otherwise
greedy is, with heuristic as a labelled panel. The rule is frozen now and
applied before any curriculum cell is scored, so the curriculum result cannot
choose its own anchor. Criteria 2 and 3 are answered by the supplementary
heuristic cohort already scored for the complexity fits (`a9d8addd`): L3
endpoints score 33.8% against heuristic versus 55.3% against greedy, seed
gains are +25.2/+30.0/+19.0, and gains rise from midpoint to endpoint. They
still go through `anchor_validation` once MTG-153's pairing exists. No new
validation games are needed here. Either way the result is reported in its own right: it says whether the
complexity contrast's greedy conclusions hold against the stronger opponent.

Games: ten 400-game cells per anchor per seed (initialization; both arms and
the control at 32, 64 and 128), so 4,000 per anchor per seed and 72,000 for
six seeds and three anchors. At the complexity
cohort's 5 process-seconds per game that is ~110 CPU process-hours, a few
hours on one many-core rented pod. Per-game speed on rented hardware is unmeasured; see Budget.

## Pre-registered analysis

- **Primary:** per-seed paired A − B gain against the validated decision anchor
  (heuristic or greedy, per the rule above) at 64, averaged equally over
  the four matrix cells; six seed effects, mean, 95% t interval (df 5).
- **Useful:** mean ≥ +5 points, interval lower bound > 0, at least five of six
  seeds positive, and no random-anchor regression: the random-anchor A − B
  six-seed interval is not entirely below zero.
- **No useful effect:** interval entirely inside (−5, +5).
- **Harmful:** mean ≤ −5 with interval upper bound < 0.
- Anything else is inconclusive and reported as such, not rerun with more
  seeds until it passes.
- **Secondary:** A − B at 128 (ceiling) and at 32; time to +10 points over
  initialization on transitions and on active seconds, right-censored at 128;
  per-cell results; B − C boundary check; whole-deal paired intervals
  conditional on each seed. Games are never treated as independent runs.

Clocks: learner transitions is the matching clock. L1 iterations run about
15% faster than L3, so A uses slightly less active time than B; a
seconds-matched comparison would favor A, which makes transitions-matching the
conservative choice. Report active seconds, wall time, games and decisions so
the imbalance is visible.

A positive result says L1 pretraining speeds L3 learning for this recipe at
this scale. It promotes no baseline and says nothing about the Academy opponent.

## Budget and placement

- **GPU:** one L4 job runs all twelve fits (A and B for six seeds), each fit
  as source then target `execute_regime` calls in the same process, so the
  transfer never crosses hosts. Expected ~80 minutes. The admitted lease is
  the watchdog bound, 12 × 1,740 s + 600 s = 21,480 s, ≤ $4.12 at $0.69/h;
  no automatic paid retry.
- **Rented CPU evaluation:** the curriculum's `evaluate` command, run on
  rented CPU through Etude's CPU launcher, which Etude is building (Jack
  Heart, 2026-10-19); no curriculum-specific launcher. MTG-135's one-off
  heuristic panel (cpu3c, 8 vCPU, $0.24/h) found pods about 2.2× slower per
  game than the Mini with 8 workers on 4 physical cores, and a 120 s game
  budget truncated a game. The reused complexity protocols keep 120 s, so run
  about one worker per physical core rather than change the protocol. Before
  science, a short bounded calibration measures seconds per game and the
  pod's hourly price from the live catalog. The cap is then frozen from those
  measurements, with a stop rule that leaves unfinished cells incomplete.
- **Allocation:** part of MTG-122 (Jack Heart, 2026-10-19), charged to its
  $35 inclusive cap (~$4.85 spent). GPU, CPU calibration, validation,
  evaluation, failed attempts and transfer all count. No automatic paid retry.

## Data structures and key functions (implemented in slice 2)

`admit_learning_state` (`manabot/training/learning_state.py`) already
continued a single-stage self-play parent on absolute iterations, checking
world, recipe, schedule, EMA, Adam and parameter order. Its only blocker was
the unchanged-match rule.

- `LearningStateImport.source_match: MatchHypers | None` declares the deck
  switch. With it, the parent's match must equal `source_match`, both matches
  must name the agent's own semantic pack, and the match must actually change.
  The parent policy is validated against the match it trained on; the
  segment's exports are written bound to the target match, so ordinary L3
  evaluation and replay admit them. Omitted, nothing changes. The declaration
  is part of the target regime, hence its digest and lineage.
- Continuations require `schedule_clock = "iteration_fraction"`. The
  complexity recipe records `run_elapsed_budget`, but Ataraxos move updates
  read only the absolute iteration, so learning is unchanged. Both arms
  declare it; every fit pins w6, including CPU fixtures.
- `experiments/runners/curriculum_campaign.py`: `Recipe`, `source_recipe`,
  `target_recipe`, `train_fit` (both segments in one process),
  `fit_checkpoints`, `evaluate_seed`, `CurriculumPlan`, the GPU `execute`
  job and the `evaluate` command. `evaluate` requires an Etude allocation
  (`ETUDE_RUN`, `ETUDE_DEADLINE`) and refuses local hosts. It does not yet
  provide that rented CPU allocation.
- `CurriculumPlan` freezes six seeds, switch 32, 128 updates, 64 streams,
  CUDA, a 1,740 s watchdog per fit split by iterations, and the frozen
  complexity plan's digest. It admits GPU rental + prior spend +
  `evaluation_usd` + $3 reserve within the $35 cap. Its protocols at 0/64/128
  are equal to the complexity cohort's L3 protocols; 32 uses deals
  1,227,132,000–049.
- Untrained initialization checkpoints from `checkpoints()` now record
  `active_training_seconds = 0`. Complexity's own `fit_checkpoints` is
  left unchanged so its frozen cells keep their meaning.

- `experiments/runners/curriculum_analysis.py` encodes the rules below before
  any data exists: `paired_difference` (pairs two runs on identical
  protocol, deals, anchor, evaluator, seed and iteration), `six_seed_summary`,
  `verdict`, `time_to_progress` and `anchor_validation`.

Not implemented yet: reusing complexity cells by digest. Slice 2 scores the
initialization and controls for every seed, about 7,200 extra greedy/random
games for the reused seeds. Report and notebook are slice 3.

## Implementation slices

1. Design — done.
2. Software — done 2026-10-19, uncommitted on this branch; see Done when.
3. Separately authorized: freeze `experiments/curriculum-plan.json` from the
   measured CPU calibration; heuristic validation; GPU job; fetch with digest
   verification; rented-CPU evaluation; notebook/HTML report beside the
   complexity report.

**Forbidden:** rewriting source bindings; bypassing loader checks; product
packs; selecting or replacing seeds; comparing raw L1 and L3 win rates;
changing recipe or capacity with the curriculum; dropping failed arms or deals;
promoting a baseline.

## Done when

Software gate (slice 2; passed 2026-10-19: 6 curriculum tests, plus 106 in
learning-state, pretrained, checkpoint-queue, complexity-campaign,
active-training, step-cadence, remote continuation/jobs and world suites):

```sh
uv run pytest tests/training/test_curriculum.py tests/training/test_learning_state.py tests/training/test_pretrained.py tests/training/test_checkpoint_queue.py tests/training/test_complexity_campaign.py tests/training/test_active_training.py tests/training/test_step_cadence.py tests/remote/test_continuation.py tests/remote/test_jobs.py tests/model/test_world.py -q
```

Science: twelve completed fits with lineage receipts, every planned cell scored
or retained as incomplete, the pre-registered verdict, cost receipts and an
editable notebook generating read-only HTML.

## Decisions (Jack Heart, 2026-10-19)

1. Budget: part of MTG-122, under its $35 inclusive cap.
2. Evaluation: rented CPU, not the Mini.
3. Heuristic is probably the opponent that matters most; validate that
   hypothesis first (above) rather than assume it.
4. Six seeds.

5. Heuristic check: MTG-153's 200-game greedy-vs-heuristic pairing is fine.
6. Rented CPU: use Etude's launcher when it lands.

Remaining dependencies: Etude's CPU launcher (then a short calibration) and
MTG-153's scientific attempt.

Check (2026-10-19): read the completed complexity results, `admit_policy`,
`PretrainedPolicy`/`LearningStateImport`, `validate_agent_setup`, Etude job
restrictions, seed usage in the MTG-137 plan, and retained GPU outputs. No
tests, games, updates or rentals.
