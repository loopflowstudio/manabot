# Ataraxos technique screens for MTG

2026-10-05. ETU-105 owns learning-rule, credit-assignment and systems screens.
Jack Heart authorized this **software/documentation delivery only**, including
source checks, declarative plans, focused tests and landing. No new training,
calibration, arena games or scientific allocation is authorized in this pass.
ETU-105 stays open for the initial empirical screen and its report/notebook.
The eight-hour value-token screen runs in its separate frozen checkout; ETU-91
is retained incomplete after its first seed. Neither is modified or restarted.

## Sources and existing evidence

The final [Nature paper](https://www.nature.com/articles/s41586-026-11036-y.pdf),
Methods (“Self-play training data generation”, “Dynamically damped self-play”)
and Extended Data Figs. 5–6, and the [supplement](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-11036-y/MediaObjects/41586_2026_11036_MOESM1_ESM.pdf),
S3.3–S3.7, were retrieved and read on 2026-10-05 (Pacific).
PDF SHA-256: paper `aa54309bf216c04708f0319c38ebfe2b89ade93bc375ab300e5a2073cd08763b`;
supplement `74f217e942cd824140dd5221c02d8c36821a715664f47aee6f489f0ca6932a08`.
The HTML fetch failed at Nature's identity redirect; the final PDF succeeded.
PDF bytes are research inputs, not checked-in training artifacts.

The source's move rule includes **clipped importance ratios**. “Damped versus
PPO” is consequently a recipe comparison, not clipping versus no clipping.
The source uses separate scalar advantage and categorical outcome estimators,
two reverse KL terms, scheduled step sizes/regularization, filtering and
post-iteration evaluation EMA. Its setup estimator and compound structure are
separate from move learning. Source experiments motivate candidates; they do
not establish their MTG benefit.

The [fidelity ledger](../docs/ataraxos.md#fidelity-table) owns implemented equations
and unresolved conventions. [Omitted controls](ataraxos-omitted-controls.md),
[compound contracts](../docs/training-regimes.md#compound-decisions),
[sampler reporting](../docs/belief-sampler.md), [local search](../docs/local-policy-search.md),
[tactical diagnostics](../docs/checkpoint-scenarios.md) and
[recovery/calibration](../docs/training-recovery.md) cover delivered ETU-92–98/100.
This inventory supersedes their prospective research ownership for learning-rule,
credit and systems screens only. Larger belief/search/exploiter studies stay
with ETU-99. ETU-101 owns W&B; architecture/input studies keep their own ownership.

Prior evidence is preserved: ETU-91's tiny PPO ablation included a timeout and
negative/ambiguous results; these are not decisive treatment verdicts. The
supplied task context establishes that disposition, but this checkout does not
contain those retained run exports, so no missing score, path or timeout cause
is reconstructed. Separately, the [historical integration table](training-regimes.md)
records a failed conflict-marker attempt and completed software smokes; that
failure must not be substituted for the scientific timeout. ETU-93's 48-game
EMA-behavior and 1,073-decision complete-game diagnostic, ETU-94's 56-game
compound proof, and ETU-92's 512-transition proof establish wiring only. Their
receipts and limits remain in the linked owners. No prior tiny result justifies
deleting a candidate, changing endpoint deals, or selecting a winning seed.

## Candidate inventory

“Test now” means admitted to the next protocol review with runnable software,
**not permission to execute now**. Every expected effect below is an MTG
hypothesis. Cost is relative to the paired baseline, not measured calibration.
Source locators describe motivation; the departures column limits fidelity.
Generator names below are accepted by `technique_screen --contrast`.

| Candidate and exact source | Current instrument / disposition | MTG departure and independent contrast | Expected effect; cost; interactions |
| --- | --- | --- | --- |
| Discount / long horizons: S3.4, Table S6 same-player return sequence | `Learning.gamma`, `discount`: **test now**, secondary family | Undiscounted outcome target is move baseline; .99 vs 1 only in delivered scalar PPO. No gamma field on categorical move rule. | Earlier reward emphasis may ease credit but bias patience; similar work; interacts with decision clock and traces. |
| Policy trace: S3.4 paragraph before (5), S7 | `AtaraxosMoveLearning.policy_lambda`, `policy-trace`: **test now** | .5 vs .95; same-viewer MTG transitions, value lambda fixed .8 | Longer credit may add variance; same collection; interacts with filtering. |
| Value trace: S3.4 (5), S7 | `value_lambda`, `value-trace`: **test now** | .8 vs 1; categorical targets in both arms. Unfinished batch tails still bootstrap at lambda=1. | Less intermediate bootstrap bias, more variance; similar compute; interacts with policy trace/critic quality. |
| Terminal-return vs bootstrap: Methods interdependent processes; S3.3 vs S3.4 | `TrainCompound.estimator`, existing four-arm compound study: **test now**, separate family | Complete games, outcome vs bootstrapped targets, both grouping choices. Ordinary lambda=1 is **not** full terminal-return collection. | May help delayed credit; complete games increase collection/storage and variance; grouping changes credit units. |
| Damped move recipe vs PPO: Methods damping; S3.4 (6) | `ataraxos.update_move_iteration` / `objectives.update_iteration`, `ppo-package`: **test now** as a package control | Scalar output in both arms because delivered PPO cannot train WDL. Normalization, reuse/minibatching, schedules, filtering, traces, coefficients and clipping differ; no isolated damping attribution. | Unknown net effect; different optimizer exposure/cost; all component interactions remain confounded. |
| Structured-reference choice: S3.4 S6/(6) | `reference_distribution`, `reference`: **test now** | Action-type uniform vs uniform offers, not Stratego piece-then-move grouping | May protect rare action classes or overweight poor choices; same support-sum cost; interacts with tau and offer counts. |
| Collection-policy reverse KL: S3.4 (6) | Saved full behavior distribution, `collection-kl`: **test now** | .1 vs 0; actual same-viewer collection probabilities retained | Stability vs slower adaptation; same data, small loss-cost change; interacts with reuse and EMA behavior. |
| Ratio clipping: S3.4 (6), S7 | `damped_policy_loss`, `ratio-clip`: **test now** | .2 vs .1 clipping radius, retain both KLs | Smaller steps may stabilize or stall; similar cost; collection KL and norm cap can dominate. |
| Gradient norm cap: S3.4/S7 | `max_grad_norm`, `gradient-clip`: **test now** | .267 vs .5, other learning settings fixed | Larger updates may learn faster or destabilize; similar cost; raw gradient norm and clipping activation needed. |
| LR schedule: S7; Extended Data Fig. 5b | `rates`, `lr-constant`: **test now only after sufficient iteration coverage** | Hold initial effective 1e-4 vs delivered clipped power law; unchanged tau clock | Benefit may appear late; enough updates to exit upper clamp required; interacts with filtering/exposures. |
| Reference regularization schedule: S7; Fig. 5b | `rates`, `tau-constant`: **test now** | Hold initial .05 vs iteration decay; learning rate unchanged | Sustained exploration vs underdeveloped policy; similar cost; inspect entropy and reference KL. |
| Coordinated schedules: Methods damping; S4 | `schedules-constant`: **deferred** to selected interaction test | Both held constant vs both scheduled; combine with preceding single removals for 2x2 | Possible complementarity; four cells, not two independent effects; budget separately. |
| Quantile threshold: S3.4 selection items, S7 | `selection_mask`, `filter-quantile`: **test now** | .75 vs .5 inclusive quantile, min magnitude fixed .01 | More training rows may help credit or waste fit time; potentially up to twice exposures; critic and trace interactions. |
| Magnitude threshold: S3.4 selection items, S7 | `filter-minimum`: **test now** | .01 vs 0, quantile fixed .75 | Retain low-signal rows; cheap unless many near-zero advantages; interacts with value accuracy. |
| No filtering: S3.4; Fig. 5 | `filter-off`: **test now** as combined filter removal | Both thresholds zero vs baseline; not threshold attribution | More coverage but more fit work; up to roughly fourfold rows absent ties/threshold effects; same collector. |
| Selection ties: S3.4 inclusive condition | `filter-ties`: **deferred** until tie-rate diagnostic | Inclusive quantile vs fixed top count; source quantile interpolation remains unspecified | Mostly software sensitivity unless ties are common; changes selected counts; estimate distribution governs cost. |
| Actor-only vs actor+critic: inspired by S3.4 common selected moves | `filter_scope`, `filter-scope`: **test now**, additional hypothesis | All critic rows vs filtered critic rows, identical actor selection | Better critic coverage may reduce selection feedback; more critic work; shared encoder can change policy even with zero actor rows. |
| Raw/EMA evaluation: S3.4/S7, Extended Data Fig. 6 | `evaluation-ema`: **test now** | One training cohort, two .999 artifacts per cutoff; EMA is evaluation averaging, not equilibrium policy averaging | Smoother evaluation vs lag; doubles arena variants, no second training; same-seed correlation mandatory. |
| EMA behavior: extension of evaluation EMA, **not a source prescription** | `TrainSelfPlay.behavior`, `behavior-ema`: **test now**, later priority | Current-self vs EMA-self, actual behavior likelihoods and bootstrap; evaluate raw and EMA for each | Smoother opponent vs policy lag; extra forward/storage costs; collection KL/trace interactions. |
| Compound credit: Methods setup decomposition; S3.3 (1)–(4) motivates only | `TrainCompound.grouping`, existing four arms: **test now** | Same recurrent legal-offer decoder, sequential conditional-factor vs grouped joint credit at gamma=1. Not flat policy vs compound architecture. | Different variance/credit cost; joint and prefix regularizers differ; cross with terminal/bootstrap. |
| Setup learning / entropy-to-go: S3.3 | **Inapplicable** to fixed authored deck setup | No learned Stratego arrangement; no invented MTG setup loss | A future deck-construction study would be a new task, not this toggle. |
| Value output, token, depth, width, history: S2/S3.6 | **Deferred to architecture owners**, implementations vary | Hold pinned model constant here; PR227 capacity remains independent | Do not charge an architecture cross to a learning-rule screen. |
| Belief learning and local search: S3.5/S3.7 | ETU-96/95/100 instruments; **deferred to ETU-99** | Hand counts, lossy commitment summary, sampled-support KL; exact-history limitations retained | Full calibration/search cost and frozen-policy matching need separate study. |
| Exploiters and tactical diagnostics: evidence qualification, not a move-loss toggle | ETU-97/frozen-opponent instrument; **deferred to ETU-99** for large scoring | Scenario checkpoints require matching setup; an unsuccessful bounded attacker is not a certificate | Use later to qualify promising strength, with new budgets. |

Implemented capability is not an “already tested” benefit disposition. None of
these learning techniques has a decisive MTG method-level verdict in this pass.
Source hyperparameters unspecified by the supplement (including quantile
interpolation, Adam details, EMA initialization and bootstrap pseudocode) retain
the explicit choices in the fidelity ledger.

### Systems inventory (separate from statistical treatments)

| Candidate / source | Disposition and instrument | Contrast, likely effect, cost and confound |
| --- | --- | --- |
| bfloat16 / mixed precision: Extended Data Fig. 5a, S3.5 Table S8 | **Deferred: unimplemented on complete training path**; `Execution` admits CPU float32 only | Exact same rule/model/workload in float32 vs supported mixed precision after numerical/reload/legality checks; hardware-specific throughput hypothesis, no laptop speedup claim. |
| Batch/stream size: S3.4 timestep batches, S7 | Existing `TrainSelfPlay.streams/transitions`; **deferred** pending matched data/update contract | Throughput may rise; changing streams also changes timestep minibatch size and gradient noise. Fixed total rows alone does not make it a pure systems test. |
| Inference batching / transfer overhead: Methods transformer tradeoff | Existing vector collector and ETU-98 calibration; **deferred** to measured bottleneck | Frozen-policy forward/step benchmarks first, then complete loop. Preserve row order, behavior bytes and bootstrap boundaries; inference SPS alone is insufficient. |
| Distributed placement: Extended Data Fig. 5a | **Deferred to ETU-108** research; no distributed learner here | Separate transport/placement from estimator and stale-data admission. No extra host access or paid compute. |
| Recovery / EMA buffers / replay: software correctness | **Already tested as capability**, ETU-98/90/100 linked contracts; no benefit claim | Retain failed attempts and immutable exports; not a statistical arm or evidence of strength. |

## Runnable plans and admission

PR225 landed at `3a29e83c`; this runner imports its `Experiment`, `Case`,
`LearningRule`, `Pipeline`, `Model` and pinned `ataraxos_mtg_v1` APIs. No new
registry, authoring language or executor is introduced. PR227 is not needed.
The preset digest is
`43e7ece151480504a795f3d355c792080d8088215151bf9fe24d64e6c22166cb`.
It fixes authored Allies/Lessons with sideboards, semantic observations,
categorical WDL, width 64/two-layer value token, CPU float32/one thread and
current-policy collection. This is a fixed local architecture, not an assertion
that its uncompleted architecture study has selected it as best.

Each primary pair differs only at its named settings; `filter-off`,
`schedules-constant` and `ppo-package` explicitly change multiple settings.
`discount` preserves ETU-93's scalar PPO baseline and is not pooled with the
categorical family. `ppo-package` holds the preset model scalar in both arms.
The two cumulative preset stages request one update each (4 streams × 64
transitions), 80 seconds/stage, 180 seconds/run. Discount reuses its two-update,
60-second/stage, 150-second/run proof recipe. These are workload ceilings,
not measured training duration or sufficient learning.

Plan export is safe in this delivery pass and fails if the destination exists:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m experiments.runners.technique_screen \
  --contrast filter-scope --out .runs/etu105-filter-scope-plan
```

It writes `plan.json` (ordinary `ResolvedStudy`), `provenance.json` (complete
resolved settings and origins) and `admission.md` (explicitly unexecuted).
The CLI lists all admitted names with `--help`. Every plan has one seed 1051,
one four-leg paired deal 951051 and random-anchor deal 952051, two checkpoints,
a 900-second whole-process/all-attempt ceiling and no inherited campaign cost.
EMA evaluation uses one recipe; EMA behavior uses two recipes × both variants.
Repeated use of these software seeds never constitutes independent evidence.

The following commands are **future execution examples**, not authorized now.
They use the existing runner; never silently scale a smoke into science:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --extra notebook \
  experiments/runners/run_training_regimes.py --study omitted-controls \
  --profile smoke --plan .runs/etu105-filter-scope-plan/plan.json \
  --out .runs/etu105-filter-scope-attempt-1
uv run --extra notebook experiments/runners/run_training_regimes.py \
  --report-only .runs/etu105-filter-scope-attempt-1
```

For compound credit, reuse the existing four exact recipes, with gamma=1,
unchanged decoder, raw evaluation and both estimators/grouping values:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --extra notebook \
  experiments/runners/run_training_regimes.py --study compound-decisions \
  --profile smoke --out .runs/etu105-compound-attempt-1
```

The existing compound smoke is capped at 900 seconds for the entire four-arm
study; individual recipe maxima are not an additive allocation. Admission
fixtures test the four-cell interpretation without executing this command.
[Selection diagnostics](ataraxos-omitted-controls.md#frozen-complete-game-selection-diagnostic)
can reuse a saved admitted policy with `--source-run` and no fitting, but still
collect games and therefore require later execution authorization.

`experiments/study/ataraxos-techniques.ipynb` inspects exported plans without
execution, and can display saved reports after the empirical pass. The existing
runner generates the full evidence notebook/report using
`experiments/study/training-regimes.ipynb`. Until a complete new cohort exists,
strength, uncertainty and treatment dispositions are unavailable.

## Ordered program and proposed budgets

1. **Now, software only:** gradient/terminal-boundary fixtures already owned by
   ETU-92; exact intervention/provenance tests, invalid model/rule rejection,
   EMA replicate counting, compound 2x2 admission and schedule activation checks.
   No games or optimizer workloads are needed for the new plan checks.
2. **First empirical cohort, proposed 4.5 laptop hours inclusive:** prioritize
   `filter-scope`, `collection-kl`, then `evaluation-ema`. Allocate 15 minutes
   total uncontended calibration, 150 minutes training (two pairs × three
   seeds × 10 minutes plus one three-seed EMA cohort × 10 minutes), 75 minutes
   arena/report/diagnostics, 30 minutes failures/recovery. This is a ceiling
   awaiting Jack Heart's scientific judgment, not an extension of ETU-91 or
   the running value-token screen. If calibration says the planned cohort
   cannot fit, reduce the plan **before scoring** or defer it; do not drop
   failures, seeds, anchors or held-out endpoints to fit after results.
3. **Subsequent independent contrasts:** filter threshold/off, reference,
   separate traces, ratio/norm caps, then discount and PPO package controls.
   Propose at most one pair per new cohort: 60 minutes training (three seeds ×
   two arms × 10 minutes) plus 45 evaluation/diagnostic and 15 recovery minutes.
   Each two-hour ceiling needs its own calibration, fixed counts and approval.
   Raw/EMA behavior requires more arena allowance; do not hide that in training.
4. **Selected interactions:** compound grouping × estimator (four cells),
   LR × tau hold/removal (four cells), then filter scope × value trace only if
   isolated diagnostics suggest it. Propose three seeds × four arms × 10 minutes
   plus 60 evaluation and 30 recovery minutes = 3.5 hours each, subject to
   calibration and approval. Do not launch every factorial or claim additive
   effects from an all-on arm. The schedule factorial needs >~2,306 iterations
   to leave the LR upper clamp; a short screen may be inapplicable at this cost.

Before **each** scientific cohort, freeze a new `ResolvedStudy`, not an edited
smoke result: resolved recipes and provenance; code/native runtime/content/setup/
observation identities; three independent paired training seeds; measured update
counts fitting each arm's equal wall cap; all attempts and previous charges;
disk estimate and reserve; development and untouched endpoint deal families;
three fixed anchors (random, scripted-greedy, PUCT-64); four seat/deck legs;
stochastic policy CPU/one-thread inference and exact artifact selection.
Tentative first-cohort training seed trios are 10501–10503, 10511–10513 and
10521–10523 respectively. Reserve new evaluation families starting at 9,105,000,
9,205,000, 9,305,000 and 9,405,000 for development-paired, development-anchor,
endpoint-paired and endpoint-anchor games; collision-check producer receipts
before freezing. Proposed 4 development and 8 endpoint deal blocks per family
are feasibility targets, not silently admitted counts. Full-game evaluation
may dominate the proposed cap; measured feasibility decides admission.

Scientific `EvaluationProtocol` must carry all anchors, at least three seeds,
untouched endpoints, paired seed schedule and `paired-seed-descriptive`
uncertainty. Freeze last-available checkpoints at common cost cutoffs (proposed
2/5/10 training minutes, to be revised before scoring if exports cannot meet
them). A smoke relabeled scientific fails admission. The existing generic
campaign plan generator does not calibrate these contrasts; no runnable
**scientific** allocation is claimed until that measured plan is frozen.

Predeclare predictions as hypotheses: actor-only filtering +.02 mean anchor
score at equal cost; removing collection KL −.02; EMA −20% cross-seed score
spread without >.02 loss in mean. These are proposed expectations, not effects.
Primary exploratory shortlist threshold: ≥.05 equal-cost mean anchor gain,
paired-seed 95% interval above zero, and no legality/information-safety failure.
EMA has a separate stability criterion above, not a second strength winner.
Report each seed and each anchor; three seeds give weak tail estimates.
An interval crossing zero is inconclusive; a complete negative cohort means
reject **at this budget**, not rule out the method. No multiplicity-adjusted
confirmatory claim arises from this screen. A selected winner needs fresh
seeds/deals and a separate confirmatory budget; never use endpoint deals to tune.

Stop on nonfinite loss, failed replay/admission, illegal action, hidden-truth
leak, deadline or exhausted storage reserve. Retain partial journals, exports,
all costs and the failure cause; do not replace a failed seed. No early efficacy
stopping, best-seed selection, budget extension or extra retry is authorized.
Existing evaluation-only recovery retains failed cells and consumes the original
remaining allocation. Unsupported process recovery remains a failed attempt.

Report complete-game strength per actual cumulative collection/learning/export
wall time, separately charging arena, replay (a subset of arena), diagnostics
and reporting. Use only overlapping observed cost support and last available
checkpoints; no extrapolation from final checkpoints or isolated inference SPS.
Retain per-seed effects and paired seed/common-deal bootstrap intervals. Raw/EMA
are correlated within seed; game-level intervals do not measure method variance.
Include entropy, full reference/collection KL, clipping/norm diagnostics where
available, value loss/residual and calibration where measured, raw/selected
advantage strata, critic/actor exposures, censored tails, full-game outcomes,
legality failures and interrupted attempts. Missing diagnostics are unavailable,
never zero. Terminal outcome residual is a noisy sample, not true value error.

## Software result and remaining acceptance

The plan generator and focused admission suite are the delivered result.
No new TrainingRun, optimizer output, checkpoint, arena score or learned benefit
exists from this pass. Historical failures remain unchanged. A passing plan
check cannot satisfy the chapter's repeatable training, improvement or human
challenge measures. ETU-105 remains open for an explicitly bounded empirical
screen, its complete report/notebook, and candidate dispositions based on that
evidence. Scientific judgment needed next: approve or revise the first cohort's
priority/cap after the running value-token study is reviewed.
