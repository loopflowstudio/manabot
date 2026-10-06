# Remaining Ataraxos transfer protocols

2026-10-05 • ETU-99 • prospective, unexecuted

Jack Heart requested this bounded research/protocol contribution while preserving
the main training priority. It allocates **no training, calibration, arena games,
hardware or paid compute**. Each study below needs a separate authorized cap and
an immutable launch manifest before execution. ETU-91's frozen campaign and
ETU-106's retained checkout are unchanged. These proposals do not satisfy the
Trained Challengers chapter or close ETU-99.

The next useful sequence is saved-game sampler quality, sampling-based search,
then distillation and attacks on identified policies. Exact enumeration is only
a tractable reference. Missing exact-history support does not gate production
sampling. The older [follow-up proposals](training-regime-followups.md) retain
their dated estimates; this document supersedes their ETU-99 launch ordering and
budgets. It is a protocol proposal, not an already frozen scientific registration.

## Coverage and evidence map

Code inspection baseline: `20b0a611ca4709be9b5aa8799e72cf82c74d533c`.
“Delivered” describes code, not scientific acceptance. Existing retained runs
below are documented by their owners; their ignored artifacts were not copied,
rescored or independently re-admitted in this contribution.

| Question and scientific owner | Delivered instrument / evidence owner | What evidence permits; remaining gap |
| --- | --- | --- |
| Learning rule, KL/reference, horizon, filtering, schedules, EMA, compound credit: ETU-105 | [Technique screen](ataraxos-technique-screen.md), ETU-92/93/94 | Declarative contrasts and bounded workflow evidence; no technique retention decision. Do not duplicate here. |
| Capacity: ETU-103; configuration/calibration: ETU-102 | [Capacity receipts](../docs/training-calibration.md), ETU-104/109 recipe APIs | Model accounting and bounded execution; no capacity winner. Fixed architecture in ETU-99. |
| History, value/pooling/topology/action representation: ETU-106 | [Value models](value-models.md), [authoring API](../docs/training-experiments.md) | Workflow evidence; broader belief architecture coordinated there. ETU-99 tests the existing sampler, not a replacement encoder. |
| Joint sampler quality, dropout and foreign-policy shift: ETU-99 B1 | [Sampler](../docs/belief-sampler.md), ETU-96 | Small frozen-policy workflow and synthetic wide-count coverage. No independent-seed calibration, adversarial robustness or wider-world strength. |
| Physical/learned search at measured cost: ETU-99 B2 | [Local search](../docs/local-policy-search.md), ETU-95/100 | Direct materialization, retained support, replay and tiny complete-loop proof. No selected-matchup improvement; learned conditional queries remain unsupported. |
| Soft/hard targets and subsequent rounds: ETU-99 D1 | Local teacher, shard readers, `TrainSupervised`, ETU-95 | Same-root target readers and cumulative fitting exist. Allocation counts are not PUCT visits. Archived-root relabeling and same-root visit comparison are not delivered by this pipeline. |
| Frozen exploiters: ETU-99 R1 | `AttackPlan`, `execute_attack_plan`, [scenario runner](../docs/checkpoint-scenarios.md), ETU-97 | Continued attacker training and replayed arena reporting exist; toy pass-policy control is not selected-world sensitivity. S1–S5 custom setups cannot admit selected-match checkpoints. |
| Full-loop cost: ETU-99 C0 | [Calibration](../docs/training-calibration.md), [recovery](../docs/training-recovery.md), ETU-98 | CPU instrumentation/recovery exists. Uncontended scaling, native peak memory and accelerator cost remain unmeasured. ETU-108 owns distributed prototype/placement. |
| Frozen campaign / product admission | ETU-91 / ETU-85; ETU-101 W&B | Reuse published identities only. No campaign rerun, dashboard implementation or demo promotion here. |

Historical motivation remains bounded by its world and controls: [exp-03](exp-03-distillation.md)
made static distillation plausible; [exp-07](exp-07-expert-iteration.md) found a
failed iteration at its label economics; [exp-09](exp-09-control-competency.md)
and [exp-10](exp-10-value-gate.md) expose continuation/value failures; [exp-11](exp-11-curriculum-exploitability.md)
contains bounded attacks, not a safety certificate. [INT-7](int-7-value-target-comparison.md),
[INT-8](int-8-student-signal-guidance.md) and [INT-17](int-17-belief-calibration.md)
retain negative or systems evidence. None decides the current-world methods.

## Primary-source check and MTG departures

Rechecked the [final Nature article](https://www.nature.com/articles/s41586-026-11036-y)
and [supplement S3.5–S3.7, Tables S8–S12](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-11036-y/MediaObjects/41586_2026_11036_MOESM1_ESM.pdf)
on 2026-10-05. The article was accessible through a direct HTTP retrieval after
the browser fetch hit Nature's cookie redirect.

The article trains beliefs from final-policy self-play, uses them generatively
at inference, and combines frozen-policy rollouts and values in a local update.
Its belief-network dropout is motivated by distribution shift; that does not
establish immunity to a deliberately misleading opponent. Its limitations
identify the bounded improvement of a single update. Search distillation and
repeated teacher/student rounds below are MTG research hypotheses, not results
established by this article.

The supplement uses teacher-forced hidden-label NLL. S3.7 evaluates forced root
actions followed by policy continuations, then applies two reverse KL penalties.
Equations (7)–(8) give logits proportional to
`(Q + alpha*log(reference) + beta*log(base))/(alpha+beta)`.
Table S12 uses alpha .002, beta .02, 1,000 rollouts and depth 40. Those are source
settings, not the implemented MTG defaults or an allocated experiment.

Code checks against that source:

- `manabot/belief/sampling.py`, `sampling_data.py` and `sampling_fit.py` implement
  autoregressive **hand counts**, physical-deal corrections and joint NLL.
  Inputs include constraints and order-invariant commitment counts, not a
  transformer over full board/history. Fitting samples training rows with
  replacement; it does not reproduce the source's every-position update scheme.
  History-feature dropout is a distinct treatment from source network dropout.
- `manabot/sim/local_update.py:regularized_update` implements the two-KL formula.
  The MTG reference is offer-uniform; current defaults are alpha .02, beta 1,
  depth 2 and one world. Defaults must not be described as paper settings.
  Signed values, physical/learned hand measures and truncated continuations are
  adaptations. V3 preserves exact base zeros and unavailable Q outside support.
- `local_sampling.py` materializes sampled counts without enumerating support.
  `sampling="belief"` is the explicit enumerated reference, with incomplete
  public likelihood history and a support check after enumeration. It cannot
  be a large-pool safety fallback. Production `True` queries do not implement
  arbitrary conditioned sampling or calibrated condition mass.
- A self-play posterior models the generating policy. A foreign opponent need
  not match it. B1 measures that shift separately from B2's playing strength;
  it is not a requirement to relabel every foreign history with a new posterior.

Move-learning clipping and estimator fidelity remain with
[ETU-92's source table](../docs/ataraxos.md) and ETU-105, not a second screen here.

## Common freeze, accounting and analysis

Each launch manifest must contain actual values, not placeholders:

1. Exact source commit, native binary and runtime/library identities; world,
   compiled content, authored main decks **and sideboards**, Lesson pool,
   Observation/action ABI and capacity; AgentSpec/architecture receipt; host,
device and threads. Start on corrected Allies/Lessons. Wider pools require a
   separately bound setup/schema and budget; synthetic wide-count tests are not
   wider-world admission.
2. Producer TrainingRun/store references, every candidate attempt, selection rule
   (last complete raw at a declared cutoff by default), checkpoint byte hashes,
   original cost and independent producer seeds. Freeze sampler dataset/schema,
   generating policy and sampler bytes; identify teacher/base/rollout/value,
   reference, coefficients, horizon, search-source hash and student target.
3. Whole-game membership and actual deal/action seed lists, generation offsets,
   shuffle/split algorithm and digest; no sibling viewer or trajectory fragment
   crosses partitions. Training, validation/development and final families are
   disjoint. Validate against actual producer receipts, not just different base
   seed integers. No endpoint-driven checkpoint or hyperparameter choice.
4. A complete ordered cell manifest, game limits, inference envelope, costs and
   analysis version. Retain attempts, partial journals, failures and unavailable
   endpoints. Deadline, support, illegal Command, hidden-input or replay failure
   stops the affected cohort; do not count incomplete games as draws or replace
   unavailable search with policy-only play. Retry is a linked attempt charged
   to the same cap, never a replacement row.

Use three independent producer/learner seeds for a first method screen. Pair
arms within seed and deals; raw/EMA, sampler sample counts and attacker rungs
are correlated observations. For score use win=1, draw=.5, loss=0. Average the
four seat/deck legs within a deal block, then blocks within a seed. Report all
seed effects and a paired seed-level 95% t interval (df=2 for three seeds), with
a separate paired block bootstrap conditional on each fixed checkpoint.
Do not bootstrap individual decisions or treat all arena games as independent
training replicates. Tiny seed cohorts will often be inconclusive.

For sampler metrics, average per decision within a whole game, then games
within a producer; pair learned-minus-prior at those same games. Resample games
with both viewers together for conditional uncertainty. Report producer and
sampler-fit seed variation separately; multiple fits on one producer do not
create independent policy populations. Use game-mean NLL as B1's primary score;
pooled decision/card Brier/ECE are descriptive secondaries. Existing aggregate
report output cannot reconstruct these intervals; retain per-game scores from
the same immutable rows before scientific analysis. Never derive an interval
from an aggregate mean alone.

An endpoint prediction below is a proposal to freeze, not an observed value.
Report positive, negative or inconclusive per primary endpoint; secondary
panels are exploratory. Cross-study method selection needs fresh confirmation,
not a pooled best result. Curves use the last available checkpoint at common
**observed** cumulative costs; no interpolation beyond coverage. Charge label
collection, all fits, initialization, reload/export, evaluation/replay, diagnostics,
failed attempts and recovery. Show shared generation once in total program cost
and in full in each standalone-method cost. Separate sunk producer cost from
incremental transfer cost without hiding either.

## Proposed budgets and calibration dependency

All figures are serial local CPU active-hour ceilings, one worker and one Torch
thread, **not authorization or measured projections**. No concurrent load against
the main training campaign. Record elapsed wall time, active time, sleep, load,
RSS and available disk separately; monitoring does not correct contention.

| Study | Prospective maximum and allocation | Calibration condition before final freeze |
| --- | --- | --- |
| C0 | 1 h: .5 h full-path calibration + .5 h failure/report reserve | Same admitted artifacts and source as the intended cohort; measure collect/fit/export/sample/materialize/search/evaluate/replay, not inference alone |
| B1 sampler | 8 h: 2 label, 3 fit (six fits), 1 offline report, 2 reserve | 256 games per producer and both deck assignments; three producers; dropout 0/.5 with paired fit seeds; reserve fits the complete saved-game report |
| B2 search | 8 h: 6 arena, 2 reserve | Three producers, three pairwise contrasts, 32 four-leg blocks each = 1,152 games; at most 18.75 s average/game including replay at the nominal 6 h allocation |
| D1 targets | 10 h: 3 labels, 3 fit (six fits), 2 arena/report, 2 reserve | Three independent producers, same-root soft/hard students; 32 four-leg blocks per pair/seed = 384 direct games, plus declared anchors inside the 2 h allowance |
| R1 attacks | 12 h: 6 continued attacker runs, 4 evaluation, 1 positive control, 1 reserve | Two fixed target artifacts × three independent attackers × 1 h; initial + three rungs × 32 four-leg blocks = 3,072 games, requiring ≤4.69 s/game for the 4 h arena envelope |

These maxima sum to 39 hours **only if each study is separately authorized**.
They do not amend ETU-91's 168 hours. Existing producers must be admitted from
published evidence; new producer training is excluded and needs its own allocation.
The R1 ceiling covers two fixed artifacts, not two three-seed policy populations.
Population-level robustness requires a new larger protocol, not silent reuse of
this cap. Hardware-cost comparisons beyond local C0 also need a separate cap.

C0 measures complete paths on non-endpoint deals after separate authorization.
Record native microsteps, learner transitions, completed games and optimizer
exposures separately; measure cloning/sampling and replay as explicit phase
costs. Python allocation peaks exclude native workspace. Record process RSS,
transfers and host contention without inferring throttling from load alone.
The current complete trainer is CPU-only; accelerator and two-host results
cannot be projected from forward-pass rates or the paper's GPU ratios.
Freeze counts only if projected total phases multiplied by 1.25 fit the relevant
cap, including reserve and all prior attempts. Estimate game costs from the full
observed distribution and include slow tails. The arithmetic rates above are
feasibility requirements, not expected throughput. If infeasible, revise counts,
latency class or scope **before** endpoint scoring and version the protocol.
Do not shorten a running cohort because a result looks decisive.
Freeze projected dataset/checkpoint/trace bytes too, retaining at least 4 GiB
free after the complete cohort and failure allowance; stop before exhausting
that reserve rather than pruning unsuccessful attempts.

Proposed seeds, pending collision checks: producers 9901–9903 if new production
is separately approved; sampler fits 9911–9913 paired across dropout arms;
student fits 9921–9923; attackers 9931–9933. Existing producer seeds override the
proposal only in the frozen manifest. Proposed disjoint deal reservations:
B1 960000–969999, B2 development 970000–970099 / final 970100–970131,
D1 development 971000–971099 / final 971100–971131,
R1 development 972000–972099 / final 972100–972131.
Numbers alone do not reserve global state or prove no historical reuse.

## B1 — Does the sampler learn transferable information?

Primary contrast: physical compatible-deal prior versus current learned sampler,
with history dropout 0 and .5 as paired fits on the same immutable dataset.
Use three independent admitted producer policies. Collect 256 complete games per
producer, alternate deck assignments, and freeze the collector's seeded whole-game
split (roughly 60/20/20; exact IDs govern). Use fit seeds 9911–9913 paired by
producer. This first screen estimates combined producer/fit variation; a later
nested multi-fit study is required to isolate the components.

Use all test games. Predeclare samples {16,64,256}, fixed inference RNG and the
64-sample panel as primary for sampled metrics. Counts are correlated and need
not be nested. Joint observed-label NLL is teacher-forced and not a Monte Carlo
posterior KL. Prediction: no-dropout lowers game-mean NLL by at least .05 nat
relative to the prior on same-policy test games. Call the effect supported only
if the seed-level interval is below zero and the mean improvement is at least
.05; otherwise preserve negative/inconclusive evidence. All arms must have zero
constraint violations; report the count and total draws, never “proved legal
for all worlds” from finite draws.

Apply each sampler to the other two producers' test games, with matching world
and vocabulary and no fitting-game overlap. Keep each foreign identity separate;
those crossed evaluations share datasets and are not six independent producers.
Prediction: dropout reduces foreign-policy game-mean NLL by .02 nat against
no-dropout without worsening same-policy NLL by more than .02. Treat that as a
secondary exploratory contrast. Report presence Brier, ten-bin ECE and adjacent
vocabulary conjunction Brier, hand size, public minima, commitment count, latency
and memory. Arbitrary typed-query calibration and full-posterior KL remain gaps.

Foreign self-play is not adversarial play. Adversarial histories must come from
an independently frozen attacker and whole held-out episodes, with population
and selection rule recorded. The current collection stage does not declare a
mixed opponent population; this stress cohort remains a separate data-collection
adapter and allocation. A tiny exact reference may check known-policy NLL/query
mass only on roots with complete supported likelihood history. Never trim failed
histories and label the survivors a full-game exact baseline.

Strongest confounds: limited history features, narrow producer behavior, and
finite sampling noise. Distinguish them with foreign-policy panels and sample
counts before proposing another architecture (coordinate with ETU-106).

## B2 — Does learned sampling improve the complete search player?

Freeze each B1 no-dropout sampler with its exact producer policy before scoring.
Do not select the sampler by endpoint NLL or arena score. Compare policy-only,
physical-prior local search and learned-sampler local search. Use all three
pairwise contrasts, 32 shared four-leg final blocks per producer. The reference
and learned teacher use identical policy/value bytes, alpha=.02, beta=1,
depth=2 and `full_probability=1`; these are initial MTG hypotheses.

The proposed search ceiling is 200 ms per decision. C0 determines fixed world
counts from {1,2,4,8} separately for physical and learned sampling so both fit
one declared realized-cost class. Freeze counts and report p50/p95/max latency,
per-game inference cost, preparation/materialization cost, rollouts and deadline
failures. A same-count panel is diagnostic only. Equal ceilings alone do not
establish matched realized compute: require mean per-game inference costs within
10%, otherwise label the contrast cost-imbalanced and show both costs. Policy-only
uses one pass and reports its smaller actual cost; do not pad it with fake work.

Primary prediction: learned beats physical search with score ≥.55 and seed-level
lower bound >.50 at matched realized cost. Comparisons against policy-only locate
whether search itself pays; failure there is retained even if learned beats prior.
Strongest confound: a poor continuation/value model can hide useful belief quality.
Keep B1 calibration separate and retain per-action Q, target entropy, support,
root value and realized compute. A target delta is not a strength result.

Current arena primitives can register and replay local search, but the ordinary
study `EvaluationProtocol` fixes policy-only inference. This prospective cohort
needs explicit arena-cell orchestration/analysis before launch; do not label a
three-player search experiment as an existing policy study to pass validation.
Search remains relevant to advice; this protocol does not register a live advisor.

## D1 — Does distilling the regularized target pay?

First compare `local_soft` and `local_argmax` from **one** immutable local-update
collection per producer, with the B2 teacher configuration fixed beforehand.
Both students start from that producer's raw policy; use equal fitting caps,
paired minibatch seeds and identical whole-game membership. Primary prediction:
soft beats hard at policy-only inference with score ≥.55 and seed-level lower
bound >.50. Also score both versus their frozen base and the declared fixed
random/scripted/search anchors; direct success alone is not improvement over base.

Generate at least enough complete games to populate the existing held-out rule
`game_index % 10 == 0`; freeze the actual collection count from C0, proposed 64
per producer. No relabeling of split membership after seeing losses. This teacher
pipeline has train/validation membership, not a second untouched target-test split;
final full-game deals provide the untouched strength endpoint. Report held-out KL
and entropy as validation diagnostics, shared-encoder value drift, optimizer
exposures, all charged labels and student cost. No held-out KL “test” claim after
using it for selection. Same-root genuine PUCT visits require another admitted
teacher/replay adapter; `local_allocation` is not a substitute.

The existing pipeline supports later collect/fit rounds and cumulative shards.
Do not run them in D1's cap. A later paired latest-only/cumulative contrast must
hold root IDs, optimizer initialization/continuation and historical label charges
explicit; relabel-versus-new-games additionally needs archived-root admission.
Selective full/cheap allocation has knobs but no measured benefit. Soft targets
preserve mixing numerically; they do not establish bluffing, equilibrium or a
repeatable improvement cycle. Strongest confound: the soft target may simply
stay closer to the base under noisy Q, rather than transfer better strategic
information. Retain base KL and the base-player control to distinguish that.

## R1 — How much can a bounded attacker exploit fixed targets?

Choose two named target artifacts by a rule frozen before attacks, for example
one producer and its soft student at declared cost cutoffs. Do not pick each
method's best evaluation seed. Attack each with independent seeds 9931–9933 using
terminal rewards and the repaired `NetOpponentTrainer` frozen-opponent path.
Freeze PPO configuration and equal architecture across targets; this is not an
ETU-105 gradient contest. Declare a cumulative update ladder calibrated to
approximately 15/30/60 minutes, preserve one collector/Adam state per attacker,
and report actual checkpoint costs rather than renaming updates “minutes.”

`AttackPlan` evaluates the untrained attacker and every rung on the same final
paired deals with all four assignments. Freeze the entire ladder before opening
any result; these final games must never drive tuning or stopping. Retain mean,
range and every attacker curve. Primary prediction: the final-rung attacker
improves score by ≥.10 over its initialization on each fixed target, with the
paired attacker-seed interval above zero. Also report endpoint score against .50;
improvement from a weak initialization alone is not exploitation. Maximum observed
attack success is selection-biased descriptive evidence, not NashConv.

Require a known-exploitable positive control before interpreting unsuccessful
attacks. The existing `test_exploiter_positive_control.py` trains against a
pass-biased checkpoint on a creature-only toy world. It is a software sensitivity
check, not this selected-world control, and is not run in this contribution.
A separately identified selected-world vulnerable policy and current-world
terminal behavior must be fixed and measured within R1's 1 h control allowance.
If no such admitted control exists or the attacker cannot exploit it, declare
attack sensitivity unresolved rather than certify targets as robust.

`validate_scenarios` reports premise/setup compatibility; it does not run the
selected checkpoint through incompatible S1–S5 decks. Preserve unsupported
scenarios. Complete-game random/scripted/search anchors qualify the attacks but
need separately enumerated cells in the evaluation allowance: `AttackPlan` itself
only evaluates attacker versus target. Its report's bootstrap varies attacker
seeds, not target-policy seeds. R1 therefore supports artifact-specific attacks;
method robustness, population training and search-player attacks require further
protocols and explicit software support. Strongest confound: attacker optimization
failure; never interpret a failed bounded attack as an exploitability certificate.

## Existing APIs and precise launch gaps

No new authoring schema or executor is needed for the supported stage graph.
The following is a **plan-only illustration** using the repository's tiny
mechanism recipe, not a selected-matchup study or authorization to execute it:

```python
from pathlib import Path

from manabot.training.experiments import Baseline, Experiment, Pipeline
from manabot.training.models import TrainingRegime, TrainSupervised

source = TrainingRegime.model_validate_json(
    Path("experiments/regimes/learned-belief-local-search.json").read_text()
)
# One producer, one saved history dataset, one sampler and one label collection.
prefix = tuple(source.stages[:4])
soft = TrainSupervised(
    id="soft", operation="train_supervised", initial="policy",
    datasets=["labels"], target="local_soft", epochs=1, batch_size=64,
)
hard = TrainSupervised(
    id="hard", operation="train_supervised", initial="policy",
    datasets=["labels"], target="local_argmax", epochs=1, batch_size=64,
)
resolved = Experiment(
    "etu99-target-example",
    Baseline.capture("existing-mechanism-proof", source),
    overrides=(Pipeline((*prefix, soft, hard)),),
).resolve()
receipt = resolved.receipt()  # Configuration/provenance only; no model or run.
```

`TrainBelief(dataset="histories", history_dropout=...)` can similarly reference
one earlier `CollectBelief` for paired fits. Stage references preserve the exact
generating policy/weights; a learned local collector requires its sampler to
come from that same policy. The existing API has no external-checkpoint import
stage for this chain. Reusing published producers without retraining therefore
needs a narrow admitted-artifact orchestration adapter; do not insert a fake
one-update producer or copy a checksum to bypass that boundary.

| API | Supported use | Gap before these scientific cohorts |
| --- | --- | --- |
| `TrainingRegime`, `TrainingRun`, VerifyStore | Collection, sampler fitting, local labels, shared-root hard/soft fits, frozen attackers; costs and failures | Published-artifact stage reuse and explicit separately seeded fitting cohorts need orchestration; implicit run seed offsets are not a crossed seed design |
| `Experiment`, `Pipeline`, `ResolvedExperiment.receipt()` | Resolve and bind existing regimes with provenance without execution | A recipe receipt is neither a cohort nor budget authorization |
| `EvaluationProtocol` / `ResolvedStudy` | Existing named policy-only studies | Restricted study IDs, player counts, policy-only inference and stage-count rules exclude B1/B2/D1 graphs; no fabricated `omitted-controls` wrapper |
| `report_saved_sampler` | Immutable own/foreign datasets, exact sampler admission, samples {16,64,256}, descriptive metrics | Per-game output/seed aggregation, arbitrary queries and native RSS unavailable; preserve original splits |
| `AttackPlan.regime_for`, `execute_attack_plan` | Exact target hashes, continued ladder, selected-match replay | No selected-world positive-control factory, anchor matrix or target-seed uncertainty; do not represent them as delivered |

After authorization, save resolved declarations, actual artifact manifests,
TrainingRun exports/store IDs, EvaluationProtocol where supported, complete arena
cell schedules, private datasets and replay digests. Generate reports/notebooks
offline from those immutable inputs. Timing fields remain fresh measurements;
statistical regeneration must bind evaluator source/runtime/seed and reproduce
its score bytes. B1 per-game analysis and B2/D1 orchestration are explicit remaining
software work, not hidden tasks performed by the code example.

This contribution performs source/code reconciliation and syntax/link checks
only. No empirical endpoint, admission of a real candidate, or scientific
conclusion has been produced.
The Python example passed syntax validation; native API admission was not run
because this checkout has no built managym extension. That check belongs before
any later execution, alongside actual artifact and budget admission.
