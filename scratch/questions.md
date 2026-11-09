# b015d5 — remaining decisions and evidence

Historical step-1/2/3 design, implementation, reviews and execution notes are at
`c2c813a475ef012bd0391e510dd7f0cb41859e83:scratch/questions.md`. Original reports,
run projections, bills and cleanup receipts remain published in
`experiments/data/training-throughput-b015d5-v1/`. Those sources retain exact
run IDs and dated direction; no historical result is being rewritten.

## Current authority and boundaries

Jack Heart replied "keep going" to the recommendation "I recommend repairing
BF16, lengthening the packed cells, and completing the missing measurements."
Comment 079b3f46 authorizes the bounded continuation. Comment 8cb4c5af records
C1 review against that authority, not a new approval. The owning protocol is
`experiments/protocols/training-throughput-b015d5-v1.md`, C1; the implementation
handoff and deletion list are in `scratch/step-3-continuation.md`.

Keep effective hashed v6 / 512 facts and the Elemental learning recipe. c5fba4
owns its shared-effect catalog repair; do not infer TLA catalog speed from the
hashed ladder. Exact quotes and external diagnostic findings remain in
`scratch/vocabulary-performance.md`.

The first pass had 11 valid solo cells, seven insufficient-overlap packed cells,
six BF16 startup failures and no a384l8 measurements. Original Float/Half error:
`index_copy_(): self and source expected to have the same dtype, but got (self)
Float and (source) Half`. CPU autocast had not covered CUDA's recurrent behavior.
The repair is implemented here for ed2300, with no second checkout; linking its
software PR and recording validation remain delivery work.

## Execution still open

- Source/world/input freeze after software integration. Original source was
  cb722dc8c / w17; C1 never relabels those results. Current runtime admission
  remains a freeze check. Real CUDA regression and a two-update w160l8 BF16
  worker/reload on one rental must pass before opening other rentals.
- Reconcile bills before allocating new liability. Step 3 $1.741 and total
  $3.781 remain estimates, not invoices. Original caps stay $10 step 3 / $50
  total; C1 adds at most min($10-prior step3, $50-prior total, $8), 15 GPU hours,
  1,600 updates inclusive of failures/setup/diagnostics. No paid work in this step.
- Same isolated client/home as the protocol. Original provider DELETE/GET 429s
  required explicit provider deletion confirmation; all three original pods
  were absent at 13:19 UTC on 2026-11-08. No controller was left running.
- Original TF32 clip-fraction shift was 3.45x the float32 repeat difference;
  finite metrics alone did not establish equivalence. Packing crossover remains
  unknown. Step 2's accepted recipe is four seeds/one producer on A4000;
  L4 recommendation remains an inference until these measurements.

## Reversible implementation choices, 2026-11-08

- Keep the definition table in embedding-weight dtype, including empty rows;
  cast recurrent outputs at the join. This is a float32 no-op with gradients.
- C1 uses the existing launch owner and distinct `c1-...-u8/u12` keys. An
  `-overlap` key is a conditional whole-cell retry, not automatic permission to
  rerun or select the fastest cell. Source freeze and cost admission are external
  execution checks; discovery never rents. Preserve legacy four-update jobs.
- The maintained C1 reader lives under `experiments/`; historical artifact
  readers stay immutable. Explicit per-cell dispositions and rental bills are
  required; steady-state cost never stands in for the whole rental bill.
- Diagnostics take one retained 256-row observation fixture. They instrument
  fixed host recording and learner-style device/host selection/transfer only,
  one child with a 120-second deadline each. They do not reproduce a complete
  update or prove the cause of the original recording stall. No speculative
  performance repair or policy/collector/learning behavior change is included.

## Older unresolved findings

- The schema-7 retained replay fixture in
  `test_resource_limit_receipt_replays_its_retained_encoding_boundary` differs
  from runtime schema 8 (`binding_mismatch`); no fixture rewrite is authorized.
- Original step-1 recording stall and step-2 M2 seed-26214 hang remain unexplained.
  Step-1 RSS near 61.05 GB was consistent with the later measured 62 GB pod
  allowance, not proof of memory-pressure cause. Cgroup memory admission was
  repaired in step 2. Etude 429 cleanup, placement lease failures and absent
  `etude-result.json` blobs remain historical launcher findings; callback results
  were recovered from attempt records, so no missing callback is inferred.
- Infrastructure memory now retains the host-confound, CUDA-admission and
  overlap lessons. The ~50k-row launch-cost diagnosis remains specific to its
  measured model/input; it is not a conclusion for all sizes.
- The earlier $100/336h planner sketch incorrectly said dollars bind at
  $.256/hour; 336h costs about $86, so hours bind for one machine. The protocol
  sketch now agrees with the retained report; dollars limit the machine count.
- Frozen Elemental recipe omitted input version but resolved to v6; investigate
  legacy loading separately if needed. C1 pins resolved v6 explicitly by admission.

## Main sync — 2026-11-08

## Historical notes retained from MTG-252's branch
## Concept review — 2026-11-05 UTC

Outcome: **no design change and no open product choice for delivery of the
bounded pilot's results**. This is not Task completion or scientific acceptance.
The existing report already distinguishes the trained endpoint from the
unfinished study; keep that usage guidance unchanged. No new skill instructions,
execution machinery or implementation direction are needed here.

The accepted interaction is: choose the frozen corpus and bounded recipe, run
self-play, inspect the exported checkpoint and paired results, and retain every
failure and cost. Jack Heart's game-local recovery decision means a failed game
is a durable attempt record, not a reason to restart the whole experiment or a
new kind of checkpoint. Normal sampling can replace its training stream without
replacing its historical evidence.

| Action | Concept and owner | Existing API / persistence |
| --- | --- | --- |
| Train the fixed population within its allowance | Protocol fixes inputs; Etude owns placement; manabot owns learning | `tla_pilot_v10.frozen()` / `runs()`, `TrainingJob`, `execute_regime`; frozen protocol/regime and run receipt |
| Continue after an admitted game failure | One `GameAttempt`; collector owns both seats' pending rows; ledger owns cumulative allowance | `SeatRoutedCollector._settle_attempt`, `GameErrorState.settle`; append-only attempt ledger and replay input contract |
| Inspect the early trained endpoint | Exported learner state, distinct from a complete process snapshot | `execute_regime` exports raw/EMA/Adam; checkpoint identity and clocks survive wall-budget pause |
| Judge full-game evidence | Fixed comparison schedule, not only surviving games | `comparison_schedule`, `play`, `tla_pilot_analysis.analyze`; original failed/missing slots remain expected |

The consequential possible mismatch is calling all of these states “success.”
Launcher success says the job completed its execution contract; trainer `paused`
says the update target was not reached; analysis `complete=false, success=null`
says the joint scientific criterion cannot be judged. Those distinctions protect
different promises and cannot be merged into one boolean. The report already
presents them together without requiring readers to reconstruct them from logs.

Normal path: a completed game supplies both seats' retained rows; the bounded
run exports a checkpoint; endpoint comparisons use the same frozen population
and matched first-64 support. Failure/recovery: a typed active-step resource
limit records the position and encoder configuration, removes only that game's
uncommitted rows, and permits ordinary resampling below the settled-game limit.
At 25/100 failures collection continues; at 26/100 it stops. Completed updates,
other games, costs and failure evidence survive. Evaluation never substitutes
new games for failed slots.

Counterexamples remain decisive: a reset/buffer/provenance failure must not
enter a recovery loop; a checkpoint without collector/RNG state cannot promise
an exact process resume; a repaired callback cannot erase its original failed
match; an 81.25% endpoint score against initialization cannot establish the
joint improvement criterion with an incomplete baseline and schedule. The
retained permanent65 and relation65 failures also refute a universal capacity
claim. Source inspection and the existing receipts agree with these boundaries;
no new games or load tests were needed for this review.

Latest direction: Task comment `81ec3f17-30e6-4821-a77e-75888b77f65d`
records Jack Heart's request to review damped learning before a full-TLA
successor. Preserve v10 and continue reporting/delivery; do not launch that
successor before the discussion is resolved. The exact future question is:
**Which objective, behavior/reference KL, iteration versus optimizer/data clocks,
sample filtering and credit horizon should the next full-TLA recipe freeze?**
Evidence prompting it is the varying training durations and the unfinished v10
schedule, not proof that a particular replacement recipe is better. This choice
is unresolved for that separate prospective outcome and does not block delivery
of the present results. Capacity enlargement and grouped/sparse representations
likewise remain unselected.

Next proof for delivery: keep the report's checkpoint identity, endpoint counts,
missing schedule and retained failures aligned with the published receipts.
Independent behavior witnesses for the two native target-selection errors and
final billing reconciliation remain outstanding work already named in the
report. This review neither resolves them nor selects new work or navigation.

## Historical context

Completed review drafts, proof requests and CI repair narration no longer direct
work. Their full pre-compression text is retained at
`6da60684cb628ec9b431dbbb9d7dc91348743622:scratch/questions.md`.
The complete local notes before this reduction, including both placement
assumptions, are preserved as git blob `0b77c45bf972280f57b4a9bb882e945afc27120d`.
Older decisions/source pins remain at
`af4227616ef2cf46bb1f0b05e690feebb28ac2e8:scratch/questions.md`.
Inherited T6/T7 notes remain at
`5cd6ed3f6e54e81f9f39c5c2bb56f322dc1576c1:scratch/questions.md` and
`4463e57f5a92ebd3b60e2fb50c046b84ac2c0372:scratch/questions.md`;
they do not replace MTG-213's direction. Cleared demo/verification plans remain
at `d2ff60e6a^:scratch/tla-pilot-demo.md` and
`d2ff60e6a^:scratch/tla-pilot-verify.md`.

The complete decisions before this reduction, including A4000/A6000 placement
assumptions and v9's identity-preserving Job.max_usd amendment, remain in git blob
`79baeb35620eddc7eea26f3978254d2e4b5d2548`.

## Task fc6ffb — batch closure and lambda (realigned 2026-11-05 UTC)

Earlier sections are MTG-213 notes inherited from main; they do not direct this Task.

Direction, by date. Jack Heart authorized $25 total on 2026-11-04 and, on the
updates-versus-arms trade, kept all five lambda arms and three seeds per study
("i think thats enough"). On 2026-11-05 he said the `seat_comparison` setup "is not
really the current recipe, the current recipe should be the elemental 10v10 set up at a
minimum", then "finish this study, then try it again with another $25 on elemental".
So this Task finishes on Allies/Lessons, labelled as such, claiming nothing about
Elemental; the Elemental repeat is a separate Task with its own $25 and does not run here.
Deleting exact quota stays out of scope. `docs/learning.md` is not on main; the closure
docs live in `docs/training.md`.

Exists, checked against the code and the published files:
- Closure separation: `TrainSelfPlay.batch_closure` (default complete games), the
  collector reads it and no longer infers, exact quota with both seats or with
  `games_per_stream` / `game_errors` is a validation error. Saved stages resolve by the
  old inference through `TrainingRegime`, serialize the field only when it differs, and
  frozen digests hold (`tests/training/test_batch_closure.py`). `AttackPlan` schema 1
  keeps exact-quota attackers; schema 2 uses the template's closure.
- Protocol `experiments/protocols/closure-lambda-v1.md`, committed before scoring, with
  calibration and Results. Both experiment files, the shared evaluation scaffolding and
  the report builder. Report, figures and `results.json` published under
  `experiments/data/closure-lambda-v1/`, untracked, manifest digests matching the local
  bytes. Spend $19.26 of $25, every failure in the report.

Decisions taken here, each reversible and recorded in the protocol:
- Omitted closure means saved; `baseline(learning_seats="one")` is new authoring, so it
  now means complete games and its digest moved; saved runs of it stay exact quota.
- Quota 106, not the 30-update-mean 119 (an opening transient); 200 and 100 updates with
  the original seeds after calibration showed 300 would cost about $41.
- Update 0 unscored, since a seed's arms share initial weights.
- The lambda study uses the two-seat recipe; the closure study is one-seat as the Task states.

Results, Allies/Lessons only, early training: closure shows no detectable difference
(−.019, seed interval −.076 to +.039 at 4.23M decisions); no lambda arm beat .5 / .8,
and .95 / 1 scored .162 lower, with critic residual far from game end roughly doubling
when value lambda rose. Evaluation covered all 2,250 deals at $4.17 with failures, 30%
of training rather than the small fraction intended. Predictions mostly refuted; see
the report's two "Against the predictions" tables.

Remaining: publication of this branch as the Task's PR. No paid run remains in this
Task. The report's wall-time rows carry host noise and the assembly-stall mechanism is
not established; both are stated as limits, not open work here. A full report rebuild
from R2 needs a rented machine or a streaming reader (the laptop reached 18 GB RSS
without finishing); the published `results.json` stands as built.

Check 2026-11-05 (realign): `uv run --extra dev pytest tests/training/test_batch_closure.py tests/eval/test_learning_signal.py tests/infra/test_experiment_data_untracked.py -q` 36 passed; local report bytes match the manifest digests (`shasum -a 256`).

Proposed experiments Wave memory (the `code` Graph does not call for writing it):
- One-seat complete games versus exact quota showed no detectable strength difference
  in 4–6M early decisions on Allies/Lessons (three seeds; interval hides ±.04–.08).
  Exact quota idles too (.47 of slots), so idle share alone does not measure closure cost.
- Value lambda above .8 hurt early learning on Allies/Lessons (.95 / 1 by −.162) and
  roughly doubled the critic residual far from game end; policy lambda .8 alone moved
  nothing. Not evidence about later training or Elemental.
- Packed `[rows, 1]` complete-game batches spend about 1,100 s per 200-update run in
  target and diagnostic scans; a wall-time comparison must separate that from closure.
- Assembly stalls (5–30 s per update, in episodes) are host-side and hit exact quota too.
- Launcher `usd` understated provider bills about 3× in calibration ($.34 against $.92):
  budget from bills and whole-rental quotes.
- Arena durable writes need memory-backed scratch on rented disks; queue mode undersizes
  windows when `parallel` exceeds pending runs; refused creates pin a run id; a pod that
  never boots ignores `cancel` (2 of 7 CPU pods here), so terminate it by machine and
  count its window; controllers need `caffeinate` on the laptop.

### Concept review — 2026-11-05 UTC (code Graph, no person)

Outcome: **no design change and no open product choice for this Task.** The
implemented contract is the accepted one, and the usage docs (`docs/training.md`
"Batch closure", `manabot/README.md`) already say what the code does; nothing is
proposed there. Checked against source, tests and the published report, with no
new games or paid runs.

| Action | Concept | API / persistence |
| --- | --- | --- |
| Author a stage: who learns, when the batch closes | `learning_seats` and `batch_closure` are separate declarations; `transitions` is a quota under exact quota and a per-stream floor under complete games; `games_per_stream` replaces the floor with whole games | `TrainSelfPlay`; serialized only when it differs from `_implied_closure`, so frozen digests hold |
| Load a saved recipe | Omission means what the run did: one seat with neither `games_per_stream` nor `game_errors` is exact quota, anything else complete games | `TrainingRegime.read_saved_stages`; `AttackPlan` schema 1 (always serialized) keeps exact-quota attackers |
| Continue from a learning state | Closure is part of the recipe identity | `admit_learning_state`, continuation validator |
| See the cost of finishing games | `idle_stream_steps` counts slots idle while the slowest stream closes, in either closure | `CollectorStats`, `StageRecord`, PERF panel |
| Read the two studies | Each conclusion stands alone, labelled Allies/Lessons | `experiments/batch_closure.py`, `lambda_sweep.py`, report under `experiments/data/closure-lambda-v1/` |

The one consequential mismatch is naming: `transitions` means two different
things by closure, and the three closure shapes (quota, floor, whole games) sit
across two fields rather than one tagged value. A clearer shape exists, but it
would restructure every saved stage and move frozen digests, which the accepted
constraints forbid; the field comment and docs carry the distinction instead.
Not pursued, recorded here so a later reader does not mistake it for an oversight.

Consequence of the "omission means saved" rule: a hand-written JSON one-seat
stage with no `batch_closure` reads as exact quota, exactly like a saved one;
only stage instances built in code take the complete-games default. The docs
state the omission reading, so JSON authors can see it; no change proposed.

Normal path, failure, recovery: a new one-seat frozen-opponent stage closes on
complete games and writes the field; exact quota with both seats, a game error
policy or `games_per_stream` is rejected at validation and again by the
collector; a saved exact-quota run continues only into an exact-quota stage.
Counterexamples held: saved regime digests pinned in `test_batch_closure.py`,
`scale_ladder` keeps its exact quota and digest, `seat_comparison("one")`'s
digest moved as recorded above.

Unresolved for later, not blocking, outside this Task by Jack Heart's
constraint ("Deleting exact quota is out of scope. Decide that after the
comparison reports"): **should exact quota be deleted now that the report
exists?** Evidence: no detectable strength difference on Allies/Lessons early
training (−.019, interval −.076 to +.039); exact quota keeps the `[quota, 256]`
fast target path (24 s against 1,114 s of scans per run); the Elemental repeat
Jack Heart directed has not run. That decision is his and belongs to a separate
Task. Nothing else is open; the next step for this Task is publication.

## MTG-252 assumptions (2026-11-05 UTC)

The fc6ffb section above is inherited from main and does not direct MTG-252.

- Implemented the optional third arm (`attention_readout`) so the protocol can
  choose 2 or 3 arms; the brief marks it optional. Reversible: drop the arm.
- Value-token initialization order left unchanged (it shifts later v7 encoder
  draws). Moving it would alter fresh initialization of existing token recipes;
  recorded as a known non-shared-init difference instead.
- Arms derive from v10's damped rule unchanged; whether to wait for the
  full-TLA learning-rule review is an open decision in the study doc.
- The pooling × floor follow-up (w4, 2026-10-06) is prior evidence the Task
  brief omitted; the study doc cites it.
## Upstream continuation context (not MTG-252 direction)
Assumption (reversible): keep this Task's b015d5 continuation notes as the
active scratch context. Main's separate 5da5f4 study decisions remain at
`e0832f67fc66a1f2033bafefdb4598e394fd5ca1:scratch/questions.md`; its study
landing authority and budget do not replace this experiment's boundaries.
