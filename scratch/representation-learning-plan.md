# V7 learning comparison — GPU plateau continuation

## Restart authorization — Jack Heart, 2026-11-04

After the cancelled eight-learner attempt and its throughput failure were
reported, Jack Heart said "start it when you can" and "seems fine". Resume
execution-design repair and the GPU study within the ORIGINAL cumulative $25
cap. This supersedes the operator's earlier return-to-review/no-replacement
restriction below; no further permission is needed for a viable replacement.
Keep the failed cohort immutable and explicitly unsuccessful. Freeze any new
attempt and execution amendment prospectively; this approval does not erase the
old no-retry denominator or allow score-based survivor selection.

Jack Heart also asked what "5–7x longer" means and why. The comparison was
against the allocation's required 32.23 seconds/update, not measured isolated
hardware regression. Single-learner admission averaged19.30seconds/update;
eight concurrent learners measured169–239seconds for their first updates and
151.24seconds for one second update. At the latter rates,2048updates would take
roughly86–136hours per learner while all run concurrently. GPU contention is
a plausible explanation, not a conclusively profiled mechanism. Spare memory
did not establish concurrent compute throughput. Do not assume that reducing
concurrency, enabling MPS or buying a larger GPU automatically solves total
cohort cost. Measure actual ordinary learner throughput and total cost for the
proposed execution arrangement before spending on the replacement cohort;
no artificial load. Consider sequential/lower-concurrency execution, supported
batching or cheaper compatible hardware as routine choices, and keep changes
bounded. Preserve independent seeds, scientific treatments, meaningful horizon,
initialization/curve evaluation, budget/teardown and results review. Reconcile
prior billing/reservations before allocation. Loopflow's installed compatible
build ac8e088f5 now reads Task state; the old Graph is stopped with no live worker.

## Current direction — Jack Heart, 2026-11-04

Jack Heart reviewed the completed short CPU cohort in this conversation and
said: "training on cpus doesnt seem long enough" and "we needt o at least get
to the point of the gpu curve where were starting to hit a plateau".
This supersedes the earlier stop at results review and restriction on further
training. Continue this Task with actual GPU learning toward a sustained
learning-curve plateau, under the existing $25 cumulative ceiling, not a new
$25 allocation. The previous CPU study remains completed, frozen and
inconclusive; do not rebind or overwrite its evidence. No merge, model/default
promotion, temporal-memory implementation or MTG-213 spending is authorized.

Provider follow-up from Jack Heart in the same conversation: "we can push an
etude sdk release and then use vast if necessary". A necessary Etude SDK
release and subsequent Vast GPU use are authorized for this continuation.
Inspect the existing Etude work/release state and reuse supported provider
implementation; do not duplicate work already underway. Coordinate any SDK
release through its repository's release procedure and verify the released
client before rental. This conditional release approval does not approve
merging the representation-study PR or spending beyond $25 total. Record SDK
and provider identities prospectively and retain the same accounting,
deadline, cleanup and scientific-evidence requirements on Vast.

Execution direction for the continuation:

- Retain pooled versus GRU and at least four independent paired training seeds.
  Use the existing authored blue-red Lessons / green-white Allies population
  to address horizon without also changing the deck distribution. This is an
  operator choice, not a user claim that these decks are sufficient for general
  architecture selection. The user was told evaluation previously held out
  deals, not decks, against a frozen heuristic.
- Define and freeze a prospective GPU protocol before scoring: initialization
  evaluation, periodic checkpoints, comparable learner-decision/game exposure,
  a substantially longer supported horizon and a sustained-plateau criterion
  across seeds and multiple checkpoints. Quantify uncertainty; a noisy flat
  point or failure to reject a slope is not positive evidence of saturation.
  Keep final held-out scoring separate from observations used for stopping.
- Inspect the inherited learning-rate/optimizer schedule: extending execution
  with an exhausted 128-update schedule must not manufacture a plateau. Bind
  any schedule changes in the new identity, with consistent arms and fresh
  initialization unless exact continuation is scientifically justified.
- Quote available compatible GPUs broadly within the cumulative budget;
  budget setup, idle time, evaluation, artifact transfer/storage, failures and
  teardown. Reconcile the prior $0.07258259388618171 bill and $6 pending
  ancillary reserve. Use configured provider access (Doppler etude/prd),
  artifact access (manabot/prd), task-local allocations, and supported Etude
  ownership/deadline/cleanup. Existing provider capabilities may be used;
  account creation or a new provider integration is not a prerequisite.
- GPU execution is now explicit direction: do not substitute another short CPU
  cohort. Capacity admission and throughput calibration are sanity checks,
  not study completion. If all compatible capacity is unavailable, report the
  concrete capacity block; if the budget ends before a credible plateau, retain
  all curves and report the plateau objective unmet. Never exceed the cap or
  select successful seeds/checkpoints after seeing scores.
- Deliver training/evaluation curves, per-seed outcomes, plateau evidence or
  its absence, exact costs and teardown receipts. Stop at results demo/review.

## GPU implementation and stopped allocation — 2026-11-04

The prospective declaration is `experiments/state_representation_gpu.py`; protocol:
`experiments/protocols/state-representation-5da5f4-gpu.md`. Scientific source:
`27a12642200a35fc0cb436a9db69ce16c505016a`. Both joint treatments retain the
authored decks, four fresh paired seeds, 2,048 CUDA updates, 256-update curves
and separate final held-out deals. Absolute LR/tau scheduling stays positive.
All six final-window paired-seed gain intervals must lie inside +/-0.05; missing
points cannot qualify. No scientific scores informed hardware selection.

Etude PR79 and release PR80 are merged; published SDK 0.2.14 controls Vast.
The scientific worker retains locked SDK 0.2.12. RunPod 6000Ada/L40S/A100
rentals refused before acquisition. Excluded Vast admission `edb504230b42`
at source `35770e55b`, instance 59138171/session `d407636829cb`, passed CUDA
12.8 numerics and four GRU updates: 128 games, 24,202 learner decisions and
733 optimizer updates. Collection 47.50s plus learning 29.70s; sampled peak RSS
4.32GB. Deletion was confirmed at 17:36:17 UTC, with subsequent provider absence.
Full-rental estimate $0.1146496 is not a settled bill; retain $0.50 reservation.
Receipts: `.runs/representation/gpu-plateau/admission/` and adjacent accounting.

Measured throughput invalidated the initial per-replicate two-hour allocation
before any scientific seed launched. Revision `8c7712bc9` retains that unexecuted
predecessor. Eight independent ordinary `execute_regime` learners now share one
GPU, each bounded at 6 GiB host memory and 10% of the CUDA caching allocator.
No weights, optimizer or RNG are shared. Training precedes evaluation, with two
evaluators at once. One Etude guardian owns the allocation; any child failure
reaps siblings. Timing describes concurrent use, never isolated GPU performance.
The job allows 20 hours plus 15 minutes setup, at most $17.20. Prior known bill
$0.07258259388618171, $6 ancillary reserve and $0.50 admission reserve bound the
cumulative reservation at $23.77258259388618171. Learners have 66,000 seconds;
evaluation retains a separate deadline reserve.

Acquisition evidence: 128GB-host and later 60GB 5090 quotes initially refused;
4090 and 3090 offers appeared and disappeared before acquisition. Launch
`ea36a15587b1` for prepared run `e4bc79b1e907` refused 3090 capacity without a
rental or job execution. Source preparation originally downloaded all Git history;
reusing local Git objects through `GIT_ALTERNATE_OBJECT_DIRECTORIES` preserved
remote verification and completed it. Interrupted local fetches were reaped.
A prospective amendment permits one 3090, 4090 or 5090 selected before execution,
identically for every seed and both arms, within the unchanged cap.

Run `9c7c8e6841a4` acquired RTX 5090 / 32 vCPUs / 64GB on Vast at 18:17 UTC,
instance 59143644, session `d123328bc691`, machine
`representation-5da5f4-cohort5090-60gb-d123328bc691`. CUDA numerics passed and
all eight scientific learners started. First-update collection/learning took
169–239 seconds per learner; a measured second update took 151.24 seconds.
The 66,000-second reserve requires 32.23 seconds/update. This invalidates the
concurrent-allocation assumption, rather than establishing learning saturation.

Operator decision, 2026-11-04: stop dependent execution and return this allocation
to design review, following implement's architectural-counterexample boundary.
The whole cohort was cancelled before evaluation; no score selected a seed,
checkpoint or stopping point. Etude's terminal cancellation is 18:26:21 UTC.
Last child manifests remain running snapshots; never rewrite them as successful
or complete. The failed attempt remains in the evidence and no-retry denominator.

Etude confirmed deletion at 18:29:35 UTC; Vast confirmed both this instance and
the earlier admission absent at 18:30:30 UTC. Full-rental retained estimates:
admission $0.1146496 and cohort $0.15609933786798355; neither bill has settled.
Together with the prior known $0.07258259388618171, estimated provider total is
$0.34333153175416525. Keep the original reservations and $6 ancillary reserve;
provider absence is not a final invoice. Output inventory contains 116 retained
files. Compact receipts live under
`experiments/data/state-representation-gpu-5da5f4/`; the launcher's output inventory
retains the complete artifact names and hashes. Local operational logs remain
under `.runs/representation/gpu-plateau/`. Interrupted redundant SCP bytes there
are partial, not admitted evidence.

Draft execution discovery, not a selected replacement: independent CUDA contexts
serialize GPU work; NVIDIA MPS can enable cooperative concurrent kernels
([NVIDIA architecture](https://docs.nvidia.com/deploy/mps/architecture.html)).
The rented host had the MPS control binary, but MPS was neither enabled nor
validated. A replacement needs complete-cohort throughput admission before any
new scientific launch, plus a prospective decision about the failed cohort and
no-retry boundary. No new allocation, production promotion or merge is selected.

Delete — do not maintain: the unexecuted per-replicate TrainingJob allocation
and worker path were removed together. Frozen CPU science and executed admission
retain historical sources; `execute_regime` and monitor remain the authorities.

Check: focused study/discovery tests passed, plus the untracked-data policy test;
all 23 published receipt entries match local size/SHA256, and every prior manifest
entry is unchanged. Ruff and whitespace checks passed. Gate owns broader suites.
Remaining: reconcile unsettled bills and review the invalidated allocation.
The plateau objective remains unmet; no new scientific launch is selected.

The historical sections below describe the completed CPU cohort, not the new
continuation's authorization or acceptance criteria.

2026-11-04 · Task 5da5f4 · results stop at demo/review; no model promotion.

## Accepted outcome and scope

Jack Heart requested the combined effective-state, shared-effect and memory
investigation on 2026-11-03. On 2026-11-04 he approved shipping the bounded
shared-effect repair, delivered in PR419, and separately authorized/resumed
replicated representation learning within $25 total. The latter stops at results
review. Neither approval selects temporal memory, semantic coalescing, a new
production default or further training after this frozen cohort.

## Delivered behavior and evidence

All eight replicates completed on `92becff6091912a3ec83ae8fdd4e1c2347ff1d45`:
128 updates and 4,096 games each, 4,056,651 learner decisions total. All 1,024
held-out games completed and replayed. GRU minus pooled score was −0.98 points
(95% paired-seed interval [−10.13, +8.18]); training-time ratio 1.06. The declared
advantage criterion failed. This is inconclusive, not equivalence or a winner.
See `experiments/state-representation-v7.md` and the published manifest-backed
receipts in `experiments/data/state-representation-v7-5da5f4/`.

Both arms use the delivered v7 contract; effect and continuation reduction change
jointly. No temporal memory, ABI/default, native rules or frozen pilot evidence
changed. The scientific source archive is retained. The maintained runner is
`experiments/state_representation_v7.py`, matching the v7 tracking name; its
Etude page shows training diagnostics separately from the scientific comparison.

## Completed cleanup

- Removed the stale v5 scientific declaration and unexecuted three-arm schedule.
  `experiments/state_representation.py` became `state_representation_v7.py`;
  historical source/receipts retain the original filename.
- Removed `manabot.verify.representation.Measurement`, `prove`, `main` and its
  three-arm timing command. It predates v7 and has no remaining consumers.
  Retained `source_effect_root` and `shared_effect_root`: current transport/model
  tests still need these native fixtures. Old proofs remain at `9dc32125f`.
- Removed the completed `runs(canary=True)` allocation path. Retained `recipe(smoke=True)`
  for bounded learner tests and `runs(revision)` for cohort/report discovery.
  Calibration receipts stay excluded and bound to their original sources.
- Consolidated completed continuation and PR419 verification notes here and in
  `questions.md`; full notes remain at `9dc32125f:scratch/`.

No remaining deletion targets. Frozen recipes, protocol hashes, run order,
allocation limits and receipts are preserved after the capacity repair below.

The 2026-11-04 sync incorporated main `5dd11cec4`: its single catalog,
schedule/remembered-subject/delayed-frequency relations, configurable card buffers
and v5 pilot entry point coexist with this study runner. The completed experiment
remains bound to its archived source; the current tree's later rules and catalog
are not a replication of that source. Retained sync checks cover integration,
not scientific rebinding. Reconciliation found that main's fresh action capacity
had grown from 64 to 512, changing both recipe hashes and rejecting retained
evaluations in `compare`. The runner now pins every original observation axis;
a regression check binds both recipes to the independently retained TrainingRun
digests. The read-only audit again admits all eight endpoints and reproduces the
retained contrast and timing exactly. No scientific bytes were rewritten.

## Cost and remaining boundary

Jack Heart authorized $25 total, separate from MTG-213. L40/L40S were unavailable;
4090 launch `8dd337797e0a` refused capacity without acquisition. The whole-cohort
CPU fallback was frozen before training: two balanced four-process batches,
one thread and 16 GiB per run. Evaluation followed training, four endpoints at
once with two active games each. No retries or score-based extensions.

Prior L4 bill: $0.07258259388618171 (supersedes $0.07083924 estimate). No new rental
charges; the $6 ancillary reserve remains pending invoices. Provider-name checks
confirmed both relevant pods absent. Etude retains refused-session metadata
without a provider receipt; its terminate operation skips refused sessions.
No Task training/evaluation process remains; MTG-213 allocations are untouched.

Prior cleanup validation recorded 38 focused passes; sync recorded 19 Python
passes and one debug Rust exact-source/zone-change pass. These results precede
this reconciliation's explicit capacity pin. Gate owns broader acceptance;
demo/review owns result judgment.

Earlier artifact tracking check passed; Etude report admitted all 8 runs and
784 metrics. `uv run --extra dev ty check` could not start because ty is absent.
No further training is needed for this step.

## Remaining acceptance and draft questions

Results demo/review remains outstanding. The measured contrast changes effect
and continuation reduction jointly and does not compare the production `rows`
continuation. Four seeds, one authored matchup and no initialization/no-update
control prevent a general architecture or improvement claim. The review decision
is whether to accept this inconclusive endpoint comparison or authorize a new,
separately frozen study; no winner or follow-up is selected here. Gate retains
broader validation, including configured typing when its tooling is available.
Ancillary invoices remain unsettled under the existing reserve.

The original broader design questions remain draft, not completed behavior:
splitting the mixed relation table, testing history or persistent memory, and
proving any semantic coalescing. The prior design and diagram remain at
`8e6cc62df:scratch/5da5f4-design-effective-state.md` and
`8e6cc62df:scratch/observation-architecture-snapshot.html`. A memory design still
requires viewer/seat ownership, reset and slot reuse, bounded-history eviction,
replay/fork identity, exact checkpoint restoration, and matching training/serving
transitions. Active obligations must remain observable when older than the
history window. No temporal-memory implementation or experiment is claimed.
MTG-240 owns the known keyword-ordering defect; it does not permit deleting
records or weakening the source/expiry/membership constraints.

Check (2026-11-04): `uv run --extra dev --extra artifacts --extra evaluation pytest tests/training/test_state_representation_study.py tests/eval/test_experiment_files.py -q` → 8 passed; `uv run --extra dev pytest tests/training/test_regimes.py -q` → 18 passed; read-only `uv run --extra artifacts python -` receipt audit reproduced all eight endpoints, contrast and timing; Ruff, `git diff --check` and `lf context` passed (both context budgets satisfied).

## Concept review — 2026-11-04

Outcome: no design change and no open product choice blocking the accepted
completed-cohort scope. Results still require the requested demo/review; this
review neither accepts a winner nor authorizes another experiment. The earlier
question about accepting the result or commissioning a follow-up belongs at that
boundary, not to a prerequisite for maintaining this completed study.

The consequential distinction is complete transport versus learned compression.
Jack Heart approved complete native effects, explicit memberships and routing
before attention. He did not require injective embeddings or approve replacing
ongoing obligations with memory. `encode_shared_effects` projects native
provenance; `SharedEffectEncoder.forward` reduces each payload and accumulates
its membership contributions into addressable objects before attention and
action-focus lookup. The study's GRU reads tokens within records, without state
across decisions. Its joint effect/continuation treatment cannot identify which
reducer caused a score difference. Existing README and result wording already
explain these distinctions; no usage-doc or skill change is needed.

Concrete interaction: nine Drill activations over fifteen permanents produce
270 legacy keyword fragments. Explicit v7 carries nine complete payloads,
eighteen keyword details and 135 membership edges, retaining separate timestamps
and original record ordinals. Header slices own bounded member lists; neither
affected-set interning nor semantic deletion is selected. Native viewer
projection owns visibility and source incarnation. Effective characteristics
describe what holds now; active effects also describe what can change later.
People and policies still receive the same authoritative legal action space.

Failure and recovery preserve that authority. Overflow or missing native
provenance rejects encoding. Replay retained commands under their bound runtime
to investigate; changed-runtime replay needs explicit identification and cannot
replace historical evidence. A future capacity/input change requires a
prospective identity. Fetching manifest-backed receipts and calling `compare`
reads the completed cohort without training: `TrainingRun`, recipe/protocol
digests and endpoint coordinates own admission. The existing explicit-axis
repair restores those identities after default drift; today's catalog cannot
be presented as the original execution.

Counterexamples remain requirements: equal current characteristics with different
source dependencies diverge when one source leaves; unequal expiries diverge at
cleanup; later entrants need not belong to an earlier fixed affected set;
repeated additive effects cannot become one contribution. An obligation older
than a history window must still be observable. MTG-240's keyword ordering
defect remains separate and supplies no license to delete records. Complete
transport does not rule out learned aliasing or establish stronger play.

Deferred product questions, with no selection here: (1) should a separately
authorized study isolate effect reduction from continuation reduction, given
the joint treatment and wide interval; (2) should a future agent carry memory
across decisions, with what viewer/seat, reset, eviction, replay/fork and exact
restoration contract? This comparison cannot answer either. Relation-table
redesign and semantic coalescing remain proposals. None is a prerequisite for
presenting this cohort honestly.

Next proof remains gate and results demo: show all endpoints, the failed
advantage criterion, uncertainty, accounting limits and frozen identities.
Concept review changed no implementation and ran no training. Its page is
`scratch/concept-review.html`; source inspection is not a browser-rendering proof.

## Publication verification — 2026-11-04

`lf sync` completed cleanly against main `5dd11cec4`. Full diff inspection
removed an exact duplicate source-duration parity test left by reconciliation.
The publication checks found that pinning the scientific 64-action axis also
made the unscored smoke incompatible with today's native 512-action buffer.
The smoke now inherits the current action capacity; scientific recipes keep
both retained digests. The report explicitly distinguishes current integration
checks from reproduction under the archived scientific runtime.

The bounded validation set passed 147 checks initially; its two learner-smoke
failures were repaired and both then passed together with eight recipe/discovery
checks (10 passes). Coverage includes model/state transport, checkpoint reload,
experiment report/runner, recipe discovery, CI/release selection and untracked
artifact policy. Ruff and diff whitespace checks passed. A read-only receipt
audit admitted all eight endpoints and reproduced the contrast/timing exactly;
all 1,024 retained replay flags passed. All 823 added manifest entries matched
local file size and SHA256, and every prior manifest entry remained unchanged.
The existing artifact publication log records those 823 published entries.
No new scientific cohort, rental, model promotion or merge was performed.

## Restart implementation — 2026-11-04

Prospective admission is appended to the GPU protocol. Measure both arms at
one and two concurrent learners, eight updates each on excluded seed 74200,
without scores; project total cohort cost with 50% allowance and separate
setup/evaluation. Latest read confirms both prior Vast instances absent and
both bills unavailable. The cheapest compatible fresh offer was a 3090 at
$0.2474074074/hour; keep old reservations and $6 ancillary pending settlement.
The additional $0.50 admission reservation gives $24.2725825939 reserved.

Delete — do not maintain: the invalidated eight-context scientific allocation
in `worker`, its 10% CUDA allocator cap and the 20-hour assumption. Preserve
scientific recipes, independent seeds, curve/held-out definitions, uncertainty,
ordinary execute_regime/monitor authorities and archived failed evidence. The
replacement execution arrangement depends on measured admission; no scientific
launch has been selected or authorized by a score.

Check: nine focused study tests, Ruff and whitespace passed. Gate retains
broader suites. Remaining: execute admission, reconcile billing/reservations,
then freeze and launch only an affordable measured replacement.
