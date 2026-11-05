# V7 representation learning — Task 5da5f4

## Latest direction — recovery cohort, routed to implement

Jack Heart, 2026-11-05: "can we schedule soem evals in paralell with the
additional training". This selects option A below and supersedes the
failure-only/no-survivor-scoring stop for a prospectively amended recovery
cohort, within the original $25. Implement and execute it; do not launch another
Graph. Merging PR84 or the study is **not** approved.

- Preserve the three completed learners and the failed attempt; the original
  protocol stays immutable. Train the five missing learners, restarting
  pool/74212 from initialization on its original seed.
- Before any scoring, freeze the amendment, retained-run admission and unchanged
  seeds, horizon, evaluator and plateau criteria. Interim scores select no
  treatment, retry, horizon or checkpoint; partial curves are labelled descriptive.
- Dispatch independent CPU evaluation jobs from durable checkpoints alongside the
  GPU learner, with bounded resources that avoid GPU contention. Require fresh
  actual-runtime/world compatibility and replay validation.
- Replace `evaluate_arm`'s hardcoded `concurrent_activity`, which claims all
  training finished. Persist each evaluation and training unit incrementally with
  hashes and idempotent job identity; do not wait for all eight learners.
- Use the launcher liveness repair before another rental. SDK repair is authorized;
  a tested exact control-client source pin may avoid changing frozen learner code.
  Expose any real compatibility or approval dependency rather than assuming merge.
- Fresh quotes and all concurrent commitments must fit the remaining allowance.
  Reusable interruption recovery stays with Etude `ff3d48`.

**Status.** Built at `c8361c675`: `RECOVERY` and the protocol's recovery amendment
(frozen before any scoring), per-unit Runs with `unit_key` identity, the
hash-verified ledger, and per-job `concurrent_activity`. `scientific_cohort`,
`ScientificAllocation` and `evaluate_arm` are gone. Local operations live in
`.runs/representation/recovery/`: `plan_*.py` run in the Manabot venv, and
`control.py` in a separate venv pinned to Etude `6bfa114ed`, with workers locked
to `64e9eaabf`. Execution state is below in **Recovery execution**.

## Recovery implementation constraints

Verified from code and retained receipts:

- **`SCORING` stays byte-identical.** `study_protocol_sha256` hashes it and
  `score_for` compares the whole protocol, so editing its "no retries" clause
  would change the evaluator identity. The amendment is a separate frozen record
  that overrides that clause for this cohort; the runner reads both.
- **`concurrent_activity` is free to change.** It is a result field outside
  `MonitorProtocol`. Pass what is actually concurrent for each job.
- **Admission uses `run.identities` and `artifact["path"]`.** `checkpoint_manifest`
  checks world, ABIs, matchup, engine source and content, and exact bytes, so
  retained `run.json` needs its paths rebased as `restore_check.py` does. This
  rebasing has not run through a real evaluation.
- **Idempotence needs a rule.** `evaluate_checkpoint` refuses an existing
  directory by design. A complete, hash-verified `monitor.json` is skipped. An
  attempt without a score may repeat in a fresh attempt directory, retained in
  the denominator; a scored endpoint is never repeated.
- **Runtime equivalence, rechecked.** `managym/` and `monitor` sources are
  unchanged since `af601877e`. The only runtime module diff is
  `manabot/model/shared_effects.py`: token-index construction yielding the same
  integers. The study runner did change in compress, structurally only: recipe
  digests at HEAD match all seven retained receipts and the job command text is
  the same. Unwatched `supervision.json` now records grace 0 rather than an
  unused 120, and the recovery runner will change again, so new learners bind
  by these identities, not by source hash equality.
- **Evaluation cost, from CPU-study receipts.** The cohort needs 11,264 games
  (72 × 128, 8 × 256). Eight retained runs took 0.59–0.68 s/game wall and
  0.83–0.92 CPU-s/game at two active games, on a different host, 64 actions and
  weaker policies. That implies about 1.9h sequential or about 15 minutes per
  learner, against about 16h of training. The sequential 5,400s bound needed
  0.48 s/game, which those receipts do not support; parallel scheduling removes
  it as a cohort-wide limit. Measure the first endpoint before committing a CPU
  allocation; no estimate here is admission evidence.
- **Control client.** PR84 head `6bfa114ed` is two commits ahead of the lock pin,
  zero behind, CI passing, still draft. It also carries the Lium launcher work,
  so a pin covers more than liveness. Run the launcher from that exact commit and
  leave `uv.lock` and the worker at `64e9eaabf`; earlier attempts already kept
  separate launcher and worker identities. Untested for this pairing. If
  they prove incompatible, that is the real approval dependency to expose.
- **Pool/74212 restart.** Compare its `initial_raw` SHA256 with the retained one
  and record the result; a mismatch needs explanation before that learner is
  scored. The GRU/pool pair at 74212 may train on different hosts. Require the
  same GPU class, record each learner's host, and state this in the amendment.

**Execution shape, accepted by the frozen amendment.** One Etude Run per learner
and per learner's evaluation. The locked runtime uploads outputs only at attempt
completion, which is why `e51f61b92931` delivered nothing. Learners use Vast RTX
3090 rentals; evaluators use RunPod CPU rentals, whose launcher still has the
single-read rule, so an evaluator can be lost and repeated under the ledger's rules.

## Execution record and remaining work

Scientific Run **`e51f61b92931` ended `lost / machine_lost` at 2026-11-05
13:09:07 UTC**, at unchanged frozen source `af601877e`. The launcher retained no
child exit code or final output inventory. The provider confirmed rental
`59194544` absent and no Task rentals live at 13:09:35 UTC. No cancellation was
requested, and the frozen deadline was still two days away. The recovery audit
below supersedes the earlier "cause unknown" reading: our own launcher deleted
the rental. What the provider reported first remains unknown.

The last hash-verified receipts, 13:00:43 UTC, retain three completed learners
(pool/GRU seed 74211 and GRU seed 74212), each with 256 updates and 65,536 games.
Pool seed 74212 had reached 83 updates and 21,248 games. Recorded exposure is
**851 updates, 217,856 games and 27,682,451 learner transitions**; work after that
snapshot is unknown. All **36 backed-up artifacts / 371,853,657 bytes** match
their receipt sizes/SHA256, including all nine scheduled raw checkpoints for
each completed learner and initialization/update-32/update-64 for the fourth.
These backups are partial recovery, not a delivered final inventory.

No curve or held-out endpoints were retained. Five learners remain uncompleted
in the evidence, and all 72 curve and eight held-out endpoints are missing.
The study therefore has **no qualified plateau, improvement or architecture
result**. No survivor analysis, scoring, retry, shortened horizon or replacement
allocation ran. The scientific objective remains unmet; this failed cohort and
its identities stay preserved under the recovery direction above.

Fresh accounting reports $2.603 for this rental versus a conservative full-life
estimate of $2.63946664. Prior per-rental reported/lifetime maxima plus settled
L4 remain $1.04590824, covered by the existing $3.50 prior-compute reserve.
Including $6 ancillary and the larger current-rental figure gives **$12.13946664
accounted/reserved**, leaving $12.86053336 under the original $25 ceiling. There
are no live commitments. This is operational accounting, not a final invoice
or authorization to spend the remainder.

Evidence is manifest-backed under
`experiments/data/state-representation-gpu-5da5f4/batched-plateau/`.
`terminal-20261105T130907Z/` retains the terminal attempt/status, carried logs,
empty delivered inventory, final accounting, all collected progress receipts,
36 independently reverified backups and operational helpers. Earlier launch
and progress directories remain immutable, including the published first
completed learner in `progress-20261105T071645Z/`.
Operational helpers remain in `.runs/representation/gpu-plateau-batched/`;
`collect.py` retained the failed terminal outcome, `final_account.py` confirmed
absence/costs, and `analyze.py` was not run because success and every endpoint
are prerequisites. The foreground watcher exited normally on machine loss;
no local monitoring command remains running. Launcher state remains owned by
`.runs/representation/etude`. Exported weights alone do not satisfy an evaluated
endpoint.

Diagnostic `911423cce90b` passed four frozen comparisons and six CUDA routing
tests, retaining both complete profiles and all 59 verified outputs. Separate
admission `37c1523a98ed` completed four updates per arm with all 49 outputs
verified. Both rentals are absent. The 55.66970-hour / $13.77310 full-cohort
projection qualified the $15.50 allocation at the same `af601877e` source.
`vast-host-diagnosis.md` owns detailed measurements and their limitations;
completed diagnosis/admission remain engineering evidence, not scientific results.

## Recovery audit and options — 2026-11-05

Evidence: `batched-plateau/recovery-audit-20261105/` (read-only, $0, no scoring).

**Who ended the rental.** Vast's audit log shows `api.instance_DELETE` for
`59194544` at 13:09:07.30 from account key `36003339` at the launcher's address.
The same holds for the other two `machine_lost` rentals, `59185580` and
`59152469`. Normal completions self-delete with instance-scoped keys. The
launcher's "deletion confirmed" followed 0.48s later. `VastMachines.alive` treats
one `offline`/`exited`/`stopped`/`error` status or a missing read as dead, and the
scheduler then calls `end()`, destroying the disk. The worker's `out/ended` was
404 at 13:09:05.96, and pool/74212 last recorded progress at 13:00:26. The
triggering status was never retained, so a host fault is neither shown nor
excluded; whether the host would have returned is unknown.

**Rental mode.** On-demand by construction: the search uses `type: ondemand` and
the create request carries no bid price. No retained provider field reads this
back. Interruption by a higher bid is not the mechanism. Account credit was
$96.20 at the audit read, so depletion is unlikely.

**Delivery fix.** Etude draft PR
[#84](https://github.com/loopflowstudio/etude/pull/84) keeps a Vast rental through
a 15-minute outage and retains the status that opened it. It is unmerged and not
in `uv.lock`, so no rental has this protection yet. RunPod and Lium keep the
single-read rule. Durable training recovery stays with Etude Task `ff3d48`.

**What the backups restore** (`restore-check.json`):

| Learner | State | Result |
|---|---|---|
| pool/74211, GRU/74211, GRU/74212 | 256 updates; raw, EMA, Adam, all nine curve points | Pass the study's curve admission, hashes and loader after path rebasing. Evaluation can use them without retraining. Max Adam step equals each receipt; lower per-parameter steps match absent gradients. |
| pool/74212 | 83 updates recorded; raw at 0/32/64 only | Exact continuation is **unavailable**: no Adam, EMA or RNG was saved. Updates 65–83 have no checkpoint at all. |

Recipe digests match this checkout. `training.sqlite` was not retained; evaluation
reads only `run.json`, but nothing has been run end to end on rebased paths.

**Selected: option A.** Keep the three learners, rerun pool/74212 from update 0 on
its seed, train the other four and evaluate all eight: about 17.7h, or 25.6h with
the 50% allowance; about $4.37, or a $6.34 ceiling, at the 3.18h measured learner
mean and $0.2474/hour. No fresh quote was taken. With per-learner setup this fits
the $12.86053336 remainder. It is not selection, because no endpoint has been
scored. Rejected: continuing from raw update 64 (resets Adam, EMA and RNG in one
arm of one pair, saving about $0.20); evaluating only the three (descriptive,
cannot meet the four-seed criterion); stopping (objective stays unmet).

## Accepted scope and remaining work

Jack Heart requires actual GPU training to the beginning of a sustained plateau;
the completed CPU comparison is insufficient for architecture selection. His
2026-11-04 22:38 direction and 23:15 continuation authorize diagnosing implementation,
measured repairs and bounded admission within the original **$25 total**, with
**$0.50 per diagnostic/admission attempt**. Necessary Etude SDK work and Vast use
are authorized. Diagnose ordinary workloads without synthetic load. Results stop
at demo/review: no study merge, model/default promotion, temporal memory
implementation or MTG-213 spending. Earlier “ship it” applied to PR419.

The complete CUDA profiles, parity reports and matched four-update admission
are now retained. No further runtime repair is justified by the first-step
profiles alone. The scientific run ended in machine loss; its missing training
and evaluations cannot be repaired by analyzing surviving learners. Teardown is
confirmed. Preserve seeds, horizon, score/plateau criteria and every failed
attempt. A missing endpoint stays missing until the amended cohort scores it; no
score-based retries, survivor-only conclusions or shortened study.
The launcher's delete of `59185580` is explained in the recovery audit; what it
observed first is not. `b153cc2e5a77` failed a
float32 tolerance before learning; independent float64 checks in `af601877e`
passed in the successful diagnostic without changing learner semantics.

## Delete — do not maintain

These cuts are complete; no further runtime deletion is justified, and frozen
learner code stays untouched for the recovery cohort. Next targets for implement:
`evaluate_arm`'s fixed `concurrent_activity` and `scientific_cohort`'s
train-all-then-evaluate-all loop, which recovery replaces. Keep the replacements
below and recover predecessor implementations only from git:

- Per-sample/per-frame projections in `TypedContinuationEncoder.forward` and
  `SharedEffectEncoder.forward`, and per-sequence padded packing in
  `RecordReducer` (`59ec1bd24`). The pre-change implementation is retained at
  `42709fa1b`, not as a fallback. Live-row validation, canonical order, duplicate
  memberships and input/parameter gradients remain required on the batched path.
- The combined diagnostic/admission job. `scripts/profile_representation.py`
  ends after two one-update profiles so Etude delivers traces before separate
  throughput work. Keep bounded log summaries, failure traces and verification
  of the final inventory; a progress log alone is insufficient.
- Total-elapsed horizon scaling and its exclusive expected-value fixture;
  `ArmThroughput` separates recurring and fixed costs. The duplicate `admit_arm`
  executor, eight-context launcher (`27a126422`), and v5 timing/canary runner
  (`9dc32125f`) remain historical. Keep native effect fixtures used by transport
  and model tests, frozen-v5 compatibility, and archived study recipes.
- Registry-only SDK pins and daily-charge timestamp-to-final-coverage conversion.
  Locked source selection and honest reported/lifetime accounting survive.
- Study-runner duplication: `compare_cohort` owns both paired comparisons,
  `excluded_recipe` both short horizons, and `ProgressWatch.stop` replaces
  `_run_child`'s separate `child_stop`. Recipe digests and job commands are unchanged.

`RecordReducer` consumes flat tokens and lengths for every caller, including
`SharedFactEncoder`; `join_record_rows` owns batch offsets and absent one-based
joins. Shared-effect token indexes remain necessary to gather header/detail
sequences without per-record device operations. They are routing, not features.
Unused continuation modules retain absent gradients: Adam treats None differently
from zero with existing momentum. No parameters or tensor contracts changed.

`OperatorTraces` captures the first ordinary forward through backward, clipping,
Adam and numerical health, with separate inference and partial-failure traces.
Four retained reference inputs and both reducers' CUDA routing gates precede
profiles. The diagnostic has a 30-minute work bound, one attempt and $0.50 ceiling;
setup/teardown and other rental costs still need a fresh whole-rental quote.
Measured CPU/CUDA costs and delivery evidence belong in `vast-host-diagnosis.md`.

## Budget and dependency authority

Jack Heart's “keep going” supersedes the operator-created final-settlement-only
gate in historical notes, protocols and Wave memory. The provider supplies no
finalization watermark. Do not hold cancelled quote caps indefinitely or infer
settlement from repeated daily totals or deleted instances.

The 2026-11-05 02:25:26 read confirmed all eleven prior rentals absent, with
$1.04590824 in per-rental reported/lifetime maxima plus settled L4. Completed
compute reserve is max($3.50, twice that fresh total), plus $6 ancillary and
full unknown/live commitments. This rule was fixed before admission completed
and releases unused ended-attempt ceilings under Jack Heart's no-double-counting
direction. It changes no historical receipt or $25 total and claims no final
invoice. The $15.50 scientific allocation was fully reserved against its
$13.77310 whole-rental quote; confirmed absence released the unused commitment.
The final operational reconciliation is above. No additional allocation ran.
Before another purchase, reconcile a fresh locked-SDK read of Task instances and
reported charges against whole rental lifetimes, quotes and disk/transfer terms.
Larger costs reduce the allowance; charges and their reservations count only
once. Unknown live allocations retain their full commitment. Historical receipts
remain immutable; no settlement watermark is required to continue implementation.

Jack Heart selected public Etude Git main for source checkouts. `uv.lock` pins
merged PR82 at `64e9eaabf016f567cb9567280bcf8730f6f5d524`; all four dependency
declarations retain `>=0.2.14` for published metadata. Refresh explicitly before
freezing a run via `docs/development.md`, never during training. Launcher and
archive-style worker source proofs agree; later admissions also verified locked
Linux installation, native build and CUDA numerics. Historical attempts keep
their released-launcher/locked-worker identities; no SDK release is claimed.
Provider credentials use Doppler `etude/prd`, artifacts `manabot/prd`; nested
injection supplies both without copying or printing keys. Allocations are
Task-local and separate from MTG-213.

## Surviving execution and acceptance

`experiments/state_representation_gpu.py` owns declaration, admission, sequential
scientific orchestration and analysis. Discovery defaults to excluded admission:
seed 74200, four complete-game updates per arm, 256 streams, CUDA float32, one
thread. Each learner has 1,800 seconds, 24 GiB host allowance and 80% CUDA
allocator allowance. Rental work is 75 minutes plus 15 minutes setup, capped at
$0.50. The longer bound avoids the earlier 600-second finalization failure;
future startup/progress/cost rules can stop it earlier and retain evidence.
No current offer or admission pass is implied by these declarations.

Science requires an explicit frozen `ScientificAllocation`, binding measured
phase costs, retained admission digest, matching hardware and remaining cap;
Etude separately checks the live quote. One learner owns the GPU at a time,
balancing arm order over paired seeds 74211–74214. Each runs 256 updates × 256
complete games = 65,536 games. `IterationSchedule(steps=256,
reference_steps=2048)` follows game exposure with nonzero rates. Wider batches
change optimization and require fresh identity; this is not numerically
equivalent to the cancelled 32-stream study. Both reducers retain the authored
Allies/Lessons population, width64/depth2, policy FiLM and v7 inputs. Pool versus
GRU jointly changes effect and continuation reduction, not temporal memory.

Every learner exports initialization, seven interior 32-update checkpoints and
final raw weights. Evaluation must complete all
72 curve endpoints (deals 85001–85032) and eight final held-out endpoints
(86001–86064), both decks/seats, with replay validation. The fixed final three
intervals must each have both arms' four-seed 95% t gain intervals wholly inside
+/-0.05. A nonsignificant slope alone does not meet this criterion; a qualifying
plateau in poor play still establishes no competence. Missing seeds, games or
checkpoints prevent a qualified result. No best-checkpoint selection or shortened horizon.

`ArmThroughput` scales only recurring collection/learning/bookkeeping with the
update horizon and 50% allowance; initialization/close, exports, outer orchestration,
setup, evaluation and teardown remain charged. `ProgressWatch` distinguishes
startup, completed-update progress and phase cost, rejects wrong placement,
retains its reason, requests stop once and reaps the child group within grace and
rental bounds. The ordinary collector consumes the executor stop/resource callback.
Fixture validation establishes these controls, not live enforcement on frozen
rentals. `ArmEvaluation.score_for` owns complete-row/protocol/recipe admission;
analyses own their schedules and duplicate detection.

## Attempt denominator and frozen evidence

Rows through `803bcc26e7de` are 2026-11-04 UTC; subsequent rows are 2026-11-05.
Evidence is manifest-backed under
`experiments/data/state-representation-gpu-5da5f4/`; amendments and exact
allocations remain in `experiments/protocols/state-representation-5da5f4-gpu.md`.

| Run / source | Retained outcome |
|---|---|
| `edb504230b42` / `35770e55b` | Excluded 5090 admission: four GRU updates, 128 games, 24,202 decisions, 733 optimizer updates; 77.19s collection/learning, 4.32 GB RSS. Deleted 17:36:17. |
| `9c7c8e6841a4` / `27a12642200a35fc0cb436a9db69ce16c505016a` | Eight-learner 5090 science cancelled before scoring at 18:26:21. Last running snapshots: 320 games, 57,258 decisions. 116 retained outputs; deleted/absent 18:29:35. |
| `332a954fd935` / `978bf3eae` | 3090 never booted: `machine_lost_before_start`, no learning. Deleted/absent 19:12:11. |
| `1fca2d156e65` / `f0669347e` | Japan 3090 passed numerics; pooled paused after 7/8 updates, 224 games, 40,241 decisions. Collection 204.40s, learning 143.74s; 4.03 GB RSS, 2.95 GB CUDA. No GRU/pair; absent 19:30:36. |
| `52617ef80df1` / `1ca17ed12` | 256-stream pooled learner entered then `lost / machine_lost`; no final exit/inventory. Exposure/cause unknown, not zero or proven OOM. Deleted 19:41:03. |
| `886233843d34` / `e5df2a58758f` | 4090 pooled paused at predictive finalization after 3/4 updates: 768 games, 143,076 decisions, 560 optimizer updates, 348.8849s collection/learning. No GRU. All 23 outputs (64,896,907 bytes) verified; controller retained failed, not cancelled. Absent 21:00:59. |
| `803bcc26e7de` / `22c10ca55197ab225546e5880cc53709db985901` | Cheaper 3090 completed both arms; result/diagnosis above. Absent 22:58:28. |
| `a46e80b31212` / `66ec957fea9fe94d6ed919e2e971fb4f63d3f468` | Both parity gates returned success; two profiled updates and one ordinary pooled update appear in carried logs. Lost at 01:12:47 without output inventory; all eight rentals absent at 01:16:18. No science or qualified throughput result. |
| `b153cc2e5a77` / `2b2061c8d` | Frozen CUDA references passed; 1/6 routing tests failed a float32 parameter-gradient tolerance. No learner. Both reports verified; rental absent, reported $0.013. |
| `911423cce90b` / `af601877e` | Four reference comparisons and six routing tests passed, both profiles completed 256 games. All 59 outputs verified; rental absent, reported $0.030. |
| `37c1523a98ed` / `af601877e` | Four ordinary updates/1,024 games per arm; all 49 outputs verified. Rental absent; reported $0.052. Complete projection $13.77310. |
| `e51f61b92931` / `af601877e` | Lost / machine_lost at 13:09:07, no final inventory or exit code. Three learners completed; fourth at 83 updates in last receipt. 36 backed-up artifacts verified; no evaluated endpoints. Rental absent, reported $2.603; no qualified result. |

Different source hashes prevent interpreting the cheaper 3090 comparison as
hardware speed. Peak memory did not establish OOM, and concurrent-cohort slowdown
is plausible contention, not causally profiled. Both 5090 charges exceeded their
estimates. Timing, refusals and charge detail: the protocol's dated sections and
`aec9f3940:scratch/representation-learning-plan.md`. Interrupted SCP bytes are not
admitted evidence.

## Delivered contract, CPU evidence and deferred design

PR419 merged as `fb5ad2c1372f9c967b550f0b8ae0f84883f75563`. Explicit v7 has
128×18 effect headers, 512×3 details and 2048×2 memberships. Native payloads
retain controller, exact source incarnation, duration/order and duplicate edges;
headers own offset/count slices and members retain one-based joins and original
ordinals. Learned vectors route into affected objects before attention/focus.
Nine Drill activations over fifteen objects use nine payloads, eighteen details
and 135 edges instead of 270 fragments. Bounds still reject overflow. Fresh
defaults remain v6; v1–v6, checkpoints and pilot evidence keep their meanings.
Lossless transport does not imply injective embeddings or stronger play.

CPU source `92becff6091912a3ec83ae8fdd4e1c2347ff1d45` completed eight runs of
128 updates/4,096 games, 4,056,651 decisions and 1,024 replayed held-out games.
GRU minus pool: -0.98 percentage points, paired-seed 95% interval [-10.13,+8.18],
time ratio 1.06. No advantage criterion passed, equivalence or winner established.
Four seeds, one matchup, joint reducer changes, extra GRU parameters, concurrent
CPU execution and no initialization control limit inference. Report/protocol/data:
`experiments/state-representation-v7.md` and
`experiments/data/state-representation-v7-5da5f4/`.

T7 sync `94b5bd38cabc44efe4b28afe9b8bdadff9e7baa3` and subsequent MTG-213
sync `8e8d702f807d61675fc1bca46417eee69316daa0` changed native rules/chooser/
capacities; unchanged recipe hashes do not establish unchanged worlds. T7's
published chooser and native information authority remain intact. Frozen CPU
recipes retain 64 actions; unscored smoke and GPU study use 512 with overflow
rejection. MTG-213's explicit first-64 pilot exception and $10 are separate.
Replacement science needs its own actual source/native freeze and admission.

Main syncs moved the selected Wave to Experiments and incorporated MTG-213
v10; none amends this study's capacities, recovery, population or separate budget.
The runtime-changing syncs require actual source/native binding at admission.
The pre-launch sync `2b2061c8d` / `51b4bd274` moved Linear bindings only.
The subsequent sync `8318f1525` incorporated main `e113602a2` and MTG-213's
v10 endpoint evidence. The recovery constraints own runtime equivalence since the
executed source. Its CUDA receipts do not certify the later checkout. Main has not moved since.
Experiments has no child memories. Standard growing-pool research stays separate.

Memory, relation splitting, membership interning and semantic coalescing remain
proposals. Coalescing requires identical future behavior across expiry, sources,
affected sets/later entrants, intervening grants and additive effects. MTG-240's
keyword-ordering bug remains separate: Indefinite LoseKeywords(Flying), Indefinite
AddKeywords(Flying), EndOfTurn LoseKeywords still lacks flying after cleanup.
The shipped expiry test uses correctly ordered LoseAllAbilities; no records
were deleted. Memory still needs viewer/seat ownership, reset, slot reuse,
eviction, replay/fork identity, exact restoration and matching serving/training;
obligations older than the history window remain observable.

## References and validation

Complete preceding notes: `2784e4283ce12011643b319968131ae5dbba9f6e:scratch/`.
Original design/diagram:
`8e6cc62df:scratch/5da5f4-design-effective-state.md` and
`:scratch/observation-architecture-snapshot.html`; PR419 verification:
`9dc32125f:scratch/shared-effects-main-verification.md`. Current CUDA gates,
traces and admission are retained and verified. Gate owns broader suites;
demo/review owns scientific judgment. No rendering proof or plateau conclusion.

Check: `recipe_digests()` at `aec9f3940` matches all seven retained receipts; recovery tests deferred to implement, broader validation to gate, scientific judgment to demo/review.
