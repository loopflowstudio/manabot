# V7 representation learning — Task 5da5f4

## Current direction, result and remaining work

Jack Heart's latest direction (2026-11-04 22:38 UTC): diagnose implementation
before assigning host fault; separate setup, collection/inference, learning,
exports and unclassified time, and add prospective startup/progress/affordability
stops. Check actual CUDA, threads, batching, memory and available telemetry with
ordinary workloads, never synthetic load. Preserve the original **$25 total**
and **$0.50 attempt cap**; no competing allocation.

The authorized cheaper 3090 admission completed both arms: source `22c10ca55`,
run `803bcc26e7de`, four updates/1,024 games per arm. All 46 outputs
(131,204,931 bytes) verified. Instance `59168556` was deleted and confirmed
absent at 22:58:28 UTC. Corrected full-cohort cost is **$71.82**, or **$47.58**
for recurring training alone without allowances, versus **$16** available.
No replacement science launched; the requested plateau remains unmet.
`scratch/vast-host-diagnosis.md` owns measured phases, attribution limits,
cleanup/accounting and the prospective controls. These are dated observations,
not a live provider inventory or a matched-source hardware comparison.

Jack Heart requested actual GPU training through the beginning of a sustained
plateau; the completed 128-update CPU study is insufficient for architecture
selection. His “start it when you can”, “seems fine”, “keep going” and later
“thats ok lets try” authorize execution repair and bounded admission under the
same cumulative allowance. Necessary Etude SDK work and Vast use are authorized.
The latest cheaper-host attempt is completed, not an instruction to repeat it.
Results stop at demo/review: no study merge, model/default promotion, temporal
memory implementation or MTG-213 spending. Earlier “ship it” applied to PR419.

Implementation update, 2026-11-05: `59ec1bd24` batches shared-effect and
continuation projections/reducers after one host metadata admission per buffer.
The pooled ordinary CPU learner completed identical 256-game/51,095-decision/
200-minibatch work before and after: collection plus learning fell from 202.86s
to 121.71s, with encoder call time 30.36s to 2.71s. CProfile ran in both;
the final run additionally captured two operator traces. This is bounded local
evidence, not CUDA synchronization attribution or scientific affordability.
Native multi-position outputs and all parameter/object/definition gradients agree
with the saved pre-change oracle; focused tests include duplicate memberships,
canonical ordering, padding and absent gradients for unused Adam parameters.

CUDA profiling is blocked by connectivity: the diagnostic quote was $0.19
under a $0.50 cap, but Etude source publication stalled on GitHub before any
run/rental was created. A bounded publication attempt also timed out, and
GitHub HTTPS/Doppler reads timed out. No new provider allocation or expense
was incurred. Details and retained local evidence are in the diagnosis.
Matched GPU source/workload admission must still demonstrate affordability
before the four-paired-seed cohort. Repeating the eight-context
failure, substituting CPU evidence, shortening the learning objective or treating
cleanup as completion does not satisfy acceptance. Every failed attempt remains
in the denominator; no score-based retries or selection.

Accepted direction remains Jack Heart's GPU plateau comparison; the proposed
profiling/optimization route does not guarantee that it fits the cap. If that
route cannot establish affordability, the unchanged objective remains unmet.
No product decision is needed for bounded diagnosis; changing the budget,
population, horizon or scientific criterion remains outside accepted scope.

## Implementation cut — encoder dispatch diagnosis

The excluded local diagnosis profiles the ordinary executor, first at smoke
size and then one update of the actual 256-stream recipe on CPU (seed 74200).
It cannot measure CUDA synchronization or qualify GPU affordability. A measured
dispatch repair must preserve checkpoint parameters, complete live-row validation,
canonical token order, joins, duplicate memberships and input/parameter gradients.

Deleted in `59ec1bd24`: per-sample/per-frame neural projections and reducer
dispatch in `TypedContinuationEncoder.forward` and `SharedEffectEncoder.forward`,
plus `RecordReducer`'s per-sequence padded packing. Malformed-input rejection and
schema/semantic tests survive on the batched path. Pre-change source remains at
`42709fa1b`, not as a runtime fallback. No parameters or tensor contracts changed.

### Delete — do not maintain

Removed `RecordReducer`'s list-of-tensors wrapper and separate `reduce_tokens`
entry point: every caller now supplies a flat token table and record lengths
through `forward`, including the frozen-v5 `SharedFactEncoder` consumer. Keep
its semantic and checkpoint tests; no exclusive predecessor fixtures remain.
One `join_record_rows` helper owns batch offsets and absent one-based references
for effects, continuation objects and parent frames. Continuation edges are
grouped once by handle, then sorted by payload; handles never become features.
No further deletion targets are identified. Frozen receipts and the existing
Wave edits are retained; broader validation belongs to gate.

## Budget and dependency authority

Jack Heart's “keep going” supersedes the operator-created final-settlement-only
gate in historical notes, protocols and Wave memory. The provider supplies no
finalization watermark. Do not hold cancelled quote caps indefinitely or infer
settlement from repeated daily totals or deleted instances.

Current conservative reservations: $2 preceding compute, $0.50 for the 4090
admission, $0.50 for the cheaper 3090 admission, $6 ancillary; **$16 remains**
within $25. Latest reported-plus-settled compute is $0.85458, not final invoices.
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
final raw weights. After all training, sequential evaluation must complete all
72 curve endpoints (deals 85001–85032) and eight final held-out endpoints
(86001–86064), both decks/seats, with replay validation. The fixed final three
intervals must each have both arms' four-seed 95% t gain intervals wholly inside
+/-0.05. A nonsignificant slope or flat poor play establishes neither a plateau
under this criterion nor competence. Missing seeds/games/checkpoints prevent a
qualified result. No best-checkpoint selection or shortened horizon.

Implemented: total-elapsed scaling and its exclusive expected-value fixture
are removed. `ArmThroughput` owns
recurring collection/learning/bookkeeping versus fixed initialization/close and
exports; only recurring work scales with update count and the 50% allowance.
Periodic exports, outer orchestration, setup, evaluation and teardown remain
charged. `elapsed_seconds` owns the measured phase total used for admission.

`ProgressWatch` uses an absent progress timestamp for startup, then completed
updates for progress; completed work without declared placement is rejected.
The child supervisor requests stop once, waits only within grace/rental bounds,
and always reaps its group and retains exit status. Nonrecoverable collection
consumes the existing executor stop/resource callback. These prospective controls
have fixture validation, not a claim of live enforcement on frozen rentals.

The duplicate `admit_arm` executor, failed eight-context launcher (`27a126422`),
v5 timing/canary runner (`9dc32125f`), registry-only SDK pins and SDK daily-charge
timestamp-to-coverage conversion are retired. Archived executions use pinned
sources. Native effect fixtures remain for transport/model tests.
`ArmEvaluation.score_for` owns complete-row/protocol/recipe admission; each
analysis owns its seed/update schedule and duplicate detection.

## Attempt denominator and frozen evidence

All rows below are 2026-11-04 UTC. Evidence is manifest-backed under
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

The cancelled cohort required 32.23s/update; concurrent first updates took
169–239s and a second 151.24s versus isolated admission's 19.30s. Contention is
plausible, not causally profiled. The 4090 pooled-only projection was $20.22
without allowances/$31.39 with them, above then-available $16.50 before GRU.
Its 17.41 GB host/18.25 GB CUDA peaks did not establish OOM. Different source
hashes prevent interpreting the cheaper 3090 comparison as hardware speed.
Earlier RunPod refusals and Vast launch `ea36a15587b1` acquired nothing; retained
records remain in the protocol. Interrupted SCP bytes are not admitted evidence.

Etude PR82 corrected inclusive UTC daily charge retrieval and removed invented
settled coverage. The repaired 20:10 read found the five earlier rentals absent,
with $0.398 reported charges/$0.4705825939 including settled L4, superseding the
$0.4487194142 lifetime estimate. Both 5090 charges exceeded estimates. The 4090
reported $0.152 versus $0.17770 estimate; cheaper 3090 $0.205 versus $0.20159.
These contrary estimates remain evidence; no final invoice claim follows.

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

Upstream reconciliation, 2026-11-04: merge `70d5893aa` moved the selected Wave
from Training to Experiments and removed the Eval memory file; the objective
is unchanged and Experiments has no child memories. The incoming Standard
research distinguishes engine architecture from measured learner throughput
(`docs/research/standard-2026-11/rl-engine-contract.md`). Its growing-pool
protocol Task is separate design work, not evidence for this comparison or
authority to replace the fixed Allies/Lessons population. The merge introduces
research and Wave changes, not an encoder optimization or new GPU result.

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

Complete preceding plan, questions and diagnosis are retained at
`c1b5c46166496434242dfb121c1a174eaf5f72c7:scratch/`; no uncommitted evidence was
removed. Earlier design/diagram: `8e6cc62df:scratch/5da5f4-design-effective-state.md`
and `:scratch/observation-architecture-snapshot.html`. PR419 verification:
`9dc32125f:scratch/shared-effects-main-verification.md`. Earlier decisions:
`1b2ddcdfadcb09df76b2dda450152cbdfa60c8c2:scratch/`; historical validation and
accounting: `37511d297611aa8720c52453a0034991c753be06:scratch/representation-learning-plan.md`.
Prior phase/watchdog/executor checks and native rebuild are recorded in the
existing diagnosis; the pre-reconciliation plan at `70d5893aa:scratch/representation-learning-plan.md`
retains the 31 passing focused checks and manifest audit. Gate owns broader
suites; results demo/review owns scientific judgment. Live watchdog enforcement
and GPU profiling remain unverified. No rendering proof is claimed.

Validation: `uv run --no-sync pytest tests/model/test_state_records.py tests/model/test_shared_effects.py -q -k 'not treatment_trains and not ordinary_learner'` — 34 passed, 4 deselected; Ruff check/format and diff checks pass. Prior saved native oracles and ordinary CPU learner results remain in the diagnosis. Gate owns broader suites; implementation owns CUDA profiling/parity and matched GPU admission after the documented connectivity failure.
