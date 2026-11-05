# V7 representation learning — Task 5da5f4

## Current direction, result and remaining work

Reconciled 2026-11-05 against source `af601877e`. Diagnostic `911423cce90b`
completed both CUDA profiles, all four frozen reference comparisons and six
routing tests, including float64-reference gradient checks. All 59 outputs
(170,244,006 bytes) match the worker inventory; rental `59191723` is confirmed
absent. Separate uninstrumented admission `37c1523a98ed` completed four updates
per arm; all 49 outputs verified and rental absent. Its full-cohort projection is
55.66970 hours / $13.77310, below the freshly reconciled $15.50 allowance.
At the retained 02:34:17 UTC handoff, scientific Run **`e51f61b92931` was running**
at that exact source; its first child was pooling seed 74211. All four paired
seeds, horizon, curve/held-out schedules and criteria are unchanged. No plateau
or architecture result exists yet. The diagnosis owns detailed accounting and
trace interpretation; `batched-plateau/` retains the prospective allocation.
At 02:34:17 UTC, instance `59194544` was the sole Task rental. The worker
acknowledged `finish when done` at 02:32:32: final delivery and self-deletion
follow its one job, with rental deadline 2026-11-07 10:09:07 UTC as backstop.
Completed scientific update counts are not available in the carried parent log.
The local wait client was stopped after this managed-run handoff. This compress
pass performs no new provider read or allocation.

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
profiles alone. The scientific run must complete all training, exports and
scheduled evaluations, retain its verified inventory and confirm teardown.
Preserve seeds, horizon, score/plateau criteria and every failed attempt. A missing
endpoint remains missing; no score-based retries, survivors or shortened study.
The historical loss of `59185580` remains unexplained. `b153cc2e5a77` failed a
float32 tolerance before learning; independent float64 checks in `af601877e`
passed in the successful diagnostic without changing learner semantics.

## Delete — do not maintain

These cuts are complete; there is no pending runtime deletion target:

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
invoice. The $15.50 scientific allocation is now fully reserved; its live
whole-rental quote is $13.77310. No competing allocation or additional funds.
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
| `e51f61b92931` / `af601877e` | Affordable four-paired-seed study running; first pooling seed 74211 started. Finish-after-work acknowledged; no qualified result yet. |

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

Main syncs moved the selected Wave to Experiments and incorporated MTG-213
v10; none amends this study's capacities, recovery, population or separate budget.
The runtime-changing syncs require actual source/native binding at admission.
The latest sync `2b2061c8d` / `51b4bd274` moved Linear bindings only. Experiments
has no child memories. The Standard growing-pool research remains separate design
work, not authority to replace this fixed Allies/Lessons comparison.

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

Complete preceding plan, questions and diagnosis:
`2784e4283ce12011643b319968131ae5dbba9f6e:scratch/`. Removed detail is committed;
no uncommitted evidence was removed. Original design/diagram:
`8e6cc62df:scratch/5da5f4-design-effective-state.md` and
`:scratch/observation-architecture-snapshot.html`; PR419 verification:
`9dc32125f:scratch/shared-effects-main-verification.md`. Current CUDA gates,
traces and admission are retained and verified. Gate owns broader suites;
demo/review owns scientific judgment. No rendering proof or plateau conclusion.

Check: `uv run --locked --extra dev --extra artifacts pytest tests/model/test_shared_effects.py -k 'memberships_share_gradients or batched_effect_routing' -q` — 3 passed, 2 local CUDA skips; compilation, Ruff and diff checks pass. Gate owns broader validation; the frozen-source remote CUDA gates remain historical evidence.
