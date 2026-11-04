# V7 representation learning — Task 5da5f4

## Current authorization and outcome

### Current accounting decision — 2026-11-04

After the primary conversation explained that the $24.97 hold was an
operator-imposed deadlock rather than spending, Jack Heart said "keep going".
Resume this Task. The requirement below to obtain a provider finalization
watermark before releasing every unused rental allowance is superseded for
operational budgeting. It was not a user requirement and the API supplies no
such marker. Do not invent settlement, alter the SDK to claim it, or rewrite
historical accounting receipts.

Before another purchase, take one fresh read of task-owned instances and
reported charges using the locked repaired Etude source. Reconcile each deleted
rental's whole lifetime, quoted rate, disk/transfer terms and reported amount.
Use a conservative operational estimate plus an explicit uncertainty reserve
instead of holding the entire cancelled 20-hour quote indefinitely. Start with
$2 reserved for ALL prior compute (last reported total $0.4705825939) and retain
the separate $6 ancillary reserve. That leaves at most $17 for ALL further
compute, including admission, setup, failed launches, idle time, evaluation and
teardown, within the ORIGINAL $25. Increase the prior-cost reserve and reduce
the new-compute allowance if fresh evidence requires it. Do not double-count
the same charge under both actuals and reservations. Unknown live allocations
still retain their full commitment until resolved.

Record this as a new operational budget revision, with unsettled reported costs
still labelled unsettled. The $2 choice is the operator's conservative reserve,
not a provider guarantee or a new user allowance. Preserve the prior receipts
and their dates. Exact final invoices can reconcile later; their unavailable
watermark alone must not cause another implementation loop. Continue useful
scientific orchestration while resolving execution constraints; accounting is
not a reason to postpone all non-spending implementation. Finish an ordinary
bounded two-arm throughput check and run the prospectively frozen replicated
study when its total cost fits. Do not repeat the known eight-context failure,
return to CPU-only evidence, silently reduce the learning objective, or stop
at cleanup/publication as though the plateau experiment were complete.

All older "stop until final coverage" instructions in this plan, the protocol,
Wave memory and prior Graph summaries are historical and superseded by this
current decision. Existing source tracking, original $25 cap, immutable failed
evidence, separate MTG-213 budget and results-review boundary still apply.

Jack Heart's 2026-11-04 dependency direction is implemented: source checkouts
consume public Etude Git main, with `uv.lock` pinning merged PR82 at
`64e9eaabf016f567cb9567280bcf8730f6f5d524`. All four dependency declarations use
`>=0.2.14`; downstream package metadata remains an index requirement. Refresh
explicitly with `uv lock --upgrade-package etude-sdk`, then `uv sync --locked`,
as documented in `docs/development.md`; never resolve main during an attempt.
Anonymous access, launcher identity, archive-style worker installation and offline
re-sync agree on the lock. Linux resolution passed only as a dry-run, not remote
GPU/native admission. Historical locks and receipts remain frozen. No SDK
release is required or claimed; source tracking cannot establish settlement.

Jack Heart requested actual GPU learning to the beginning of a sustained plateau
on 2026-11-04; the completed 128-update CPU comparison is insufficient for
architecture selection. His subsequent “start it when you can” and “seems fine”
authorize execution repair and a viable prospective replacement within the
ORIGINAL cumulative $25. They supersede earlier no-further-training and
return-to-review restrictions, but do not erase unsuccessful attempts or permit
score-based retries. A necessary Etude SDK release and Vast use are authorized;
Etude PR79/80 and release 0.2.14 already supply that capability.

The plateau objective remains unmet. One scientific GPU cohort was cancelled
before scoring and three restart admissions were unsuccessful. No replacement
scientific cohort has launched. At the last retained provider read, all five
relevant Vast instances were absent. No allocation is selected until accounting
is reconciled and a complete two-arm admission demonstrates affordability.
Reconciled against implementation and retained receipts on 2026-11-04.
The 20:37 UTC operational revision below supersedes the historical settlement
gate. Final invoices remain unknown; they do not prevent bounded execution.

Stop at results demo/review. No study merge, model/default promotion, temporal
memory implementation or MTG-213 spending is approved. The repair's earlier
“ship it” approval applied to PR419 only.

## Surviving implementation and next proof

`experiments/state_representation_gpu.py` declares the 256-stream candidate and
defaults to an excluded admission: four ordinary complete-game updates per
arm, sequentially, seed 74200, CUDA float32. Each learner has 600 seconds,
24 GiB host allowance and 80% CUDA allocator allowance. The rental allows
30 minutes plus 15 minutes setup, originally at most $0.20, on a compatible RTX 3090 with
at least 32 GB host RAM, included disk and free transfers. These are retained
candidate bounds, not permission to reuse its spent reservation.

Wider minibatches change optimization. Any later science needs a new prospective
identity and fresh paired seeds 74211–74214; 256 updates × 256 complete games
preserve 65,536 games per learner. `IterationSchedule(steps=256,
reference_steps=2048)` maps schedule progress to game exposure with nonzero
rates. Pooled versus GRU jointly changes effect and continuation reduction;
keep the authored Allies/Lessons population and other treatment dimensions
fixed. No score selected this candidate, and lost admission output does not
establish its speed, memory sufficiency or affordability.

Remaining work before scientific execution:

1. The operational budget revision is complete: fresh 20:37 UTC locked-SDK
   read reports $0.4705825939; five instances absent, later complete Vast
   inventory empty. $2 prior compute + $6 ancillary leaves $17 new compute.
   Receipts are in `.runs/representation/gpu-resume/` pending artifact retention.
2. Complete both ordinary admission learners and project total sequential cost
   with 50% training allowance, 90 minutes evaluation and 15 minutes setup.
   No qualifying 3090 offer is currently available; the cheapest compatible
   4090 quote is $0.611111111111111/hour, disk included, free transfers. Admission
   now has a $0.50 total cap; retain at most $16.50 for subsequent science.
3. Scientific orchestration is implemented in `state_representation_gpu.py`:
   explicit frozen allocation, sequential balanced seed/arm order, all exports,
   existing checkpoint evaluation and strict full-cohort analysis. Validate and
   run it only if the measured total arrangement fits. No completed GPU cohort
   or plateau exists yet. Flat poor play does not establish competence.
4. Retain every attempt, deadlines, teardown and costs. If capacity or measured
   throughput prevents a credible plateau within the cap, report it as unmet.

`execute_regime` owns learning; existing monitoring owns evaluation. Earlier attempts used a released
0.2.14 launcher and locked 0.2.12 worker; those identities remain historical.
Future jobs prepared from this branch use the exact Etude Git commit in its lock. Provider
credentials use Doppler `etude/prd`, artifacts `manabot/prd`; nested injection
supplies both without copying or printing keys. Task-local allocations remain
separate from MTG-213. Frozen protocol and amendments:
`experiments/protocols/state-representation-5da5f4-gpu.md`.
Its original 32-stream/eight-process sections describe source `27a126422`;
the later batched amendment describes the current candidate. The old allocation
and seed IDs are not instructions for the current executable.

## Historical attempt denominator and accounting (2026-11-04 UTC)

The final-coverage-only restrictions in this historical section are superseded
by the operational revision above; retained receipts are not rewritten.

| Run / source | Retained outcome |
|---|---|
| `edb504230b42` / `35770e55b` | Excluded RTX 5090 admission passed CUDA numerics and four GRU updates: 128 games, 24,202 learner decisions, 733 optimizer updates; 77.19s collection/learning, 4.32 GB peak RSS. Instance 59138171 deleted at 17:36:17. |
| `9c7c8e6841a4` / `27a12642200a35fc0cb436a9db69ce16c505016a` | Scientific eight-learner RTX 5090 cohort cancelled at 18:26:21 before scoring. Partial running snapshots retain 320 games and 57,258 decisions. Instance 59143644 deleted at 18:29:35; provider absence confirmed. |
| `332a954fd935` / `978bf3eae` | RTX 3090 instance 59148889 never booted before startup deadline; `machine_lost_before_start`. No learning. Deleted; absent at 19:12:11. |
| `1fca2d156e65` / `f0669347e` | Japan RTX 3090 instance 59149701 passed numerics; sequential pooled learner paused at finalization after 7/8 updates, 224 games, 40,241 decisions. Collection 204.40s + learning 143.74s (49.73s/update), 4.03 GB host RSS, 2.95 GB CUDA reserved. No GRU or concurrent pair. Deleted; absent at 19:30:36. |
| `52617ef80df1` / `1ca17ed12` | 256-stream admission entered its pooled learner, then `lost / machine_lost`, no final exit status or output inventory. Cause/exposure unknown, not zero or an established OOM/native/model fault. Deletion confirmed at 19:41:03. |

The scientific allocation required 32.23 seconds/update for 2,048 updates in
66,000 seconds. First concurrent updates took 169–239s and a second took
151.24s, versus isolated admission's 19.30s/update. “5–7× longer” compared with
the required pace, not an isolated hardware regression. Contention is plausible,
not conclusively profiled. At those rates the cohort needs roughly 86–136 hours.
Cancellation invalidated the allocation, not the scientific treatments; keep its
no-retry denominator and last running child snapshots unchanged.

Earlier RunPod 6000Ada/L40S/A100 refusals acquired nothing. Vast 3090 launch
`ea36a15587b1` for prepared run `e4bc79b1e907` also refused before acquisition;
4090/5090 selection was prospective, not score-based. The unexecuted per-run
allocation survives at `8c7712bc9`. Acquisition details and complete inventories
remain in the protocol and historical plan cited below.

The 20:10 UTC repaired SDK read is the current accounting evidence. All five
rentals remain absent and terminal; reported daily charges in the table's order
are $0.168, $0.164, $0.003, $0.042 and $0.021 ($0.398 total). Including the settled
L4 bill of $0.07258259388618171 gives $0.47058259388618171 reported plus settled,
not final expenditure. This supersedes the $0.4487194142 billed-plus-estimated
provider total; both 5090 charges exceed their earlier lifetime estimates.
Conservative reservations remain $24.97258259388618171, including the pending
$6 ancillary reserve. Release is $0; $0.02741740611381829 remains unreserved,
so another $0.20 admission does not fit. Reserved dollars are not money spent.

The 19:55–19:57 UTC investigation reproduced two Etude 0.2.14 defects: its query
excluded the creation day's charges, and its elapsed-interval coverage guard
rejected daily rows (`start == end == 1793750400`, November 4 UTC midnight).
The [provider API](https://docs.vast.ai/api-reference/billing/show-charges) documents
date-range charge reports, not exact settled-through timestamps. The repair
replaces `VastAPI.bill` with typed `VastAPI.charges`, including the creation UTC
day and retaining inclusive dates and observation time. `VastMachines.bill`
correctly supplies no settled coverage. The single repaired read at source
`109fa4128052972699ec2a63fba5a3958fac430f` returned the same amounts. Its tree
matches merged [Etude PR82](https://github.com/loopflowstudio/etude/pull/82),
now pinned above. The task-local repair left other Etude checkouts unchanged;
the patch, source identity and both accounting reads are retained as artifacts.

Missing evidence: provider-confirmed final GPU, disk and bandwidth totals covering
each contract through its retained teardown timestamp (17:36:17, 18:29:35,
19:11:26, 19:27:24 and 19:41:03 UTC, respectively). Daily dates, instance absence,
exhausted pagination and repeated identical totals do not establish this coverage;
the documented API exposes no finalization watermark. Do not release the $17.20
cancelled-cohort cap or poll unchanged charges. No rental, replacement scientific
implementation, launch or score followed this accounting gate. Full preceding
accounting details remain at
`075378dd87600428ad3efa60b7288b327c32cee7:scratch/representation-learning-plan.md`.

Manifest-backed evidence: `experiments/data/state-representation-gpu-5da5f4/`,
including `restart/`, compact receipts, all attempt inventories, quotes and final
accounting. The first cohort inventory names/hashes 116 outputs. Operational
logs remain under `.runs/representation/gpu-plateau/` and `gpu-restart/`;
interrupted SCP bytes and lost outputs are not admitted evidence.

## Delivered contracts and frozen CPU evidence

PR419 merged as `fb5ad2c1372f9c967b550f0b8ae0f84883f75563`. Explicit v7 carries
128 × 18 shared-effect headers, 512 × 3 details and 2,048 × 2 memberships;
headers own offset/count slices, members keep one-based public object joins
and original record ordinals. Complete native payloads preserve controller,
exact source incarnation, duration, order and duplicate contributions. Route
learned vectors into affected objects before attention/action focus. Effective
characteristics alone cannot describe future expiry or source dependencies.
Nine Drill activations over fifteen objects become nine payloads, eighteen
details and 135 edges instead of 270 legacy fragments. Bounds still reject
overflow; old captures cannot supply missing provenance. Lossless transport
does not establish injective learned embeddings or stronger play. Fresh defaults
remain v6; frozen v1–v6 and pilot evidence retain their meanings.

The CPU cohort completed on `92becff6091912a3ec83ae8fdd4e1c2347ff1d45`:
eight runs, 128 updates/4,096 games each, 4,056,651 learner decisions and 1,024
held-out replay passes. GRU minus pooling: −0.98 percentage points, 95% paired-seed
interval [−10.13, +8.18], training-time ratio 1.06. No advantage criterion passed;
this establishes neither equivalence nor a winner. Four seeds, one matchup,
joint reducer changes, extra GRU parameters and absent initialization control
limit inference. Full report/protocol/receipts remain in
`experiments/state-representation-v7.md` and
`experiments/data/state-representation-v7-5da5f4/`.

Main sync `94b5bd38cabc44efe4b28afe9b8bdadff9e7baa3` integrates T7 at
`4463e57f5a92ebd3b60e2fb50c046b84ac2c0372`, following T6's catalog,
scheduling/delayed-frequency relations, configurable cards and v5 pilot. T7
adds player-control relation kinds 13/14 without widening columns; published
`ActionSpace.player` owns the chooser and native inspection owns visibility.
These upstream rules changes do not refresh scientific evidence. A replacement
must pin and admit its actual source/native runtime; no unchanged-world claim
follows from unchanged recipe hashes. Original CPU recipe axes remain pinned
(64 actions); smoke uses today's native 512 actions. The earlier default drift changed both recipe
hashes despite passing discovery checks; regression hashes and eight-endpoint
read-only admission now protect retained identities. Re-execution needs archived
source/native runtime, not rewritten evidence.

## Retired implementation and surviving ownership

Delete — do not maintain: the SDK's `VastAPI.bill` timestamp-to-coverage
conversion and its elapsed-interval fixture were removed in the dependency
repair. Daily-charge, pagination and unknown-settlement behavior survive on
`VastAPI.charges` and `VastMachines.bill`. The four 0.2.12 release pins and
registry-only SDK lock entry are replaced coherently by the Git source and
exact lock. No remaining deletion targets. The admission-only worker dispatch is replaced
by explicit admission/science dispatch; both use the same child lifetime owner.
The failed eight-context launcher (`27a126422`)
and v5 three-arm timing/canary paths (`9dc32125f`) are retired; archived
executions use their pinned source. The native `source_effect_root` and
`shared_effect_root` fixtures still serve transport/model tests.

The sequential admission owns one child and its deadline, stop, reap, log and
exit receipt. `ArmEvaluation.score_for` owns complete-row and protocol/recipe
admission for both analyses; each analysis owns its seed/update schedule and
duplicate detection. Frozen identities and retained receipt bytes stay unchanged.
Live accounting and dependency status live here; questions link here instead of
duplicating the operational history.

## Deferred questions and evidence references

Persistent memory, relation-table splitting, affected-set interning and semantic
coalescing remain proposals. Coalescing needs proof across expiry, sources,
affected sets/later entrants, intervening grants and additive effects. MTG-240's
keyword-ordering defect is separate: Indefinite LoseKeywords(Flying), Indefinite
AddKeywords(Flying), then EndOfTurn LoseKeywords still lacks flying after cleanup.
The shipped expiry fixture uses correctly ordered LoseAllAbilities; no records
were deleted. A memory design still needs seat/viewer ownership, reset, slot
reuse, eviction, replay/fork identity, exact restoration and matching serving
transitions; obligations older than the history window must remain observable.

Complete prior plan/decisions/validation:
`1b2ddcdfadcb09df76b2dda450152cbdfa60c8c2:scratch/`. PR419 verification:
`9dc32125f:scratch/shared-effects-main-verification.md` (23 Python/four debug Rust
checks after rebuild; no PyPI release claimed). Original design/diagram:
`8e6cc62df:scratch/5da5f4-design-effective-state.md` and
`:scratch/observation-architecture-snapshot.html`. Main's inherited T6 notes:
`5dd11cec4:scratch/questions.md`; PR416 notes: `3b122de907:scratch/questions.md`.
No uncommitted evidence was discarded. Earlier publication validation and receipt
hash audits remain in that committed plan; gate owns broader acceptance and
results demo/review owns scientific judgment. No rendering proof is claimed.

Historical validation: `c23aacd8f95f5cea16301653fd05df72ff3156cf:scratch/representation-learning-plan.md`.
The manifest-backed `reconciliation/` evidence includes the original offline
Etude reproduction; `reconciliation-repaired/` retains the corrected read.
Runtime admission and scientific execution follow the current operational budget.

Prior implementation validation at `bdb567cc0`: 16 focused study/discovery tests
passed; read-only analysis reproduced all eight retained CPU endpoints and
rejected missing, duplicate, incomplete and wrong-identity inputs. This prose
reconciliation does not require repeating learning or those tests.

Retained dependency validation: isolated locked-source artifact/Etude integration
— 28 passed; release staging/untracked-artifact checks — 2 passed. SDK
network-isolated suite — 595 passed, 6 skipped, 3 deselected; mypy/Ruff plus PR
and merge-queue CI passed. Commands and source proof remain in
`reconciliation-repaired/source-proof.json` and `sdk-validation.json`.
This reconciliation updates Training memory's stale SDK and billing status;
no frozen receipt changed. Two-arm admission and the actual scientific cohort
remain outstanding; the sequential orchestration is now implemented.

Check: focused study checks pending; gate owns broader suites. The operational
budget permits admission; measured affordability must precede scientific launch.
