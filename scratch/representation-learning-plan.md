# V7 representation learning — Task 5da5f4

## Current authorization and outcome

Latest dependency direction, Jack Heart, 2026-11-04: switch Manabot to Etude's
current source rather than requiring SDK releases. Use a Git-main uv source,
lock its resolved commit for reproducible local/remote runs, and document the
explicit refresh command. Reconcile all duplicated release pins and extras;
validate isolated Etude/artifact integration and remote source accessibility.
Preserve historical experimental locks and evidence. Source consumption does
not publish the local Etude billing patch or resolve its settlement limitation.
The live implementation owns this dependency change as part of the continuation.
The four base/extras/play-runtime pins now use valid `>=0.2.14` package metadata.
`tool.uv.sources` selects public Git main; `uv.lock` pins
`64e9eaabf016f567cb9567280bcf8730f6f5d524`, the merged Etude PR82 billing repair.
The registry-only lock entry is removed. `uv lock --upgrade-package etude-sdk`
then `uv sync --locked` is the explicit update workflow, documented in
`docs/development.md` and `docs/training.md`; no in-run branch resolution occurs.
Anonymous Git access, installed launcher identity, a clean archive-style locked
worker-dependency install and offline re-sync all agree on that commit. Linux
locked dependency resolution passed as a dry-run only; no remote GPU or Linux
native execution is claimed. Installed package metadata remains a normal index
requirement. Historical source locks and receipts remain frozen.

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
The latest accounting evidence is the repaired SDK read at 20:10 UTC below.
Daily-charge retrieval now works; authoritative coverage through teardown is
still unavailable. That external evidence, not product approval or another
unchanged query, is required before releasing reservations.

Stop at results demo/review. No study merge, model/default promotion, temporal
memory implementation or MTG-213 spending is approved. The repair's earlier
“ship it” approval applied to PR419 only.

## Surviving implementation and next proof

`experiments/state_representation_gpu.py` declares the 256-stream candidate and
launches ONLY an excluded admission: four ordinary complete-game updates per
arm, sequentially, seed 74200, CUDA float32. Each learner has 600 seconds,
24 GiB host allowance and 80% CUDA allocator allowance. The rental allows
30 minutes plus 15 minutes setup, at most $0.20, on a compatible RTX 3090 with
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

1. Obtain provider-confirmed final GPU, disk and bandwidth totals covering each
   of the five contracts through its retained `ended_at`. The repaired SDK read
   in `reconciliation-repaired/accounting.json` returns daily totals but no such
   coverage. Stop paid execution until that evidence exists; do not repeat the
   unchanged query or treat an SDK release as settlement. Reconcile against
   `restart/final-accounting.json` and both reconciliation receipts, release
   superseded unspent caps only against authoritative settlement, then quote the
   entire arrangement including setup, idle time, evaluation, artifacts,
   failures and cleanup.
2. Complete both ordinary admission learners. Measure their combined wall time,
   including startup and exports; project four paired seeds at the full horizon
   with 50% extrapolation allowance, 90 minutes evaluation and 15 minutes setup.
   Do not divide sequential time by assumed concurrency. Spare GPU memory does
   not prove compute throughput. No artificial load or automatic eight-context
   retry; MPS was investigated but neither enabled nor validated.
3. Implement the replacement scientific orchestration after admission. Current
   `runs()`/`worker()` execute only admission; recipes, evaluation protocols and
   `plateau()` exist, but no current entry point trains the eight scientific
   learners, exports their curve checkpoints, or executes their evaluations.
   This is substantial remaining implementation, not a launch-command choice.
   Freeze affordable execution and scoring prospectively. Retain independent
   RNG/optimizers, initialization and 32-update curve checkpoints, separate
   final held-out deals, actual game/decision/optimizer exposure and uncertainty.
   Both arms' final three paired-seed gain intervals must each fit within
   ±0.05 (95% Student-t, df3); all 72 curve endpoints must complete. A noisy flat
   point or nonsignificant slope alone is insufficient. Flat poor play does
   not establish competence. No survivor or checkpoint selection.
4. Retain every attempt, supervised deadlines, complete teardown and costs.
   If capacity or budget prevents a credible plateau, report it as unmet.

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

## Attempt denominator and accounting (2026-11-04 UTC)

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

Known L4 bill: $0.07258259388618171. Unsettled full-rental estimates: original
admission $0.1146496; scientific cohort $0.15609933786798355; restart attempts
$0.0187264764, $0.0581526294 and $0.0285087767. Billed-plus-estimated provider
total: $0.4487194142. The $6 ancillary reserve remains pending. Conservative
reservations, including prior unspent caps, total $24.9725825939: this is NOT
money spent or proof that the scientific budget is consumed.

Fresh accounting investigation, 2026-11-04 19:55–19:57 UTC: all five provider
instances remain absent and all five Etude runs are terminal. Released Etude
0.2.14 returns no bills, but the authenticated full-UTC-day `/api/v0/charges`
query returns all five Task rows (pagination exhausted). In table order above,
reported charges are $0.168, $0.164, $0.003, $0.042 and $0.021: $0.398 total,
or $0.47058259388618171 including the settled L4 bill. This is current provider
charge evidence, not the earlier $0.4487194142 lifetime estimate and not final
settlement. Both 5090 charges exceed their earlier lifetime estimates.

The SDK has a reproducible query/coverage defect: `VastAPI.bill` starts its
`day` filter at the instance's intra-day creation timestamp. Actual charge rows
have `start == end == 1793750400` (UTC midnight on November 4), so that filter
misses them. Even supplying the nonempty row through an offline transport returns
`None`, because the coverage guard requires `instance.start_date < row.end`.
The provider's [charge API](https://docs.vast.ai/api-reference/billing/show-charges)
documents a UTC date-range query and per-contract GPU/disk/bandwidth charges;
these daily dates cannot be treated as exact settled-through timestamps.

Required settlement evidence is still missing: no charge finalization marker
or authoritative coverage through teardown was supplied. Reservation release is
$0, leaving $0.0274174061 unreserved; the $0.20 admission cannot be allocated.
The dependency is Etude/provider billing reconciliation with correct daily
semantics, not another approval or an architecture choice. Do not manufacture
coverage or release the $17.20 cancelled-cohort reservation from these partial
records. No new paid work or scientific score ran in this pass. Evidence and
the offline reproduction are published in the `reconciliation/` artifact directory;
prior `restart/final-accounting.json` and all failed receipts remain unchanged.

SDK repair and single follow-up read, 2026-11-04 20:10 UTC: Etude source
`109fa4128052972699ec2a63fba5a3958fac430f` replaces the erroneous
`VastAPI.bill` conversion with `VastAPI.charges`. The query includes the creation
UTC day; typed results carry inclusive dates and observed time, never a fabricated
settled-through timestamp. `VastMachines.bill` returns no settled coverage.
The five Task contracts report $0.168/$0.164/$0.003/$0.042/$0.021, unchanged at
$0.398, and all five remain absent. Including the settled L4 gives
$0.47058259388618171 reported plus settled, not final expenditure. Reservations
remain $24.97258259388618171 and released dollars remain zero. No rental,
scientific implementation/launch or score followed the blocked accounting gate.

The exact missing evidence is a provider-confirmed final GPU, disk and bandwidth
total with coverage through each retained teardown timestamp (17:36:17,
18:29:35, 19:11:26, 19:27:24 and 19:41:03 UTC). UTC dates, instance absence,
pagination exhaustion and repeated identical totals do not establish it. The
provider's documented charge API exposes no finalization watermark. Per the
iteration boundary, execution stops here without polling unchanged charges.
The SDK repair landed as [Etude PR82](https://github.com/loopflowstudio/etude/pull/82)
at `64e9eaabf016f567cb9567280bcf8730f6f5d524`, after local, PR and merge-queue
checks. Its tree matches the repaired source used for the accounting read.
The dependency clone is under this checkout; other Etude checkouts remain
unchanged. The SDK patch and read-only source remain with the reconciliation
evidence. Jack Heart's subsequent source-consumption direction is implemented
above; no PyPI release is needed or claimed. Source tracking cannot supply the
missing provider coverage, and no new scientific attempt was launched.

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
exact lock. No remaining deletion targets.
The failed eight-context launcher (`27a126422`)
and v5 three-arm timing/canary paths (`9dc32125f`) are retired; archived
executions use their pinned source. The native `source_effect_root` and
`shared_effect_root` fixtures still serve transport/model tests.

The sequential admission owns one child and its deadline, stop, reap, log and
exit receipt. `ArmEvaluation.score_for` owns complete-row and protocol/recipe
admission for both analyses; each analysis owns its seed/update schedule and
duplicate detection. Frozen identities and retained receipt bytes stay unchanged.

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
Runtime admission and scientific execution await authoritative settlement.

Prior implementation validation at `bdb567cc0`: 16 focused study/discovery tests
passed; read-only analysis reproduced all eight retained CPU endpoints and
rejected missing, duplicate, incomplete and wrong-identity inputs. This prose
reconciliation does not require repeating learning or those tests.

Check: final locked-source `uv run --isolated --locked --extra dev --extra artifacts pytest tests/training/test_artifact_storage.py tests/training/test_etude_sdk.py tests/training/test_etude_source.py tests/training/test_etude_job.py -q` — 28 passed; release staging/untracked-artifact checks — 2 passed. SDK network-isolated suite — 595 passed, 6 skipped, 3 deselected; mypy/Ruff plus PR and merge-queue CI passed. Source proof retained in `reconciliation-repaired/source-proof.json`; provider final coverage still blocks paid admission.
