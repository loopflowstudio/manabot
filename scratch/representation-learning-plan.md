# V7 learning comparison — completed, awaiting results review

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
