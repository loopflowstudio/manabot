# MTG-245 — implementation handoff

## Accepted direction

Jack Heart selected A on 2026-11-04 (“i agree A is the place to start”): authored
functional reference decks, frozen independent per-seat tickets and stable
anchors. B generation and C adaptive search remain alternatives. Selecting A did
not waive actual training-exposure capture. Exact lists, 50/25/25 shares, seeds
and numeric bounds are proposed settings for Task 04's authored freeze.

Jack Heart accepted the four-core-Wave/workflow work inside this Task (“nah i
think jsut leave that in this”), then resolved consolidation: “training is the
main one and the evals tasks get moved into it”. The original Training identity
54bc9162-c883-4574-bab1-af589526c1ad survives as Experiments. Academy stays separate.
The latest reconciliation steer e6c3e03c-65a1-4b6a-8ec2-abed2aa9964f covers the
concurrently recreated Training Wave. No scientific campaigns, rentals, spending
or shipping were authorized here. Existing Task Graph/review boundaries remain.

## Implemented in this continuation

Shared native CardCapture optionally records complete nontrivial positional
choices using the observation owner's action focus, plus completed effect frames
at native finalization. Card capture types moved from sim/card_evidence into
env/card_capture, preserving historical serialization and one owner. No policy
state, input widths, native legality, RNG or rules behavior changed. Search forks
still drop instrumentation. Forced decisions remain skipped in actual runners.
Final inspection found a shared capture counterexample: a face change could make
its recorded name/definition violate the reader's physical-copy invariant. Native
snapshots now retain identity from setup/creation while zone/incarnation remain
current. A controlled face-switch fixture covers this repair; rules and policy
observations are unchanged.

TrainingJob explicitly enables capture for the reference league; omitted/default
jobs retain their configuration. CapturedVector saves terminal captures before
native auto-reset changes their seed/setup, and records interrupted current
prefixes on normal close. On a native failure the existing private failure owner
retains the original error; invalid partial capture cannot replace it. Unobserved
failed prefixes remain missing. The private digest-named JSONL receipts retain
sampled decks, seed/actions, exact copy/incarnation, runtime/input identity and
run/update context. TrainingRun registers these as exposure artifacts, never
checkpoints. Replay inputs use the existing failure-replay owner; these receipts
are **not replay-verified**.

The offline report accepts downloaded exposure directories with repeatable
`--training-evidence`. It validates digest, run/seed/world/source/decks and reports
terminal versus interrupted games, per-stream budgets, both-seat decision
denominators, opening/draw
copies, action-type opportunity/choice counts, per-definition focus counts and
completed frames. Observed zero is distinct from absent receipts. Missing capture
makes `complete` false independently of the primary `modest_check` criterion.
Resolution counts describe completed effect frames by the physical source copy's
initial definition; they do not assert mechanic-specific success, dynamic copied
program identity, strategic benefit, or contribution to an optimizer update.

Training → Experiments, Eval's two open Task moves, and Eval retirement used lf.
History/memory provenance is in docs/workflows/2026-11-04-migration.json. The
concurrent fc6ffb Task created a new generic Training Wave; inspection confirmed
its experimental outcome and separate $25 authority. It moved to Experiments
without altering its live Graph or comments, then its empty Wave was retired.
Final shared reads show the four core Waves plus Academy; abandoned Study remains
historical. The migration receipt records preservation of PR #421's then-published
head and uncommitted Training-memory bytes. Upstream `fe83a6b06`, integrated through
`e5c3dd947`, now supplies the Eval archive reference and TLA recovery lessons in
Experiments' memory, retaining the shared-effect evidence and reservations.
Later PR #421 contributions still need integration review; this pass did not
audit its separate checkout. Provider synchronization was pending in the receipt;
no auth repair or moved-backlog execution occurred. Shared graph defaults await delivery;
main's old `iterate` selection is not a replacement of existing captured graphs.

## Remaining ownership and limits

- The concept-review binding cut is implemented. `ProtocolBinding` selects the
  importable entry module and protocol path/digest; Etude's file stem supplies
  the experiment name. Thin entries validate their discovery path and delegate
  to `growing_pool_runtime.py`. Declarations freeze the selected entry/manifest,
  callbacks validate experiment and protocol conditions, and offline analysis
  opens the selected store. `--experiment` replaces path-only `--protocol`.
  No registry or copied scientific pipeline was added. Controlled two-binding
  fixtures exercise actual Etude discovery/job readers, imported callbacks and
  sealed offline stores, including crossed identities and a misnamed entry.
  These fixtures are not runtime admission or scientific results.
  Regime compilation is separate from runtime admission: offline health and
  exposure reconstruct frozen settings, while job creation, evaluation and CLI
  admission still reject an incompatible engine. Analysis shares one parsed
  declaration and delegates checkpoint ownership checks to the schedule builder.

- Reconciled 2026-11-04: the frozen proxy is not currently admitted. Its engine
  source predates training-exposure capture; the same world label and valid decks
  cannot establish compatibility. The protocol documentation now states this at
  its entry point and commands. V1 JSON remains unchanged. Task 04 must freeze the
  integrated prospective protocol, actual Adventure/prepare lists and shared
  probe IDs, numeric settings and execution authority. Neither synthetic fixtures
  nor the proxy lists establish learning, throughput, real exposure or playable
  coverage.
- Gate owns the affected Rust/Python suites and full automated acceptance checks;
  focused checks cover the directed debug fixtures and Python capture/report path. Demo/review remains the next product boundary.
- Task 08 owns operational Validation B. The deeper strength/generalization track
  begins separately after full FDN; its execution Task is still unassigned.
- Tasks 01/03 remain owners of broader shared history/crash receipts. This change
  extends their existing native capture boundary, adds no mechanics taxonomy and
  launches no competing Task. MTG-243's observation/action version stays intact.
- MTG-210/211/230/231 retain memberships pending an outcome-based mapping; moving
  Eval and resolving the recreated Training identity does not select that backlog.
- PR #421's empirical reservations and fc6ffb/MTG-213's separate budgets confer no
  scientific execution authority on MTG-245.

The 2026-11-04 shared Wave read still has the four accepted core identities plus
Academy, with Study abandoned. No replacement Training or Eval Wave is active.
The migration receipts preserve memory provenance. Reconciled 2026-11-04:
Experiments' memory now references `19bdee944^:.lf/wave/eval/MEMORY.md`, the same
archived blob named in `docs/workflows.md`; the previously planned link repair is
complete upstream. The renamed goal files retain Training/Engine headings and
their earlier chapter scopes. Proposed follow-up within an owning Wave Graph:
incorporate applicable archived evaluation lessons and reconcile headings/current
responsibility with Jack Heart's accepted
2026-11-04 boundaries without overwriting historical chapter scope or PR #421's
contributions. This unbound Task selects no Wave for memory curation; those files
remain unchanged. Relevant Experiments and Coverage sections were read, including
the new capacity-recovery lessons; unrelated sections and Wave memories were not
audited. MTG-213's opt-in recovery and first-64 support do not change this protocol:
its compiler still supplies no game-error recovery and requires strict overflow.

## Assumptions and deletion cut

For the first reference-league handoff, all surfaced flat-policy seats are
captured. Opportunity means a card is in the native action's focus, counted once
per definition/chooser/decision, not a claim that using it is strategically good.
Resolution means the native effect-frame program completed, not that each effect
changed state. Retaining all choices trades artifact size for auditable bounded
pilot evidence; performance/cost remain for the authorized pilot to measure.
Downloaded job-output directories are explicit analysis inputs so the instrument
does not invent a second Etude checkpoint/artifact store or assume bucket layout.

Delete — do not maintain (completed; no remaining targets): implicit runtime
admission inside the offline regime compiler; duplicate analysis declaration
parsing/export checks; implicit v1 defaults
in shared execution/analysis, the path-only CLI selector and monolithic entry
point; the constant “training exposure unavailable; acceptance
unresolved” report field; the old sim-owned native capture model definitions;
the retired Eval graph config. Existing historical capture bytes and evaluation
readers retain their meanings. The current review replaces the prior page at
scratch/concept-review.html; its earlier acceptance questions were resolved by the
named 2026-11-04 steers. Earlier exploration detail is at
190b0c603d70e790f2217c61c970c38f82544d08:scratch/explore-design.md, and earlier
protocol/workflow implementation context at a72b5ef1a:scratch/questions.md.

Unchanged native capture/report and workflow checks retain their prior passing
evidence at `e5c3dd947:scratch/questions.md`. The post-sync inspection also passed
10 focused checks covering capture, strict compilation and incompatible-input
rejection; capacity recovery remains opt-in. Frozen manifests and migration
receipts are unchanged. No complete game or scientific campaign was run.

Prior implementation validation: 52 protocol/analysis tests and Ruff passed at
`bfac9958e:scratch/questions.md`. No executable code changed in this reconciliation;
Gate owns the full affected suites and acceptance, and demo/review remains required.

Check: `git diff --check` — passed; both Eval archive references resolve to the same blob; `lf context` remains within both limits.

## Sync provenance — 2026-11-04

The merge from main conflicted only in this shared scratch filename. Retain
MTG-245's implementation handoff here; main's MTG-213 pilot decisions remain at
`ee85d6fc82ed631d7b460e68c73e7ac27f0458ff:scratch/questions.md` and its
results at `experiments/tla-trophy-pilot.md`. Those separate pilot decisions and
execution budgets do not extend MTG-245's authority. Incoming code and protocol
changes merge unchanged; this resolution changes only task-local notes.

## Publication verification — 2026-11-04

`lf sync` completed cleanly, preserving the existing reconciliation edits. The
publication review inspected the committed range against `origin/main` plus the
previously uncommitted work. No implementation changes were needed in this step.

- All seven repository graphs compile and their skill, loop, delivery and
  experiment boundaries pass `scripts/check_loopflow_workflows.py`.
- Rust formatting, all-target/all-feature Clippy with warnings denied, and the
  complete workspace debug suite pass (`CARGO_PROFILE_DEV_OPT_LEVEL=1`, `-j 2`).
  Clippy needed the absolute pinned interpreter path; the documented relative
  path did not resolve from its build subprocess.
- 185 Python checks pass across growing-pool protocol/analysis/exposure,
  card evaluation, experiment discovery, regimes, CI/release selection and
  package boundaries. The exact two-seat/evaluation CI invocation collects
  all 383 tests without module-name collisions.
- The deck/native-buffer/recovery regression selection initially had 96 passes
  and four failures because the local extension predated main's recovery
  argument. Rebuilt the release wheel from the repository root and atomically
  installed its cp312 extension. All four failures and all four exposure checks
  then pass (8 checks). No source change or weakened assertion was needed.
- Ruff lint/format and `git diff --check` pass. A fresh shared Wave read retains
  the four core identities plus Academy, with Study abandoned.

These are automated software fixtures and regression suites, including the
repository's bounded native-game and recovery fixtures. No scientific campaign,
pilot, allocation or spending was performed. No learning, throughput or replay-
verification claim follows from them. The frozen v1 proxy still rejects the
changed engine source; Task 04 must author and freeze its integrated prospective
binding, lists/probes and settings before authorized execution. Engine-budget CI,
demo, delivery and post-merge verification retain their own boundaries. Publish
for review only; do not merge, arm auto-merge or complete MTG-245.

## Gate integration and validation plan — 2026-11-05

Syncing main `41981b3abb326e656d06d3b6aacea9a2d73f79ed` preserves upstream Wave IDs and Linear bindings while
selecting the accepted `coverage` and `experiment` graphs (retiring `workflow:
iterate`). Preserve upstream's removal of docs/README.md and put this branch's
workflow guide link in the root guide map. Follow the moved Standard milestone
in Coverage's research directory. These are mechanical integration choices.
MTG-213's separate assumptions remain at `41981b3abb326e656d06d3b6aacea9a2d73f79ed:scratch/questions.md` and
its durable pilot report; this file continues to own MTG-245 only. The shipping
direction in explore-design.md supersedes the historical publish-only boundary.

Comparison: Task base `fe83a6b0` before sync; main `41981b3abb326e656d06d3b6aacea9a2d73f79ed` after integration.
The pre-sync published head `a1927892d` passed all CI checks, including debug
Rust, both Python integrations, package/install, Academy and Engine Budget.
No implementation defect was found in the capture/reporting review.

Focused plan (one runner at a time):
- experiments/growing_pool*.py and protocols, capture/failures, TrainingJob and
  execution, card-evidence imports: protocol/analysis/exposure, card evaluation,
  experiment discovery, regimes, package-boundary and failure/recovery tests.
- Native capture, env/vector, frame finalization and bindings: directed debug
  card_capture and vector_env tests; retain the unchanged full debug and engine
  budget CI results. Incoming native test changes only relocate existing data.
- .lf graphs/skills/configs and docs/workflows.md: real lf discovery/compilation
  and boundary validator. No Wave migration or other Task execution is repeated.
- CI integration path addition: exact two-seat/evaluation collection plus CI
  selection/release tests. The changed-aware selector is .github/ci/select-tests.sh;
  it selects all jobs for native source and has no local result cache. Keep the
  local plan focused and leave the wider matrix to CI on the integrated revision.
- Ruff lint/format for changed Python, cargo fmt and all-target/all-feature Clippy,
  git diff --check, and local links in branch documentation. No UI changes.
- The four CLI demo cases already passed; reuse that dated evidence rather than
  claiming it ran against the new source. Focused analysis tests construct fresh
  source-bound fixtures after sync. No measured learning/performance claim.
