# MTG-177 — resolution legality and the continuing rules audit

Draft implementation; projection corrected 2026-10-27. The resolution contract is accepted;
final integrated-world identity and acceptance remain pending.

## Requested stack — 2026-10-27

Jack Heart directed "stack on 195": MTG-177 is to build on MTG-195's
trigger-ordering implementation, retaining the resolution repair and reconciling
the shared continuation/model contracts and final world identity. This supersedes
waiting for an unrelated main-branch delivery. Existing review and no-merge
boundaries remain; MTG-195's unresolved label choice belongs to its conversation.

Attempted `lf checkout MTG-177 --stack-on MTG-195 --json`; it returned
`stack parent PR is missing`. Fresh inspection shows MTG-195's implementation is
still uncommitted with no parent PR available to that operation. Its checkout was
left intact. Current lf 0.14.17 also fails Task status/checkout reads with
`missing field interactive`; Session and process inventories remain readable.
No live audit Graph was observed, and no replacement Graph was launched.
The stack has not been established. The next supported action is to establish
MTG-195's parent PR handoff, recover working Task reads, then repeat the requested
stack operation and reconcile the combined source. Do not apply the old stash
from the failed runner message: the current implementation edits are already
present and the cited stash belongs to the earlier recovered audit.

## Accepted direction and boundary

Jack Heart requested `pursue` on 2026-10-26 after reviewing the three audit
candidates. Implement the first priority: one CR 608.2b resolution-legality
authority across spells, activated abilities, triggers and suspended resolution.
This supersedes the earlier audit-only restriction for this repair. Return the
implementation and evidence for review; ready publication, merge and whole-Task
completion remain unselected. The earlier “ship it” delivered PR354 only.

Deathtouch history and earthbend destination identity remain proposed follow-ups.
MTG-195 owns controller-selected trigger ordering; MTG-196 owns affected-player
replacement/prevention ordering. Those mechanisms are outside this repair.
The whole-rules audit remains unfinished.

## Resolution contract and implementation

CR 608.2b evaluates each chosen target against its targeting clause when
resolution begins. Preserve that result while instructions execute or suspend:
gained hexproof, changed control or changed type cannot retrospectively retarget.
Selection, acceptance and resolution entry share `target_is_legal`; legitimate
response-time checks remain. Live object-presence checks are separate (CR 400.7).

`EffectFrame` retains ordered `ResolutionTarget { target, requirement, legal }`
slots. Keep illegal positions because the final-target damage instruction is
positional. All three entry paths reject nonempty all-invalid sets before any
effect; zero optional targets still resolve. Trigger context is untargeted and
cannot replace an illegal chosen slot. PermanentId and graveyard ObjectRef retain
the existing incarnation guards; no stack-target ABI migration is selected.
Clone, undo and canonical serialization carry the slots. Continuation projection
now retains `legal_at_resolution_start` with each reference, clause and ordinal.

The retained hexproof regression is enabled. Independent bounded fixtures cover
both controllers, all entry paths, mixed/all-invalid and multi-clause targets,
suspended choices, characteristic changes, clone/undo and departure/reentry.
The two unrelated ignored proposals retain their CR-required outcomes. Synthetic
fixtures establish obligations, not admitted-deck reachability.

### Delete — do not maintain

Completed: per-handler `target_is_legal` rechecks, separate spell/trigger
all-invalid predicates, frame `target_req_indices`, and the hexproof probe's
ignore attribute. Positional damage, ForEachTarget and continuation consumers
use the surviving frame contract. No remaining production deletion target.

Completed projection correction: removed the boolean native encoder `relations`
selector and its single-width assumption. The existing policy-input version
contract selects the layout; the same continuation encoder consumes v3. Frozen
v1/v2 rows and checkpoint omissions survive. No parallel legality computation
or new lifecycle was introduced.

Removed `resolution.rs::permanent_on_battlefield` and `spell_controller`:
callers now use the exact-object authority and the existing stack controller
accessor after one lookup. Positional targets now use `split_last`;
owned inline effects travel together with their counter key instead of cloning
and branching twice. Presence checks remain between damage events;
all target slots are preserved. These reductions introduce no new serialized state.

Retired one-shot audit authors (`scratch/update_review.py`, `finalize_audit.py`,
`run_audit.py`) remain archived at
`.cache/rules-audit/retired-authoring-20261026.tar`; their charges are not refunded.
`reviews.json` is the authored source; `scripts/audit_rules.py` generates the
index. Exact retirement rationale/hash and prior validation are
in `a9dd41142:scratch/plan.md` and `line-accounting.json`.

## Concept review and projection correction

The 2026-10-26 concept review changed the design: native authority retained
`ResolutionTarget.legal`, but `ClauseTarget` dropped it, making different entry
eligibilities indistinguishable after characteristics converged. Jack Heart's
accepted 2026-10-21 shared-observation direction and 2026-10-26 resolution
contract settle the correction; no new product decision or UI redesign is implied.
The original review remains in `scratch/concept-review.html`.

Implemented 2026-10-27: `ClauseTarget.legal_at_resolution_start: Option<bool>`
projects directly from suspended slots. Stack/announcement targets have no
entry result. Acting-viewer gating, original slot order and exact-object joins
are unchanged. True eligibility cannot restore an expired object reference.
Python bindings and native portable validation retain the fact. Missing historical
fields mean unknown; malformed facts and eligibility on nonresolution frames
fail validation. Capture v2 cannot acquire known eligibility retroactively.

Draft policy-input v3 appends column 32 to `relations[64,33]`: unknown=0,
legal=1, illegal=2. Columns 0–31 preserve v2 semantics, including row validity
at 31. Scalar/vector buffer widths bind the version; reference encoding and the
model use the configured width. V1/v2 checkpoint layouts, initialization order
and serialized omissions remain intact. Draft capture v3 and text representation
`manabot-text-3` expose “Legal when resolution began”; embedded Observation and
Command schema 6 are unchanged. Final version reservation remains integration work.

The native paired witness covers both controllers: hexproof gained during
resolution versus present at entry, with a second legal clause allowing both
effects to suspend. Current observations and v2 tensors are identical; retained
eligibility, v3 tensors, portable text and model values differ. Independent
expected slot values precede encoder parity assertions. Checked generated
captures live in `tests/fixtures/resolution-eligibility-captures.json`; these are
synthetic observation evidence, not executable Academy setups or proof of deck
reachability. Existing witnesses now assert capture/clone/rollback preservation,
nonacting privacy and absent replacement links for departed permanent/graveyard
objects. No target reselection, cancellation, undo action or browser authority
was added. Command/reconnect acceptance remains gate-owned after integration.

## Integration and remaining acceptance

The 2026-10-26 reconciliation confirmed MTG-195 still has uncommitted trigger-order,
action/observation, hashing, rollback and w14 fixture changes in its own checkout.
Its accepted design and local implementation notes now report completed local
validation and availability for review, but no supported commit/PR handoff.
The Task API's planning snapshot is marked stale and still describes design-only
authorization; the dated local notes record Jack Heart's later “implement for sure”
decision. Neither the API snapshot nor reported tests establish integrated acceptance.
Local main remains `2f5b7cbcd`, already merged here; no newer upstream source was
inspected. The adjacent implementation and its historical receipts remain separate.

This isolated draft uses `w15-resolution-draft` and canonical hash 10 because
EffectFrame serialization changed; MTG-195 locally reserved w14/hash 9. This is
not final world assignment or admission. Remaining work is a supported handoff,
integration, final identity selection and fresh conformance/Academy prefixes.

Gate owns broader affected acceptance once: full debug Rust/clippy,
world/admission/monitoring, maintained model/choice-separability, Command/recovery
and consumer/replay checks in `docs/development/validation.md`. This includes
`tests/academy/test_choice_navigation.py`, `test_learn_setup.py`, the model suite
and focused browser interactions against refreshed world-bound positions.
MTG-195's accepted maintained-learnability interpretation agrees with this plan;
the retired verifier remains excluded. Historical worlds, checkpoints,
experiments, source pin and failure evidence remain frozen. Integration is a dependency,
not a claim that acceptance passed.

## Audit findings and historical evidence

The October 26 audit inspected all 150 individual entries in CR 115, 400.7,
603, 608 and 704: 74 partial, 56 missing, 8 outside admission, 2 implemented,
10 unknown. Overall 2,840 of 3,322 entries remain unknown. Unknown means
unreviewed/unproved, not necessarily missing code. The index contains 3,166
numbered rules/subrules and 156 headings. Family coverage and agreement between
shared callers do not establish individual child obligations.

Remaining candidate order after resolution:

1. Consume deathtouch damage history at the shared SBA boundary (CR 704.5h),
   preserving simultaneous checks, new damage, lethal damage, indestructibility
   and undo. Coordinate stabilization with MTG-195. Keep positive first-check and
   negative later-check witnesses.
2. Retain earthbend's first destination incarnation through trigger resolution
   (CR 400.7e, 603.6/603.7c). Keep normal graveyard/exile returns; reject a second
   move or same-zone round trip. Coordinate trigger context with MTG-195. Current
   direct-zone probe does not establish real-card reachability or justify a
   general stack-target ABI migration.

True modal-trigger placement, ability stack lifetime and full characteristic LKI
remain inspected limits; resolution-time color/token/may choices must not move
to placement just because they are named `Modal`. Ten unresolved obligations
and the other rule families still need evidence. See the coverage map and
per-rule reviews for exact remaining obligations and authority references.

PR354 merged as `4a2744705e0f493ce05c0643e5eafe59a5749c08`, delivering shared
targeting/hexproof and the combined w13 payment integration. Prism/Treasure now
passes explicit nested payments; its earlier failures remain historical.
The subsequent audit Graph stopped during sync after merging main as `a52bfc5d3`.
Its complete draft was recovered from retained stash
`2690e564e06d34988f8bb7e62a19c49ed8b7e270` and committed in `6b18ee3d9`;
candidate review is `f16ec7886`, authorization `e35982be4`, repair `a9dd41142`.
Full prior narrative: `a9dd41142:scratch/plan.md`;
inherited MTG-184 questions: `324db65f2:scratch/questions.md`.

## Resource limits and validation

The approximately 10k authored-churn-line ceiling remains; 9,096 are charged, including
94 lines for the inherited reduction and 574 for this correction. Reports/index/receipts/design notes are excluded; runtime demos
and helpers count. Deleted code retains its prior charges. Current
accounting is `docs/rules/audit-2026-10/line-accounting.json`.

Validation uses debug Rust with two Cargo jobs, uv-managed Python and controlled fixtures.
Limits: 900s cold Rust, 180s warm Rust, 120s per Python check. No paid compute,
training experiments, artificial load, random-game searches, assertion weakening,
unrelated card/format expansion or unrequested CI-cost research.

Pinned September 25, 2026 TXT:
`.cache/rules-audit/MagicCompRules-20260925.txt`; metadata/digest in `source.json`.
Rules' objective remains one faithful shared world for human play and training;
agreement between consumers alone cannot prove rule correctness. Its GOAL.md was
read; MEMORY.md is absent, and directory inspection found no child scopes or
child memories in this checkout. No Wave files changed.

Earlier repair and reduction receipts remain historical source snapshots:
`docs/rules/audit-2026-10/resolution-validation.json` and
`scratch/compress-*.json`. The current correction's bounded command outputs and
source digests are in `scratch/eligibility-{native-tests,encoding-tests,python-tests,native-build,native-install}.json`.
The generated capture receipt is consumed by the maintained Python proof.

Check: focused debug resolution/continuation/encoder/capture tests → 18 pass;
focused Python continuation/text tests → 26 pass; release wheel built and cp312
extension installed atomically; Ruff, format, diff and context checks pass.
Full debug/clippy, world/admission/model/training/replay/Academy acceptance and
final version identity remain deferred to integration and gate as above.
