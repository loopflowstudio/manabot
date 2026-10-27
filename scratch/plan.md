# MTG-177 — resolution legality and the continuing rules audit

Draft implementation integrated with merged MTG-195 on 2026-10-27.
Final development identity is w15/hash 10 with policy/capture/text v4.
Fresh evidence is recorded and focused checks pass; broader acceptance remains gate-owned.

Integration checkpoint: `b25b90610` incorporates MTG-195 PR386 and main
`37e5ccd8d`; `4f0511233` fixes the remaining validity-column lookup. Prior
combined checks cover eight resolution and seven trigger tests, plus 30 Python
continuation/text cases. The native extension was rebuilt for that snapshot.
Finalization selects w15, refreshes affected receipt paths and reconciles live
claims. The final native wheel is built and installed atomically. No full gate or final admission pass is claimed yet.

Return for review without ready publication, merge or whole-audit completion.
Deathtouch and earthbend remain follow-ups. The approximate 10k authored-churn
budget retains the earlier 9,096-line charge; integration accounting is separate
from inherited MTG-195/main changes and generated evidence.

## Requested stack — 2026-10-27

Jack Heart directed "stack on 195": MTG-177 is to build on MTG-195's
trigger-ordering implementation, retaining the resolution repair and reconciling
the shared continuation/model contracts and final world identity. This supersedes
waiting for an unrelated main-branch delivery. Existing review and no-merge
boundaries remain; MTG-195's unresolved label choice belongs to its conversation.

The retry now reads both Tasks successfully with lf 0.14.18. MTG-195 is
completed: PR386 merged as `e742e8bffe622ddb43009d1586ceec25d44cf342`.
The stack selector found the old PR354 merge intent and rejected reparenting;
no historical delivery was canceled. Since the parent is merged, the supported
integration is `lf sync --manual` onto main (`37e5ccd8d`), which includes PR386.
The child remains based on main and inherits MTG-195 through ancestry.

Merge resolution preserves native trigger-placement frames/action joins together
with resolution-start eligibility. The shipped v3 input/capture contract belongs
to MTG-195; this repair now uses v4, retaining 32-column v2/v3 layouts and adding
eligibility at column 32 only for v4. Row validity remains column 31 for pooling,
frame joins and padding. Embedded decision schema 7 and trigger ordering survive.
Historical capture v3 lacks eligibility; v4 carries the fact. No fixed-order
fallback or second legality authority is introduced. Parent w14 evidence stays
unchanged. The integrated development world is w15/hash 10; fresh evidence and
gate acceptance bind that final identity.

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

Completed finalization cut: replaced all current w14 conformance/demo/navigation/trigger-root
selectors with w15 selectors in CI, native checks, Academy and their tests. Retain
the old receipt files as frozen evidence; they are not a parallel execution path.

Completed projection correction: removed the boolean native encoder `relations`
selector and its single-width assumption. The existing policy-input version
contract selects the layout; the same continuation encoder consumes v4. Frozen
v1/v2/v3 rows and checkpoint omissions survive. No parallel legality computation
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
fail validation. Capture v3 cannot acquire known eligibility retroactively.

Policy-input v4 appends column 32 to `relations[64,33]`: unknown=0,
legal=1, illegal=2. Columns 0–31 preserve v2 semantics, including row validity
at 31. Scalar/vector buffer widths bind the version; reference encoding and the
model use the configured width. V1/v2/v3 checkpoint layouts, initialization order
and serialized omissions remain intact. Capture v4 and text representation
`manabot-text-4` expose “Legal when resolution began”; embedded Observation and
Command schema 7 are unchanged. Final development identity is w15; release certification remains gate-owned.

The native paired witness covers both controllers: hexproof gained during
resolution versus present at entry, with a second legal clause allowing both
effects to suspend. Current observations and v2 tensors are identical; retained
eligibility, v4 tensors, portable text and model values differ. Independent
expected slot values precede encoder parity assertions. Checked generated
captures live in `tests/fixtures/resolution-eligibility-captures-v4-w15.json`; these are
synthetic observation evidence, not executable Academy setups or proof of deck
reachability. Existing witnesses now assert capture/clone/rollback preservation,
nonacting privacy and absent replacement links for departed permanent/graveyard
objects. No target reselection, cancellation, undo action or browser authority
was added. Command/reconnect acceptance remains gate-owned after integration.

## Integration and remaining acceptance

MTG-195 PR386 is merged and included through main; no parent handoff remains.
The final development identity is w15/hash 10. Policy/capture/text v4 extends
the parent's v3 contract without changing embedded decision schema 7. Fresh
`semantic-kernel-w15-v1`, Academy w15 demo/navigation/trigger-root fixtures and
w15 eligibility captures bind the combined runtime. Their w14 and provisional
predecessors remain unchanged. Recording replays the existing bounded inputs,
checks exact offer labels and decision outcomes, and never searches new seeds.

Implementation sanity checks cover fresh native recording, the paired eligibility
witness and a focused Python consumer check. Gate owns broader acceptance once:
full debug Rust/clippy, current-world admission/monitoring, maintained model and
choice-separability, Command/recovery/replay checks, complete choice navigation
and Learn setup suites, and focused browser interactions. Follow the affected
checks in `docs/development/validation.md`; reuse the implementation checks when
still applicable. The retired learnability verifier remains excluded. Historical
worlds, checkpoints, experiments, source pin and failure evidence stay frozen.
No paid compute, strength experiments or current-world release certification is
implied by fixture generation.

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

Authored churn totals 10,104 lines against the approximate 10k guideline
(104 over, about 1%). The historical 9,096 charge is retained; merge/follow-up
integration adds 238 and this finalization adds 770. The six fresh runtime
fixture files contain 10,413 physical lines, with 550 changed lines against their
inherited w14 inputs; unchanged inherited bytes are excluded from authored churn.
Reports, indexes, conformance/observation receipts and design notes are excluded.
The accounting records both measurements in
`docs/rules/audit-2026-10/line-accounting.json`. No earlier charges were refunded.

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

Earlier repair/reduction evidence is historical at `resolution-validation.json`
and `scratch/compress-*.json`; projection evidence is in `scratch/eligibility-*.json`.
Finalization receipts are `scratch/w15-{native-build,native-install,conformance-record,
conformance-handoff,eligibility-native,fixtures-record,python-focused,academy-focused}.json`.
The initial fixture recorder rejected the manifest's expected world change;
`scratch/w15-fixtures.json` retains that failure. The corrected recorder permits
exactly w14 → w15 and asserts equality of every other manifest field, all prefix
labels and final navigation projections. No expected outcome was relaxed.

Check (2026-10-27): release wheel built in 9m38s and installed atomically;
14 conformance cases / 3,420 Commands recorded; paired debug eligibility and public
handoff tests passed (2); focused Python registration/capture/Academy checks passed
(9); all six Academy fixtures replayed; pinned inventory, Ruff, format and diff
checks pass. Full affected suites/browser acceptance remain gate-owned above.
