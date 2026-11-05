# MTG-241 — composable rules execution

## Outcome and boundary

Jack Heart's accepted directive (2026-11-04): one PR through the code Graph,
stopping at interactive review; no merge, auto-merge, training or rewriting frozen
evidence. The Task remains incomplete until the remaining implementation acceptance
below is met. No human judgment is missing.

The selected Wave is coverage; its context is read-only. The approved PRINCIPLES /
README guide from PR #425 remains integrated. Published MTG-244 `beaacaa198acddb4162e5081cd570b1ec133ba05`
is integrated at `457a779fa`, preserving its whole-board layer reads and this Task's
prospective entering controller and exact replacement identities. Its live speed
work is independent; no competing writer or Task was launched. Historical plans,
constraints and prior checks remain at `02200f0ed:scratch/mtg-241-composable-rules-execution.md`
and `f1ce0ce13:scratch/mtg-241-composable-rules-execution.md`.

## Surviving authorities

Native `Effect` is the only authored program. `EffectFrame` / `run_frame` own its
execution; `Decision` owns the question. `FrameCompletion` retains exactly one
return destination: caller frame/decision, operation event, spell cleanup, paid
announcement, activated ability, rule continuation or priority. Iteration retains
remaining targets, isolates child results and defers the parent suffix/finalization.
Branches share their scope. `ObjectSelector::Targets` applies a verb to all eligible
slots simultaneously; `ForEachTarget` runs a sequential child program per slot.
Leaves from the Vine uses `Targets` for simultaneous counter placement.

`EventBatch` owns proposals, original ordinals, exact replacement identities
(source incarnation, definition index, namespace), cursor and result binding; it
retains nothing derivable. Recheck applicability after each replacement; the
affected player chooses. No member commits until all are ready. The board cannot
change between proposal and commit, so departing LKI and pre-event trigger sources
are read as the batch commits (MTG-241 swing, 2026-11-05), never stored. Token
creation finishes its own replacements, then its allocated entries take its place
in the same batch for their entry replacements; there is no child batch.
`advance_event_batch` returns the results and the consequences of the committed
event (cards returning as a linked exile ends); `run_frame` binds the results, then
runs the consequences as the instruction's next batch, which never replaces them.
`bind_event_results` owns postambles; `run_frame` schedules and resumes.
`continue_resolution(frame, next)` owns batch-before-decision scheduling. Automatic
and chosen replacements share `apply_replacement`, including identity tracking.

Only departure batches snapshot triggers, and only the indexed sources with a
departure or sacrifice ability. Departures match pre-batch sources; entries match
the finished battlefield. Sacrifice facts match subjects before zone moves; trigger
placement waits. Paid tap/life events match before sacrifices remove sources.
Damage lifelink/counter loss stays in the same batch. Entry counters pass through
replacement before entry, including intrinsic Saga lore. Immediate operations use
this executor and run every consequence with an explicit no-question precondition,
never pending-as-success. A spell's `FrameCompletion::Spell` runs its stack
departure through the ordinary batch and announces resolution once the card has
left the stack; there is no separate resolved state.

Mana activation transfers its caller by ownership. `advance_mana_upfront` commits
automatic mana, tap and life costs, retains output and original payment, then runs
sacrifice. A linked return can suspend on entry replacement without exposing output,
paying again or granting priority. Activation metadata is captured before sacrifice;
card-ability source incarnation binds after discard. Spell cleanup, combat, SBA,
Earthbend, library placement and chosen groups use retained batches. First strike
retains normal damage across SBA; SBA progress is Stable/Changed/Pending. Cleanup
and `turn_tick` honor pending work.

## Hidden selection and capture lifetimes

`SelectCards` serves public results and private hand/library selections.
`ZoneSelection` retains owner, zone and predicate. Both samplers use one exact-object
assignment across active selections, caller aliases and owned `ObjectBinding`s.
Bindings retain incarnation, captured zone/predicate and per-seat exact knowledge
through waiting, queued, detached placement, stack and resolving occurrences. Known
bindings pin exact copies; departed refs never follow a later incarnation; moved
captures take the new zone. Selecting `Remembered` refreshes candidates/selected
aliases even without a zone marker. Viewer projections never expose unknown identities.

Requirements are collected before hidden-zone reassignment; remapping follows
selected/remembered/result aliases. Incompatible count worlds are excluded, and
materialization rejects impossible commitments rather than dropping them. Direct
sampling retries bounded compatible deals and errors on exhaustion. Ordinary roots
avoid an extra full clone. Mixed foretell assignments retain only compatible physical
fills (the two-selected-instant / two-foretold-slot witness retains 14 of 20).
The frozen count-only learned sampler rejects retained selections/foretell before
inference; its saved schema is not silently extended.

The final inspection audit found viewer-independent library repinning. It is now
after joint remapping. Scry, Surveil, LookAndSelect and OrderLibrary preserve a known
viewer's exact inspected prefix, remap unknown inspected identities, and refresh the
current offer in both samplers. LookAndSelect minimum matching requirements participate
in joint assignment. Pending public battlefield entrants remain pinned. Whole-library
search inspection is stable definition/object order, never hidden library order.

`CardsThisWay` retains the latest mill/discard independently of later moves. Optional
discards commit their completed subset together; draws read committed `Moved` results.
Library-to-hand movement reveals and updates known-hand minima. LibraryEnd belongs
to rules types; movement commits explicit placement before child linked returns,
with shuffle a later instruction. OrderLibrary top retains its full inspected prefix;
bottom keeps immediate placement and remaining-only visibility. Surveil retains its
partition until committing the simultaneous move, then orders kept cards. Scry/Surveil
completion events occur only after all choices, including empty-library completion.
Planetarium recognizes both keywords. No registered card prints Surveil above one;
constructed programs prove arbitrary kept ordering, not a fictitious real-card witness.

## Event-model correction implemented — 2026-11-04

Pinned CR 614.16 (`4dc4fead596288be1ca6aa9c913ea1739c680a29`) requires more than the
initial owner of a counter placement. `CounterProvenance` retains original cause
(Effect, Cost, TurnAction) and whether a replacement has produced an effect. Costs
remain costs for payment ownership. `ModifyCounters.effects_only` gates applicability;
rechecking reads updated provenance. Effect constructions are classified explicitly;
precombat lore is TurnAction. The Cost case is represented/tested despite no registered
positive-counter cost today. Intrinsic entry lore/loyalty/defense are replacements,
unlike precombat lore. No card-specific execution branch was added.

`ProposedEvent::CreateTokens` precedes all allocation. It owns final count, registered
definition or immutable copy snapshot/exceptions, controller and entry terms. Once
creation replacements finish, allocate the final group and run the ordinary simultaneous
entry batch. That child returns the created objects; linked-return children still
preserve their parent's results. Copy characteristics remain fixed across suspension.

`EffectFrame.created` now owns the entire exact created set, not the last token.
`CreatedThisWay` selects it completely and sampling rebinds it. The implicit token
`attach_source` flag is deleted. `AttachSourceTo` uses the common ObjectSelector;
Kyoshi Battle Fan creates, selects one created Ally, then attaches. Native forced
skipping handles a singleton; doubled tokens expose both strategic choices. Both
Trebuchet copies retain tapped/attacking state and delayed sacrifice bindings.

Complete first-printing definitions are registered for Caradora (DFT,
`1256d22d-a2a9-41fb-b669-0661ba230bc7`) and Doubling Season (RAV,
`c7e71299-98f6-494e-b187-8d22ce5f50af`), verified against Scryfall. Caradora includes
optional Mount/Vehicle search, reveal/hand/shuffle, and creature-or-Vehicle +1/+1
replacement. Doubling Season includes token multiplication and effect-only counter
doubling. Leaves' real-card simultaneous 3-versus-4 witness now uses both actual cards.
Directed checks also cover declined search/no shuffle, entry lore versus turn lore,
doubled Food, token-copy exceptions, six entry choices and once-only suffix completion.
Learning tokens distinguish replacement causes, predicates and both transformations.

## Retained-state projection and frozen inputs

The published MTG-244 v8 is extended, not replaced. Object/relation/shared widths
remain fixed; effects 128×18, details 4096×3, members 2048×2; 512 local facts by
default. V1–v7 and fresh default v6 stay frozen. Saved omitted configurations retain
their existing meaning. No capacity or legal-action prefix changes.

Native `agent/execution.rs` derives viewer-safe payloads from the retained owners:
remaining instructions, scoped selected/moved/this-way/created results, iteration
progress, completion events, paid/rule callers, logical creation, proposals, causal
counter provenance, applied replacements, LKI/trigger snapshots and child results.
Every Decision variant is matched; private references go through native visibility.
No raw Game or storage-ID serialization enters inputs. Copy definitions retain their
programs. A final audit added departed source/trigger/target LKI: SourcePower can
read a captured value after the source leaves, so a definition reference alone
was insufficient. The sixth same-offer pair proves that captured power reaches
v8; existing catalog-only admissions retain their frozen meanings. Dynamic rules tokens now retain definition-reference values rather than
silently dropping token-name joins.

Fact kind 14 carries execution as a separate ordered occurrence from a captured
source program (ordinal reset); relation kind 10 links one-based visible reference
slots. An explicit controller-holder relation lets focus-free actions read their
fact-bearing player row. The paired completion-event test found that missing join:
critic values changed but action logits did not. The corrected path distinguishes
all five pairs through rows/typed_pool/typed_gru using fixed nonzero fact readout
weights; fresh fact weights intentionally start at zero. This is representability
and gradient evidence, not training or learning efficacy.

Native/Python encoding use the same ordered occurrences and links. LLM text includes
complete retained details and visible slot labels. Academy's `DecisionContext.pending`
uses plain-language summaries from the same native projection; it does not invent
choices or automate them. Historical omission stays valid. Cross-viewer validators
reject private pending summaries. Current and replay/Study/advice schemas embed this
optional field; UI exposes it under Pending resolution.

`SelectionOrigin` preserves former codes 5/6/7 on v1–v7 and their previously unset
mode/decline columns. Only v8 writes shared code 28. New codes 27/29, new selection
domains and novel retained completion/batch state reject earlier inputs before writes
with UnsupportedExecutionInput. Ordinary migrated searches are not blanket-rejected. A real Fabled Passage probe
found that its implicit LibrarySearched completion still rejected v6 despite
retaining code 6. Admission now recognizes completion events already implied by
the matching library-search/Scry/Surveil obligation and rules player. Different
events retained under the same Modal offers still require v8. Native witnesses
cover all eight versions; the real fetch search checks native/Python parity.
The per-version native witness covers frozen codes/header columns. Typed Capacity
errors remain distinct from fatal version admission. #426 recovery discards overflowed
games, so default overflow is a sampling bias, not neutral recovery.

Six native paired captures (same present offers) cover selected object, remaining
iteration, completion event, counter cause, token count and departed-source power. Their other-viewer frames
match. Native/reference arrays, text, actual model forward/backward and typed overflow
are checked. The six-copy fixture originally needed 660 facts and overflowed 4/7 pauses.
Ordered compact replacement records and omission of default entry/empty batch fields
reduce the same peak to 489, with 0/7 overflow; relation peak stays 12. No payload,
choice or capacity was removed. This small corpus does not prove arbitrary depth fits.

## Delete — do not maintain

Completed: local non-suspending iteration; independent parent/finalize state;
cloned mana callers; FrameFinalize; OrderLibraryBottom; MillAndSelect,
ReturnMilledThisWay, MilledBonus, SelectMilled; optional-discard/draw and basic/tutor
search variants and dedicated helpers; synchronous linked return/sacrifice/destruction
wrappers and result postambles; duplicate run_batch_before_decision scheduling;
bare remembered-ref transfers and frame-only binding discovery/refresh shortcuts;
eager token allocation; last-created-only storage; implicit token attachment.
Their required behavior survives in the authorities above. No adapter/flag/second
executor remains. Original completed-capture failure evidence stays at
`f1ce0ce13:scratch/completed-private-capture.patch`.

No further predecessor path is selected for deletion. Surviving projection now
dispatches completion once, shares target-reference formatting and renders each
effect description once. Ordinary visibility precedes retained-entry discovery;
damage/counter replacement scans share an immutable layer-board read while
preserving candidate order and the debug oracle. These remove repeated work
without changing stored state or input semantics. Inherited sibling notes are
consolidated in [MTG-244 context](mtg-244-layers.md), with exact historical sources.

Check: `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test --manifest-path managym/Cargo.toml -j 2 --lib resolution_tests` — 50 passed; all 12 regenerated paired captures/digests/arrays unchanged. Rebuilt/atomically installed extension: `uv run --extra dev pytest tests/verify/test_retained_execution_input.py tests/semantic/test_text_observation.py -q` — 64 passed. Formatting/diff checks pass; final suites and performance acceptance remain gate-owned.

## Implementation closeout and remaining acceptance

The bounded reference / published-layers / combined comparison and mitigations are
recorded in [risk validation](mtg-241-risk-validation.md). Counts/full digests agree;
the net reference cost remains above the standing budget in these diagnostics.
This is an explicit review/gate risk, not an accepted cost or a reference reset.

The extension was rebuilt and atomically installed after the final LKI/admission
repairs; current receipts and focused consumer checks are refreshed. Current w17 authority, parity, belief and Academy fixtures are refreshable;
frozen experiment evidence and prior world corpora remain unchanged. The native
catalog now has 340 supported cards, verified from the runtime.

Gate owns final-head affected suites, full debug Rust/Clippy, schema consistency,
current-world arena admission and the unchanged 10% Engine Budget. Demo/review owns
judgment on the residual net cost and general nested-capacity/learning risks. No
training, merge, publication or continuation past interactive review is selected.

Check: focused debug rules/projection/hidden-selection and Python model/text/
reconnect/trigger-order checks pass; Svelte check and pending-summary render pass.
Current-world producers have completed: 14 semantic replay cases / 3,556 Commands;
authored parity 135 Commands / 816 viewer checks; public commitments 66 / zero gaps.
The semantic producer compares explicit-step and skip-trivial executors but shares
rules primitives. Four Elemental tapes change Command counts (80→83, 196→209,
212→282, 149→146); they are not part of the unchanged-work benchmark claim.
Frozen prior corpora remain untouched. Gate owns final affected conformance checks.

## Integration evidence and limits

MTG-242 owns payment resources/search, MTG-243 naming, MTG-244 layers. Their latest
independent work is not selected here. Published #427 supplied the concrete v8
contract; its later speed work remains live. Its first CI report at `1930146b0`
matched counts/digests but regressed search 168.192→453.756 s, worst reply
4.909→15.349 s, and throughput 12,817→5,654.8 steps/s. `beaacaa19` repairs that
source but has no clean timing acceptance in this record. Preserve this contrary
evidence; compare net cost with main/reference, not only already-modified cuts.
No reference reset, speedup or accepted cost is implied. Jack Heart's raised speed
bar on the siblings is useful context, not a change to this Task's authorization.

The migrated fetch-land heuristic is repaired at `02200f0ed`; both Fabled Passage
and Prismatic Vista keep their trigger. It is the only Python consumer of retired
Effect names. Inherited LLM field-coverage omissions (colors, keyword timestamps,
program/description) are repaired in the test inventory; text already renders them.
Trigger-order-w17 is refreshable after final rules. Three non-CI sim failures also
reproduce on `main` (`fe83a6b06`): random invalid fact validity, seed-197 checkpoint
without world binding, and old conditional-search digest. They are not repaired
here or confused with this branch's further digest change.

Rules evidence and historical detailed checks are retained in the committed plan
references above, including Guilds release notes for completion, Aetherdrift /
Foundations notes for 3/4 ordering and costs, and the April 2026 Saga-entry bulletin.
The approved research milestone's later costs/history/object-lifetime cohorts do
not extend this Task into another content rollout, executor or experiment campaign.
