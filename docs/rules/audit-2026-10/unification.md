# Authority adjudication — concrete draft

Historical adjudication for the increment delivered in PR354. World labels,
ownership reads and validation below describe those checkpoints. The
[October 26 coverage map](coverage-map.md) supersedes current-scope claims,
including the newly distinguished intra-resolution target-recheck failure.

This increment starts at `ededb325`, preserving the original inventory and
`2b6e2b1b7` draft. It inspects the accepted boundaries, rather than rerunning
the whole inventory. Rules are the pinned September 25, 2026 text and digest
in [source.json](source.json). New tests use local synthetic fixtures, not new
cards or frozen content.

## Confirmed overlap: stack-target mana-value restrictions

**Definitions:** `triggers.rs::stack_spell_targets` independently compared a
spell's mana value with the minimum while `resolution.rs::target_is_legal`
implemented the same restriction for `TargetSpec::SpellOrPermanent`.
`legal_targets_for_spec` called the former for stack spells and the latter for
permanents. Trigger acceptance and resolution used the latter again.

**MTG verdict:** both implementations correctly implement this restricted
printed-cost subset. CR 202.3 defines the values; CR 601.2c requires chosen
targets to satisfy restrictions; CR 608.2b rechecks them. Lightning Bolt (1),
Gray Ogre (3) and Mountain (0) distinguish unrestricted, below-minimum, equal,
and above-minimum offers. The exact expected lists are written independently
of either implementation in `stack_target_mana_value_boundaries_have_explicit_expected_choices`.
A separate witness rejects Bolt and counters Ogre for minimum 2; another
removes the only target before resolution and proves that even an untargeted
life-gain instruction must not execute.

**Chosen authority and removed logic:** `target_is_legal` now owns the stack
restriction for offers too. Enumeration retains reverse stack order and sends
all candidates through that predicate once. The independent threshold comparison,
`stack_spell_targets` helper and separate per-kind filters are removed. Player
and graveyard witnesses retain controller restrictions, predicates and ordering;
mixed targets retain their player/spell prefix before battlefield candidates.
Spell and ability choices inherit
this through existing `legal_targets_for_spec` callers. Per-requirement duplicate
exclusion, casting-source exclusion, minimum/maximum target counts, and current
state rechecks remain stage-specific responsibilities. No legal action, order,
serialization, Command binding, viewer projection or replay identity is intended
to change; no world version is changed. This is a bounded consolidation, not a
claim that the entire predicate is MTG-correct.

## Confirmed inconsistent rule: hexproof across target specifications

**Original definitions:** in `resolution.rs::target_is_legal`, creature/permanent branches
called `hexproof_blocks`, but the permanent branch of `SpellOrPermanent` checked
only battlefield presence and mana value. A 3-mana opposing Gray Ogre with
temporary hexproof is rejected as `Creature` yet offered and accepted as
`SpellOrPermanent { min_mana_value: 3 }`.

**MTG verdict:** CR 702.11b prohibits both targets. The creature/permanent
branches are faithful on this fact; target wording does not bypass hexproof.
`audit_spell_or_permanent_must_respect_opponents_hexproof` records the independent
expected exclusion and was explicitly run to failure. This is a real competing
definition inside the shared predicate, not offer/resolution disagreement.

**Applied correction:** `target_is_legal` checks battlefield presence and
opposing hexproof before matching any permanent target specification. The four
branch-local hexproof guards and redundant battlefield checks are removed;
controller, type, predicate and mana-value restrictions remain branch-specific.
Matching target kind first gives the permanent branches one validated object;
the single-use `hexproof_blocks` helper and repeated object/presence lookups are
removed. Nontargeted effects retain their own battlefield-presence helpers.
No offer-time legality cache is introduced. The original distinguishing witness
is enabled. New bounded witnesses cover both controllers, own targets, predicate
and threshold rejection, noncreature lands, response-time hexproof, mixed/all
invalid targets, and leaving/reentering the same physical card.

The live runtime is w12 with a fresh `semantic-kernel-w12-v1` corpus; w11 and
older receipts, authored content and `erd1` evidence remain frozen. Tensor shapes
and learning schema are unchanged. See `scratch/plan.md` for current validation
and limitations; historical receipts below describe the earlier consolidation.

## Trigger collection/order/placement: missing choice, no duplicate found

Inspected `process_game_events`, `snapshot_trigger_sources`,
`check_triggers_for_event`, `pop_next_pending_trigger`, `flush_triggers`,
`place_triggered_ability_on_stack` and `stabilize_before_priority`. Collection
matches committed facts and source snapshots; placement selects targets and
pushes stack objects. Ordering has one owner: `(APNAP rank, enqueue_order)`.
Target selection is a separate choice, not a second trigger-order authority.

CR 603.3b requires each controller's ordering choice (and separately treats
triggers whose condition is another ability triggering). At 2 life, simultaneous
“gain 2 life” and “deal 2 damage to this controller” have two legal orders:
last-stacked gain resolves first and the player survives at 2; last-stacked
damage resolves first and the player loses at 0 before gain can resolve
(CR 405.2, 117.5, 704.5a). A passing tiny witness independently verifies both
outcomes by varying the injected enqueue order. The ignored
`audit_same_controller_triggers_require_an_order_choice` fails because flush
stacks both without returning an order decision. Existing
`cr_603_3b_pending_triggers_flush_in_apnap_order` separately covers both players'
APNAP order and target-choice ownership.

Do not consolidate collection with placement or implement a browser ordering
workaround. A future shared decision must retain the CR 603.3b two-part process,
complete controller permutations, targeted-trigger sequencing, actual suspended
choices on reconnect, Command binding/stale rejection, viewer safety and exact
replay/fork restoration. This draft adds evidence, not that new decision ABI.
The refreshed Task list contained no Task title specifically for trigger order;
that is not proof of absent ownership in every external system.

## Targets and zone identity: existing authority, legitimate rechecks

Offers, spell/activation acceptance via legal-target lists, direct trigger
acceptance, and resolution converge on `target_is_legal`. The confirmed
stack-value duplication above was the exception inside enumeration.
`identity.rs` owns `current_object_ref` and exact lookup; committed movement
advances incarnations in `zones.rs`. Battlefield targets retain their departing
PermanentId (removed slots are not rebound); graveyard targets carry ObjectRef.
Existing `cr_400_7_manowar_target_does_not_follow_reentered_card`,
`cr_608_2b_trigger_with_illegal_target_does_nothing`, and the Sokka one-illegal-
target / all-illegal-target cases in `elemental_cards.rs` already cover the
requested distinctions. We did not add copies of those tests. The new only-
target witness adds the otherwise-untargeted effect distinction.

The follow-up inspection traces `Target::StackSpell(CardId)` through
`push_spell_to_stack`, `commit_zone_move`, `find_spell_on_stack_index` and
resolution. Zone movement advances ObjectRef incarnations, but the target
retains only CardId, and lookup finds any current spell with that card. This
representation loses the old/new distinction while the recast spell is present.
That observation alone does not establish a reachable supported-play defect:
stack insertion is at the top and resolution is LIFO. The new directed
`recast_spell_resolves_before_the_old_targeted_ability` witness bounces and
recasts a targeted Ogre, resolves the new Ogre first, then proves the old
countering ability neither removes that permanent nor grants its untargeted
life gain. General non-LIFO insertion, copy/retargeting effects and all CR 400.7
exceptions remain uncertified. An exact-object migration should be decided
with a reachable counterexample and its semantic serialization/receipt impact,
not by adding a second target identity field beside the existing authority.

Five additional individual entries now carry inspected obligations rather than
family triage: **115.1c, 115.1d, 115.2, 115.3 and 115.6**. The bounded evidence
covers activated/triggered target timing, explicit target zones, clause-local
distinctness and optional zero-target selection. Existing Azula and Dai Li
tests distinguish repeated targets across clauses; Jet excludes duplicates
within one clause. The new direct-submission witness also rejects the duplicate
without discarding progress, then accepts reuse in the next clause. These are
stage-specific responsibilities sharing target metadata, not contradictory
authorities. Tests cited in `reviews.json` were read; only the commands in the
validation receipt were executed in this increment.

Two authoring limits remain explicit. Activated selection currently supports
one target and always excludes the source permanent; this fits Disciple's
“another” wording but is not CR 115.5's general self-target rule. Activated
resolution also lacks the spell/trigger whole-object all-invalid gate, although
current targeted effects recheck locally. Before admitting compound activated
effects with independent untargeted instructions, prove that an invalid sole
target suppresses the entire ability. No current-card failure or additional
world change is claimed from these observations.

## SBA, replacements and committed movement: duplication not established

`stabilization.rs` owns the pre-priority fixed point. `sba.rs` selects a batch
from one pre-batch state/source snapshot, distinguishes zero toughness from
destruction, and sends exact objects to `destroy_object` or
`move_object_to_zone`. `proposed_event.rs` validates proposals, transforms them,
and calls `commit_zone_move`; ordinary `move_card_with_entry` also enters this
pipeline. `zones.rs` captures departure/LKI, commits zone membership and emits
facts. The stack-departure flashback transform and object-local death-exile
transform apply different replacements at different stages, not competing
versions of the same replacement. No second direct rules movement authority
was established in these paths. Low-level Zones methods remain storage and
scenario setup primitives, not replacement decisions.

Existing `cr_704_sba.rs` covers the shared-source simultaneous death batch;
`cr_614_replacement.rs` covers entry replacement before observation and
noncommuting prevention/doubling outcomes. Its deterministic CR 616 ordering
is explicitly narrower than the affected player's choice required by MTG.
That is another missing-choice boundary, not evidence of two competing
replacement engines. No general replacement-choice redesign is included.

## Combat and static-query follow-through

`eligible_attackers` and `declare_attacker` both use `Permanent::can_attack`
plus `combat_condition_met`; blocker offers and acceptance use the same
`blocker_can_block_attacker`. `Game::effective_keywords` delegates to
`Permanent::effective_keywords`, rather than defining keywords again.
SBA and combat query `Game::effective_pt`. `Permanent::local_power` and
`local_toughness` have no source callers and explicitly omit whole-game effects;
they are not evidence of two active characteristic authorities. Additive
layer-7c terms commute in the supported subset. No consolidation is forced;
timestamps, dependencies and broader combat completeness remain unproved.

## Refreshed ownership and delivery boundary

Read-only `lf task status MTG-174 --json`, `lf task status 4ba5dd --json`
and `lf task list --json` were refreshed October 22 UTC. Raw snapshots remain
in scratch. MTG-174 resolves to `task_f721516493004baa9b5a9025b1ef81c8`, the
Elemental $10 training Task, owned by Jack Heart / Intelligence, state started,
execution idle after its Flow. It is **not** the shared-payment Task described
by the original audit. That historical ownership reference is stale; this
increment does not guess a replacement Task or change it.

`4ba5dd` remains Rules' composable-payment Task, owned by Jack Heart, state
started. Its recorded stopped landing Flow reports no auto-merge request,
while its PR #344 also records merge commit
`43d807036ed86e1899ca6ffc1bc2e54fe559dfdd` (already in this checkout's history).
The recorded lifecycle is not completion authority and was not repaired here.
No live Flow was restarted and no payment code was modified. Locating the
current shared-payment owner is still required before any future payment work.

Exact checks, failed probes, toolchain failure and incremental cost are in
[unification-validation.json](unification-validation.json). No paid compute,
required-check weakening or training-recipe/deck/experiment change is included.
Jack Heart approved shipping the bounded increment on 2026-10-22, superseding
the older draft-only holds. Gate does not publish or land; the parent conversation
owns delivery. MTG-177 stays open for the unfinished substantive audit.

## Integrated runtime validation

The full [integrated w12 diagnostic](learnability-w12-integrated.json) passes
in 43.5s after the upstream collector integration, with all capacity seeds and
reward-driven production PPO/checkpoint reload. PPO collected 65,536 learner
transitions; after/no-update scores are 1.0/0.640625. Fresh dependency-blocked
play/Study protocol passes 17 tests. Semantic retry, all seven Study branch
cases and all 20 monitor cases now pass across bounded invocations. The branch
fixture uses returned frames and recorded priority automation; complete-match
creation and assertions are unchanged. Two additional native witnesses bring
the debug targeting result to 10 passed, with the separate known trigger-order
proposal still ignored.

The original collector timeout, the 99.6s historical pass, and the new branch
fixture timeout remain in the receipts. A stack sample placed the branch
fixture delay in attempt persistence; that does not establish the cause of
every historical timeout. No production runtime, training budget, timeout or
assertion changed in the `bda9b0aeb` acceptance follow-up. Its conformance, arena
and other consumer receipts were reused where production source and fixtures
were unchanged at that checkpoint.

The enumeration cleanup now has fresh affected validation: 11 targeting and
300 rules cases in debug, 6 w12 conformance checks, 11 Command/recovery and
decision-surface cases, 21 dependency-blocked play/Study protocol/runtime cases,
all 7 Study branch cases, all 11 arena cases and all 20 monitor cases. The
extension was rebuilt and installed atomically; existing w12 corpus/identity
and pre-w12 frozen evidence remain unchanged. The earlier full acceptance and
failures remain historical records. See the gate entry in
[validation receipts](unification-validation.json) for the tested source/native
digests, remaining scope and measured cost.

The fresh [gate diagnostic](learnability-w12-gate.json) passes in 43.0s with
144 capacity assertions and reward-driven PPO/checkpoint reload (65,536
transitions; after/no-update 1.0/0.640625). Total measured gate command duration
is 331.942s, including one corrected Clippy failure; budgets and assertions
remain fixed. No required affected check is deferred. The full CI matrix
remains CI-owned; the parent conversation can land the bounded increment.
