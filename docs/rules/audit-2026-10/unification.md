# Authority adjudication — concrete draft

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
spell candidates through that predicate. The independent threshold comparison
and its numeric helper parameter are removed. Spell and ability choices inherit
this through existing `legal_targets_for_spec` callers. Per-requirement duplicate
exclusion, casting-source exclusion, minimum/maximum target counts, and current
state rechecks remain stage-specific responsibilities. No legal action, order,
serialization, Command binding, viewer projection or replay identity is intended
to change; no world version is changed. This is a bounded consolidation, not a
claim that the entire predicate is MTG-correct.

## Confirmed inconsistent rule: hexproof across target specifications

**Definitions:** in `resolution.rs::target_is_legal`, creature/permanent branches
call `hexproof_blocks`, but the permanent branch of `SpellOrPermanent` checks
only battlefield presence and mana value. A 3-mana opposing Gray Ogre with
temporary hexproof is rejected as `Creature` yet offered and accepted as
`SpellOrPermanent { min_mana_value: 3 }`.

**MTG verdict:** CR 702.11b prohibits both targets. The creature/permanent
branches are faithful on this fact; target wording does not bypass hexproof.
`audit_spell_or_permanent_must_respect_opponents_hexproof` records the independent
expected exclusion and was explicitly run to failure. This is a real competing
definition inside the shared predicate, not offer/resolution disagreement.

**Separate unvalidated correction proposal:** route that permanent branch
through the same hexproof guard. No behavioral repair is applied here. It would
change legal offers and resolution in existing worlds; before admission it
needs a new versioned conformance/world identity, retained historical receipts,
acceptance and resolution regressions (including gaining hexproof in response),
Command stale rejection/replay/fork evidence, current-world arena/monitor checks,
and the full learnability diagnostic required by AGENTS.md. None of that
admission evidence is supplied by these local synthetic probes. Current w11
and all frozen evidence remain intact. The ignored probe is explicitly failing
evidence, never passing conformance or a waived required check.

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

This inspection does not prove every CR 400.7 exception or stack-spell
leave/reenter case. In particular, `Target::StackSpell(CardId)` deserves a
separate incarnation review; no claim about its complete identity behavior is
made here. Current-state checks at selection and resolution must remain.

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
CI selection/required-check change, training/deck/experiment change, publication,
merge, auto-merge request or Task completion belongs to this increment.
