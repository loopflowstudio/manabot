# MTG-430 questions and assumptions

- Assumed: emblems are catalog definitions keyed `emblem.<name>` (like
  tokens) and registered as `ObjectDefinition` rows, so learned tables bind
  them by key. Simpler alternative: inline ability lists on the effect, with no
  catalog identity. Chose the catalog because consumers bind by key.
- Assumed: empower is a composition behind one constructor, not an `Effect`
  variant. The open question is whether the text should say "Empower Jace N".
  Default: the template recognises the constructor's shape.
- Assumed: the faithful planeswalker rule is current CR (no damage
  redirection, CR 306.7 removed in 2018). The Task text's "redirecting damage"
  is read as "damage reaching planeswalkers".
- Assumed: Chandra, Chill of Compliance and Gideon's Memorial wait as whole
  cards for card-type `ManaUse` restrictions rather than shipping partial
  definitions.

- Moved the ten Commons and Annex lands ahead of the activation-ledger work.
  PR #593 was open at `c133c0308f`. These lands use the existing planeswalker
  type predicate and modal mana, so they can be played and tested independently.

- Fixed loyalty lowers to the existing source-counter payment components. Both
  produce the signed counter event with `EventCause::Cost`; zero pays no
  counters and still spends the permanent's loyalty activation.
- The declaration model has a second producer. Fire Navy Trebuchet and Suki
  create tokens already attacking. CR 508.4 requires their controller to
  choose who or what each attacks as it enters. The current resolution path
  (`finish_created_token`) has no declaration or defender choice.
- Ordinary trample against a planeswalker uses CR 702.19b and 702.19f.
  CR 702.19c and 702.19e cover the separate “trample over planeswalkers”
  ability. [Comprehensive Rules, November 6, 2026](https://media.wizards.com/2026/downloads/MagicCompRules%2020261106.txt).

- Use the source counter’s signed 32-bit quantity for announced activation X.
  The payment planner still prices mana symbols in its existing mana units.
  A loyalty-only X can use every counter left after other source-counter costs.

- At the MTG-420 sync, keep main's v11 `face_down` and
  `cast_from_graveyard` columns at their existing indices and append the
  loyalty columns. Regenerate the five Academy opening fixtures from the
  combined catalog using their recorded Command prefixes.

- Assumed (realign, pass 3): no historical offer authority for combat replay.
  Main defines no planeswalker card, so only this branch's own snapshots can
  hold a newly legal defender; positions without one keep identical offers.
  Simpler than resolving a historical engine from a v1 snapshot that omits
  its source revision.

- Treat battle defenders as part of battle support. The current catalog has
  no battle and the rules state has no protector. The stress case uses ten
  attackers and six opposing planeswalkers.
- Tomik changes the catalog digest used for cold admission. Check the old
  player-only snapshot before registering the card, and keep its exact Command
  and offer fingerprint as a regression case against the new catalog.
- The advice schema’s new attacked_permanent property matches the Pydantic
  generator. At 1679c477fd, the checked-in advice schema already differed
  from that generator in eight definitions, including LegacyPermanentView.
  This slice updates the combat property only.

- CR 701.71a specifies a Jace **planeswalker token**. Use all three
  properties when creating and choosing. The rules program uses the shared
  frame's existing chosen-object decision. Spell and trigger text abbreviate
  the exact pair of instructions to “Empower Jace N”.
- Use the existing count of moved objects filtered by destination for
  Overwrite. Token creation writes a separate result and preserves that count.

- Emblems use a public “Emblem — <planeswalker>” label. Their rules name
  stays empty in naming and predicate evaluation. Chandra’s first +1 uses
  the paid resolution cast decision and the exact exiled object.

- Way grants use `StaticAbility::Continuous` over `YouControl(planeswalker)`
  with `Modification::GrantActivated`, the path Tomik's static grant uses.
  They stack as program abilities (`granted_by: None`), so the policy sees
  their effects as tokens. No granted Way ability reads its grantor, and a
  catalog join would put an index the frame can't name on the grantee's
  list. `AttachedHas` stays because folding it into the same static would
  change TLA's stored learning tokens and move the pinned pilot.
- "If you removed two or more loyalty counters to activate it" is a field
  of the event (`ActivatesLoyaltyAbility { min_counters_removed }`), not an
  intervening-if. Trigger-time intervening-if checks cannot read the event's
  amount, and the cost paid cannot change before resolution.
- Ajani Unrelenting and Garruk, Curse Breaker move to slice 7. Ajani's −3
  ("each creature except for tokens you control") needs a controller-scoped
  exception, which `CardPredicate` lacks. Garruk's −4 is an until-your-next-turn
  attack trigger.
- Slice 8 pending authorities (not shortcutted):
  - **Return to the Light Realms**: non-cast Aura batch entry requires CR 303.4f
    attachment choices (`02_resumable_results_choices` authority); currently only
    cast Aura spells carry `attach_to` and non-cast Auras die to CR 704.5m SBA.
  - **Loot, the Anomaly**: negative-power combat damage assignment substitution
    ("assigns combat damage as though his power were positive") belongs to
    combat-damage rules/characteristics authority (`07_characteristics_identity` /
    `08_replacements_rules_procedure`).
- Slice 8 orthogonal primitives: added `TriggerModes.min` (skipped when 0) for
  exact multi-mode triggers (Craftwork Crusher), `RevealUntilToHand.to_battlefield`
  (skipped when false) for Identity Echo, and `Effect::Fight { first, second }`
  (sharing `fight_permanents` with `FightLastTarget`, leaving existing tokens
  unchanged) for Mind Meanderer. Added `("Mouse", Creature)` to `subtype.rs`.

- Coverage decomposition: the five older `DelayedEvent` variants keep their
  spelling when delayed watchers move onto `TriggerCondition`. Converting them
  would change TLA's stored tokens and retained state hashes; the simpler call
  is a closed legacy set. Folding them belongs with the 0.3.0 removals.
- Coverage decomposition: no general zone-change replacement yet. Every death
  replacement in the batch is "would die → exile"; Overwrite's
  battlefield-to-exile diversion is the first card needing another axis.

- Adopted for review (R3 concept review G3): accept that `ExileLink.ability_set`
  moves state-hash bytes for any state holding an exile link (new CR 607.5
  state main never recorded); note it in the PR. Passed over: auditing retained
  TLA receipts for mid-game exile links first, since no checked-in receipt
  failed for it.

- Adopted for review (R4 concept review G5): accept that granted abilities on
  the stack or pending write `acquired` into the state hash; note it in the PR
  with G3. Passed over: auditing retained TLA receipts first.
