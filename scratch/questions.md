# MTG-297 questions

- Fixture repair (2026-11-12): funded card-behavior fixtures spend ordinary
  colorless mana for generic pips to preserve colors for later spells. Hybrid
  pips and regular/combat expiry are chosen explicitly where they arise.
  Abandon Attachments' prowess fixture chooses blue for its hybrid pip, then
  colorless for its generic pip; either hybrid color exercises the same trigger.
  This is a test-player choice, not an engine payment preference; shared
  casting, priority, mode and target helpers still expose pending decisions.

- A fixed (non-activated) mana source producing two types under Heartbeat
  would need a bonus type choice inside automatic payment; the planner leaves
  such a source out. No current fixed source produces two types.
- The tapped permanent answers trigger predicates from the record taken when
  it tapped (`TappedForMana`), checked against predicates of triggers on the
  board then. A triggered mana ability that arrives between the tap and the
  production with a predicate no earlier trigger had would not see the
  tapped permanent. No current content can do that. (Simpler choice taken.)
- The iteration direction named "the existing kind-14 frame `source`
  reference (... ManaChoice arm)". Kind 14 is the delegated-decision frame;
  the `ManaChoice` arm is the kind-6 obligation frame (decision code 3), so
  the sources are projected there: output choice `source` = the mana
  ability's source; bonus choice `source` = the trigger's source, `context` =
  the tapped permanent.
- Mana abilities do not support `ActivatedAbilityDefinition.sacrifice`
  (sacrifice-another) costs, so the witness sacrifices Heartbeat through the
  ability's non-mana instructions, which run before the output is produced.
- The spend-effect pool program and the cast receipt reuse fact kind 11 and
  the kind-14 execution occurrence rather than new fact kinds: a new kind
  grows the model's kind embedding in three implementations (torch, the
  numpy routes, MLX) for no added meaning. (Simpler choice taken.)
- The receipt is projected only when a rule reads it, so ordinary spells add
  no rows. A card that reads another object's spent mana would need the
  receipt projected for that object too; none exists.
- Expiry is a `combat` property of a class in combined payment search and
  projection; live regular/combat pools keep their existing ownership. This
  keeps ordinary regular mana allocation-free and needs no input shape change.
  Automatic class/expiry preferences have been removed.

- Sync onto `285cd49ce` (2026-11-12): retain main's unrelated w17 fixtures,
  parity hashes and belief receipts as directed by sync-conflicts. Neither
  parent's hashes describe the combined catalog; gate must refresh current-world
  evidence after rebuilding. Keep mana behavior in main's new printed/card/token
  structure, including the four new definitions and Badgermole production.
