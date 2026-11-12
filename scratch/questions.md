# MTG-297 questions

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
- Automatic spending order for mana with a spend effect: before ordinary
  mana when the effect applies to the spell being cast, after it otherwise.
  This stands in for the design's spending choice until that is built.
