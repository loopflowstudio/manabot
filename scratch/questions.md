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
