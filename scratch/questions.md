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
