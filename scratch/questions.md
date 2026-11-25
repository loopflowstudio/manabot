# Open assumptions (MTG-431)

- MTG-428 ("Mark Demonic Pact and Maze's End as defined in the FDN ledger") is
  unstarted. Assumed: this Task updates those two ledger rows because they are in
  its scope; MTG-428 can then complete with nothing to do. Reversible.
- Dragon Mage: "each player discards their hand" is one simultaneous action.
  Assumed: use one batch over every hand if a selector exists for it; otherwise
  two APNAP batches. That only differs for a "one or more players discard" batch
  trigger, and none is authored. Reversible.
- Run Away Together: built with the faithful `DifferentControllerFrom` relation
  (Slice 4), not the two-player "yours + theirs" approximation.
- Perforating Artist: the opponent's three-way choice is a nested optional
  (discard? else sacrifice? else lose 3). The choice is complete but takes two
  steps rather than one three-way pick.
- Build the independent stack-copy cohort first. The entry-choice cohort needs
  a complete CR 205.3m vocabulary before its prompt can offer every legal type.
  For repeated copies, compose a counted instruction with the existing copy
  verb so authored single-copy programs retain their serialized form.
- Creature names use the straight apostrophes found on card type lines.
  The November 6 CR 205.3m list has 367 entries. Scryfall’s creature-type
  catalog contained all 367 and ten extra names, including Automaton and Naga.
  Only the rules list supplies legal choices.
- The 168-byte permanent budget still holds. A shared entry answer takes one
  pointer, and the rarely populated exhaust history uses a boxed slice with
  the same serialized list. Academy displays the private look from the native
  known-hand counts and viewer-visible definition names.
- The new advice schema property matches the Python model. Comparing the
  whole permanent schema also found earlier differences in seven unrelated
  properties, including missing `face_down` and `keywords`. Those properties
  match the checked-in schema before Slice 1.
- Biogenic Upgrade uses one simultaneous counter batch for distribution and
  another for doubling. Each recipient reads its own pre-batch counter count.
  Sequential child programs would split one instruction into separate events.
- Divine Resilience uses a kicked “up to” bound. This changes both the minimum
  and maximum. Its zero-target cast is legal on an empty battlefield when the
  kicker is payable. Deadly Brew applies `Except::Moved` in the public-zone
  selector so the ordinary graveyard selection keeps its existing behavior.
