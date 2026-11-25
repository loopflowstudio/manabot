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
- Abyssal Harvester waits for MTG-420 (PR #593, auto-merge requested at
  2026-11-25T01:31Z). If it has not merged when Slices 1–9 are done, publish
  without Harvester and leave its row pending.
- Build the independent stack-copy cohort first. The entry-choice cohort needs
  a complete CR 205.3m vocabulary before its prompt can offer every legal type.
  For repeated copies, compose a counted instruction with the existing copy
  verb so authored single-copy programs retain their serialized form.
