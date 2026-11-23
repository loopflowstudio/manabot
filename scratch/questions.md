# MTG-422 scope choices

- The Task uses “Cosmic Cube” as a control example. The MSH ledger assigns next-turn control to Construct a Cosmic Cube. Use that Plan and keep the artifact's separate MSH-05 casting work in its own batch.
- For the open-ended cross-set cohort, use both non-planeswalker EOE player-procedure cards and both MSH-09 cards, with the existing Secret of Bloodbending as the control regression. These cover a second user of opening entry and poison as well as both named control cases.

- Build turn ending first. The existing Sundial instruction gives a second consumer for Time Stop, and the pre-cleanup state-based check can exercise a resumable procedure before changing reset for every deck.

- The cleanup fixtures need eight independently selectable hand objects. Draw eight from their seeded decks instead of planting particular basic lands.

- Use OpeningHands, Playing and Finished as lifecycle positions. Keep the end-turn return point in its existing frame. For ordered bottoming choose deepest first and name that direction in the offer.

- Use turn zero for pregame entries. Store the turn a permanent entered so power-up keeps its discount through any priority during repeated cleanup.

- Retain every simultaneous loss cause per seat. Preserve the historical empty-draw precedence in the single-reason Env field so one consumer's projection does not erase facts from the committed result.

- Use byte-sized opening counters (at most seven) and allocate the combat workspace only during combat. Compacting the opening state alone still failed the synced 4096-byte assertion. The paired budget at gate will measure the combat allocation.

- Return the same terminal info from indexed and semantic Env steps. Derive it from the committed native outcome so restoring a snapshot needs no second stored receipt.

- Use Sakashima of a Thousand Faces to keep two legendary Bracelets on the same creature. Use Doubling Season to cross seven plan counters with one placement.

- Use match-state hash v19 for the smaller serialized victory record. Keep the previous hash in git and leave the recorded games at their original revisions.

- For the local measurement slice, pair duplicate-creatures with one Herald per seat on diverse static boards. This isolates a common SBA consumer and the combat workspace at bounded sizes.

- Count a rules draw separately from simulation timeout. Credit a committed win or draw on the last allowed step and keep all games in the win-rate denominator.
