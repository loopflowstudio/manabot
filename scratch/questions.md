# Questions and assumptions (MTG-432)

- Assumed: this Task owns changeling (`Keyword::Changeling` + all-zone creature types) and
  `Effect::DiscardHand`, which MTG-431 also needs (Three Tree Mascot, Dragon Mage). When they
  land, hand MTG-431 the commit and paths; if MTG-431 lands one first, sync and reuse it.
- Assumed: first printings follow Scryfall literally, promos included (Giada `psnc`, Heroes' Bane
  `pjou`, Dryad Militant `prtr`), matching the existing `pm15.rs`.
- Assumed: control becomes a real layer-2 computation (slice 1) with `Permanent.controller` as its
  cached result, rather than adding a static-only control path beside resolution records.
- Kept printed `NoMaximumHandSize` distinct. The existing observation field
  describes a rest-of-game record. A battlefield grant can expire when its source leaves.
- Slice 1 retains the entry controller as one bit relative to the current controller
  in the two-seat permanent record. This preserves the existing permanent size
  and leaves ordinary unchanged-control state hashes free of a redundant seat.
- Harmless Offering reads its opponent through a clause index. Existing player
  selectors keep the active clause, including the binding used inside modal effects.
- Coupled blocker declarations keep their sequential structured offers. Independent
  blocker choices can form an illegal group under menace or competing requirements.
- Protection supplies a `PreventAll` damage replacement now. Slice 8 can use the
  same transform. Numeric prevention remains meaningful for smaller shields.
