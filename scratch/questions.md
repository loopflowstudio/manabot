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
- The main sync (#593, `b3485ac26c`) keeps `face_down` at its published v11 column and appends the
  branch's control and blocked columns. Source quality reads main's battlefield
  LKI and exact graveyard arrivals. Academy opening fixtures use main's recorded
  Commands replayed through the combined engine.
- The three existing counted bonuses retain their serialized field names so
  checkpoints read the same card-program tokens after the Rust type changes.
- Changeling uses the creature types in CR 205.3m (2026-11-06), including
  types without a card in the catalog, such as Time Lord.
- CR 701.20b and 701.20d distinguish a zone change from a new object.
  A milled Colossus stays in the library, reveals, and gets a fresh
  appearance when shuffled. The old reference expires without a move event.
- The existing benchmark name `graveyard-routes` measures casting permissions.
  The slice 5 milling position is named `militant-mill`.
- A discard paid during resolution of an opponent's ward ability takes its
  cause from that ability. A discard paid to announce your own spell takes
  its cause from your cost. Both keep a single receipt after replacement.
- The existing mixed-life-gain benchmark orders triggered abilities. The
  Angel replacement position uses its own name so both measurements remain
  reproducible. The Giada position uses Sakashima to keep four legendary
  Giadas on the battlefield.
- Pass 6 uses Giada for the Confiscate position. Two Angels on the new
  controller's side make the pre-entry count observable, and Giada supplies
  the only white mana for two cards with the same cost.
