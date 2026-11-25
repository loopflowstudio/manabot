# Questions and assumptions (MTG-432)

- Assumed: this Task owns changeling (`Keyword::Changeling` + all-zone creature types) and
  `Effect::DiscardHand`, which MTG-431 also needs (Three Tree Mascot, Dragon Mage). When they
  land, hand MTG-431 the commit and paths; if MTG-431 lands one first, sync and reuse it.
- Assumed: first printings follow Scryfall literally, promos included (Giada `psnc`, Heroes' Bane
  `pjou`, Dryad Militant `prtr`), matching the existing `pm15.rs`.
- Assumed: control becomes a real layer-2 computation (slice 1) with `Permanent.controller` as its
  cached result, rather than adding a static-only control path beside resolution records.
- Open: whether `StaticAbility::NoMaximumHandSize` folds into `ControllerHas` cleanly; only fold
  it if its readers become simpler.
