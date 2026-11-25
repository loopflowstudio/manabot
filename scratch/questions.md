# MTG-439 questions

- **Resolved by steer `1952c51e`:** Labels follow Jack Heart's suggestion —
  `"Tap <Name> for {G}/{W}"` for 2–4 explicit modes, `"Tap <Name> for any color"`
  for 5-color sources, and unit/restriction suffixes for multi-mana/restricted
  simple taps (e.g. `"Tap Hermitic Herbalist for 2 mana (Lesson spells only)"`).
  Open polish detail: whether long `ManaRestriction::describe()` strings on
  `Jasmine Dragon Tea Shop` should stay verbatim or be shortened.
- **Pass 2 scope:** The interchangeable-sources skip applies to basics-only
  payments too (`Mountain × 4` for `{2}{R}`, `Plains × 3` for `{W}{W}`), moving
  decision counts across games. Assumed wanted from steer `5d5904ae` (b) to
  keep one rule for basics and duals.
- **LLM text (QA after Pass 5, reversible):** Pass 5 rendered simple choice
  taps as `"Tap Kyoshi Village for {G}/{W} (P1); ability 2"` while basics at the
  same stop read `"Tap for mana P2 (Plains). Native description: Tap Plains for
  {W}"`. QA gave choice taps the basics' shape, `"Tap for mana P1 (Kyoshi
  Village). Native description: Tap Kyoshi Village for {G}/{W}"`, so both read
  alike without rewording basics or convoke taps. `REPRESENTATION_VERSION`
  stays `manabot-text-10`; should this wording change bump it?
- **Non-simple mana abilities** (pain lands, Treasure, filter lands, tap-N)
  keep `"Activate ability: <Name>"` and their modal so extra costs/effects stay
  clearly distinct from free mana taps. Should they get a `"for mana"` hint?
- **Pass 2 multi-unit guard:** Use tap counts only when every participating
  source produces one unit per useful tap. A broader skip for mandatory
  multi-unit sources would need to account for mana left floating after each
  legal tap order.
- **Post-build concept review choice 1 (Pass 4, reversible):** Relax
  `summoning_sick` equality in `interchangeable.rs` for plain non-creature,
  non-animating lands (still checked for creatures, non-lands, and manlands
  with `Modification::BecomeCreature` such as `Soulstone Sanctuary` / `Treetop
  Village`; Earthbend grants haste, CR 702.182a) and relax `entered_this_turn`
  when the permanent has no `power_up` ability. Without this, playing a land
  for turn defeated Pass 2 on nearly every real turn (as QA observed).
- **Post-build concept review choice 2 (Pass 4, reversible):** Delete
  `resolution_mana_abilities` so `Decision::Payment` (`TurnFaceUp`, `Ward`)
  omits simple choice taps whenever `can_pay && help.waterbend == 0 &&
  !offers_extra_payment_taps`, matching how `Decision::Payment` already treats
  fixed sources (including `Ward {2}` with `Kyoshi Village + Sol Ring` matching
  `Plains + Sol Ring`).
- **QA (outside MTG-439):** The Academy Moves log shows `Tap Kyoshi Village
  for {G}/{W}` with raw braces and the offer label rather than the color added
  (pre-existing for basics).
