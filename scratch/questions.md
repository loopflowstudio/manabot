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
- **LLM text** (`manabot/semantic/text_observation.py:658`) still says
  `"Activate ability of <Name>"`; left alone as frozen prompt wording. Should
  it follow native `action.description`?
- **Non-simple mana abilities** (pain lands, Treasure, filter lands, tap-N)
  keep `"Activate ability: <Name>"` and their modal so extra costs/effects stay
  clearly distinct from free mana taps. Should they get a `"for mana"` hint?

- For resolution payments, preserve a simple activation when accepting the
  cost can leave a different source or floating pool. Ward {2} with Kyoshi
  Village and Sol Ring distinguishes the two payments.

- For pass 2, use tap counts only when every participating source produces
  one unit per useful tap. A broader skip for mandatory multi-unit sources
  would need to account for mana left floating after each legal tap order.
