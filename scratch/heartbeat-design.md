# MTG-297 design decisions (review 2026-11-12)

The design is `scratch/tle-production-design.md` (copied from MTG-286's
5f6ae4633), with the decisions below. The review page is the published
Artifact "Heartbeat of Spring Design".

## Review answers, verbatim

- A, on narrowing to production only: "no include all here"
- B, on reusing `ManaChoice` plus a `producer` field with no new v10 block:
  "yeah if it works thats fine"
- C, on firing bonuses when mana enters the pool rather than at tap time:
  "Follow the mtg rules exactly"

## What that means

**Scope (A).** This Task implements the whole production and spending
contract, not just Heartbeat. It defines the four undefined cards that need
it, with Scryfall ids, directed tests, observability and learning tokens:

- Heartbeat of Spring (CHK, `chk.rs`): "Whenever a player taps a land for mana,
  that player adds one mana of any type that land produced."
- Animal Attendant (TLE): "{T}: Add one mana of any color. If that mana is
  spent to cast a non-Human creature spell, that creature enters with an
  additional +1/+1 counter on it."
- Toph, Greatest Earthbender (TLE): "When Toph enters, earthbend X, where X is
  the amount of mana spent to cast her." "Land creatures you control have
  double strike."
- Uncle's Musings (TLE): "Converge — Return up to X permanent cards from your
  graveyard to your hand, where X is the number of colors of mana spent to cast
  this spell." "Exile Uncle's Musings."

It also moves The Great Henge and Tarnished Citadel suffixes (already on main)
onto the production transaction, migrates Badgermole Cub, and makes the
deletions the design lists. Remove those cards' rows from the "not yet
defined" table in `docs/engine.md` if they appear there, and update its count of corpus decks that fail admission.

**Encoding (B).** Choose the simplest projection that carries every choice and
receipt without ambiguity. Reuse the existing `ManaChoice` with a `producer`
where that is enough. CLAUDE.md allows no encoder or model version branch
beyond v10's turn-state append, so new pool-class, spend-effect and receipt
state goes into existing fact rows (open vocabularies are rows), shared by
v9 and v10 or refused by `managym.input_v10_state` admission. Keep parity
across native scalar/vector, Python, text and tokens. Do not add the
proposal's `mana_choices[A,4]` block unless the existing action joins cannot
carry a choice; if so, stop and ask.

**Timing (C).** Follow the Comprehensive Rules exactly. No pinned CR file is in
the repository, so cite the current CR in code comments and tests:

- 106.12: to tap a permanent for mana is to activate its mana ability that has
  {T} in its activation cost. Waterbend and Food taps are not.
- 106.12a: "is tapped for mana" triggers when that mana ability resolves and
  produces mana. No mana produced means no trigger. "Type that land produced"
  reads the mana that ability actually produced, colorless included.
- 605.1b / 605.4a: a triggered mana ability resolves immediately after the
  mana ability that triggered it, without the stack.
- Heartbeat's "that player adds": the tapping player gets the mana and chooses
  its type, whoever controls Heartbeat.
- Spending (Toph, converge, Animal Attendant): use mana actually spent to cast,
  including additional costs, and exclude mana spent on other abilities along
  the way (CR 601.2h and the cards' rulings).

Where the design and the CR disagree, the CR wins; record the difference in
the PR description.
