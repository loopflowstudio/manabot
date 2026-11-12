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

## Coordination with MTG-242 (2026-11-12)

The user's request: "you may also want to coordinate with mtg-242 which is also touching mana payment".
MTG-242 (`jack/mtg-242-generalize-mana-spending-restrictions-and-payment-resources`,
worktree `../manabot.mtg-242-generalize-mana-spending`, PR #496 open on 2026-11-12) reworks
`state/mana.rs` restricted-pool classes into typed `ManaRestriction` /
`SpendingContext`, and makes the encoders refuse restricted mana outside the
representable list (its `scratch/questions.md` proposes a
`FactKind::RestrictedMana` row).

- Production first: the trigger, `ManaChoice` producer, the Henge/Citadel suffix,
  Badgermole and Heartbeat touch `flow/zones.rs`, `flow/mana.rs` resolution and
  `state/card.rs`, not the pool's storage.
- Spending (pool classes, receipts, Animal Attendant, Toph, Uncle's Musings)
  builds on MTG-242's types: extend its `ManaRestriction` / `SpendingContext`
  and restricted-mana fact row instead of a parallel canonical-class
  representation. Do not rewrite `state/mana.rs` pool storage before MTG-242
  lands on main; read its branch, then rebase onto main once it merges.
- If spending can't wait, stop after the production half and say so.

## Concept review of the production half (2026-11-12)

The production cut matches the design: one trigger record (`who`, predicate,
typed output), bonuses resolved off the stack after the output, the planner
folding bonus combinations into the same output alternatives, the old tap-time
bonus path deleted. Two gaps, both settled by the design and answer C, are the
next implement direction. Neither changes the v9/v10 layout. Both are
built (see `mtg-297-heartbeat-of-spring.md`).

1. **Trigger sources at production, not at tap.** `questions.md` took the
   simpler choice of selecting trigger sources when the source taps. Answer C
   was "Follow the mtg rules exactly", and CR 106.12a triggers when the mana
   ability resolves and produces mana, so a trigger source removed while costs
   are paid (a "{T}, Sacrifice a permanent: Add" ability sacrificing Heartbeat or
   Badgermole Cub) must not trigger. No current card has that cost, so behavior
   today is unchanged. Capture only the tapped permanent's LKI at tap (it may be
   the sacrificed source, which still triggers); select trigger sources from
   the board at production, after the sacrifice cost and suffix. The
   planner's `production_alternatives` keeps reading the pre-activation board
   as its estimate, and play is the authority. Witness: a test ability in
   `tests/cards/chk.rs` that sacrifices the Heartbeat as its cost adds no
   bonus; the existing sacrificed-land witness still adds one. Remove the
   first `questions.md` entry once built.
2. **A bonus choice names its source.** The design's first counterexample is
   that `ManaChoice` "drops the production source", and `produced` alone still
   leaves a bonus type choice ({G} or {C} from Heartbeat) indistinguishable
   from the land's own output choice in the continuation frame and the
   decision text. Carry the trigger's source permanent with each pending
   bonus (`bonuses` entries become output plus source) and, on the output
   choice, the tapped source. Project it through the existing kind-14 frame
   `source` reference (`agent/continuation.rs`, `ManaChoice` arm), and give
   the bonus choice its own instruction in `decision_context.rs` (a triggered
   mana ability's type, chosen by the tapping player). This is Academy's
   shared decision surface: the choice must be readable as Heartbeat's
   without the browser inventing it. Keep scalar/vector, Python, text and
   token parity; it adds no buffer or fact kind.

The spending half is unchanged: it waits for MTG-242 (PR #496) per the
2026-11-12 steer quoted in `mtg-297-heartbeat-of-spring.md`.
