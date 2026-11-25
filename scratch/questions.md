# MTG-487 questions

Choices made at kickoff with nobody present; each is reversible. MTG-439's
questions are in that Task's branch history.

- **Reference census over an enumerated scan.** Objects are distinguished by
  a walk of all rules state with exhaustive destructuring, not by a list of
  places to check. Costs more to write; a new id-holding field cannot be
  forgotten.
- **Incarnation is not compared.** Two identical hand cards with different
  zone histories fold, though the owner's observation shows different
  `incarnation` values. Is that number meant to be strategic information?
- **Permanents with layered (including granted) static abilities or keyword counters never fold**
  (timestamps order their effects). Could be relaxed to "no other effect's
  timestamp lies between".
- **Hand and graveyard order are not compared.**
- **Attack/block symmetry is broken by not offering "attack" after an
  identical creature declined**, with `combat_preview` moving a draft's mark
  to the creature the engine asks. Alternative passed over: exposing forced
  declines to the browser and normalizing the draft there.
- **Aliases ride beside `current_action_space`, not inside `ActionSpace`**,
  and reach Academy as an optional `aliases` list on the experience offer.
  `SEMANTIC_DECISION_VERSION` stays 8.
- **Compound offers keep duplicate candidates**; only lowering resolves
  aliases.
- **Folded offers keep their plain label** (no "×3").
- **Not measured at kickoff:** how often real games offer interchangeable
  duplicates. No build cache existed in this checkout; Pass 1 reports the
  share of decisions folded and mean actions per decision.
- **`TurnState.graveyard_from_battlefield`** is read by rules but absent from
  `CanonicalTurn` (`managym/src/state/hash.rs:79-151`). Bundled into Pass 3's
  hash bump; becomes its own fix if Pass 3 is split off (concept-review 4b).
- **Concept review options 1a–5a** were taken headless as recommended; see
  the plan's concept-review section. 4a (combat in this Task, hash v32) is
  the one with the largest cost.
