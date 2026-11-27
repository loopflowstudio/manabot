# Questions

## Settled (Jack Heart, 2026-11-27)

Recorded from [scratch/design-guide.html](design-guide.html), [scratch/explore.html](explore.html), and [scratch/explore-belief-memory.html](explore-belief-memory.html):

- **Choices (`1a, 2a, 3a, 4a, 5a, 6a`):**
  - Type & journal consistency: `GameStateEdit`, `"op": "game_state_edits"` in `Intervention`, `env.edit_game_state(edits) -> Observation`, `RestoreOptions(..., game_state_edits=...)`.
    - Jack Heart: *"BoardEdit / might want to put things in graveyard or hand right? / GameStateEdit?"*
    - Jack Heart: *"interventions should be called game_state_edits right"* / *"or whatever naem we use, more consistent between journal and api"*
  - `move_permanent` with deck conservation instead of `remove_permanent`:
    - Jack Heart: *"i feel like remmove_permanent should be move_permenant and take a zone, remove_permanent just yolos it out of the game entirely ?"* / *"or do we want to not do that so things like the cards in the deck still all accounted for"*
    - Settled on `move_permanent(permanent, to="graveyard" | "exile" | "library_bottom")` (plus `set_tapped`, `set_damage`, `set_counters`, `set_life`). (A later code read showed `shuffle_libraries` skips pinned library cards, so a `library_bottom` card stays on the bottom; see review choice Q2 below.)
  - Policy history: expire live object links without appending a synthetic `Arrival` row (`2a`, preserving a full 32-row window).
  - Doors: `env.edit_game_state(edits)` + `RestoreOptions(..., game_state_edits=...)`; legacy `scenario_*` untouched (`3a`).
  - Belief boundary: dropped `edit.hidden_neutral` (Jack Heart: *"what is edit.hidden_neutral exactly"* → *"drop it"*); `GameStateEdit` stays pure data and `manabot.belief` compares `PossibleWorldSpace` directly. `BeliefTracker` carry-over stays a separate `manabot` Task.
  - Atomicity/perf: pre-validate batch before mutating in place (`6a`, 1 clone + 1 republish).
  - Proceeded with: *"sounds good lets proceed"*.

## Adopted for ship-review, not settled (concept review, 2026-11-27)

Headless realign took the review's recommendations; each binds replay digests, so confirm before merge. Detail and passed-over options in `scratch/plan.md`.

- Q1 → 1a: tokens cease on any `move_permanent`; `exile_links` cards stay exiled, link dropped.
- Q2 → 2a: `game_state_edits` run before `RestoreOptions.mode`; edits + `forced_first_action` rejected; `library_bottom` stays pinned.
- Q3 → 3a: at declare steps, newly eligible creatures join the remaining declarers.
- Q4 → 4a: phased-out permanents rejected.
- Melded permanent moved: both cards go to the destination (realign's own call).

## Implementation choices

- The `set_counters` journal entry uses `"kind": "set_counters"` and a separate `"counter"` field. The Python constructor keeps `kind`.
- A bottom move keeps existing library facts. For a face-down card the owner receives a printing fact only if they could inspect the permanent. For a melded permanent the second component is bottommost.
- An empty live batch returns the current observation at a supported decision without republishing or journaling. An empty restore batch is equivalent to omitting it.
- A declared blocker has used its declaration even when an edit moves its attacker away. Retain that fact through later edits in the same step. Determinization may replace a known bottom card with another physical copy of the same printing, as it does for existing library facts.
- Combat declarations use the first offer to identify the creature and rules player. This relies on each published declaration asking one creature to answer, including when another seat controls the decision.

## Open / deferred (for follow-up Tasks)

- **Manabot `BeliefTracker` carry-over & double-scoring:** teach `BeliefTracker` to carry posteriors across `hidden_neutral` edits via canonical hand-count transport, and support Codex's held-vs-recomputed pair for board-sensitive belief models.
- **Pre-existing bug found during exploration (outside MTG-502):** `BeliefTracker.observe` raises `"unexplained hand exit"` when any public card enters the opponent's library, an ordinary rules move included (`manabot/belief/tracker.py:362`). Not filed yet.
- **Legacy `scenario_force_battlefield` return value:** returns storage index while `GameStateEdit` takes observed `Permanent.id`. Left untouched since legacy `scenario_*` is kept separate.
- **Future `GameStateEdit` variants:** hand/graveyard/exile/battlefield placement, mana pool, summoning sickness, control, opt-in `settle_sbas`, and suspended-decision boundaries (`ChooseTarget`, `Modal`, `PayOrNot`).
