# MTG-216: open and closed decklists as a game option

Status: **accepted** (Jack Heart, 2026-10-30): one open/closed decklist
option, open by default. Implementation 2026-10-30 (headless). The four-way
main × sideboard draft is superseded. Its history is in this Task's comments.

> "default should be opendecklist yes. open decklist mean open maindeck and
> open sideboard. no such thing as open one but not the other" (Jack Heart)

## Problem

At main @ 10eeb1f, decklist visibility was fixed, and the surfaces disagreed:

- `Observation` JSON/text/Academy: no main-deck field (closed); both players'
  `sideboard_counts`/`remaining_sideboard_counts` (open).
- Policy tensor v5: the player's own outside copies only; the opponent's
  lists never appeared.
- `determinize`, `PossibleWorldSpace`, belief, flat MC/PUCT: the opponent's
  unseen pool came from their real cards ("Decklists are public").

## Decisions

- **D1 Authority.** Native `DecklistDisclosure { Open (default), Closed }` on
  `PlayerConfig.disclosure`, copied to `Player.disclosure`. It means "what
  this player's opponent knows of this player's main deck and sideboard". It is
  fixed for the game and serialized only when not the default, so default
  digests and fixtures stay byte-identical. Python `MatchConfig.decklists`
  (`"open"|"closed"`, default open, omitted from serialization when default) and
  Academy `GameConfig.decklists` are the product controls. They write the same
  value to both players. A format rule is not world or encoder state.
- **D2 Default open.** Older records without the field read as open.
- **D3 Observation.** `PlayerData` gains `main_counts` and `unseen_counts`;
  `sideboard_counts` and `remaining_sideboard_counts` become `Option`. All four
  are `Some` for the viewer's own lists and for an open opponent, and `None`
  for a closed opponent. `unseen_counts` counts the owner's cards in Hand ∪
  Library ∪ face-down foretold exile, minus the viewer's own hand, so for the
  opponent it is the possible-world pool. One owner: `Game::unseen_cards`
  (moved from `possible_worlds`).
- **D4 `remaining_sideboard_counts`.** The derivation is correct: these are
  sideboard copies in no zone, which Learn retrieval reduces and undo
  restores. A retrieved copy becomes an ordinary card. The count includes
  copies Learn cannot fetch ("outside the game", not "retrievable"). Fixes:
  the opponent's copies reach the tensor; the experience protocol keeps `{}`
  (known and empty) distinct from absent (hidden); the guard row is corrected.
- **D5 Refusal.** For a viewer whose opponent is closed, `Env` determinize,
  possible-world, hidden-hand and flat-MC entry points return `AgentError`
  ("decklist closed"). `Game::determinize` and
  `HiddenHandConstraints::for_viewer` assert. `manabot/sim/registration.py`
  refuses searcher and belief players in closed matches when they are built.
- **D6 Policy input, open v5 extended in place.** New buffer `decklists`
  `(2, max_cards_per_player, 5)`, with rows `(definition+1, main, unseen,
  sideboard, remaining_sideboard)` over the union of main and sideboard
  definitions, sorted. The buffer is present at v5 only (length 0 below, like
  `relations` below v2). A closed opponent has all-zero rows. An open main deck
  is never empty, so no player-row flag is needed and the player width stays
  60. The model pools definition embeddings weighted by each count column
  (÷60), per seat, into the player embedding at v5. LLM text lists or says
  "closed".
- **D7 Wire/Academy.** Experience `PlayerView` carries the four `Option` maps
  with no empty-skipping. Academy `GameConfig.decklists` is recorded in traces
  and replays and set in the new-game form; `PlayerArea` shows the decklist or
  "Decklist closed".

## Delete: do not maintain

- `possible_worlds::unseen_cards` (now `Game::unseen_cards`).
- The guard row `sideboard_counts, remaining_sideboard_counts → Elsewhere("agent
  sideboard rows in agent_cards")` in `managym/tests/cards/observability.rs`.
- The unconditional "Decklists are public" line in `flow/search.rs`; the
  learn-lesson.md line becomes a statement about disclosure.

## Exclusions

Priors over unknown decks for search in closed games; asymmetric product
setting; partial disclosure; Etude table columns; between-game sideboarding.
f871a8 (PR #395) owns `Deck`. Disclosure is not part of deck identity; the
second PR to land threads `disclosure` through `Deck.player()`.

## Remaining work

Compress (2026-10-30): `Observation::player_data` counts through one
`definition_counts` helper; the Python reference row builder is private
(`_decklist_rows`). Nothing left on the delete list. Check: `cargo test`
(debug) `decklist_disclosure` 5 + `learn_tests` 13, clippy clean;
`tests/env/test_decklist_disclosure.py` + `test_observation.py` 59 pass;
PlayerArea vitest 2 pass. Gate owns the full suites.

Implemented 2026-10-30: engine authority, encoding (native, Python, vector env,
model, inference, text), surfaces (MatchConfig, registration, experience,
Academy form/PlayerArea/trace) and records (choice-navigation w17 fixture,
authored-match receipt and parity re-recorded; WORLDS.md, learn-lesson.md,
READMEs, AGENTS.md). Left for gate/demo: browser walkthrough of the new-game
select and opponent panel; `public-commitment-parity` and
`test_ur_lessons_increment` (see check line).

Check result: `cargo test` (debug) all pass, including `rules_tests
decklist_disclosure` (5); fmt/clippy clean. Python: decklist, Learn, model,
observation, world/observation-shape suite and Academy re-recorded fixture
tests pass; PlayerArea vitest (2) and `npm run check` pass.
`tests/academy/test_public_commitment_parity.py` needs local
`experiments/data/rul-9-played-workloads-v1.measurement.json` (not present);
`test_ur_lessons_increment` asserts permanent width 29 against main's 32
(fails without this change). Also failing before this change: stale width
asserts in `test_temur.py::…production_training_observation_path` (60 vs 40)
and `test_observation_parity.py::…indestructible…` (55 vs 41); belief
`test_keystone_demo…` and `test_native_projection_preserves_canonical_receipt_bytes`
fail identically on the afbd6a checkout (base content). tests/env, model,
semantic, verify and belief: 584 pass otherwise. The full `tests/` run was
stopped after 45 min; gate/CI owns it.
