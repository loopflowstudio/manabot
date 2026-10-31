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
  for a closed opponent. `unseen_counts` counts the owner's cards the viewer
  has not seen: its own library; for an opponent, hand ∪ library ∪ face-down
  foretold exile minus `known_hand` (concept review, below). The physical
  pool has one owner, `Game::unseen_cards` (moved from `possible_worlds`);
  the possible-world space keeps that pool and `known_hand` as its minimum.
- **D4 `remaining_sideboard_counts`.** The derivation is correct: these are
  sideboard copies in no zone, which Learn retrieval reduces. A retrieved copy becomes an ordinary card. The count includes
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
Disclosure is not part of deck identity. f871a8's `Deck` landed first on
main (2d2f45105); the sync (1e2e817e7) seats `Deck.player()` open, so training
deck pools are always open and closed games come from `MatchConfig.decklists`
or Academy.

## Remaining work

Implemented 2026-10-30 and synced onto main @ b73a6b5db (Deck import, engine
undo removed): engine authority, encoding (native, Python, vector env, model,
inference, text), surfaces (MatchConfig, registration, experience, Academy
form/PlayerArea/trace) and records (choice-navigation w17 fixture,
authored-match receipt and parity re-recorded; WORLDS.md, learn-lesson.md,
READMEs, AGENTS.md). The delete list is complete.

Left for gate/demo: a browser walkthrough of the new-game select and the
opponent panel; the full `cargo test`/`tests/` suites after the sync.
Known outside this change: `tests/academy/test_public_commitment_parity.py`
needs local `experiments/data/rul-9-played-workloads-v1.measurement.json`;
pre-sync, `test_ur_lessons_increment`, `test_temur.py::…production_training_observation_path`,
`test_observation_parity.py::…indestructible…`, belief `test_keystone_demo…`
and `test_native_projection_preserves_canonical_receipt_bytes` failed on base
content too.

`unseen_counts` means unseen (concept review below) implemented 2026-10-30:
the opponent's map drops revealed `known_hand` copies; the possible-world
pool is unchanged. Re-recorded: choice-navigation w17 DISCARD projection and
the authored-match parity receipt (projection hashes only); the
authored-match receipt bytes did not change.

Check (2026-10-30): `cargo test` (debug) `rules_tests decklist_disclosure` 6,
`learn_setup_tests`/`possible_worlds_tests`/`cards` 321 pass; `uv run --extra dev pytest tests/env/test_decklist_disclosure.py tests/env/test_observation.py tests/training/test_training_decks.py tests/academy/test_learn_setup.py tests/academy/test_choice_navigation.py tests/academy/test_authored_match_parity.py tests/academy/test_authored_match_authority.py tests/academy/test_learn_evidence.py` 96 pass.

## Concept review (2026-10-30, headless)

Reviewed the implemented change against the accepted design (one
open/closed option, default open; Jack Heart).

- **Settled, implemented: `unseen_counts` means unseen.** The opponent's
  map was `Game::unseen_cards` verbatim, including hand copies already
  revealed (`known_hand`), so Academy ("N unseen of M") and text ("Unseen
  cards") told a human a card they saw was unseen. The observation now
  subtracts `known_hand` from the opponent's map (one derivation,
  `ObservationBuilder::unseen_counts`; own map is library only and
  unaffected), and the field comment holds for tensor, text and Academy. `PossibleWorldSpace` is unchanged: it
  already carries the full pool with `known_hand` as a minimum. Proof: a
  directed test where an opponent's revealed hand card is absent from
  `unseen_counts` and still present in the possible-world pool; re-record the
  choice-navigation w17 fixture and authored-match receipts only if bytes
  change (open world, nothing frozen).
- **Kept: per-player native authority.** The product control is one game
  option (`MatchConfig.decklists`, Academy `GameConfig.decklists`), stored as
  `PlayerConfig.disclosure` on both players. The per-owner meaning ("what the
  opponent knows of this player's lists") is exactly what
  `decklist_known(owner, viewer)` reads, it rides on `PlayerConfig`/`Deck` and
  failure receipts without a new game-level argument through every `Env`,
  vector-env, rollout-pool and replay constructor, and an asymmetric native
  config is still well defined. Asymmetry stays excluded as a product setting;
  no validation is added for it.
- **Kept: the two names.** `decklists: "open" | "closed"` names the game
  option; `disclosure` names the per-player record. CLAUDE.md states the
  mapping.
- Lifecycle checked: new closed game → opponent maps `None`, tensor rows zero,
  search players refused at build; reconnect/replay → trace and failure
  receipts retain the setting; records without it → open.

## Concept review 2 (2026-10-30, headless): next implement direction

The engine, encoding and records match the accepted design. One interaction
does not: Academy's new-game form offers a combination the server refuses.

- **Mismatch.** The opponent select defaults to `search-64`
  (`OpponentSelector.svelte`), and the new Decklists select offers Closed
  unconditionally (`routes/+page.svelte`). Choosing Closed with the default
  opponent fails only after "New game", with the server's D5 refusal
  (`academy/server.py` `_parse_game_config`). A person has to learn, by
  failing, that search opponents sample hidden cards from open decklists.
- **Direction (follows from D5; not a new product choice).** Show the
  constraint in the form, using `OpponentSelector`'s existing pattern of
  disabled options that carry their reason:
  - Decklists Closed: the three Search options stay listed, disabled,
    labelled "— needs open decklists".
  - A search opponent selected: the Closed option stays listed, disabled,
    labelled "— search opponents need open decklists".
  Neither select changes the other's value; nothing is silently selected
  (CLAUDE.md, Academy's shared decision surface). The server refusal stays as
  the authority for other clients. `OpponentSelector` takes `decklists` as a
  prop; the store needs no new state.
- **Proof.** Vitest: Closed disables the three Search options with the
  reason, and a search choice disables Closed; the server test for
  closed + search refusal stays. Gate/demo browser walkthrough: pick a
  non-search opponent, choose Closed, start a game, see "Decklist closed" in
  the opponent's panel.
- Checked and unchanged: live advice serves static fixtures (no live
  possible-world call to refuse); the hero's own panel always shows its
  lists; `unseen_counts` per concept review 1.
- **Built (2026-10-30).** `OpponentSelector` takes `decklists`; the Decklists
  select moved into `DecklistsSelector.svelte` (takes `opponent`); both read
  `isSearchChoice` in `game.svelte.ts`. Check: `npx vitest run
  src/lib/components/DecklistsSelector.svelte.test.ts` 4 passed; svelte-check
  0 errors. Browser walkthrough and full suites left to gate/demo.

No open product choice.
