# MTG-505 — `create_token` and `add_to_battlefield` on `GameStateEdit`

Follow-up to MTG-502 (PR #637, `23212c88f5`), which added `GameStateEdit`
(`move_permanent`, `set_tapped`, `set_damage`, `set_counters`, `set_life`).
Those edits can only remove or adjust what is already on the board. This
change adds the other direction: put a permanent onto a live battlefield at
the same decision, without wiping policy history, decision memory or replay
provenance (which `scenario_force_battlefield` does).

## User-visible outcome

At a Priority, DeclareAttacker or DeclareBlocker decision, a Python caller
asks "what if this were on the board?" and gets the same decision back,
refreshed:

```python
edits = [
    managym.GameStateEdit.create_token("token.spirit", obs.agent.player_index),
    managym.GameStateEdit.add_to_battlefield(hand_card.id),                    # observed Card.id
    managym.GameStateEdit.add_from_library(1, "Gray Ogre", ready=False),
    managym.GameStateEdit.add_to_battlefield(pacifism.id, attached_to=ogre.id),
]
obs = probe.edit_game_state(edits)
branch = game.snapshot().restore(managym.RestoreOptions.exact(game_state_edits=edits))
```

The new permanents appear in the returned observation with fresh
`Permanent.id`s, their static abilities apply, a ready creature added during
DeclareAttacker joins the remaining prompts, and the batch journals as one
`game_state_edits` intervention that cold replay reapplies.

## API (choices made for review)

```python
GameStateEdit.create_token(
    key: str, controller: int, *, ready: bool = True, tapped: bool = False
) -> GameStateEdit
GameStateEdit.add_to_battlefield(      # a card by its Card.id: hand, graveyard,
    card: int, *, controller: int | None = None, ready: bool = True,   # exile, or an
    tapped: bool = False, attached_to: int | None = None,              # inspected library top
) -> GameStateEdit
GameStateEdit.add_from_library(        # one copy of a deck-entry name from `player`'s library
    player: int, name: str, *, controller: int | None = None, ready: bool = True,
    tapped: bool = False, attached_to: int | None = None,
) -> GameStateEdit
```

Rust (`managym/src/flow/state_edit.rs`), flat like the existing variants
(`tag = "kind"`, snake_case, `deny_unknown_fields`); the wire keys are
permanent replay surface once shipped:

```rust
CreateToken { key: String, controller: usize, ready: bool, tapped: bool }
AddToBattlefield { card: i32, controller: Option<usize>, ready: bool,
                   tapped: bool, attached_to: Option<i32> }
AddFromLibrary { player: usize, name: String, controller: Option<usize>,
                 ready: bool, tapped: bool, attached_to: Option<i32> }
```

`GameStateEdit` stops being `Copy` (it already sits behind
`Arc<[GameStateEdit]>`); the `for &edit in edits` loops iterate by reference.

Choices and what was passed over (concept review, adopted headless, open to
reversal before the PR lands):

- **Three monomorphic constructors.** `add_to_battlefield(card)` takes the
  observed `Object.id` (`agent/observation.rs:2074`), which names the
  physical card across zones (the id `policy_history::expire_links` takes)
  and covers an inspected library top (`observation.rs:1667`).
  `add_from_library(player, name)` names the zone, mirrors
  `set_life(player, ...)`, and uses the name because "a deck entry names the
  one eligible card of that name" (CLAUDE.md). Passed over: one overloaded
  `add_to_battlefield(int | str, owner=)` with a nested
  `CardSource::{Observed, Library}` (hides the library zone, puts a
  viewer-relative "observed" tag in a viewer-independent journal), and one
  `(owner, zone, name)` selector (cannot tell same-named graveyard/exile
  copies apart).
- **Tokens by key**, never printed name (four tokens share "Spirit"). One per
  edit; no `count`.
- **Auras take `attached_to`** (an observed `Permanent.id`), checked after
  the batch by the real state-based rule; refusing Auras would make "what if
  this creature were Pacified?" unaskable.
- **A batch that strands an added Aura is refused**, including
  `[add(pacifism, attached_to=ogre), move_permanent(ogre)]`. Passed over:
  checking at installation and letting the next SBA drop it.
- **A permanent added in a batch cannot be edited later in the same batch**
  (its id is unknown to the caller); use a second batch.

## Entry behavior

All three edits are quiet, like `move_permanent`: no replacement effects, no
`CardMoved`/`CountersPut` events, so no ETB, landfall or constellation
triggers, no policy-history arrival row, no turn-history change
(`land_entered`, lands played). Other state-based actions (legend rule, 0
toughness) wait for the next ordinary check.

| Aspect | Behavior |
|---|---|
| Timestamp | `Permanent::new` takes `id_gen.next_id()` for `id` and `timestamp`, so each added permanent is the newest, in batch order. |
| `ready` | `True` (default): already on board before this turn: `summoning_sick = false`, `entry_turn = 0` (the `Permanent::new` default, as `scenario_force_battlefield` leaves it), so `entered_this_turn` is false. `False`: entered this turn: summoning-sick, `entry_turn = turn_number`. |
| `tapped` | Sets tap state directly. |
| Controller | Token: required seat. Card: its owner unless `controller` is given. |
| Face | Printed front face; no back face. A face-down exiled card enters face up. |
| Instant / sorcery | Refused (`battlefield_entry_face` is `None`, CR 304.4/307.4). |
| Planeswalker | Printed loyalty (intrinsic to the face, `proposed_event.rs:2010`). |
| Saga | One lore counter (`zone_move_proposal`); chapter I does not trigger. |
| Aura | Requires `attached_to`; after the whole batch, `aura_attachment_illegal` (`sba.rs:440`, made `pub(super)`) must be false on the layered board, so an accepted Aura never falls off at the next check. `attached_to` on a non-Aura is refused. |
| Equipment | Enters unattached. |
| Clone, "enters with counters / tapped", "as … enters, choose" | Not applied: printed, no choice. A Clone is a 0/0 that dies at the next check. |
| Combat | Never attacking or blocking; declaration eligibility is recomputed by `refresh_declaration`. |

## Source of truth

`GameStateEdit` in `managym/src/flow/state_edit.rs` and
`Game::edit_game_state`. Everything else derives from it: the
`Intervention::GameStateEdits` journal (`agent/env/provenance.rs:59`), PyO3
`PyGameStateEdit` (`python/bindings.rs:68`), the pydantic mirror in
`managym/recording.py:199`, and the stub in `managym/__init__.pyi`.

## Mechanism

**One transaction for every decision.** `edit_game_state` edits
`let mut edited = self.rollback_snapshot()` at Priority too (today only
declarations do, `state_edit.rs:86`): apply the batch, `refresh_control()`,
check each added Aura with `aura_attachment_illegal`,
`assert_library_knowledge()`, refresh priority or the declaration, and swap
`*self = edited` only on `Ok`. Atomicity no longer depends on validation
predicting post-batch rules state.

**One pass through the edits.** Resolve input ids and library preferences
from the original position. Apply each edit to the rollback copy after
checking that its target is still available. Check Aura legality on the
edited board after the batch. Refusals leave state and journal untouched:

- unknown token key (a card key or a printed name included; checked before
  `prepare_token`, which `debug_assert!`s); invalid seat
- unknown card id; a card on the battlefield, stack, command zone or
  sideboard; a non-card object (spell copy, prepared inset copy); the same
  card named twice
- no copy of `name` left in `player`'s library, counting earlier claims
- instant or sorcery; Aura without `attached_to`; `attached_to` on a
  non-Aura or an unknown/phased-out host; after the batch, an illegal
  attachment

The Python constructors reject what they can see without a game (unknown
token key, seat outside 0/1), as `set_counters` rejects an unknown counter.

**Reshape `commit_zone_move` (`flow/zones.rs:883`) first**, no behavior
change to real moves:

1. *Departure knowledge* (`zones.rs:896–923`): library pin removal and
   `known_library` update, `forget_hand_card` for a hand departure.
2. *Permanent installation* (the `entered_permanent` closure,
   `zones.rs:1052–1081`): face, `Permanent::new`, controller, `entry_turn`,
   tapped, counters, `attached_to`, the `permanents` push and
   `object_to_permanent` link — nothing else. `land_entered` (`:1073`),
   `prepare_inset` (`:1077`) and entering attacking stay in
   `commit_zone_move`.

`scenario_force_battlefield` (`flow/scenario.rs:155`) keeps its contract
(it resets history on purpose) but builds its permanent through the shared
installation instead of its own copy (`scenario.rs:183–199`).

**Quiet card entry:** `add_from_library` takes the bottom-most copy that is
neither pinned nor covered by `known_library.public`/`own`, falling back to
the bottom-most copy, so scry and `move_permanent(..., "library_bottom")`
facts survive. Then the departure-knowledge function (a library departure
calls `known_library.remove_position`, `state/knowledge.rs:39`; a hand
departure `forget_hand_card`), `policy_history::expire_links(game,
object.id, None)` (`agent/policy_history.rs:126`), `mint_appearance(...,
Battlefield, false)`, and the shared installation.

**Quiet token:** `prepare_token(key, controller)`, `mint_appearance`, and
shared installation. All quiet zone changes use `capture_object_move`, which
also retains the original token definition for resolution attribution. Capture
records both meld components and departing tokens. Exile departure clears
links to the old appearance.

## Risks answered

- *Public id across the move:* `policy_history::record` reads
  `game.state[arrival].id` (`policy_history.rs:160`); `Object.id` names the
  card across zones and the permanent gets its own `Permanent.id`.
- *Removing a library card:* `remove_position` shifts both knowledge views;
  `assert_library_knowledge` (debug) checks every fact.
- *Old journals:* additive variants; `protocol/*.schema.json` does not
  mention the type.
- *Academy:* reads only the branch kind (`recording.ts:54`,
  `replay.ts:260`); no frontend change.

## Risks the first pass must settle

- **Engine budget.** The extracted functions must leave
  `bench_search`/`bench_steps` digests and timings unchanged (`#[inline]` if
  needed). Priority edits now clone the game; that path is not benchmarked.
- **Cached views.** Debug recomputes every cached view on read; run the Rust
  tests in debug.

## Acceptance check

```bash
cd managym && CARGO_BUILD_JOBS=2 cargo nextest run --profile ci --test-threads 2 state_edit
# rebuild the extension from the repository root (CLAUDE.md), then
uv run python -m pytest tests/env/test_game_state_edits.py tests/academy/test_game_records.py -q
uv run python scripts/public_api.py check
```

One Python scenario carries the whole path: at a DeclareAttacker decision,
a batch creates a `token.spirit`, adds a creature from the hand
(`add_to_battlefield`) and one from the opponent's library with `ready=False`
(`add_from_library`); the returned observation shows
all three with the requested tap and readiness, the hand and library counts
drop by one each, the spirit is offered as an attacker and the unready
creature is not; `snapshot().to_replay_json()` round-trips through
`GameSnapshot.from_replay_json` to the same `state_digest`; the same edits
through `RestoreOptions.determinize` and `SlotLoad.snapshot` produce a
playable game; `repr` reads as the constructor call.

## Tests (what would be noticed if broken)

Rust, `flow/state_edit/tests.rs` and `agent/env/state_edit_tests.rs`, debug:

- token: enters with the newest timestamp, no ETB trigger from a watching
  permanent, policy history keeps its rows; a card key and a printed name
  are refused without a panic; with statistics on, a quiet Treasure's later
  sacrifice ability resolves attributed to its definition
- `ready=True` leaves `entered_this_turn` false; `ready=False` sets it and
  summoning sickness
- hand source lowers a `known_hand` minimum; library source keeps a known
  top fact true and skips a pinned known bottom copy for an unknown one;
  graveyard and exile sources work
- history rows that linked the departed card lose their live link
- Liliana enters with printed loyalty and survives the next state-based
  check; Pacifism with `attached_to` stays attached and stops the attack;
  Pacifism without it, Pacifism whose host a later edit moves, an instant, a
  battlefield card and a doubled card id are refused at Priority and at a
  declaration and leave the state digest unchanged
- cold replay of a journal containing both kinds reproduces the digest

Python, `tests/env/test_game_state_edits.py`: the acceptance scenario,
error cases through `managym.AgentError`, pydantic `GameStateEdits` accepts
the three new kinds and still rejects untyped payloads.

## Affected surfaces

| Surface | Change |
|---|---|
| `managym/src/flow/state_edit.rs` (+ `tests.rs`) | three variants, one transaction for both decisions, validation, quiet entry |
| `managym/src/flow/zones.rs`, `scenario.rs`, `sba.rs`, `card_capture.rs` | two extracted functions; scenario uses the shared installation; `aura_attachment_illegal` visible to `state_edit` |
| `managym/src/python/bindings.rs` | three constructors, `repr`, native token validation |
| `managym/__init__.pyi`, `tests/public_api.json` | additive; `uv run python scripts/public_api.py record` |
| `managym/recording.py` | `CreateToken`, `AddToBattlefield`, `AddFromLibrary` models in the union |
| `docs/engine-python.md`, `.lf/qa/engine.md` | constructor table rows, entry-behavior paragraph, refusal list, token keys from `managym.content_pool("world")`, the library-only limit under `determinize` |
| Academy UI, `trace.py`, `protocol/` | none |

## Exclusions

- Choosing a back face, a copy target, entry counters or an "as enters"
  choice at creation.
- Entering attacking or blocking.
- Token copies of a permanent, emblems, cards from the sideboard or stack.
- Editing a just-added permanent inside the same batch.
- Recording a mid-game edit on a `ScalarCapture`-tracked scalar `Env`
  (unchanged limit from MTG-502).

## What goes

- The hand-rolled permanent construction in `scenario_force_battlefield`
  (`scenario.rs:183–199`), replaced by the shared installation. Its
  signature, history reset and callers stay.
- `Copy` on `GameStateEdit`.
- The in-place Priority branch of `edit_game_state` (`state_edit.rs:99–103`).

- The token-only capture registration and the separate batch validation
  passes. The existing edit kinds keep their journal spelling.

## Passes

1. **Engine complete** (`4e2a85351`). 14 focused debug tests passed.
2. **Python surface complete** (`fddd9a4d8`). 20 focused Python/Academy tests and `public_api.py check` passed.
3. **Review fixes complete** (`853464096`). 15 native tests, 21 Python/Academy tests and the public API check passed. Edited replay yields complete card facts, live and retained statistics agree, and rejected batches preserve state and capture.

Synced with MTG-500 (`606e79e90a`). The native performance budget and full
suites are deferred to CI.

## Docs written

- `docs/engine-python.md` (`### Counterfactual state edits`): updated example,
  constructor table (`create_token`, `add_to_battlefield`, `add_from_library`),
  token-key discovery via `managym.content_pool("world")["definitions"]`, entry
  behavior, refusal list, and the `add_from_library` limit under `determinize`.

## Done when

A Python caller can add a token, a hand card, a library card and an attached
Aura to a live decision in one batch, read them in the returned observation,
keep playing, and replay the branch cold to the same digest; an invalid
batch raises `managym.AgentError` and changes nothing.

## Concept reviews — disposition

- Pre-build: all findings adopted above as choices made for review.
- Post-build (`20d7b59d18`): pass 3 addressed capture continuity, exile
  links, and the one-pass transaction. Quiet moves use plain
  `capture_object_move` with the existing capture cause and replay keys.
  Token-catalog discovery and the hidden-hand limit are deferred below.

## Deferred findings

- **Counterfactual card the opponent holds in a hidden hand.**
  `RestoreOptions` applies `game_state_edits` before `determinize`
  (`agent/env/snapshot.rs:167–180`), so `add_from_library` refuses when every
  remaining copy sits in the opponent's true hand, and the refusal leaks that.
  Fix needs one of: `add_from_hand(player, name)`, a catalog conjure edit
  (breaks deck-copy conservation), or edits applied after determinization.
  Separate Task if counterfactual callers need it; this change documents the
  limit.
- **Token discovery.** `content_pool("world")["definitions"]` gives key and
  name only ("Spirit" x4), so `token.spirit` and `token.plain_spirit` cannot
  be told apart without reading Rust. Expose token P/T, colors, keywords and
  text in Python catalog discovery; separate small Task.
