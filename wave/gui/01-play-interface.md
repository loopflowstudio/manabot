# The Play Interface

Human-vs-bot play in the browser: Jack as one player, an agent as the other,
on the INTERACTIVE_DECK mirror. This is the demo that anchors the project — a
Magic player sitting down and playing full games against the thing we train.

## How to launch

Two processes: backend (FastAPI/uvicorn, port 8000) and frontend (Vite dev
server, proxies `/api` and `/ws` to port 8000).

```bash
# one command (repo root) — starts both, Ctrl-C stops both,
# npm-installs frontend deps on first run:
uv run scripts/play.py
```

Manual equivalent (all Python through uv, per AGENTS.md):

```bash
uv run uvicorn gui.server:app --port 8000     # terminal 1 — backend
cd frontend && npm install && npm run dev     # terminal 2 — frontend
```

Open http://localhost:5173 — that root route IS the play page. Pick an
opponent, hit New Game, click actions. The replay viewer is at
http://localhost:5173/replay and lists every finished game automatically.

Traces land in `gui/traces/` (override with `MANABOT_GUI_TRACES_DIR`).

## What already existed (commit e234933 and before)

- `gui/server.py`: FastAPI WebSocket server (`/ws/play`) with sessions,
  resume tokens, TTL cleanup; villain auto-play loop; trace persistence;
  trace list/load API (`/api/traces`).
- `gui/villain.py`: villain policies — but only `passive` and `random`.
- `gui/trace.py`: trace dataclasses, JSON persistence, hand redaction for
  the trace API.
- `frontend/`: full Svelte 5 play UI at `/` (board, hand, battlefield, life,
  turn/phase indicator, stack when non-empty, clickable actions with
  board-target filtering, game log, game-over overlay with rematch) and a
  replay viewer at `/replay` with timeline playback.

## What this step added

**Pluggable opponents** (`gui/villain.py` rewritten; policy signature is now
`(env, obs) -> action_index`):
- `search` — flat determinized Monte Carlo via `managym.Env.flat_mc_scores`
  (same engine path as `manabot/sim/flat_mc.py`). Strength dial
  `villain_sims` (1–4096), default 64. Determinization redraws the
  *hero's* hand from the villain's viewpoint, so search plays by human
  hidden-info rules. Measured live: ~27 ms mean per hero move at N=64.
- `checkpoint` — a trained `Agent` from a `.pt` training checkpoint
  (`villain_checkpoint` path), stochastic by default, argmax with
  `villain_deterministic: true`. Torch is imported lazily so search/random
  games never pay for it.
- `random`, `passive` — kept as baselines.
- New-game config wire format:
  `{villain_type, villain_sims?, villain_checkpoint?, villain_deterministic?, seed?, hero_deck?, villain_deck?}`.
  The server validates everything and returns a friendly error frame.

**Default matchup**: `DEFAULT_DECK` is now the INTERACTIVE_DECK mirror
(kept in sync with `manabot.verify.util.INTERACTIVE_DECK` by a test).
UI defaults to Search 64.

**Hidden-info integrity** (`gui/server.py: hero_view`, `gui/trace.py`):
- Every live payload is hero-perspective and redacts the villain's hand
  (count only). The engine already omits opponent hand cards from
  observations; the redaction layer is defense-in-depth, and the
  perspective swap closes a real leak: terminal observations can be
  villain-perspective, which previously put the villain's full hand in the
  `agent` slot of the game-over payload.
- Same fix applied to the trace API: `prepare_trace_payload` normalizes
  every stored event to hero-perspective before redacting, so replays no
  longer flip sides on villain decisions and no longer leak the villain's
  hand through villain-perspective events. `?reveal_hidden=true` still
  shows everything (post-game review).
- Libraries are never serialized (counts only) in any payload.

**Action labels in Magic terms** (`gui/server.py: _format_action`):
"Play Mountain", "Cast Lightning Bolt", "Attack with Grey Ogre",
"Block Grey Ogre with Wind Drake", "Wind Drake: do not block",
"Target Villain", "Pass priority". Attack/block focus ids are *permanent*
ids, which the old id→name map missed — attack/block buttons were
previously unlabeled.

**UI**: opponent selector with Search 16/64/256, checkpoint (path + argmax
toggle), random, passive; "Your move" indicator on the action panel;
trace config records the exact opponent for every game.

**Bug fixes along the way**:
- Trace filenames contained `+` (timestamp normalization stripped `:`
  before matching `+00:00`), which `TRACE_ID_PATTERN` rejects — every
  saved trace 404'd through the API. Fixed ordering.
- `hand_hidden_count` now comes from `zone_counts.HAND` (the hand list in
  an opponent observation is always empty at the engine level).

## Validation

- `pytest tests/gui` — 17 green: existing server/session/trace tests plus
  new `tests/gui/test_play_modes.py` (full games vs search / random /
  checkpoint villains driven through the WebSocket as a scripted human,
  hidden-info assertions on every payload, config validation, action-label
  unit tests, terminal-perspective swap test).
- Live (real uvicorn + TCP WebSocket): two full games to terminal, zero
  illegal actions, zero leaks; search-64 villain mean 27 ms / max 282 ms
  per hero move; traces listed and loaded through the API, redacted and
  hero-perspective.
- `cd frontend && npm run build` clean; `npm run check` 0 errors;
  `npm test` 12 green.

## Fix note: the frozen-DOM reactivity bug (July 2026)

The play page rendered once and then never updated: the connection badge sat
on "disconnected" even though the WebSocket was open, and New Game visibly
did nothing while the server happily played games underneath. Nothing in the
validation above could catch it — every check ran at the WebSocket/protocol
or unit level, never against the rendered DOM.

**Root cause**: `+page.svelte` and `replay/+page.svelte` were Svelte
*legacy-mode* components (top-level `$:` statements, no runes) reading the
*runes-mode* stores (`game.svelte.ts` / `replay.svelte.ts`, `$state` class
fields). In legacy mode Svelte 5 deliberately preserves Svelte 4 semantics:

- Template reads of imported objects compile to
  `$.untrack(() => gameStore.connection)` — imports were never reactive in
  Svelte 4, so legacy mode explicitly opts out of tracking them.
- `$:` statements compile to `legacy_pre_effect(deps, fn)` where `deps` only
  reads the compile-time-detected identifiers (the `gameStore` object
  reference — never `.actions`) and the body runs inside `untrack(fn)`.

So no DOM effect ever subscribed to the stores' `$state` signals. Store
updates happened; the DOM never followed. Both routes were affected from the
first GUI commit (the replay page's "worked" status was equally never
DOM-verified).

**Fix**: every component converted to runes mode
(`$props`/`$derived`/`$state`/`$effect`, `onclick` instead of `on:click`),
so all store reads happen in tracked contexts.

**Interop rule going forward**: never read a runes store from a legacy-mode
component — in particular, no legacy `$:` over `$state` fields. Any new
`.svelte` file in this frontend must be runes mode (a stray top-level `$:`
or `export let` silently flips the whole component back to legacy).

**Regression net**: `npm run test:e2e` (Playwright, headless Chromium)
drives the real stack — backend on `MANABOT_API_PORT` (default 8011,
uvicorn binary override `MANABOT_UVICORN`), vite dev on 5183 — and asserts
DOM *mutation*: badge reaches "connected", New Game renders a board vs the
random villain, a full game is played with random legal actions (at least
ten decision points) where the game log must grow after every click and the
board HTML must change across the game, and any console error (except the
dev favicon 404) fails the run. A second spec loads `/replay`, requires the
fetched trace list to render, and steps through frames asserting the frame
counter and board follow. The play spec fails against the pre-fix page
(badge frozen at "disconnected") and passes now.

## Known gaps Jack will hit

- **Checkpoint quality**: there is no trained checkpoint in the repo; the
  checkpoint option is validated with a synthetic untrained agent. Point it
  at a real `step_*.pt` from a training run (path is typed into the UI).
- **No mid-combat granularity in the log**: combat damage and state-based
  deaths show up via the derived board-diff notes ("X left the
  battlefield"), not as explicit "Y deals N damage" lines.
- **Mulligans/side choice**: game always starts hero as player 0 (on the
  play) with fixed 7-card hands; no mulligan decisions exist in the engine
  action space.
- **Session TTL is 15 min** of inactivity; a long think can expire a game
  (the UI offers resume while the session lives, then requires New Game).
- **Draws**: the engine observation carries only a `won` bool; a drawn/step
  -capped game reports a winner rather than a draw.
- **Search strength ceiling**: N=256 is the max exposed in the UI
  (~100-300 ms/decision); the server accepts up to 4096 via the wire config
  if you want to feel the difference.
