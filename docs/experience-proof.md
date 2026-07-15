# Game experience proof baseline

This is the first repeatable proof gate for the deliberately narrow **UR
Lessons (human/hero) versus GW Allies (bot/villain), Search 64** experience. It
records current behavior and catches regressions; it is not a Phase-parity or
release-readiness claim.

The machine-readable source of truth is
[`frontend/e2e/experience-proof-baseline.ts`](../frontend/e2e/experience-proof-baseline.ts).
It pins the reference profile, sample counts, initial measurements, regression
budgets, and explicit exclusions. Each proof run attaches its full observed
distribution as `experience-proof.json`.

## Reference command

Prepare the repository's pinned CPython 3.12 extension as described in
`AGENTS.md`, then run:

```bash
npm --prefix frontend run test:e2e -- experience-proof.spec.ts
```

The complete scoped verification is:

```bash
uv run pytest tests/gui
npm --prefix frontend run check
npm --prefix frontend test
npm --prefix frontend run build
npm --prefix frontend run test:e2e -- experience-proof.spec.ts
```

All Python entry points remain behind `uv` (or the uv-managed interpreter path
used by the existing Playwright configuration).

## What the harness proves

The Playwright proof uses the current uvicorn + Vite development stack and
headless Chromium. It performs five fresh-context warm launches, then at least
twenty serialized keyboard interactions in one measured game session. It
records:

- warm navigation through the first authoritative legal action;
- Enter keydown through the matching local hero-log acknowledgement;
- the same Enter keydown through the next authoritative update sequence;
- `requestAnimationFrame` pacing during play and reconnect; and
- Chromium renderer `JSHeapUsedSize` samples.

At a surfaced decision it forcibly closes the browser WebSocket. The check
requires the connection to recover under the same session credentials, with
the same board text, legal actions, and log, before another keyboard action can
advance the game. It also requires every visible enabled button in the sampled
state to have a computed accessible name and fails on unexpected console or
page errors.

The reference profile is a MacBook Pro with Apple M4 Max (16 cores), 128 GB,
arm64 macOS 26.0.1, Node 25.8.0, Playwright 1.61.1, and the Chromium version
recorded in the baseline module. Numeric values live only in that module so
the test and documentation cannot acquire competing budget sources.

## Selected-matchup prompt inventory

`ActionSpaceKind` is the engine authority for prompt meaning. The exact named
decks can reach these nine non-terminal families:

| Prompt family | Selected-matchup source | Browser proof before this baseline |
| --- | --- | --- |
| `PRIORITY` | Core land/cast/activate/pass flow | Explicit full-game and stops coverage |
| `DECLARE_ATTACKER` | Core combat | Explicit server surfacing with a custom deck; incidental selected-match coverage |
| `DECLARE_BLOCKER` | Core combat | Incidental full-game coverage only |
| `CHOOSE_TARGET` | Targeted Lessons, removal, combat tricks, and Ally effects | Explicit server formatting/surfacing; incidental selected-match coverage |
| `SCRY` | Compassionate Healer | Rule-level proof; villain-side in the default orientation |
| `LOOK_AND_SELECT` | Accumulate Wisdom and Water Tribe Rallier | Rule-level proof; non-deterministic browser observation |
| `PAY_OR_NOT` | Firebending Lesson and It'll Quench Ya! | Rule-level proof; non-deterministic browser observation |
| `DISCARD_THEN_DRAW` | Learn effects | Rule-level proof; non-deterministic browser observation |
| `WATERBEND` | Water Tribe Rallier | Rule-level proof; villain-side in the default orientation |

`GAME_OVER` is terminal rather than a prompt. `MODAL` exists in the engine and
frontend framework, but its proof card, Crossroads of Destiny, is not in either
selected deck and therefore is not part of this matchup inventory.

Existing backend coverage lives in `tests/gui/test_server.py`,
`test_play_modes.py`, `test_stops.py`, and `test_trace_api.py`. Frontend unit
coverage exercises the game/replay stores, socket parsing, stops, and target
mapping. Existing Playwright specs cover reactive full-game DOM mutation,
replay stepping, priority stops/F6, and at least one encountered
mid-resolution choice. Those tests do not yet deterministically cover all nine
families; absence is reported rather than inferred from frontend support.

## Boundaries and next evidence

This baseline deliberately measures warm development-stack launch and
renderer JavaScript heap. It does not measure cold or release-build startup,
backend/browser/GPU/asset RSS, semantic animation quality, offline assets,
multiple sessions, or large boards. Frame deltas only describe scheduling
around the current DOM updates.

It also does not prove protocol-v1 stale/duplicate command handling, checkpoint
recovery, replay equivalence, deterministic browser scenarios for all prompt
families, screenshot stability, contrast, reduced motion, touch, or
screen-reader behavior. Those remain explicit Game-wave gates; a passing first
baseline must not be promoted into a broader quality claim.

