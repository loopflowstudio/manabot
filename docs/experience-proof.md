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

## Terminal release-build matrix

The deterministic follow-on gate is defined by
[`frontend/e2e/release-prompt-matrix.json`](../frontend/e2e/release-prompt-matrix.json).
It runs the built Svelte application through Vite preview, proxies the shipped
HTTP/WebSocket paths to uvicorn, and drives two fixed-seed games to terminal.
The default UR-hero orientation covers the seven UR-side families; a proof-only
reverse-seat game surfaces GW's `SCRY` and `WATERBEND` prompts through the same
table and protocol. The product default remains UR Lessons versus GW Allies.

Run the release gate with:

```bash
uv run pytest tests/gui/test_release_prompt_matrix.py
npm --prefix frontend run test:e2e:release -- release-prompt-matrix.spec.ts
```

There is no runtime seed search or retry. Each scenario pins its prompt counts,
terminal winner, turn, and command count. The gate fails on incomplete or
unexpected family coverage, a non-terminal run, a missing terminal trace, or
any browser console/page error.

The same two trajectories run entirely through Enter/Space command activation
with the browser's reduced-motion preference enabled. Every prompt occurrence
must expose a named and described Actions region, exact accessible names for
all legal offers, and focus on the next legal choice. The first occurrence of
each family additionally proves tab-boundary behavior, reduced-motion computed
styles, and automated WCAG A/AA checks including color contrast. The terminal
overlay announces the result and contains focus. A bounded socket close checks
only the accessibility of the already-existing reconnect status and focus
restoration; it adds no recovery protocol claim. The run also rejects public or
failed play-asset requests, curated-pack fallbacks, and browser errors.

This gate does not claim replay equivalence, stale/duplicate/checkpoint
recovery semantics, screenshot stability, touch/mobile/zoom certification,
real assistive-technology automation, or a new performance baseline.

## Boundaries and next evidence

This baseline deliberately measures warm development-stack launch and
renderer JavaScript heap. It does not measure cold or release-build startup,
backend/browser/GPU/asset RSS, semantic animation quality, offline assets,
multiple sessions, or large boards. Frame deltas only describe scheduling
around the current DOM updates.

It also does not prove protocol-v1 stale/duplicate command handling, checkpoint
recovery, replay equivalence, screenshot stability, touch, or real
assistive-technology behavior. Those remain explicit Game-wave gates; the
browser accessibility-tree proof must not be promoted into a broader quality
claim.
