# W2-186 — Establish the game experience proof harness and budgets

## Directive and scope

Directive v1 is incorporated. This Task establishes an independent baseline on
the current GUI protocol and existing Playwright stack. It does not depend on
W2-183, alter the authority protocol, or claim Phase parity.

The work stays in this Task's current worktree and one focused PR. It does not
start another Task Session, select backlog work, or create another worktree.

## User-visible outcome

A developer or reviewer can run one documented proof command for the exact
**UR Lessons (human/hero) versus GW Allies (bot/villain), Search 64** experience
and see:

- a versioned inventory of the prompt families this matchup can produce and
  which current tests actually prove;
- measured warm launch-to-playable, input acknowledgement, authority response,
  animation-loop frame pacing, and Chromium renderer-heap baselines on a named
  reference profile, with numeric regression budgets;
- a browser-level reconnect fault that resumes the same live session and
  preserves the current board/prompt; and
- one keyboard-only action path with a non-empty accessible name that reaches
  an authoritative response.

The proof output reports observed measurements and uncovered prompt families.
It must not describe an unobserved family, an unavailable metric, or a passing
baseline as Phase parity.

## Current inventory

### Selected matchup prompt families

The engine's `ActionSpaceKind` is the prompt vocabulary. Cross-referencing it
with the exact named deck constants and registered card definitions yields nine
reachable non-terminal families:

| Family | Selected-matchup source | Current proof |
| --- | --- | --- |
| `PRIORITY` | Core turn/priority flow; land, cast, activate, pass | Full server/browser games and stops E2E exercise it. |
| `DECLARE_ATTACKER` | Core combat | `test_non_priority_spaces_always_surface` asserts it with a small custom deck; selected-match browser games only exercise it incidentally. |
| `DECLARE_BLOCKER` | Core combat | Action-label unit coverage and incidental full games; no selected-match explicit browser assertion. |
| `CHOOSE_TARGET` | Firebending Lesson, Igneous Inspiration, Divide by Zero, It'll Quench Ya!, Earth Kingdom Jailer, Allies at Last, Yip Yip!, Fancy Footwork, and other targeted effects | Server label coverage and a custom-deck surfacing assertion; selected-match browser coverage is incidental. |
| `SCRY` | Compassionate Healer's becomes-tapped trigger | Explicit Rust rule test and frontend prompt support; default UR-hero/GW-villain play resolves this on the villain side, so the human UI does not reliably render it. |
| `LOOK_AND_SELECT` | Accumulate Wisdom; Water Tribe Rallier | Explicit Rust rule tests and frontend prompt support; current two-deck E2E may observe it but does not require it. |
| `PAY_OR_NOT` | Firebending Lesson kicker; It'll Quench Ya! payment | Explicit Rust rule tests and frontend prompt support; current two-deck E2E may observe it but does not require it. |
| `DISCARD_THEN_DRAW` | Learn on Pop Quiz, Igneous Inspiration, and Divide by Zero | Explicit Rust rule tests and frontend prompt support; current two-deck E2E may observe it but does not require it. |
| `WATERBEND` | Water Tribe Rallier activation payment | Explicit Rust rule tests and frontend prompt support; default orientation resolves it on the villain side and does not reliably render it to the human. |

`GAME_OVER` is a terminal state, not an input prompt. `MODAL` is supported by
the engine and frontend, but its proof card, Crossroads of Destiny, is
explicitly excluded from the Milestone-1 decks; it is not a selected-matchup
prompt family. The current `two-deck.spec.ts` candidate set includes `MODAL`
and therefore must not be treated as the authoritative reachable-family list.

### Existing GUI/server proof surfaces

- `tests/gui/test_server.py`: new-game/action loop, resume, invalid and expired
  resume credentials, and wire log batching.
- `tests/gui/test_play_modes.py`: full WebSocket games against search, random,
  and checkpoint opponents; hidden-information integrity; exact named deck
  selection; trace attribution; action labels.
- `tests/gui/test_stops.py`: server-authoritative auto-pass, stack stops, live
  stop updates, F6, non-priority surfacing, and invalid configs.
- `tests/gui/test_trace_api.py`: trace list/load, redaction, and invalid IDs.
- Frontend unit tests: store replacement/game over/resume-failure behavior,
  socket parsing, stop persistence, action targeting, and replay frames.
- Existing Playwright: reactive full-game DOM mutation, replay stepping,
  priority-stop click reduction/F6, and one non-deterministically encountered
  mid-resolution choice in UR versus GW.

The backend resume test proves the registry primitive, but no browser test
currently severs a live WebSocket and proves the Svelte client returns to the
same session. No current check records the requested experience measurements,
enforces their regression budgets, or proves a general keyboard action.

## Source of truth

Game semantics remain authoritative in:

- `managym/src/agent/action.rs` for `ActionSpaceKind`;
- the registered card definitions in `managym/src/cardsets/`; and
- the named deck constants currently mirrored by `gui/server.py` and
  `manabot/verify/util.py`, whose equality already has a test.

This PR will add one small machine-readable TypeScript baseline module under
`frontend/e2e/` as the authoritative proof record. It owns:

- schema version and exact matchup/opponent/build scope;
- reference device, OS, architecture, Node, Playwright, and runtime browser
  version;
- the nine reachable prompt-family names and explicit `MODAL` exclusion;
- metric definitions, sample counts, initial p50/p95/max observations, units,
  and numeric regression thresholds; and
- explicit limitations for metrics not measured by this first harness.

The Playwright proof spec imports that record and emits an observed JSON
attachment. `docs/experience-proof.md` explains the commands, inventory,
interpretation, and gaps while linking to the machine-readable values rather
than becoming a second numeric source of truth.

## Smallest build

1. Add the baseline record and `docs/experience-proof.md`.
2. Add one `frontend/e2e/experience-proof.spec.ts` using the existing
   Playwright config, WebSocket API, page DOM, and Chromium CDP session. Do not
   introduce a new E2E framework or application protocol fields.
3. Run the exact default named matchup with Search 64. Use five fresh-page
   repetitions for warm launch/new-game measurements and at least twenty
   serialized legal actions for input/authority samples. A deterministic hero
   policy chooses a playable land first, then a cast/activation, then pass, so
   measurement does not use `Math.random()`.
4. Measure:
   - **warm launch-to-playable:** navigation start until the WebSocket badge is
     connected and the first authoritative legal action is visible;
   - **input acknowledgement:** trusted keyboard/click activation until the
     matching hero log entry is rendered locally;
   - **authority response:** the same activation until `data-update-seq`
     advances from the server response;
   - **frame pacing:** `requestAnimationFrame` deltas and long-frame count while
     the sampled interactions and reconnect occur; and
   - **memory:** Chromium CDP `JSHeapUsedSize`, sampled after launch, first
     playable state, each response, and reconnect; record the peak renderer
     heap only.
5. Instrument `WebSocket` with `page.addInitScript`, close the active socket at
   a surfaced decision, and assert the badge transitions through disconnect/
   reconnect, session-storage credentials remain identical, the board and
   prompt are restored without a duplicate hero action, and another legal
   action still advances the authority sequence.
6. For the accessibility slice, focus a legal action button by keyboard,
   assert its accessible name is non-empty, press Enter, and require both the
   local acknowledgement and authoritative sequence increment. Also fail on
   any unlabeled enabled button in the measured game state or any page/console
   error. This is one useful baseline check, not an axe, contrast, reduced-
   motion, or screen-reader claim.
7. Correct the two-deck proof inventory so `MODAL` is not represented as
   reachable. Preserve its existing full-game behavior; this Task does not
   make every prompt family deterministic in the browser.

Initial numbers are measured before thresholds are committed. Regression
thresholds are derived mechanically and their observed values remain beside
them:

- timing p95 budget: the larger of `1.5 × initial p95` and
  `initial p95 + 50 ms`;
- input acknowledgement p95 budget: the larger of `1.5 × initial p95` and
  `100 ms`;
- frame-delta p95 budget: the larger of `1.5 × initial p95` and `34 ms`, with
  a separately recorded worst-frame guardrail of the larger of
  `1.25 × initial max` and `100 ms`; and
- renderer-heap peak budget: the larger of `1.2 × initial peak` and
  `initial peak + 8 MiB`.

Round committed thresholds upward to stable human-readable values. These are
regression guardrails for this reference profile, not aspirational product or
Phase targets. A pathologically high initial value remains documented as a
gap; it must not be described as good merely because the guardrail passes.

## End-to-end proof

On the reference profile, a clean proof run loads the root route, reaches a
playable UR Lessons versus GW Allies Search-64 decision, activates a named
legal action by keyboard, observes the immediate hero acknowledgement and the
subsequent authoritative response, collects frame/heap samples, forcibly
disconnects the live socket, resumes the same board and prompt under the same
session credentials, and successfully continues play.

The focused proof command is:

```bash
npm --prefix frontend run test:e2e -- experience-proof.spec.ts
```

The complete PR verification target is:

```bash
uv run pytest tests/gui
npm --prefix frontend run check
npm --prefix frontend test
npm --prefix frontend run build
npm --prefix frontend run test:e2e -- experience-proof.spec.ts
```

The reference record starts with this machine profile: MacBook Pro (Apple M4
Max, 16 cores, 128 GB), arm64, macOS 26.0.1, Node 25.8.0, Playwright 1.61.1,
headless Chromium. The spec records the exact Chromium runtime version. The
existing Playwright configuration runs uvicorn plus Vite dev; therefore this
Task's launch number is explicitly a warm development-stack baseline, not a
cold release-build measurement.

The Python extension is a hard prerequisite. If the pinned CPython 3.12
`managym` extension is absent, startup fails with the repository's normal
build instructions from `AGENTS.md`; the proof must not skip Python/server
checks or substitute a different interpreter.

## Affected surfaces and consumers

- **Engine/deck definitions:** read to establish reachability; unchanged.
- **FastAPI/WebSocket session registry and resume endpoint:** exercised by the
  new fault test; wire DTOs remain compatible and unchanged.
- **Svelte socket controller, game store, DOM log/action panel, and
  sessionStorage credentials:** exercised as the reconnect and timing
  consumers; application changes are only warranted if the proof exposes a
  missing observable hook.
- **Playwright configuration and browser runner:** reused; no parallel harness.
- **Proof baseline module:** authoritative machine-readable values consumed by
  the spec.
- **Documentation/reviewers/CI:** use the reference commands and interpret the
  emitted JSON attachment against the committed record.
- **Existing GUI, unit, build, and E2E suites:** remain compatible and green.

## Absent and error states

- A prompt family not deterministically observed is reported as uncovered; it
  is never counted from frontend support or an incidental prior run.
- `MODAL` remains explicitly not applicable to this exact matchup.
- Missing CPython 3.12 extension, failed backend/frontend startup, absent legal
  actions, too few timing/rAF samples, a hidden/background document, missing
  CDP heap data on the reference Chromium run, page errors, and unexpected
  console errors fail the proof rather than producing zero or `N/A` values.
- A metric above its committed regression threshold fails and attaches the
  observed distribution for diagnosis.
- Reconnect that creates a new session, loses or advances the prompt, changes
  the board, duplicates an action, or cannot accept the next action fails.
- Invalid/expired resume credentials retain the current server error semantics;
  this Task does not redesign recovery UI.
- The measurement artifact is a test receipt. It does not overwrite the
  committed baseline automatically.

## Operational boundary

The first baseline is intentionally narrow: one concurrent local session,
exact named matchup, Search 64, headless Chromium, a foreground page, current
uvicorn/Vite dev stack, five fresh-context warm launches, and at least twenty
serialized interactions in one measured play session.
The action response wait remains bounded by the existing 30-second E2E timeout,
but the numeric budget is much tighter and derived from the initial reference
distribution. Reconnect must complete within the existing client's bounded
backoff window and before the 15-minute server session TTL.

Only renderer JavaScript heap is measured. Backend RSS, browser-process RSS,
GPU memory, asset memory, cold process startup, release-build startup, and
multi-session scale are named gaps, not silently folded into this value.
Frame deltas describe browser scheduling around the current DOM updates; the
semantic animation system does not exist yet, so this is not an animation
quality claim.

## Exclusions

- Protocol-v1 revision, prompt, offer, command-ID, stale-command, duplicate-
  command, checkpoint-recovery, or replay-equivalence work owned by W2-183 and
  later dependent tasks.
- Deterministic browser scenarios for all nine prompt families. This PR records
  the matrix and establishes the harness seam; later work fills the scenarios.
- Production/release-stack cold-start and total-process memory gates.
- Screenshot baselines, axe/contrast audits, reduced motion, touch, responsive
  viewport coverage, screen-reader assertions, and visual redesign.
- Curated offline asset packaging or network-denied play, owned by W2-185.
- Soak, multi-session, worker/WASM, or large-board/event-storm benchmarks.
- Any comparison to or parity claim with Phase.

## Pursue finish line

Pursue is complete when the focused PR contains the authoritative baseline
record, inventory/documentation, and one Playwright proof spec; real reference
measurements and derived numeric guardrails are committed; the exact scenario
passes including keyboard action and same-session reconnect; every command
above is green; and the handoff explicitly lists the prompt, release-stack,
memory, visual, and protocol gaps that remain.
