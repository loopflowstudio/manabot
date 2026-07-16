# Phase vs. Etude Fantasia: full game-experience technical deep dive

Research date: 2026-07-15
Phase revision: [`553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d`](https://github.com/phase-rs/phase/tree/553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d)
Etude revision inspected: `bbb5a0a38f8b90efeb87829b60847fb40c5d55d4` (`rescue-set-seed`)

## Executive judgment

**Phase is a real, heavily engineered game-experience platform, not a vibecoded facade.** Its strongest work is not merely its rules engine: it has a serious client runtime with atomic state/action snapshots, worker-isolated WASM, local/server/P2P/replay adapters, reconnection and persistence, guarded PWA updates, desktop packaging, explicit interaction coverage, and a presentation system built to survive Magic's combinatorial UI surface.

It is also **large, accreted, and expensive**. At the pinned commit, `client/src` contains roughly 162,000 TS/TSX lines, 451 component files, and 245 test files. Its experience layer includes 123 registered player-facing `WaitingFor` variants, a 3,373-line choice modal, a 3,404-line game page, and a 3,057-line manually mirrored Rust/TypeScript type file. A complete client game-state representation crosses the WASM boundary as JSON and is then structured-cloned from a worker on every action. Much of its visual semantic reconstruction lives in a 779-line TypeScript event normalizer. That is not evidence of fraud; it is evidence of a broad general-purpose Magic client paying the full generality tax.

**Etude today is the opposite trade:** a small, working, intentionally utilitarian AI-learning instrument. Its frontend is about 3,200 source lines. It exposes an unusually clear legal-action inspector, real learned/search/random/passive opponent identities, fully attributable decision traces, and replay as an AI-observation surface. Its current UI is not Phase-level product polish or portability: state snaps, networking is revisionless, card art is remote-only, the Python backend is required, reconnect is process-local, and there is no animation, PWA, desktop shell, large-board strategy, or production frontend CI.

The correct “great artists steal” conclusion is therefore:

1. **Steal Phase's experience invariants**: atomic snapshot + legal choices, monotonic revisions, one command queue, adapter boundaries, background execution, semantic presentation events, fail-loud interaction coverage, full-snapshot recovery, update safety, overflow/grouping, and exhaustive flow testing.
2. **Do not steal Phase's surface area or accidental architecture**: the Commander cockpit, generic deck/format UI, manual type mirror, 123 branch-specific dialogs, dual dispatch paths, all-state JSON copies, animation timing as summed sleeps, or DOM measurement of every visible object on every action.
3. **Build a third thing**: a curated-deck, AI-native “experience runtime” whose rules/AI process emits a compact, versioned, clone-friendly frame, stable interaction offers, and semantic presentation events. Direct manipulation and Etude's action inspector should be two views of the same offers. That can feel better than Phase precisely because the selected deck/mechanic inventory is finite and art-directable.

This report concerns the game experience boundary—client architecture, state delivery, rendering, input, portability, resilience, replay, and perceived quality—not the internal correctness of either rules engine.

### Evidence labels and limits

- **Observed** means read directly in the pinned source or executed in this checkout.
- **Source claim** means a repository comment or document says it; I did not independently measure it.
- **Inference** means a consequence derived from code structure and is labeled as such.
- **Not measured** means the source supports architectural conclusions but not a perceptual/performance claim.

I could not launch the pinned Phase client. The sparse checkout lacks `node_modules`, generated WASM JS/`.wasm`, and generated card-data assets. The configured in-app browser-control runtime was also unavailable, so there was no hands-on Phase visual QA. Accordingly, this report does **not** claim that Phase subjectively feels smooth or meets a frame-time target; it establishes that the source contains a substantial experience platform and identifies where that platform should or should not scale.

## System understanding

### Architecture

#### Phase: an adapter-driven client runtime

Phase's React/TypeScript client is organized around several different kinds of state rather than one undifferentiated store:

- an authoritative displayed game snapshot, legal actions, logs, history, and sequence commit gate in [`gameStore.ts`](https://github.com/phase-rs/phase/blob/553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d/client/src/stores/gameStore.ts#L205-L330);
- disposable interaction state—selection, open dialogs, targeting—in the UI store;
- a presentation-only animation store with captured element positions, a queue, active animation, and post-animation state in [`animationStore.ts`](https://github.com/phase-rs/phase/blob/553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d/client/src/stores/animationStore.ts#L25-L81);
- persisted user preferences and separate multiplayer/session stores.

The client consumes an `EngineAdapter` interface rather than binding the board directly to one transport. The interface covers initialization, dispatch, snapshotting, serialization/rehydration, replay-related state, takebacks, and AI work ([`adapter/types.ts`](https://github.com/phase-rs/phase/blob/553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d/client/src/adapter/types.ts#L2975-L3057)). Implementations include local WASM, WebSocket, P2P host/guest, and replay-worker paths. This is a genuine portability seam even though the implementations do not share equally compact protocols.

The local engine normally runs in a Web Worker; `WasmAdapter` falls back to the main thread when workers are unavailable. It deliberately shares a singleton initialized adapter to preserve the expensive WASM/card database and V8 optimization across games, with a single in-flight initialization promise to avoid duplicate large instances ([`wasm-adapter.ts`](https://github.com/phase-rs/phase/blob/553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d/client/src/adapter/wasm-adapter.ts#L106-L193)). Separate AI workers form a small pool and receive game-scoped card subsets. The source explicitly disables that pool on iOS/Android phones due to per-worker module memory pressure ([`wasm-adapter.ts`](https://github.com/phase-rs/phase/blob/553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d/client/src/adapter/wasm-adapter.ts#L25-L42)).

**Source claims, not measurements:** comments describe a roughly 90–93 MB card database, 3–5 second fetch/parse initialization, and roughly 48 MB per WASM AI worker. These explain several design choices but require profiling on a production build before use as a baseline.

Phase is deployable as a web/PWA client, a server-backed client, and a Tauri desktop app. The Tauri package includes a `phase-server` sidecar plus card/draft data and has signed updater configuration ([`tauri.conf.json`](https://github.com/phase-rs/phase/blob/553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d/client/src-tauri/tauri.conf.json#L1-L52)). The source README refers to a `TauriAdapter`, but no adapter of that name exists at this revision; local Tauri play appears to use the WASM adapter. That is minor documentation drift, not a platform absence.

#### Etude: a thin Svelte instrument over one authoritative Python session

Etude's experience is much smaller and easier to understand:

- SvelteKit renders one live-game route and one replay route.
- FastAPI owns a `managym.Env`, advances the villain automatically, normalizes a hero-view observation, and pushes a full observation plus positional legal actions over a WebSocket ([`gui/server.py`](/Users/jack/src/etude/gui/server.py:603)).
- [`GameStore`](/Users/jack/src/etude/frontend/src/lib/game.svelte.ts:60) owns the currently displayed observation/actions, logs, focus IDs, target selection, connection state, stops, chosen decks, and opponent identity.
- [`GameSocketController`](/Users/jack/src/etude/frontend/src/lib/socket.svelte.ts:66) reconnects, resumes an in-memory server session, and applies each full server update directly.
- Replay loads recorded full observations into the same board components and provides previous/play/next/scrub/speed controls.

The previous charter explicitly called this an **instrument** for learning what the bot is, obtaining a human-anchored benchmark, and producing human-game data; visual product polish and public hosting were marked out of scope ([`wave/game/legacy-gui-charter.md`](/Users/jack/src/etude/wave/game/legacy-gui-charter.md:1)). Therefore “Etude is visually plain” describes the old scope, not failed execution. The new Game wave retains the instrument's epistemic identity while making the game itself portable, reliable, and authored.

### Data flow

#### Phase local action path

Phase's most important experience invariant is an atomic `EngineSnapshot { state, legalResult, seq }`. A globally monotonic stamp is attached to snapshots; `gameStore` has a single sequence-gated writer that applies displayed state and corresponding actions together. The commit gate was added to prevent “split epoch” softlocks where a new state could be paired with old legal actions or vice versa ([`gameStoreCommitGate.test.ts`](https://github.com/phase-rs/phase/blob/553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d/client/src/stores/__tests__/gameStoreCommitGate.test.ts#L1-L130)).

The animated dispatch path roughly does this ([`dispatch.ts`](https://github.com/phase-rs/phase/blob/553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d/client/src/game/dispatch.ts#L192-L443)):

1. capture current object DOM positions;
2. optionally save a safe undo checkpoint;
3. submit the action to the engine/adapter;
4. obtain a new atomic snapshot;
5. save the new authoritative state before presentation;
6. normalize engine events into visual beats and audio cues;
7. keep rendering the old board while overlays animate toward the new state;
8. commit the new snapshot after the presentation duration, unless a newer sequence already won.

Local and remote updates share a module-level queue/mutex. The queue captures the current `WaitingFor` prompt object and drops an action if the prompt changed while it waited, a useful stale-interaction defense ([`dispatch.ts`](https://github.com/phase-rs/phase/blob/553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d/client/src/game/dispatch.ts#L517-L581)).

There are two architectural blemishes:

- components can use both `gameStore.dispatch` and the richer animated `dispatchAction`; comments/tests acknowledge both routes. Two mutation paths make it harder to guarantee identical ordering, persistence, animation, and error semantics.
- presentation completion is based primarily on accumulated durations/timeouts, not completion callbacks from actual visual work. That is deterministic and simple, but can diverge under background-tab throttling, reduced motion, or dropped frames.

At the WASM boundary, the Rust state getter serializes the client state to a JSON string and immediately calls JavaScript `JSON.parse`; the worker then `postMessage`s the resulting object to the main thread ([`engine-wasm/src/lib.rs`](https://github.com/phase-rs/phase/blob/553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d/crates/engine-wasm/src/lib.rs#L64-L76), [`engine-worker.ts`](https://github.com/phase-rs/phase/blob/553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d/client/src/adapter/engine-worker.ts#L114-L120)). Rust's borrowed `ClientGameStateRef` avoids cloning the internal game graph before serialization, but the complete client representation still incurs serialization, parse allocation, and worker structured clone. No delta, transferable binary format, or shared memory is used.

The state contains engine-authored `DerivedViews`: stack grouping/details, keyword badges, commander damage, player status/aura, remaining payment, auxiliary variants, and turn order. This is a good ownership decision: the engine supplies compact display meaning instead of forcing React to reinterpret all rules. The view builder still walks battlefield/state collections each time ([`derived_views.rs`](https://github.com/phase-rs/phase/blob/553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d/crates/engine/src/game/derived_views.rs#L194-L404)).

#### Etude live action path

The server emits a hero-normalized `Observation` plus `ActionOption[]`. Each option has an array index, type, description, and `focus` object IDs. The UI uses those IDs both to highlight cards when an action is inspected and to filter ambiguous target actions when a board card is selected. This is a compact and valuable **interaction offer** already hiding in the current design.

The client sends only `{type: "action", index}` ([`socket.svelte.ts`](/Users/jack/src/etude/frontend/src/lib/socket.svelte.ts:147)). The client does not disable existing actions while a normal action is in flight, attach the observed update sequence, provide a prompt ID, or use a command ID. Offline messages—including actions—are queued and flushed after reconnect ([`socket.svelte.ts`](/Users/jack/src/etude/frontend/src/lib/socket.svelte.ts:166)).

**Inference:** a rapid double click can send two indexes. If the first advances the engine to another action space, the same integer can now mean something else. Likewise, an action queued against an old decision can be flushed after a resumed fresh snapshot. WebSocket ordering does not prevent semantic staleness. This is the highest-priority experience correctness flaw because an apparently legal click can become a different legal click.

The villain search/step loop runs synchronously inside the FastAPI WebSocket flow. **Inference:** at larger search budgets or multiple sessions, CPU-bound search blocks the event loop, delaying heartbeats and unrelated clients. There is no structured thinking-progress event, deadline/cancel protocol, or interaction shell beyond the F6-specific `fastForwarding` flag.

### Key abstractions worth preserving or replacing

| Concern | Phase abstraction | Etude today | Recommended abstraction |
|---|---|---|---|
| Authoritative display | atomic state + legal-result snapshot + sequence | full observation + action list, applied together but no server revision | versioned `ExperienceFrame` with projection + offers + status |
| User choice | `WaitingFor` variants, legal actions, many special dialogs | positional `ActionOption` with description/focus IDs | stable semantic `InteractionOffer` IDs; direct manipulation and inspector share them |
| Command | adapter action call, prompt-aware local queue | `{action, index}` | idempotent `Command {command_id, expected_revision, prompt_id, offer_id, selection}` |
| Presentation | frontend normalizes raw events, captures DOM, timed queue | immediate state replacement + textual log | engine-authored `PresentationEvent` stream independent of state commit |
| Recovery | full snapshots, protocol versions, persisted sessions | process-local session + token in `sessionStorage` | versioned full `RecoveryEnvelope`, recent event tail, accepted command IDs |
| Replay | initial data/action log reconstructed in a replay worker | full observation/action snapshot at every event | commands + periodic checkpoints + research sidecar; deterministic hashes |
| Portability | WASM worker, WS, P2P, PWA, Tauri | Python/FastAPI + Vite dev orchestration | one compact wire contract across server/native worker/replay; WASM optional |

## Game-experience deep dive

### Render, update, animation, and large-board behavior

Phase does considerably more than render cards. Its event normalizer interprets engine events into presentation semantics: it drops non-visual events; separates casts and turn changes; merges related zone/life changes; pairs combat direction; groups deaths/sacrifices; collapses large token/counter runs; and converts sufficiently large player-hit sequences into a damage flurry ([`eventNormalizer.ts`](https://github.com/phase-rs/phase/blob/553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d/client/src/animation/eventNormalizer.ts#L1-L779)). The overlay layer uses Framer Motion for card movement/cast arcs, reveal/mill, death effects, particles, floating values, screen shake, and synchronized sound. Resolve-all has batching/progress logic and large storm sequences can be condensed.

This creates spectacle and readable pacing, but too much presentation meaning is reconstructed in TypeScript. The client needs deep knowledge of event combinations and fixes. A better contract would let the engine emit semantic, viewer-safe presentation intents such as “spell cast,” “three attackers declared,” “sweep destroyed these render IDs,” or “repeat counter addition 17 times,” while leaving exact timing/art direction to the client.

Before each animated action, Phase runs `querySelectorAll("[data-object-id]")` and calls `getBoundingClientRect()` for each visible object ([`animationStore.ts`](https://github.com/phase-rs/phase/blob/553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d/client/src/stores/animationStore.ts#L56-L64)). **Inference:** this is O(rendered objects), can force layout, and becomes an input-latency risk on distinct-permanent boards. Phase mitigates board explosion by grouping identical permanents and switching to overflow/scroll representations using distinct-stack thresholds, with `data-grouped-ids` preserving animation targets for unrendered group members. That is a strong technique, but it is not DOM virtualization and no quantitative large-board browser test was found.

Etude directly replaces a Svelte observation. It has no animation/presentation queue, audio, semantic batching, object-position transition system, token grouping, battlefield overflow mode, or level of detail. Cards wrap at fixed sizes. That is adequate for the current selected games, but “Phase parity” requires explicit budgets for distinct-permanent boards and event storms; it should not be answered by importing Phase's DOM-scanning approach.

### Interaction grammar and core flows

Phase's broad mechanic coverage is serious engineering. Its `WaitingFor` registry contains 123 unique player-facing variants. Tests parse the Rust enum and require every variant to be registered, then heuristically require a render/dispatch source for each ([`waitingForRegistry.ts`](https://github.com/phase-rs/phase/blob/553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d/client/src/game/waitingForRegistry.ts#L1-L189), [`waiting-for-handler-parity.test.ts`](https://github.com/phase-rs/phase/blob/553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d/client/src/adapter/__tests__/waiting-for-handler-parity.test.ts#L10-L75)). Unknown variants fail loudly in a diagnostic modal rather than silently softlocking. This **coverage invariant** is worth stealing even though the variant count and modal architecture are not.

The dialog host distinguishes:

- centered modal decisions;
- inline controls;
- click-through target/payment interactions that must leave the board interactive;
- narrow-screen sliding/peek overlays versus wide-screen side placement.

The comments and tests show attention to pointer stacking and hit-testing failures ([`DialogHost.tsx`](https://github.com/phase-rs/phase/blob/553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d/client/src/components/modal/DialogHost.tsx#L14-L215)). Source inspection shows flows for mulligans and hand-bottoming, priority/full-control/stops, alternate costs/modes/splice/targets, manual or automatic payment, attackers/blockers/damage, takebacks, game over, and replay. This supports “real platform.” The 3,373-line `CardChoiceModal` and numerous mechanic-specific branches support “expensive platform.”

Etude's core grammar is unusually coherent for its size: a board card and a textual legal action are linked by stable object IDs. The action sidebar always shows the truth about available decisions, and clicking a target narrows ambiguous options. Seven non-priority action-space kinds receive authored prompt copy; unknown kinds still fall back to generic legal buttons rather than becoming impossible ([`game.svelte.ts`](/Users/jack/src/etude/frontend/src/lib/game.svelte.ts:47)). MTGO-style stops and F6 eliminate priority churn. The missing piece is not a Commander-style universal UI; it is a finite **selected-deck interaction inventory** and an exhaustive contract that every offer in that inventory is executable and understandable.

The ideal polished form keeps the inspector as a first-class secondary view. The board can offer direct casting, targeting, drag/confirm, and payment previews, while the inspector exposes the exact same stable offer IDs for learning, debugging, accessibility, and trust. Phase mostly optimizes for playing Magic; Etude can uniquely optimize for understanding a machine playing Magic.

### Touch, keyboard, accessibility, and responsive behavior

Phase has a meaningful responsive/input layer:

- keyboard shortcuts for help, pass/confirm, full control, undo, turn pass, escape, and debug;
- a 500 ms / 10 px long-press recognizer with pointer capture and context-menu handling;
- hover styles guarded with `(any-hover: hover)` for hybrid devices;
- dynamic viewport, safe-area, short-landscape, and narrow-screen adaptations;
- reduced-motion accommodations in some overlays;
- widespread buttons and ARIA labels.

However, this is not proof of fully accessible play. The central `PermanentCard` is an interactive `motion.div` with `onClick` but no equivalent `role`, `tabIndex`, or keyboard handler ([`PermanentCard.tsx`](https://github.com/phase-rs/phase/blob/553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d/client/src/components/board/PermanentCard.tsx#L636-L663)). No automated axe/Lighthouse/accessibility browser suite was found.

Etude has a useful accessibility seed: the core [`Card.svelte`](/Users/jack/src/etude/frontend/src/lib/components/Card.svelte:1) uses a real button with an `aria-label`, and its selects/buttons/labels are mostly native controls. It has only the F6 gameplay shortcut, no focus management/live announcements, no safe-area/reduced-motion/touch inspection system, and its hover preview disappears below the `xl` layout with no long-press replacement. Some non-clickable cards are still enabled-looking buttons. Phase supplies patterns, but an explicit input-equivalence matrix is still required for either project.

### Images, asset caching, offline behavior, and updates

Phase's card image hook maps metadata to Scryfall URLs and keeps in-memory request/refcount, negative, and printing caches to avoid repeated fetch/rerender storms ([`useCardImage.ts`](https://github.com/phase-rs/phase/blob/553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d/client/src/hooks/useCardImage.ts#L1-L240)). Contrary to a README claim of IndexedDB image caching, the pinned service worker explicitly does **not** runtime-cache remote Scryfall images because a prior CORS caching incident broke them ([`vite.config.ts`](https://github.com/phase-rs/phase/blob/553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d/client/vite.config.ts#L303-L315)). Same-origin imagery, fonts, audio, WASM, metadata, and content-addressed card data receive PWA caching strategies. Thus Phase is robustly installable, but card art is not proven fully offline.

Its update code is notably battle-tested: it distinguishes initial service-worker control from updates, checks periodically and on visibility, breaks iOS reload loops, reports progress/errors, and defers activation/reload during multiplayer. A Vite preload-error handler has similar guarded self-recovery ([`registerServiceWorker.ts`](https://github.com/phase-rs/phase/blob/553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d/client/src/pwa/registerServiceWorker.ts#L1-L299), [`chunkReloadHandler.ts`](https://github.com/phase-rs/phase/blob/553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d/client/src/pwa/chunkReloadHandler.ts#L1-L81)). This is exactly the kind of invisible polish Etude should learn from.

Etude renders direct `api.scryfall.com/cards/named?format=image...` URLs with browser lazy loading and a text fallback ([`scryfall.ts`](/Users/jack/src/etude/frontend/src/lib/scryfall.ts:1)). It has no request dedupe, persistent manifest, service worker, or offline asset path. Existing E2E tests intercept card images with a one-pixel fixture, so they validate flow without validating production image availability.

Curated decks turn this weakness into an opportunity. For each allowed deck/version, Etude can build a content-addressed manifest containing exact printings, art crops, tokens, emblems, sounds, interaction schemas, and hashes. Preflight can guarantee every selected match's assets are locally available before “Play.” This can be more reliable and more art-directed than Phase's general Scryfall dependency without becoming a deck builder.

### Reconnect, persistence, replay, and failure modes

Phase versions its WebSocket and P2P protocols, sends explicit mismatch diagnostics, and reconnects with full viewer-filtered snapshots. P2P frames over 256 bytes use a small gzip envelope ([`network/protocol.ts`](https://github.com/phase-rs/phase/blob/553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d/client/src/network/protocol.ts#L188-L230)). Local/host games persist a trusted engine envelope to IndexedDB after actions; comments note game state can exceed 5 MB ([`gamePersistence.ts`](https://github.com/phase-rs/phase/blob/553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d/client/src/services/gamePersistence.ts#L84-L145)). P2P host sessions persist separately and offer a grace period; guests retry. WebSocket reconnect attempts are bounded. Engine panics/`STATE_LOST` have special classification and rehydration behavior.

The server and P2P steady-state protocols still send full snapshots, legal actions, derived views, and events rather than state deltas ([`server-core/protocol.rs`](https://github.com/phase-rs/phase/blob/553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d/crates/server-core/src/protocol.rs#L285-L345)). P2P does only top-level message validation. Full snapshots are a good recovery primitive and an easy correctness baseline; they are not necessarily the right forever transport for large states or spectators.

The engine-worker watchdog has an important compromise: gameplay operations report “slow” at 60 seconds but remain pending, AI has a longer timeout, and `resolveAll` deliberately has no timeout to avoid interrupting engine work ([`engine-worker-client.ts`](https://github.com/phase-rs/phase/blob/553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d/client/src/adapter/engine-worker-client.ts#L28-L50)). This protects correctness but permits an indefinitely stuck resolving overlay. A user-facing cancel/recover/resync state machine would be stronger.

Etude keeps sessions in an in-memory registry with a 15-minute TTL and stores credentials in browser `sessionStorage`. Reload/reconnect can recover while the Python process and registry survive; a server restart loses all sessions and a browser restart loses credentials. Client reconnect is indefinite exponential backoff capped at five seconds. Server messages are only checked for a recognized top-level `type` before a TypeScript cast ([`socket.svelte.ts`](/Users/jack/src/etude/frontend/src/lib/socket.svelte.ts:49)). There is no protocol/content/engine version or recovery from sequence gaps.

Etude's replay is excellent in product intent and expensive in representation. A trace records the full observation and legal action list before every event plus the final observation, as pretty, uncompressed JSON written directly to disk ([`gui/trace.py`](/Users/jack/src/etude/gui/trace.py:1)). This enables O(1)-style snapshot seeking in the browser and retains exact decisions, but grows approximately with turns × visible state, has no atomic write/schema migration, and mixes player replay material with research data. The trace API redacts hidden information by default but accepts an unauthenticated `reveal_hidden=true`; appropriate for a local instrument, unsafe if exposed as a hosted product.

The better design separates three needs:

1. deterministic playback: initial configuration + versioned stable commands + periodic checkpoints;
2. polished playback: optional semantic presentation events;
3. AI research: observations, legal masks, chosen action, policy/value/search diagnostics, reward, identity, and provenance in a sidecar dataset.

Phase has the more compact “reconstruct from actions in a replay worker” direction. Etude has the more important attribution and analysis direction. Preserve the latter and adopt the former's checkpointed reconstruction.

### Tests, CI, and evidence of polish

Phase's source contains substantial automated engineering gates: Rust formatting/lints/tests/card-data checks/WASM compilation/Tauri checks and frontend lint/type/unit tests. The client has 245 test files, including important cross-language `WaitingFor` parity and split-epoch regression tests. Release workflows cover web hosts, server artifacts, and desktop builds.

I found no Playwright/Cypress dependency or browser job, screenshot regression suite, automated accessibility audit, Lighthouse budget, or large-board browser benchmark at the pinned revision. Therefore source-level confidence in mechanics and state management is much stronger than evidence for visual polish, real-device input behavior, or performance.

Etude's local tests are small but live:

- **Observed:** `npm test -- --run`: 24 unit tests passed across 5 files.
- **Observed:** `npm run check`: 0 errors and 0 warnings.
- **Observed:** `npm run build`: succeeded; largest emitted client JS chunks were about 26.5 kB raw / 10.3 kB gzip, and client build took about 320 ms on this machine. SvelteKit warned that `adapter-auto` could not choose a production target.
- **Observed:** targeted Playwright `play.spec.ts` + `replay.spec.ts`: 2 tests passed in 11.1 seconds, covering a random full game and replay stepping. This was repository automation, not hands-on browser QA.
- **Observed:** `uv run python -m pytest tests/gui -q`: 27 tests passed in 2.10 seconds. The module form avoids a stale `.venv/bin/pytest` entry-point shebang while preserving the repository's uv-managed interpreter.

The current GitHub CI runs engine Rust/Python work but does **not** install/check/test/build `frontend` or run `tests/gui` ([`.github/workflows/ci.yml`](/Users/jack/src/etude/.github/workflows/ci.yml:1)). The working game experience can regress outside local checks. Adding existing tests to CI is a prerequisite, not Phase imitation.

## Experience contract for a curated, AI-native Etude

This is the proposed non-negotiable contract. It deliberately does not promise generic Commander, arbitrary decks, or deck construction.

1. **Fast first meaning:** from launch or deck selection to the first understandable decision is bounded and measured; exact-match assets can prefetch before the match.
2. **Atomic decisions:** every displayed authoritative frame includes the legal offers for exactly that revision and prompt. No mixed epochs.
3. **Stable commands:** every command identifies match, expected revision, prompt, stable offer, and an idempotent command ID. A stale or duplicate command is rejected, never reinterpreted.
4. **Complete selected-deck grammar:** every prompt reachable by an allowed deck pairing has an authored or intentionally generic interaction renderer. Unknown interactions fail loudly with diagnostics; they do not softlock.
5. **Two views, one truth:** direct board manipulation and the decision inspector are projections of the same `InteractionOffer`s. The inspector remains available for learning, accessibility, QA, and AI analysis.
6. **Input equivalence:** every legal decision is performable and understandable with mouse, touch, and keyboard; state changes and errors are announced; reduced motion preserves semantics.
7. **Truth does not wait for theater:** authoritative state advances independently of animation. Presentation can play, accelerate, skip, or recover from interruption without changing game truth.
8. **Semantic presentation:** transitions consume viewer-safe semantic presentation events, not guesses from arbitrary state diffs. Stable render IDs distinguish object incarnations.
9. **Legible AI:** the UI identifies the actual policy/search/checkpoint and deterministic mode, shows one bounded thinking state per decision composition, stays responsive, and supports deadline/cancel/recovery.
10. **Exact recovery:** disconnect, reload, sleep/wake, or update returns to the exact current decision. Gaps cause explicit resync from a full snapshot; updates never strand a live match.
11. **Research-grade replay:** a completed game is deterministically reconstructible and attributable to decks, engine/content versions, seed, policy/checkpoint hash, search budget, decisions, rewards, and human/bot actor.
12. **Curated offline completeness:** a chosen supported matchup can be made fully playable with its exact art, tokens, fonts, sounds, and data offline. Missing assets fail preflight.
13. **Bounded board cost:** identical objects group, distinct-object rendering has level-of-detail/virtualization thresholds, and large event runs condense without hiding meaningful decisions.
14. **Portable frame contract:** web/server, desktop/native worker, replay, and a possible future WASM implementation consume the same versioned experience contract; transport is replaceable.

## Proposed new technical approach

Phase's strongest ideas and Etude's strongest identity fit into four wire concepts:

Concrete types and acceptance/client flows are worked through in the
[experience protocol code sketch](experience-protocol-code.md).

```text
rules + AI authority
    │
    ├── ExperienceFrame ───────────┐
    ├── PresentationEvent[] ───────┼──> thin experience client
    └── RecoveryEnvelope ──────────┘       ├── direct board manipulation
           ▲                              ├── decision inspector
           │ Command                      ├── animation/audio scheduler
           └──────────────────────────────└── replay/analysis overlays
```

### 1. `ExperienceFrame`

```text
match_id, revision, prompt_id, viewer
public entity projection keyed by stable render_id/object incarnation
zones, players/HUD, priority/phase, display-derived groups/status
InteractionOffer[]
transient authority status: ready | thinking | resolving | reconnecting
content/asset manifest hash
```

This is not the complete internal `GameState`, not training tensors, and not a handwritten mirror of every Rust type. Generate the schema and bindings from one definition (for example JSON Schema/TypeSpec/Protobuf plus conformance fixtures). Use a full frame for boot/recovery/checkpoints. Only introduce patches after instrumentation proves full-frame bandwidth/parse/commit is a bottleneck; patches must carry base and result revisions and fall back to resync.

### 2. `InteractionOffer` and `Command`

An offer contains a stable `offer_id`, semantic verb, source and eligible render IDs, count/ordering constraints, authored display copy, payment/preview capability, and keyboard/touch intent. The client replies with:

```text
Command {
  command_id, match_id, expected_revision, prompt_id,
  offer_id, selection
}
```

The authority accepts a command once, rejects stale/duplicate/invalid commands explicitly, and returns the accepted command ID in the next frame. This preserves Etude's action clarity while removing positional-index ambiguity.

### 3. `PresentationEvent`

Events carry sequence, causing command ID, semantic intent (`cast`, `move_zone`, `damage`, `die`, `tap`, `reveal`, `attack_group`), actor/source/destination render IDs, importance, pacing group, and optional sound cue. The client owns timing, composition, reduced motion, skip, and level of detail. The authority owns game meaning. This avoids Phase's large frontend event interpreter without coupling the engine to CSS animations.

The client maintains three separate layers:

- last committed authoritative frame;
- local disposable interaction draft;
- presentation queue/overlay.

Animation may visually bridge frame N to N+1, but reconnect or “skip” commits N+1 immediately. Do not scan/layout-read every card by default; capture the source/destination elements named by the event, and use CSS/layout transforms or a retained render registry.

### 4. `RecoveryEnvelope`

```text
protocol/content version + match ID
complete current ExperienceFrame
recent PresentationEvent tail
last accepted command IDs
replay/checkpoint cursor
```

Use the full envelope for reconnect/resync. Keep steady-state optimization optional. Store it server-side or in a native authority with a durable match lease; `sessionStorage` may point at it but must not be the only continuation record. Updates and transport reconnects defer presentation, not authoritative recovery.

### Scaling posture

The key scaling choices should be similar to Phase only at the invariant level:

- **Same:** clone-friendly client projection, worker/process isolation, stable object incarnations, derived display views, full-snapshot recovery, grouping, background AI, and adapter seam.
- **Different:** generated narrow schema instead of a 3,057-line manual mirror; semantic stable offers instead of 123 UI prompt classes; targeted presentation events instead of TS event archaeology; command/revision identity at the wire; full-frame first with measured patch escape hatch instead of unexamined full JSON copies; exact-deck asset packs instead of a world card database; AI server/native authority first, optional WASM rather than WASM as ideology.

WASM is a distribution choice, not the experience architecture. Phase demonstrates that local browser rules can be portable, but also demonstrates the memory/startup/copy costs of a large WASM + full card database. Etude's learned agents and search are likely better isolated in a server or native worker initially. Keep the experience frame portable enough that a smaller rules-only WASM or selected-deck local bundle can arrive later without rewriting the client.

## Core-flow experience matrix

| Flow | Phase at pinned source | Etude now | Curated target |
|---|---|---|---|
| Cold start | WASM worker/fallback, large DB initialization, progress/error paths, shared warm adapter; PWA/Tauri | `scripts/play.py` starts backend + Vite dev; tiny client, backend required | fast shell; explicit AI/authority readiness; exact-match asset preflight; measured cold/warm first-decision budget |
| New match | broad format/deck/AI/local/online/P2P setup | selected named decks + search/checkpoint/random/passive selectors | deliberately small authored matchup gallery; policy identity visible; no general deck builder |
| Mulligan | dedicated flows including bottoming/special cases | engine actions rendered generically; no richly authored mulligan surface found | authored only for reachable selected-deck rules, covered end to end |
| Priority/stops | priority controls, full control, stops, resolve-all | strong MTGO stops, auto-pass, F6/pass-turn | preserve stops; make priority state spatially obvious; one busy/resolve composition |
| Cast/target/payment | many specialized dialogs plus board click-through/manual payment | readable action list + focus IDs; direct target filtering; generic choices | stable offers drive both direct board interaction and inspector; authored previews/payment for finite inventory |
| Combat | dedicated attacker/blocker/damage interactions and animations | legal actions through inspector/board; state snaps | board-first group selection, combat preview, inspector equivalence, semantic beats |
| Resolution | normalized event queue, effects/audio, batching/skip | direct snapshot + log | authority emits presentation events; skippable/accelerable without delaying truth |
| AI turn | worker pool/local heuristics; memory-conscious phone fallback | real search/checkpoint/passive/random server-side; no general progress/cancel | retain policy identity/provenance; isolated deadline-bound worker; responsive thinking composition |
| Disconnect/reload/update | versioned protocols, full snapshot, persisted host/local, guarded PWA updates | in-process 15-minute session; `sessionStorage` token; no revision/resync/update model | durable recovery envelope; gap detection; idempotent commands; live-match-safe update |
| Game over | broad postgame/rematch surfaces | winner + trace/replay availability | concise result plus policy/deck provenance and immediate analysis/replay |
| Replay | reconstructs actions in dedicated worker | first-class full-snapshot timeline, same board | deterministic command replay + checkpoints + presentation + AI research sidecar |
| Offline | PWA assets/WASM/data cached; remote card art not guaranteed | none; remote Scryfall + Python server | exact selected matchup fully cacheable; optional native/local authority |
| Mobile/a11y | responsive/touch/shortcuts, partial ARIA; core permanent keyboard gap | semantic card buttons; little touch/responsive equivalence | all offer types covered by mouse/touch/keyboard; focus/live announcements; reduced motion |
| Large board | identical grouping/overflow/event collapse; no measured browser budget | fixed wraps; no grouping/LOD | selected-deck adversarial fixtures; grouping + LOD/virtualization + perf gates |

## Comparison matrix: what matters

| Axis | What Phase does that matters | What Etude already does that matters | Direction |
|---|---|---|---|
| Client architecture | separates authoritative, interaction, animation, preference, multiplayer state; adapter seam | very small shared live/replay components; comprehensible single-creator system | adopt separation and adapters without framework-scale proliferation |
| State contract | atomic state/legal snapshot, monotonic commit gate, engine-derived display views | full hero-normalized observation + legal actions already arrive together | add authority revision/prompt/command identity and generated compact schema |
| Object identity | object IDs/grouped IDs support targeting and transitions | focus object IDs bind action inspector to cards | promote to stable render/incarnation IDs across frames and presentation events |
| Presentation | semantic-looking animation/audio/batching/overflow machinery | readable action/log truth; no theater that can lie | preserve truthfulness, add engine-authored presentation intent and authored pacing |
| Reliability | protocol versions, full reconnect snapshots, persistence, engine-state recovery, update guards | resume token/reconnect and replay exist | durable recovery, runtime validation, idempotency, resync, existing tests in CI |
| Portability | web/PWA/server/P2P/Tauri | tiny web bundle but requires dev server + Python authority | thin web/PWA first; packaged native authority next; optional selected-deck WASM later |
| AI identity | game AI is a feature, largely fixed/hand-engineered | AI is the product: checkpoint/search budget/determinism/baselines | preserve and make policy/search provenance central in live and replay UX |
| Research | player replay/save | exact configs and per-decision traces/rewards/actors | keep attribution; split compact playback from analysis data |
| Scope | arbitrary-card/format/multiplayer generality | curated decks and finite interaction surface | use constraint as an artistic/QA/performance advantage |
| Maintainability | massive broad client with parity tests | ~3.2k frontend lines and one clear grammar | generated contracts, small primitive vocabulary, deck-specific data—not deck-specific component forests |

## What Etude must keep doing that Phase does not

1. **AI as the work, not a convenience opponent.** Search budgets, checkpoints, deterministic mode, and passive/random baselines correspond to actual research systems ([`game.svelte.ts`](/Users/jack/src/etude/frontend/src/lib/game.svelte.ts:20), [`gui/villain.py`](/Users/jack/src/etude/gui/villain.py:1)). A polished UI must not flatten them into “Easy / Medium / Hard.”
2. **Attributable games.** Traces preserve exact deck lists/names, villain type/sims/checkpoint/determinism, seed/stops, observations, legal actions, chosen index/description, reward, and actor. Improve the storage format; retain the provenance contract.
3. **Decision legibility.** The visible legal-action inspector linked to board objects answers “what exactly can I do?” and makes engine behavior inspectable. It should survive as a player aid, learning layer, debugging console, and accessibility route.
4. **Replay as an AI-learning surface.** The trace browser and timeline are already part of the product, not a late match-viewer feature. Extend them with policy scores, search alternatives, counterfactuals, and competency hypotheses.
5. **Curated scope as taste.** Exactly selected decks provide a finite asset set, finite prompt inventory, exhaustive golden-flow matrix, authored visual language, and stronger performance bounds. Do not import Phase's deck builder, format legality, or Commander dashboard merely because the infrastructure can support it.
6. **A light, comprehensible client.** The current production output is tens of kilobytes of JS rather than a roughly 48 MB engine plus large universal data set. Heavy AI/rules can remain off-main-thread and off-client while the shell stays fast.
7. **Single-creator coherence.** Etude can define a dozen orthogonal interaction primitives and compose them artistically. It should not grow one component per Magic mechanic until it resembles Phase's giant switches.
8. **The broader training/evaluation loop.** Vector environments, learned policy checkpoints, search, experiment provenance, and human benchmark design are outside Phase's core product. Polish must serve that loop rather than displacing it.

## Tensions

### Generality versus authorship

Phase proves that arbitrary Magic needs many escape hatches. Etude's refusal to allow arbitrary decks is not a missing feature; it is the mechanism that makes exhaustive interaction polish possible for one creator. The risk is allowing “curated” to become hard-coded deck-name branches. Keep mechanics in typed engine/experience primitives and keep *selection, assets, authored copy, camera, sound, and pacing* in content manifests.

### Full snapshots versus scaling

Both projects favor full viewer snapshots, which are excellent for correctness, cloneability, reconnect, debugging, and deterministic tests. Phase's path serializes/parses/structured-clones a large representation; Etude repeats full observations in wire messages and traces. Optimize only after measuring. The durable invariant is “a complete frame is always available”; the optimization may be structural sharing, binary encoding, patches, or checkpoints.

### Truth versus animation

Phase keeps the old visual board during an animation and commits afterward; Etude commits immediately with no transition. The new runtime should commit authoritative truth independently while allowing a presentation layer to display a controlled transition. Input targets must always bind to the authoritative prompt, and skip/reconnect must instantly converge.

### Local portability versus AI capability

WASM makes Phase self-contained but produces startup/memory/copy challenges. Etude's serious learned/search opponents naturally fit a server or native process. “Portable” should first mean a stable client contract plus web/PWA and packaged desktop authority—not forcing all AI into browser WASM. Selected-deck local rules can be a later transport implementation.

### Inspector clarity versus visual magic

Direct manipulation can hide available legal choices; an action list can feel like operating a debugger. Treat them as synchronized views of one offer model. Default to the authored board, allow the inspector to explain/highlight, and make it central in analysis/replay mode.

## Observations

### Complexity

- **Observed:** Phase's client breadth is real and strongly tested at unit/source level, but its largest files and 123 prompt variants show that generic mechanic-by-mechanic UI does not preserve single-creator agility.
- **Observed:** Phase has useful scale mitigations—worker isolation, grouping, event collapse, shared warm initialization—but still performs full state serialization/copy and broad DOM measurement.
- **Observed:** Etude's client is dramatically smaller and already shares passive board components between live/replay. Its protocol and trace formats, not its component count, are the first scaling hazards.

### Quality

- **Observed:** Phase's commit gate, prompt-aware queue, fail-loud registry, PWA update defenses, protocol mismatch handling, and reconnect snapshots are platform-grade patterns.
- **Observed:** Phase lacks browser-level evidence for the very qualities the user wants to match: no visual regression, accessibility, real-device, or performance gates were found.
- **Observed:** Etude's local unit/type/build/targeted E2E/GUI tests pass, but CI does not protect the frontend or GUI server.
- **Inference:** Etude's revisionless action indexes can execute unintended decisions after double click/reconnect; this is a correctness defect disguised as an interaction detail.

### Potential

- **Inference:** curated deck packs allow Etude to outperform Phase on cold asset certainty, interaction completeness, authored animation, and accessibility because the reachable surface is enumerable.
- **Inference:** an AI-native presentation stream can outperform Phase's frontend event reconstruction while making replay and policy analysis richer.
- **Inference:** preserving the inspector means Etude can be simultaneously more beautiful and more intellectually honest than clients that hide the action model.

## Risk register

| Risk | Probability | Impact | Evidence | Mitigation |
|---|---:|---:|---|---|
| Stale/double-click action index executes against a new prompt | High | Critical | action command contains only index; actions persist in UI; offline queue flushes after resume | revision/prompt/offer/command IDs; one in-flight command; reject stale/duplicate; clear old queue on reconnect |
| CPU-bound AI blocks FastAPI event loop and UI status | High as search/multi-session grows | High | synchronous villain/search advance in WebSocket session | dedicated process/worker pool; deadline/cancel; progress/heartbeat; per-match serialized authority |
| Session disappears on backend restart | Medium | High | in-memory registry + browser `sessionStorage` | durable recovery envelope/checkpoints; lease-based sessions; explicit unrecoverable state |
| Trace growth, corruption, version drift, or hidden-data exposure | High | High if hosted | full pretty observation per event; direct write; reveal query | versioned append/atomic finalize; command+checkpoint replay; research sidecar; auth/capability for hidden view |
| Card art fails offline or under remote service behavior | High | Medium | direct Scryfall URLs; no SW/manifest | selected-deck content-addressed asset packs; preflight; local fallbacks; printing pinning |
| Large board or event storm freezes UI | Medium | High | fixed wrap/no grouping in Etude; Phase's DOM scan/full copies show likely future cost | adversarial deck fixtures; grouping/LOD; targeted element registry; event condensation; measured budgets |
| Copying Phase creates type/modal explosion | High if imitation is literal | High | 3,057-line mirror, 3,373-line modal, 123 prompt variants | generated narrow wire schema; orthogonal offer primitives; selected-deck reachability gate |
| Full snapshot JSON/copy dominates state update | Medium | High at large state/frequency | Phase stringify→parse→worker clone; both send full state | instrument bytes/serialize/parse/commit; preserve full resync; add binary/patch only when justified |
| Animation and authoritative state race | Medium | High | Phase dual dispatch and timeout-based presentation; Etude no command identity | single command pipeline; separate authoritative/presentation layers; event sequence; skip/resync invariant |
| Visual/input/performance regressions ship | High today | High | Etude frontend absent from CI; Phase lacks browser-quality gates | add existing tests now; then screenshot/a11y/perf/network/interruption suites |
| Curated deck definitions drift across server/client/assets | Medium | Medium | current server/client selected-deck knowledge is partly duplicated | one versioned matchup manifest generated into bindings and asset build |
| Update or cached-version mismatch strands a match | Medium once PWA ships | High | Phase needs extensive mitigation; Etude has no update protocol | protocol/content handshake; defer activation during match; recovery envelope; circuit breaker |
| “Accessible controls” mask an inaccessible board | Medium | High | Phase core permanent is click-only div; Etude lacks full input-equivalence tests | offer-level keyboard/touch mappings; semantic renderers; focus/live-region tests; axe + manual screen-reader pass |

## Proof and benchmark plan

Do not declare “equal or better than Phase” from code review. First obtain a full Phase checkout/build and run the same instrumented scenarios. Where Phase cannot execute a selected Etude matchup, compare the common experience mechanics, not card coverage.

### Measurements

1. **Boot:** cold/warm time to interactive shell, authority ready, assets ready, and first meaningful decision on a low-end phone profile and desktop. Record download, parse, heap, and cache state.
2. **Decision latency:** input→command-send, command→authoritative frame, frame→first visual response, and frame→settled presentation p50/p95/p99.
3. **AI responsiveness:** main-thread INP/long tasks while search runs; thinking-state update cadence; deadline and cancel recovery.
4. **Transport/state:** bytes per action, serialize/parse/validation/commit time, complete-frame size, patch hit/fallback rate if patches are later added.
5. **Memory:** heap after first match, ten sequential matches, replay/live switching, and reconnect/update cycles; verify worker/asset release.
6. **Boards:** 20/100/500/1,000 distinct permanents and 10,000 identical tokens. Measure DOM nodes, input latency, layout/paint, frame consistency, and inspectability.
7. **Event storms:** 1/100/1,000/5,000 presentation events. Verify condensation, skip, progress, no lost semantic beat, and no authoritative blockage.
8. **Recovery fault injection:** disconnect before command, after accept/before frame, during animation, during AI thought, on frame gap, duplicate command, reordered frame, browser reload, server restart, sleep/wake, and update mid-match.
9. **Replay:** 100/1,000/10,000 decisions; file size, load, random seek, checkpoint reconstruction time, deterministic state hash, migration/error behavior.
10. **Assets:** fresh install then network-off for every allowed matchup; verify exact printing, tokens/emblems, fonts/audio, fallbacks, and update invalidation.
11. **Input:** every reachable interaction primitive by mouse, touch, keyboard, reduced motion, high zoom, and screen reader on phone/tablet/desktop sizes.

### Minimum proof gate before a “Phase-parity” claim

- instrumented Playwright flows for every interaction primitive reachable by every selected deck pairing;
- screenshot baselines at desktop, tablet, phone portrait, and short landscape for each major frame/prompt;
- automated axe checks plus manual keyboard/touch/screen-reader passes;
- production-build performance traces and enforced frame/INP/DOM/heap/network budgets;
- offline and constrained-network runs with exact asset packs;
- 100-game soak with randomized disconnect/reload/duplicate-command/animation-skip injection;
- deterministic replay hash checks across current and previous supported schema/content versions;
- frontend build/unit/E2E and `tests/gui` in CI immediately, before the visual rebuild.

The first benchmark should capture Phase as it exists rather than choosing flattering Etude targets. The next step is to turn those distributions into explicit budgets. No Phase runtime number in this report is measured.

## Recommendations

### 1. Freeze the experience protocol before expanding visual mechanics

**Observation:** Etude's action index has no revision/prompt identity, while Phase's atomic commit gate solves a closely related class of softlocks.
**Cost:** medium; touches Rust/Python serialization, WebSocket messages, traces, store, and tests.
**Benefit:** prevents unintended commands and gives every later interaction/animation/reconnect feature a sound foundation.
**Verdict:** **do first.** Introduce `ExperienceFrame`, stable offers, command IDs, expected revision, prompt ID, runtime validation, and explicit stale/duplicate responses.

### 2. Preserve the inspector and make direct manipulation another projection

**Observation:** Etude's focus-linked action list is a differentiator; Phase's broad direct UI often needs many mechanic-specific branches.
**Cost:** medium.
**Benefit:** combines smooth play with exact legibility, accessibility, and AI analysis.
**Verdict:** **core product principle.** Never create a board gesture that bypasses or invents semantics outside `InteractionOffer`.

### 3. Add a semantic presentation stream, not state-diff animation archaeology

**Observation:** Phase's polished effects require a large client event normalizer and DOM capture; Etude currently has no transitions.
**Cost:** medium-high across engine/experience boundary and art direction.
**Benefit:** authoritative, replayable, skippable animation with a small client vocabulary.
**Verdict:** **build after protocol identity.** Start with move/cast/damage/die/tap/reveal/attack-group and composition/importance metadata.

### 4. Treat selected decks as compiled experience packs

**Observation:** deck/format generality is not a goal; remote art and unbounded prompts undermine reliability.
**Cost:** medium build tooling/content work.
**Benefit:** exact offline assets, exhaustive prompt coverage, authored visuals/audio/copy, reproducible tests, smaller downloads.
**Verdict:** **high leverage.** One manifest should generate server selection, client metadata, asset preload, reachability fixtures, and version hashes.

### 5. Isolate the authority without prematurely forcing WASM

**Observation:** Phase gains local portability from WASM but pays large startup/memory/copy costs; Etude's learned/search agents are naturally process-heavy.
**Cost:** medium-high for worker/process lifecycle and packaging.
**Benefit:** responsive UI, bounded AI, web/native portability, and an eventual optional WASM transport.
**Verdict:** **server or native worker first; WASM later if measured/product need justifies it.** Keep one adapter/frame contract.

### 6. Separate replay, presentation, and research records

**Observation:** Etude's full-snapshot traces are attributable but large and fragile; Phase reconstructs playback from actions.
**Cost:** medium.
**Benefit:** compact deterministic replay, fast checkpointed seek, schema migration, and richer AI diagnostics without leaking hidden data.
**Verdict:** **preserve the data ambition, replace the container.** Initial config + stable commands + checkpoints; presentation log optional; research sidecar access-controlled.

### 7. Adopt Phase's invisible reliability patterns selectively

**Observation:** full-snapshot reconnect, protocol mismatch errors, fail-loud handler coverage, PWA update deferral, reload circuit breaking, and overflow grouping are proven-by-source investments.
**Cost:** incremental but nontrivial.
**Benefit:** the “smoothness” users notice mostly when something goes wrong.
**Verdict:** **steal these patterns.** Implement against Etude's smaller contract; do not copy the surrounding generic product surface.

### 8. Put today's working experience under CI, then add perceptual gates

**Observation:** local Etude checks pass but CI omits the frontend and GUI; Phase's own weak spot is browser-level proof.
**Cost:** low immediately, medium for fixtures/performance stability.
**Benefit:** converts polish and reliability from taste claims into release criteria.
**Verdict:** **immediate.** Add existing build/unit/E2E/GUI tests, then screenshot, accessibility, interruption, offline, and performance budgets.

## Open questions

1. Which exact deck pairings form the first “compiled experience pack,” and what complete set of reachable interaction primitives do they imply?
2. Is the near-term product primarily local desktop, hosted single-player web, or both? This changes authority persistence/packaging, not the client contract.
3. Which parts of an observation must be available for player display versus AI analysis? The public experience frame should not inherit the research tensor/state schema.
4. What is the canonical stable object-incarnation ID across transform, blink, copy, token grouping, control change, and replay reconstruction?
5. How much semantic pacing should originate in engine events versus authored per-card/per-match content? The engine should state meaning; the experience pack may state emphasis.
6. Should AI thinking expose only policy identity and elapsed budget, or also safe live search diagnostics? Analysis richness must not leak hidden information into play.
7. What replay compatibility window matters? Pin exact engine/content builds forever, migrate schemas, or support only artifacted releases?
8. Which Phase runtime baselines emerge once a complete build can be launched on the same machines? Source comments are not sufficient performance evidence.

## Bottom line

Phase's game experience is not a mirage. It is a credible general platform with several excellent invariants and a huge breadth tax. Etude should aspire to its reliability, portability, response discipline, and invisible lifecycle polish—not to its product breadth or component count.

The artistic opening is to make the restriction to selected decks do real technical work: compile exact assets and interactions; prove every path; present AI identity honestly; retain the decision inspector; make replays research-grade; and give a small semantic event vocabulary exceptional animation, sound, responsiveness, and recovery. With a versioned experience frame and stable offers beneath it, that can be both **more agile than Phase and better at the game experience it intentionally chooses to provide**.
