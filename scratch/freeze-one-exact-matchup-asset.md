# W2-185: Freeze one exact matchup asset pack with offline fallback

## Decision and user-visible outcome

Freeze the current default play-table matchup as
`tla-ur-lessons-vs-gw-allies@1.0.0`: **UR Lessons** is the hero deck and
**GW Allies** is the villain deck. This interpretation follows the server and
client defaults and the Milestone-1 play-interface documentation; the legacy
named `interactive` mirror remains selectable but is not part of this pack.
The source matchup is 41 cards for UR and 40 for GW; the manifest preserves
that current behavior rather than silently changing the UR deck while packaging
it.

After this Task, a player can open the existing table, start the default match,
play, reload, and view its replay while all visible card faces come from the
installed pack. Play makes no Scryfall or other public-network request. The 29
deck identities plus the reachable `Ally` and `Clue` token identities receive
local authored treatments. If any identity is absent or malformed, the player
still sees a stable name-bearing fallback tile and can continue playing.

Version 1 deliberately contains no third-party card scans or art crops. A
manifest-defined treatment is the asset for each identity. This is the smallest
authored pack seam allowed by the Task's “art/crops or treatments” scope and
avoids treating Scryfall delivery as a redistribution license. The pack states
that third-party art is absent and records rights/provenance without claiming a
license the repository does not have.

## Source of truth

Add one committed manifest at
`frontend/src/lib/packs/tla-ur-lessons-vs-gw-allies/v1/manifest.json` and its
pack-local `NOTICE.md`. The manifest is authoritative for:

- schema version, pack ID, semantic pack version, and display title;
- oriented matchup identity and exact `{engine card name: count}` deck lists;
- the complete reachable identity inventory: both decks plus `Ally` and `Clue`;
- one local treatment recipe per identity;
- the deterministic fallback recipe and its version;
- identity-data provenance and treatment/rightsholder/license metadata.

The manifest must use stable keys, not array position, and must distinguish
`card` from `token`. Real cards pin their Scryfall Oracle identity (Oracle ID,
canonical name, source URI, and retrieval date); the two engine tokens pin a
managym-local token identity. A treatment includes only authored presentation
data such as palette, motif, and stable seed. It contains no remote URL.

Every rights record explicitly includes `asset_kind`, `contains_third_party_art`,
`creator`, `copyright_notice`, and an SPDX expression or `NOASSERTION`.
`NOASSERTION` is required for the authored treatment until the repository has
an applicable project license; it must not be rewritten as MIT or public domain
by implication. `NOTICE.md` identifies Magic/card identity ownership and links
the applicable Wizards fan-content policy, identifies Scryfall as the source of
identity metadata rather than a rights grant, and states that the pack is
unofficial. Per-printing artist attribution is excluded because v1 ships no
printing art.

Product views are derived from this record:

- a small Python loader supplies the two named deck definitions, display names,
  pack reference, and manifest SHA-256 to `gui.server`;
- a TypeScript loader supplies the same deck choices/defaults and card-treatment
  resolver to live play and replay;
- the existing `manabot.verify.util` deck constants remain an engine/test mirror
  and must be sync-tested against the manifest, not treated as a competing GUI
  source of truth;
- a match trace records `{id, version, manifest_sha256}` so asset provenance is
  durable even after a later pack version is installed.

The manifest hash is SHA-256 of the committed UTF-8 manifest bytes. It is
computed by the backend loader and verification code, not stored inside the
hashed document.

## Build target

Keep this as one focused serial PR.

1. Add strict Python and TypeScript manifest loaders. Reject duplicate identity
   keys, non-positive deck counts, missing deck identities, missing reachable
   tokens, remote treatment URLs, incomplete rights data, or a fallback version
   unknown to the client.
2. Derive `ur_lessons` and `gw_allies` server/client deck metadata from the
   manifest. Preserve the existing `new_game.config.hero_deck` and
   `villain_deck` inputs, display names, selector shape, and legacy/custom deck
   compatibility.
3. Replace `scryfallImageUrl()` use in `CardImage.svelte` with a local treatment
   resolver shared by live play, hover preview, and replay. A pack treatment
   renders a recognizable, name-bearing face from committed palette/motif/seed
   data. An unknown or invalid identity renders `fallback-v1`: hash the exact
   UTF-8 engine name with FNV-1a 32-bit, select from the manifest's fixed
   palette/pattern table, and render the escaped name plus existing power/
   toughness overlay. Neither path performs I/O or retries a remote source.
4. Add an additive `asset_pack` object (`id`, `version`, `manifest_sha256`) to
   observation/game-over payloads and current traces when the oriented default
   matchup is selected. Update frontend DTOs to accept it. Older payloads and
   traces with no field remain valid; other deck combinations report
   `asset_pack: null` and use installed identity treatments where available,
   then deterministic fallback.
5. Remove the runtime Scryfall URL helper and the E2E Scryfall response stubs.
   Do not add a replacement fetcher. Add the focused manifest, resolver, wire,
   replay, and offline tests below.

## Affected surfaces and compatibility

- **Backend authority:** `gui/server.py` named-deck/default resolution and all
  observation/game-over payloads; a new focused loader owns manifest validation.
- **Trace/replay:** `gui/trace.py` persists the pack reference. Existing trace
  JSON without it still loads and renders. A trace referring to another or
  missing installed version retains that provenance and renders an installed
  treatment when its exact card name is present, then deterministic fallback;
  it never fetches or refuses replay.
- **Frontend selection:** `frontend/src/lib/decks.ts` derives the two curated
  deck entries and defaults from the manifest. The existing `interactive`
  option remains as a legacy entry and receives local fallback treatment for
  identities outside this pack.
- **Frontend card consumers:** `CardImage.svelte`, `Card.svelte`, hand,
  battlefield, graveyard, exile, stack, target hover preview, live table, and
  replay all use the same resolver. Card backs and hidden-information behavior
  do not change.
- **Wire DTOs:** `new_game` input is unchanged. `observation` and `game_over`
  gain only the optional/additive pack reference; old servers/recordings remain
  accepted.
- **Automation:** GUI sync tests, frontend unit tests, existing browser tests,
  and a dedicated clean-cache/offline browser test consume the manifest. Build
  and launch commands remain unchanged.
- **Engine/training:** no Rust card behavior or observation change. Training
  deck mirrors are checked for drift but asset presentation is not imported
  into the engine.

## Absent and error states

- Missing, unreadable, or schema-invalid manifest at backend startup makes the
  curated default pack unavailable with a precise startup error. The server
  must not silently substitute a stale hard-coded default deck.
- Missing deck identity, token identity, treatment recipe, provenance, or rights
  metadata fails manifest verification and the frontend build/unit gate.
- A malformed treatment encountered despite those gates renders `fallback-v1`
  and emits at most one local diagnostic; it never attempts Scryfall.
- A card outside the pack (legacy `interactive`, a custom wire deck, or an old
  trace) is a supported absent state and always receives `fallback-v1`.
- Missing `asset_pack` on an old payload/trace means “unversioned legacy
  presentation,” not corruption. Unknown pack ID/version/hash remains recorded
  provenance; replay resolves exact installed names and otherwise remains
  legible through fallbacks.
- Empty hand, empty zones, hidden opponent cards, unavailable localStorage, and
  WebSocket recovery keep their current semantics.

## End-to-end proof

Add `frontend/e2e/offline-pack.spec.ts`. It starts with a fresh Playwright
browser context (empty HTTP cache and storage), permits only loopback frontend,
API, and WebSocket traffic, and aborts every public-network request. Through the
real UI it starts the default match against the random opponent, observes
`UR Lessons vs GW Allies`, asserts visible known cards report pack-backed
treatments rather than fallback, takes legal actions, reloads/resumes, and opens
the resulting replay. The test fails on any Scryfall/public-network request,
missing treatment, page/console error, or loss of the installed pack after
reload.

Exhaustive unit/contract checks complement that scenario:

- validate the exact current 41-card/40-card decks, all 31 reachable identities,
  the two engine tokens, metadata completeness, and absence of remote URLs;
- verify backend named decks and `manabot.verify.util` mirrors equal the
  manifest and that payload/trace pack references use the manifest hash;
- resolve every identity through the TypeScript treatment loader and snapshot
  `fallback-v1` for a missing name twice to prove determinism;
- retain live/replay component tests so both projections use the same resolver.

Verification commands:

```bash
uv run pytest tests/gui/test_curated_pack.py tests/gui/test_play_modes.py
cd frontend && npm test
cd frontend && npm run check
cd frontend && npm run build
cd frontend && npm run test:e2e -- e2e/offline-pack.spec.ts
```

The observable finish line is a clean-context default match and replay that
remain fully legible with public networking blocked, while the browser records
zero Scryfall requests and every reachable manifest identity passes exhaustive
local resolution.

## Operational boundary

- Runtime public-network requests for card presentation: exactly zero.
- Installed identity inventory: exactly 31 for v1 (29 deck identities and two
  reachable tokens); additions require a new manifest version.
- A fresh browser context must reach a connected default board within 60
  seconds after the local processes are launched, and reload/resume without a
  warm browser cache. The test records elapsed time and fails above the bound.
- Treatment lookup is in-memory and O(1) by exact engine name. No subprocess,
  asset download, service worker, IndexedDB cache, or runtime integrity scan is
  added.
- The normal `uv run scripts/play.py` launch path remains the operator surface.

## Exclusions

- No deck builder, collection browser, format legality, or arbitrary-card
  fetch/cache path.
- No Scryfall runtime fallback, bulk-data refresh automation, or image proxy.
- No service worker, PWA install/cache policy, IndexedDB asset cache, worker
  authority, WASM, or adapter benchmark.
- No third-party scans, crops, card-frame replicas, audio, fonts, animation,
  protocol-v1 redesign, card-rules change, or broad visual-table redesign.
- No claim that the legacy `interactive` deck or arbitrary custom decks are
  curated packs; they remain playable with deterministic fallback tiles.
