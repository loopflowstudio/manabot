# MTG-443 — Battlefield card badges that say what matters now

## Outcome

In Academy (Play, Watch, StepBoards, and Study), a permanent's battlefield chips
and Inspect panel share one effective authority, one CR 613 provenance model,
and one Sepia Etude vocabulary that reads cleanly in light and dark — including
on real 1280×800 live-game boards with lands and cards in hand:

- **Platypus-Bear** with a Lesson in its controller's graveyard shows
  **Defender struck through**, and Inspect says `"Defender · can attack anyway"`.
  Without a Lesson: plain `Defender`.
- **Mai, Jaded Edge** shows **Prowess** (and `Double strike · 1 double strike counter`
  after Exhaust, on one chip); an Ally beside **Sokka, Tenacious Tactician**
  shows granted **Menace** and **Prowess** (with `--gain` edge) and Inspect names
  **Sokka, Tenacious Tactician** (never `"Another ability"`); **Fire Nation Cadets**
  shows **Firebending 2** (or `Firebend 2` on an 80 px upright card, never bare
  `+1`) only while a Lesson is in its controller's graveyard; **Avatar Kyoshi**
  and other self-scoped conditional keyword grants stay plain (never false
  `--gain` / `"Another ability"`); printed and granted **Ward** show (`Ward {2}`,
  `Ward—Pay 7 life`, `Ward—Discard a card`, falling back to `Ward` when a narrow
  card cannot fit the full cost); a vanilla 1/1 **Soldier** token never collides
  with a Firebending 1 **Soldier** token.
- **Ty Lee, Chi Blocker** shows no **Flash** chip on the battlefield; tapped at
  112×80 px with `Doesn't untap` + `Prowess`, it shows both chips whole (never
  bare `+2`); Inspect still lists **Flash**.
- A summoning-sick creature shows a **16×16 spiral mark**, never truncated words;
  no chip is ever clipped mid-word (`scrollWidth <= clientWidth` at every card
  width), with anything that does not fit whole counted in a trailing `+N` that
  carries a `title` tooltip listing the hidden badges.
- Active battlefield restrictions — whether from resolving continuous-effect
  records (`effects[]`) or static abilities (freeze/Pacifism Auras naming the
  Aura, own printed restrictions like Cat-Armadillo without `"Static ability"`
  noise, conditional unblockable, `CantBlock` tokens, `AttacksEachCombat`) —
  show on the board and in Inspect.
- `Observation`, the RL policy tensor (`PERMANENT_DIM`), default
  `serialize_observation()` (`include_details=False`), `CardFace`, pinned
  pilots, and frozen evidence stay **100% unchanged**.

## Architecture: One Effective Authority & One Provenance Model

`managym` separates three contracts:

1. **`Observation` (`PermanentData`)** — the frozen-digest RL/policy contract.
   Direct Comprehensive Rules state only; untouched by MTG-443.
2. **`decision_surface_view(env, viewer)` (`managym/experience_projection.py`)**
   — the viewer-safe Academy projection contract used by **Live Play, Watch
   replays (`server.py:1585`), StepBoards (`step_boards.py`), and Study
   (`study_branch.py`)**. Calls `env.permanent_details_json(viewer)` and joins
   effective battlefield details by public permanent id.
3. **`CardFace` (`managym/academy/card_faces.py`, `/api/card-faces`)** — pure
   printed decoration (cost, colors, type line, printed keywords, printed P/T).
   Never a battlefield ability authority.

## Design

### Unified Layer 6 provenance in Rust (`AbilitySource`)

Introduce one internal enum `AbilitySource` (in `managym_state::ability`,
`#[serde(skip)]` on `EffectiveTrigger` so serialization and hashes stay
identical):

- `AbilitySource::Own` — seeded from the permanent's copiable face OR from the
  permanent's own printed static ability (`node.key.len() == 2 && node.source == id`,
  such as Avatar Kyoshi, Aang A Lot to Learn, and `tla.rs:3216/4280`).
- `AbilitySource::Effect(PublicId)` — direct resolved record
  (`node.key.len() == 3 && node.key[1] == usize::MAX`) OR a record-granted
  static ability applying to its own host (`node.key.len() > 3 && node.key[1] == usize::MAX && node.source == id`).
- `AbilitySource::Counter(CounterKind)` — keyword counter node
  (`node.key.len() == 3 && node.key[1] == usize::MAX - 1`).
- `AbilitySource::Permanent(PermanentId)` — another permanent's static ability
  (`node.source != id`, such as Sokka, Swiftfoot Boots, Mechanical Glider, or an
  Aura/Equipment).

In `managym/src/flow/layers.rs`:
- `Characteristics.keyword_grants` stores `Vec<(Keyword, AbilitySource)>` and
  records only grants where `source != AbilitySource::Own` and the keyword was
  not already present in `view.keywords` (so printed keyword + matching counter
  stays `Own`, and self-scoped `StaticScope::This` stays `Own`).
- `EffectiveTrigger` carries `source: AbilitySource` (replacing
  `effect_timestamp: Option<PublicId>`). This eliminates the positional
  `i >= own` index check in `permanent_details.rs`.

In `managym/src/flow/statics.rs` & `combat_actions.rs`:
- Keep `has_restriction` allocation-free on combat hot paths.
- Add a helper returning the external static source permanent (`source != target`)
  for each active `Restriction` (`DoesntUntap`, `CantAttackOrBlock`,
  `CantBeBlocked`, `AttacksEachCombat`) so `permanent_details.rs` can attach the
  granting permanent's public effective name (`views[&grantor].name`) when an
  Aura or other permanent imposes the restriction, and `None` when it is the
  permanent's own printed static rule (e.g. Cat-Armadillo).

### Effective battlefield details on the wire

`Env.permanent_details_json(viewer)` returns a map keyed by public permanent id.
`PermanentDetails` (flattened onto `LegacyPermanentView`) carries:

- `keyword_grants: Option<BTreeMap<String, AbilityGrantView>>` — present when
  the permanent has at least one effective evergreen keyword (empty `{}` means
  all are its own; absent on pre-change recordings). Each granted keyword maps
  to `AbilityGrantView { effect_timestamp?: u32, source?: String, counter?: bool }`
  (omitting false/None fields).
- `keyword_abilities: Vec<KeywordAbilityView>` — one entry per effective
  trigger instance: `{ label: String, granted?: bool, effect_timestamp?: u32, source?: String }`.
  Multiple Ward costs and repeated Prowess remain separate; Firebending X keeps
  its printed X.
- `attacks_despite_defender: bool` — true while the native Defender exception
  applies (omitted when false).
- `restrictions: Vec<RestrictionView>` — `{ kind: PermanentRestriction, source?: String }`,
  one per active restriction (`cant_be_blocked`, `cant_block`,
  `cant_attack_or_block`, `doesnt_untap`, `must_attack`), with `source` set to
  the granting permanent's public name when another permanent's static ability
  imposes it.

Regenerate all six `protocol/*.schema.json` files and keep `managym/experience.py`
and `academy/ui/src/lib/types.ts` aligned via the protocol conformance tests.

### One UI badge vocabulary & real-board fitter (`academy/ui/src/lib/board/badges.ts` + `BoardCard.svelte`)

`badges.ts` takes `(permanent: PermanentState, face: CardFace | null, blocks?: string | null)`
and feeds `BoardCard.svelte`, `Inspect.svelte`, `BoardCard.svelte`'s
`aria-label`, and `view.ts:pileKey`:

- **Exact provenance (zero guessing)**:
  - Evergreen keywords read `p.keyword_grants[keyword]`: `counter` →
    `counterTerm(keyword, count)`, `effect_timestamp` → `effectLabel(record)`,
    `source` → permanent name. Absent key in a present `keyword_grants` map
    always means own/printed (`granted: false`, no `--gain` ring), fixing both
    `StaticScope::This` and printed-keyword + counter.
  - Keyword abilities read `trigger.effect_timestamp` or `trigger.source`.
  - Restrictions merge static `restriction.source` with matching `p.effects[]`
    `effectLabel(effect)` sources (never dropping the Aura when an effect also
    applies, and omitting `"Static ability"` noise on own-printed restrictions).
  - Pre-change historical frames (missing `keyword_grants` / `keyword_abilities`
    / `restrictions`) keep their graceful fallback.
- **Board vs Inspect**:
  - Order on the card: damage, summoning-sick spiral mark, statuses/restrictions,
    counters (keyword counters stay Inspect-only when the permanent has that
    effective keyword), abilities/keywords.
  - `Flash` stays Inspect-only; Attacking/Blocking stay in Inspect + `aria-label`
    (board uses lean, tilt, block line); `"Struck first"` stays on the card
    during the first-strike window.
- **Real-board whole-chip packing (`fitBadges` + `BoardCard.svelte`)**:
  - `.chip` keeps `12px/16px` DM Sans (`DESIGN.md` scale), zero
    `text-overflow: ellipsis`, `padding: 0 4px`, `.chips` inset `2px`
    (`boxWidth - 6` inner width, `chipWidth` adds `10px` = `8px` pad + `2px`
    slack).
  - **Bottom row beside P/T plate**: when a card has a P/T plate and does NOT
    display Watch's bottom-center `.policy` pill, `.chips` anchors at
    `bottom: 3px` and passes per-row widths to `fitBadges`: full width
    (`boxWidth - 6`) for every row above the P/T plate, and
    `boxWidth - 6 - ptReserve` for the bottom row beside the P/T plate.
    `BoardCard.svelte` renders each fitted row as an explicit `.chip-row` so
    DOM wrapping matches `fitBadges` 100%. (When Watch's `.policy` pill is
    present on a card, chips stay strictly above `below`.)
  - **Compact `shortLabel` fallback**: `Badge` supports optional `shortLabel`
    used *only* when `measure(badge.label) > fullRowWidth` (the full label
    cannot fit on any row of the card):
    - `Firebending N` / `Firebending X` → `Firebend N` / `Firebend X`
    - `Ward—…` (non-mana Ward costs like `Ward—Waterbend {4}`, `Ward—Pay 7 life`,
      `Ward—Discard a card`) → `Ward`
    - `Doesn't untap` → `No untap`
    - `Can't be blocked` → `Unblockable`
    - `Can't attack or block` → `No combat`
    Wider/turned cards always show the full label; `title`, `aria-label`, and
    `Inspect` always use the full `detail ?? label`.
  - **Overflow-aware packing & `+N` tooltip**:
    - Fast path: if all board badges fit within the available row widths with no
      overflow chip, render all of them.
    - Otherwise, scan badges in priority order, placing each badge whose width
      (plus the `+remaining` overflow chip on the remaining row budget) fits,
      skipping oversized badges so shorter subsequent badges (e.g. `Prowess`)
      still render beside `+1` instead of collapsing to bare `+2`.
    - Return `{ rows: FittedChip[][], hidden: Badge[] }`; render `+N` with
      `title={chips.hidden.map(badgeReading).join(' · ')}`. When `rowWidths` is
      empty and `hidden.length > 0`, show `+N` (with `title`) beside the name.

## Choices made for review

- **Effective details in `decision_surface_view` (`include_details=True`), not
  `Observation` or `CardFace`**: keeps `PERMANENT_DIM`, pinned pilots, and
  default `serialize_observation()` untouched.
- **Unified `AbilitySource` on the wire (`source` names granting permanent for
  static grants and static restrictions)**: fixes self-conditional keywords,
  printed-keyword + counter collision, and `"Another ability"` / `"Static ability"`
  in Inspect.
- **Bottom-row chips beside P/T (except when `.policy` pill is shown) +
  overflow-aware packing + width-triggered `shortLabel` fallback + `+N`
  tooltip**: makes headline badges readable at real 1280×800 game card sizes
  (80×112 upright, 112×80 turned) without changing `Board.svelte` table layout
  or breaking `DESIGN.md`'s 12 px minimum font size.
- **Drop Attacking and Blocking chips; keep `"Struck first"`; keep `CardFace`
  unchanged; spiral mark for summoning sickness only**.

## Docs updated in Pass 3

- `docs/academy.md` — Inspect naming static grant/Aura sources (Sokka, freeze
  Auras), compact board labels on narrow cards, `+N` tooltip.
- `academy/ui/DESIGN.md` — `## Card state`: 4 px chip padding, bottom-row beside
  P/T (yielding to `.policy`), width-triggered compact labels, `+N` tooltip.
- `managym/academy/README.md` — updated `AbilityGrantView` / `RestrictionView`
  fields on `permanent_details_json`.

## Passes

1. **Done (74d1830899).** Native `permanent_details_json`, shared combat
   restriction helpers, `decision_surface_view` join, protocol schemas.
2. **Done (63583fc3ff).** `badges.ts`, `BoardCard.svelte`, `Inspect.svelte`,
   `view.ts:pileKey`, Vitest, Playwright light/dark e2e, live-game QA.
3. **Done (e41607db14).** `AbilitySource` provenance on grants, triggers and
   static restrictions; real-board fitter (bottom row beside P/T, `shortLabel`
   fallbacks, overflow-aware packing, `+N` tooltip); 1280×800 and Watch e2e; docs.
   Avatar Kyoshi's land grants name `source: "Avatar Kyoshi"`; self-scoped
   `StaticScope::This` grants stay `Own`. `wholeChips` awaits `document.fonts.ready`
   and measures in one `page.evaluate` (webfont refit raced `evaluateAll`).
   Implement recorded no test-run results for this pass; the gate runs them.
4. **Done (291bb7aa9f).** Gate before publish: full debug nextest 2105/2105;
   Vitest 509/509; Svelte check clean; card-badges e2e 5/5; public API check;
   pinned pilot and package boundaries pass. `tests/academy` has 28 failures
   that main (3be5a0cfb2) shares; the branch's 4 extra were stale fixtures
   (`keyword_grants: {}` in retained frames), refreshed with their writers.

## Acceptance check at gate

1. `cargo nextest run --manifest-path managym/Cargo.toml -E 'test(platypus) | test(permanent_detail) | test(keyword_abilit) | test(experience_protocol) | test(study_protocol) | test(study_index) | test(canonical_replay)'`
2. `uv run python -m pytest tests/academy/test_experience_protocol.py tests/academy/test_decision_surface.py tests/academy -k "badge or defender or permanent_detail" -q`
3. `npm --prefix academy/ui test -- src/lib/board/badges.test.ts src/lib/board/view.test.ts src/lib/design.test.ts src/lib/protocol-conformance.test.ts`
4. `npm --prefix academy/ui run check && npm --prefix academy/ui run test:e2e -- e2e/card-badges.spec.ts`
5. `uv run python scripts/public_api.py check`

## Done when

Playing or watching a real TLA game in Academy at 1280×800 (with creatures,
lands, and cards in hand): Platypus-Bear reads as able to attack when a Lesson
is in the graveyard; Fire Nation Cadets shows Firebending instead of `+1`;
turned Ty Lee shows `Doesn't untap` and `Prowess` instead of `+2`; Unagi shows
`Ward` instead of `+1`; Sokka's Allies and Aura-locked creatures name their
actual source permanent in Inspect; self-conditional keywords and printed
keywords with matching counters never falsely turn green; piles never merge
permanents with differing badges; and no chip clips or overlaps P/T or Watch's
policy pill in either theme.
