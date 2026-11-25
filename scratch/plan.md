# MTG-443 — Battlefield card badges that say what matters now

## Outcome

In Academy (Play, Watch, StepBoards, and Study), a permanent's battlefield chips
and Inspect panel share one effective authority and one small Sepia Etude
vocabulary that reads cleanly in light and dark:

- **Platypus-Bear** with a Lesson in its controller's graveyard shows
  **Defender struck through**, and Inspect says `"Defender · can attack anyway"`.
  Without a Lesson: plain `Defender`.
- **Mai, Jaded Edge** shows **Prowess**; an Ally beside **Sokka, Tenacious
  Tactician** shows granted **Prowess** (with the `--gain` edge) alongside
  granted **Menace**; **Fire Nation Cadets** shows **Firebending 2** only while
  a Lesson is in its controller's graveyard; printed and granted **Ward**
  (`Ward {2}`, `Ward—Pay 7 life`, `Ward—Discard a card`) show; a vanilla 1/1
  **Soldier** token never collides with a Firebending 1 **Soldier** token.
- **Ty Lee, Chi Blocker** shows no **Flash** chip on the battlefield (only
  **Prowess**); Inspect still lists **Flash**.
- A summoning-sick creature shows a **16×16 spiral mark**, never truncated words;
  no chip is ever clipped mid-word (`scrollWidth <= clientWidth` at every card
  width), with anything that does not fit whole counted in a trailing `+N`.
- Active battlefield restrictions — whether from resolving continuous-effect
  records (`effects[]`) or static abilities (freeze/Pacifism Auras, conditional
  unblockable, `CantBlock` tokens, `AttacksEachCombat`) — show on the board and
  in Inspect.
- `Observation`, the RL policy tensor (`PERMANENT_DIM`), default
  `serialize_observation()` (`include_details=False`), `CardFace`, pinned
  pilots, and frozen evidence stay **100% unchanged**.

## Architecture: One Effective Authority per Surface

`managym` already separates two contracts:

1. **`Observation` (`PermanentData`)** — the frozen-digest RL/policy contract.
   It carries direct Comprehensive Rules state (definition tokens, graveyard
   cards, Layer 6 keywords, `effects[]`, `ward_*`, `abilities_lost`,
   `granted_firebending`) and is encoded into fixed policy tensors. We do
   **not** add derived UI flags (like `ATTACKS_DESPITE_DEFENDER`) to
   `PermanentData` or `PERMANENT_DIM`, and we do **not** alter default
   `serialize_observation()` (`include_details=False`) used by
   `manabot/sim/frames.py`.
2. **`decision_surface_view(env, viewer)` (`managym/experience_projection.py`)**
   — the viewer-safe Academy projection contract used by **Live Play, Watch
   replays (`server.py:1585`), StepBoards (`step_boards.py`), and Study
   (`study_branch.py`)**. Just as it already calls `env.decision_context_json`,
   `env.public_schedule_json`, and `env.player_controls_json` to attach
   human-readable explanations from `Game` without touching `Observation`, it
   now attaches per-permanent effective battlefield details from `Game`.
3. **`CardFace` (`managym/academy/card_faces.py`, `/api/card-faces`)** — stays
   pure printed decoration (cost, colors, type line, printed keywords, printed
   P/T). It is name-keyed and asynchronously fetched, so it must **never** own
   effective battlefield abilities (multiple tokens share names like `"Soldier"`
   and `"Dragon"` with different abilities, and static/conditional grants cannot
   be answered from a printed catalog).

## Design

### Effective battlefield details on the wire

`Env.permanent_details_json(viewer)` returns a map keyed by the public permanent
id. Academy joins it to both battlefield rows in `decision_surface_view`.
The optional fields on `LegacyPermanentView` are:

- `keyword_abilities` is an ordered list of `{label, granted?}`, one per live
  trigger instance. Multiple Ward costs and repeated Prowess remain separate.
  Firebending X keeps its X because the amount is evaluated on resolution.
- A browser with no card faces cannot infer whether effective Menace is printed or granted. When a permanent has evergreen keywords, `keyword_grants` identifies the granted ones and their record timestamps (null for static or counter grants). Granted trigger instances carry `effect_timestamp` when a record supplies them. Inspect joins those timestamps to the permanent’s own effects.
- `attacks_despite_defender` is true while the native Defender exception applies.
- `restrictions` lists the active `cant_be_blocked`, `cant_block`,
  `cant_attack_or_block`, `doesnt_untap` and `must_attack` rules.

An empty `keyword_grants` on a permanent with keywords means all are its own. Other false or empty details are omitted. The UI reads these
directly from `PermanentState` (`academy/ui/src/lib/types.ts`). The catalog test checks registered definitions,
where printed Ward has already become a trigger.

### One UI badge vocabulary (`academy/ui/src/lib/board/badges.ts`)

Create one pure module `badges.ts` taking `(permanent: PermanentState, face: CardFace | null, blocks?: string | null)` and producing the ordered list of badges and facts used by:
- `BoardCard.svelte` (board chip subset + width/row fitter),
- `Inspect.svelte` (all abilities, keywords, statuses, provenance & durations),
- `BoardCard.svelte` `aria-label` (so screen readers hear keywords, abilities, overridden Defender, and statuses), and
- `view.ts:pileKey` (so permanents with differing effective keywords, `keyword_abilities`, `attacks_despite_defender`, `restrictions`, or `granted_firebending` never collapse into one `×N` pile).

`keywordsOf`, `statusesOf`, and `STATUSES` in `board/inspect.ts` fold into
`badges.ts`; the inline `chips` array and `keywordRoom` heuristic in
`BoardCard.svelte` are removed.

Where each thing shows:

| Thing | Battlefield chip | Inspect | Hand / stack |
|---|---|---|---|
| Damage marked | `"N damage"`, `--damage` ink | yes | — |
| Summoning sick | 16×16 SVG spiral mark (`title="Summoning sick"`) | words | — |
| Active restrictions (`cant_be_blocked`, `cant_block`, `cant_attack_or_block`, `doesnt_untap`, `must_attack`) + record-only statuses (`Saddled`, `CantActivateAbilities`, `MustBeBlockedByAllAble`, `ProtectionFromEverything`, `PreventCombatDamage`, `ExileIfWouldDie`, `ExileIfWouldLeave`) | words (deduplicated between `restrictions[]` and `effects[]`) | words + duration + source (when from `effects[]`) | — |
| `abilities_lost` | `"No abilities"` when no effective keywords/abilities remain; `"Lost base abilities"` when later-timestamp grants exist | yes | — |
| First-strike participant | `"Struck first"` | `"first-strike participant"` | — |
| Counters | `"+N/+N"`, `"Kind ×N"` | yes | counters in Inspect |
| Effective evergreen keywords (flying, reach, haste, vigilance, trample, first strike, double strike, deathtouch, lifelink, menace, hexproof, indestructible, and any unknown future keyword) | words; granted ones keep `--gain` ring | words + source | Inspect only |
| Defender | words; **struck through** (`--text-tertiary`, `line-through`) while `attacks_despite_defender` | `"Defender · can attack anyway"` | Inspect only |
| Effective `keyword_abilities` (Prowess / `Prowess ×N`, `Ward …`, `Firebending N`; fallback to `granted_firebending` on pre-change frames) | words; granted ones keep `--gain` ring | words + source/duration when matching `effects[]` | Inspect only (rules text covers hand/stack) |
| Flash | **never** | yes | Inspect only |
| Attacking, blocking | **no chip** — lean, tilt, and block line already carry them | yes | — |
| Token, entered this turn, attacked this turn | no | yes | — |

**Order on the card**: damage, summoning-sick mark, statuses/restrictions,
counters, abilities/keywords.

**Whole-chip fit rule (zero mid-word truncation)**:
- Remove `text-overflow: ellipsis` from `.chip`.
- Compute the chip row's inner width (`boxWidth - 6`) and available row count
  from `boxHeight` (accounting for the name bar and `--below`).
- Measure each chip's rendered width with a cached Canvas 2D text-width helper
  (with a conservative font-metrics fallback for jsdom/SSR unit tests; 16 px
  fixed for the spiral mark; 10 px horizontal padding + 2 px safety margin).
- Pack chips whole into the available rows in priority order. Any chip wider
  than `innerWidth` or past the row budget is omitted and counted into a
  trailing `+N` chip (reserving width for `+N` before placing the last chip on
  the final row). Every rendered chip satisfies `scrollWidth <= clientWidth`.

**Visual terms (all existing `DESIGN.md` tokens)**:
- Chips: 12/16 DM Sans on `--bg-surface`, 1 px `--border` ring, `--radius-sm`.
- Summoning-sick mark: 16×16 chip containing an inline SVG spiral stroked in
  `currentColor` (`--text-secondary`).
- Overridden Defender: `--text-tertiary` with `text-decoration: line-through`.
- Granted keyword or keyword ability: 1 px `--gain` ring.
- Add a concise "Card state" section to `academy/ui/DESIGN.md` and update the
  summoning-sickness note in `docs/academy.md`.

## Absent and error states

- **Pre-change / historical frames** (no `keyword_abilities`, `restrictions`, or
  `attacks_despite_defender`): plain Defender, effective `keywords`, `effects[]`
  statuses, and `granted_firebending` still render cleanly.
- **Face not loaded or unknown name** (`/api/card-faces` slow or failed):
  battlefield chips are 100% unaffected because every effective ability and
  status comes from `PermanentState`.
- **`abilities_lost` + later-timestamp grant**: never hides later Layer 6
  `keywords` or `keyword_abilities`; labels the status `"Lost base abilities"`
  (or `"No abilities"` when none remain).
- **Face-down permanents**: use the face-down copiable values plus effective
  grants. Hidden printed abilities never supply a label.
- **Opponent's permanents**: identical badges; every input is viewer-safe.

## Choices made for review

- **Effective details in `decision_surface_view` (`include_details=True`), not
  `Observation` or `CardFace`**: keeps `PERMANENT_DIM`, pinned pilots, and
  default `serialize_observation()` untouched while covering Play, Watch,
  StepBoards, and Study and fixing token collisions, Sokka's Allies, Fire Nation
  Cadets, granted Ward, and static restrictions.
- **Drop Attacking and Blocking chips**: lean, tilt, and block line already show
  combat roles (`DESIGN.md`: "No status that says nothing"); both remain in
  Inspect and `aria-label`. Keep `"Struck first"` for `first_strike_participant`.
- **Keep `CardFace` unchanged**: Hand/Stack Inspect already shows evergreen
  keywords plus full rules text.
- **Words for abilities, spiral mark for summoning sickness only**: matches
  Sepia Etude's typographic voice.

## What goes

- Inline `chips`, `keywordRoom`, and the text `sick` chip in `BoardCard.svelte`;
  `.chip`'s `text-overflow: ellipsis`.
- `keywordsOf`, `statusesOf`, and `STATUSES` in `academy/ui/src/lib/board/inspect.ts`
  (folded into `badges.ts`; tests move with them).

Must survive: byte-identical default `serialize_observation()` (`include_details=False`)
for every game; `Observation` / `PermanentData` / `PERMANENT_DIM` unchanged;
`CardFace` fields unchanged; Inspect's P/T provenance; accessible names' facts
(now enriched with keywords, abilities, and statuses).

## Docs

Updated in place before implementation so the next pass builds directly to them:

- `docs/academy.md` — battlefield card state chips, concrete examples
  (Platypus-Bear, Fire Nation Cadets, Mai, Sokka's Ally, Ward, Ty Lee, Soldier
  tokens), spiral summoning-sickness mark, whole-chip `+N` fit, piling, Inspect
  provenance, and older-recording fallback.
- `academy/ui/DESIGN.md` — `--gain` role note and `## Card state` section
  defining chip styling, spiral mark, overridden Defender, board vs. Inspect
  split, and the whole-chip fit rule.
- `managym/academy/README.md` — `permanent_details_json` projection seam
  alongside `decision_context_json` and `decision_offer_subjects_json`.

## Passes

1. **Done (74d1830899).** Native `permanent_details_json`, combat legality sharing the restriction rules, public-id join in `decision_surface_view`, protocol schemas, `PermanentState` types. No test results were recorded in scratch; gate reruns the acceptance commands.
2. **Done.** Board cards, Inspect, accessible names and piles use the same effective badges. The fitter measures whole chips and preserves the exact overflow count even on a short tapped card. The dark-mode browser fixture with no card faces confirmed that grant provenance comes from the frame.
3. **Gate.**
   Full targeted Rust, Python, protocol conformance, `scripts/public_api.py check`
   (additive `record` for the new `Env` method if in public surface), Vitest,
   and Playwright e2e in both themes. The card state docs are updated.

## Pass 2 checks

- Native release filter `permanent_details_` passed both card tests, including static Menace and record-granted Ward after layer evaluation.
- The recovery-after-Lesson test and Python/Rust field conformance test passed with the rebuilt CPython 3.13 extension. The recovery test checks historical observation bytes.
- Badge, pile, Inspect and design Vitest filters passed (315 tests). TypeScript protocol conformance passed. Svelte check reported zero errors and warnings.
- Playwright passed both 1280×800 themes. Inspected battlefield, Inspect and crowded screenshots in `academy/ui/test-results/`. The fixture exercised a face-service outage, later grants after ability loss and a crowded row of upright and turned cards. Every measured chip and row fit.

## Acceptance check at gate

1. `cargo nextest run --manifest-path managym/Cargo.toml -E 'test(platypus) | test(permanent_detail) | test(keyword_abilit) | test(experience_protocol) | test(study_protocol) | test(study_index) | test(canonical_replay)'`
   — Platypus-Bear, Mai, Sokka's Ally, Fire Nation Cadets, Soldier vs vanilla
   Soldier token, freeze Aura, `abilities_lost` + later grant, viewer-safety,
   and all schema conformance tests pass.
2. `uv run python -m pytest tests/academy/test_experience_protocol.py tests/academy/test_decision_surface.py tests/academy -k "badge or defender or permanent_detail" -q`
   — `decision_surface_view` emits exact effective details; default
   `serialize_observation()` is byte-identical even with an active Platypus-Bear.
3. `npm --prefix academy/ui test -- src/lib/board/badges.test.ts src/lib/board/view.test.ts src/lib/design.test.ts`
   — all badge placement, override, `abilities_lost` + later grant, fit-rule,
   `pileKey`, and design-lint checks pass.
4. `npm --prefix academy/ui run test:e2e -- e2e/card-badges.spec.ts` — fixture
   board (active and inactive Platypus-Bear, Mai, Sokka's Ally, Fire Nation
   Cadets, vanilla vs Firebending Soldier tokens, Ty Lee, narrow summoning-sick
   crowded row) in light and dark at 1280×800: every rendered chip has
   `scrollWidth <= clientWidth`, no `"Flash"` on the battlefield, spiral mark
   present, screenshots verified visually.
5. `uv run python scripts/public_api.py check` passes cleanly.

## Done when

Playing or watching a TLA game in Academy: an attack-ready Platypus-Bear reads
as able to attack (struck-through Defender), Mai and Sokka's Allies read
Prowess, active Fire Nation Cadets reads Firebending 2, vanilla Soldier tokens
never show false Firebending, Ty Lee shows no Flash on the field, a fresh
creature carries the spiral mark, static and record restrictions both surface,
piles never merge permanents with differing badges, and no chip on any card at
any board density is clipped in either theme.
