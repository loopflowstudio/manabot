# Etude Fantasia Visual Design System — Sepia Etude

The design contract for Etude Fantasia's player-facing surfaces. It
describes one visual language in three artifacts: **the Sheet** (Play),
**the Plate** (the card), and **the Score** (Replay, and Study as it
grows). `frontend/src/app.css` is the single source of truth for every
token value; this document explains what the values mean and the rules
that keep them coherent. If the two disagree, `app.css` wins and this
document is the bug.

---

## Brand Foundation

**An illuminated score.** The name is the brief: an *étude* is practiced
discipline, a *fantasia* is free invention, and the product is the study
of a game that improvises. The interface holds both, the way an
illuminated manuscript does — two hands on one page:

- **The scribe (étude)** — classical, academic, classy. Sepia ink on aged
  paper: adaptive grounds, ruled regions, three quiet inks, one bronze
  display serif, one red for urgency. The scribe's world changes with the
  light (light and dark modes) and never shouts.
- **The illuminator (fantasia)** — vibrant, creative, chaotic,
  improvisational. Full-saturation color, real art, gradients, gold: the
  banner weave, the player bars, the card plates, the mana pips. The
  illuminator's world is fixed in both modes — pigment doesn't care what
  time it is.

### The Frame Law

The two hands never touch except at a frame. Everything fantasia is
**framed and fixed**: bounded by a border, unaffected by color scheme,
carrying literal ivory text under a whisper of scrim. Everything étude is
**paper**: adaptive, ruled, never saturated, never inside a frame. An
unframed vivid element or an adaptive color inside a frame is a defect —
the one existing tension either pole feels with the other should be the
scribe *annotating* the illuminations: printer's marks, tallies, and
hatching are the scribe's hand on the illuminator's work, and that is the
product's whole gesture — you study the game's improvisation.

### The Echo Rule

Every fantasia color has an étude echo. The vivid pie
(`--vivid-w/u/b/r/g`) is the illuminator's palette; the five registers
(`plains/island/swamp/mountain/forest`) are the same hues ground down to
library volume for the scribe's use — washes, rules, marks. New color
pairs must keep this rhyme: if the illuminator gains a color, the scribe
gains its echo, and neither is used in the other's world.

| Visual choice | Hand | Brand signal |
|---------------|------|--------------|
| Parchment grounds, ruled regions | Scribe | The game as an open book |
| Bronze display serif | Scribe | Margin annotations, studied notes |
| One Mountain-red accent | Scribe | The live decision is the only urgent thing |
| WUBRG registers | Scribe | The vivid pie, quoted at library volume |
| Banner weave, player bars | Illuminator | The brand and the players in full color |
| Printed plates with real art | Illuminator | The cards are the illuminations |

### The two metals

- **Bronze** (`display`) is for reading: headings, rubrics, marginalia.
- **Mountain red** (`action`) is for acting: buttons, focus, links, the
  prompt's rule, the stack, targeting marks, errors.

Urgency has one color. A prompt is a red-ruled italic annotation; an error
is a red-washed banner; they share the family but never the shape.

---

## The Language

### Grounds: page, panel, field

Three ground levels per mode, so surfaces separate by value, not just by
outline: the **page** (`ground`), **panels** on it (`panel`, with `panel-muted`
wells), and the **field** fill inside bordered controls.

Light — Parchment:

| Token | Hex | | Token | Hex |
|-------|-----|-|-------|-----|
| `bg` | `#EBDFC6` | | `text` | `#3A3122` |
| `bg-surface` | `#F7F0E0` | | `text-secondary` | `#5F553D` |
| `bg-muted` | `#DED0AF` | | `accent` | `#973427` |
| `bg-field` | `#FCF9EE` | | `accent-hover` | `#AB4634` (lighter) |
| `border` | `#C9B892` | | `accent-text` | `#6D5A35` |
| `border-strong` | `#A5926A` | | `accent-soft` | `#EAD6C6` |

Dark — The library after dark:

| Token | Hex | | Token | Hex |
|-------|-----|-|-------|-----|
| `bg` | `#191510` | | `text` | `#ECE4D0` |
| `bg-surface` | `#221C14` | | `text-secondary` | `#B3A88E` |
| `bg-muted` | `#2B241A` | | `accent` | `#B24D38` |
| `bg-field` | `#2F2717` | | `accent-hover` | `#A34531` (**darker**) |
| `border` | `#423A2B` | | `accent-text` | `#C3A568` |
| `border-strong` | `#625741` | | `accent-soft` | `#3F2B20` |

> **Dark hover deepens.** A red lighter than `#B24D38` cannot hold ivory
> button text at WCAG AA, so the dark-mode hover moves down, not up.

`ivory` (`#F8F1E0`) is constant across modes: it is the text on accent
fills and nothing else.

### Three inks

Text is `ink`, `ink-2`, and `ink-3` — reading, secondary, and whisper
(placeholders, empties, status words, fine print) — plus the bronze
`display` tier for headings and rubrics. `ink-3` (light `#6E6247`, dark
`#9A9077`) holds AA on page and panel but **not on muted grounds**; nothing
tertiary sits on `panel-muted`. The pie speaks through container washes,
rules, and marks — never through body text. The `-ink` register companions
exist for the rare moments a register must speak in text (error banners, a
hot rubric); treat each new use as an exception to justify.

The vivid pie (`--vivid-w/u/b/r/g`) is the full-saturation Magic palette,
constant in both modes, reserved for the fixed rich worlds: the banner and
the player bars.

### The color pie — five registers

Each mana color owns a semantic register. Assign new colored elements to
the register they *belong* to, never the one that looks best. Play
whispers the pie (deck rules, the red stack, status chips); the Score
sings it (every line carries its register). There is no quota — there is
fidelity.

| Register | Mana | Base | Carries |
|----------|------|------|---------|
| `mountain` | Mountain | `#973427` | Action and urgency: the accent itself — prompts, the stack, targeting, damage tallies, errors |
| `plains` | Plains | `#9F7A1C` | Order and structure: lands and turn-structure lines in the score, reconnecting, targeting highlights |
| `island` | Island | `#2E6DA8` | The engine thinking: spells in the score, auto-passing, and the study apparatus to come |
| `swamp` | Swamp | `#6A5F86` | The hidden and the spent: card backs, game over, deaths in the score |
| `forest` | Forest | `#4C8040` | Presence and health: connected, hero's lines in the score |

Each register ships a contrast-safe `-ink` text companion per mode and a
`-wash` container tint.

**Washes are tokens, not arithmetic.** `--{register}-wash` is pre-mixed
with the panel ground per mode — 10% on parchment, 26% by lamplight — so a
wash can never silently vanish when the ground changes beneath it. Free
alpha modifiers on register bases are for borders and marks, not grounds.

### The vocabulary

Components speak semantic utilities and no others — there is no stock-name
shim, so a stale Tailwind color class fails to resolve instead of silently
meaning something:

```
ink / ink-2            reading text
ground / panel / panel-muted / field   the ground levels
line / line-strong     hairline and emphasized borders
action / action-hover / action-soft    the interactive accent
display                bronze display text
ivory                  text on accent fills
forest / plains / island / swamp / mountain   register bases
{register}-ink         contrast-safe register text
{register}-wash        pre-mixed register ground
```

The migration recipe and full mapping live in
`frontend/docs/semantic-utilities.md`.

### The voices

One text style per role, defined once in `app.css`; components never set
font sizes, families, or tracking inline. This is the discipline that
separates designed from decorated:

Every voice pins all four axes, and every computed line-height is a
multiple of 4 — the law that keeps rows on one rhythm:

| Voice | Face | Size/Leading | Tracking | Job |
|-------|------|--------------|----------|-----|
| `type-display` | Cormorant 600 | 20/28 | −0.01em | Player names, page-scale titles |
| `type-title` | Cormorant 600 | 16/20 | 0 | Panel headings: Actions, Game Log, Stops, The Score |
| `type-label` | Lato 600 | 11/16 | +0.04em | Field labels, button-adjacent labels, chips |
| `type-caption` | Lato 400 | 12/16 | 0 | Library counts, empties, hints, timestamps |
| `type-rubric` | Mono 600 · caps | 10/16 | +0.14em | **Notation only**: staff rubrics, tempo lines, frame counters, actor columns, "life" |
| `type-annotation` | Cormorant italic | 13.5/20 | 0 | **Prompts and marginalia only** |
| `type-numeral` | Lato 600 | 24/28 | −0.01em | Life totals and score-scale numbers |
| `type-brand` | Cormorant 600 | 30/36 | −0.01em | The banner brand mark only |

Body text is the default Lato 14/20, and the whole page sets
`font-variant-numeric: tabular-nums` — numbers are identity in a game
interface and never jitter. Ten pixels is the small floor; nothing renders
below it. Mono outside notation, or italic serif outside annotation, is a
defect.

Scoped component CSS may implement a voice only value-exactly, with a
comment naming the voice it copies. Inside the fixed worlds (plates,
backs, bars, banner) print objects carry their own literal type — the
illuminator's hand is exempt from the scribe's voices but not from the
10px floor's spirit at readable sizes.

**Do:** give a new label `type-label` and pick its ink.
**Don't:** write `text-[10px]`, a `tracking-*`, or a `font-*` family class
in a component — if a voice is missing, add the voice.

### Buttons

Three roles, two sizes, one shape — nothing else is a button:

- `btn btn-primary` — the red action; at most one per surface.
- `btn btn-secondary` — field fill, hairline border, hover turns the
  border red.
- `btn btn-ghost` — quiet text that gains a field on hover.
- `btn-sm` compacts either to 26px for margin-column density; base is
  32px (44px at touch widths).

Two interactive species live outside the button system, deliberately:
**rows** (action options, score lines — full-width field rows whose border
answers hover) and **quiet links** (`type-label`, underlined, ink-2 to
ink, padded to the 24px hit minimum). Marginalia affordances ("+ note") are ghost annotations, not
buttons.

---

## The Shell

The site around the book. The **banner** is the brand's one moment of full
color: the W→U→R→B→G Lotus Cobra weave, a fixed rich world in both modes,
carrying the ivory brand and the nav — the active page sits in a filled
ivory chip (`aria-current="page"`). The **colophon** closes every page the
way a book closes with its imprint: identity line, table of contents, and
the Fan Content attribution the WotC policy asks us to display. Everything
between them belongs to the sheet.

## The Sheet — Play

One continuous leaf. Regions are ruled, never boxed.

- The game header is the sheet's **masthead**, in conscious levels: the
  matchup names the players — "You (UR Lessons) vs Search 64 (GW Allies)",
  serif with mana pips; configuration fields appear only while they matter
  (before a game, and again at game over) with small-caps labels set above
  them; New Game is the one red action; and the connection is a register
  dot and a whisper of mono beside it. Nothing on the masthead competes
  with the sheet below.
- The turn is a **tempo marking**: small caps between flanking rules, in
  human notation ("Turn 6 · Combat · Declare Blockers"). Raw engine
  identifiers live in data attributes for Study, never on the player's
  table. Identical legal actions coalesce with a count.
- Each player is a **region**: bronze serif heading with the deck name in
  italics, the deck's color identity as a thin **pie rule** beneath it,
  life as bare numerals. Zones are **staves** — hand, battlefield,
  graveyard — named by small-caps **rubrics** in a margin column, divided
  by hairlines. The opponent reads hand-first; the hero battlefield-first.
- **The stack is the one thing on the sheet allowed to burn red**: a hot
  rubric and red-framed spell lines between the two regions.
- The **margin column** holds the apparatus behind a single vertical rule:
  the prompt as a red-ruled italic annotation, action options as field
  buttons, "Your move" in red small caps, stops, and the log as ruled
  lines with actor rubrics.
- The log records **game vocabulary only** — actor and action; key hints
  and control labels never enter the record.
- **Hidden information is counted, not drawn**: hidden cards are the
  violet-framed back with its blind lozenge, unlabeled. **Empty states are
  silent**: one italic serif aside in `ink-2` with a staff's worth of
  breathing room — no borders, no chips.
- Modals sit on the **scrim** (`color-mix(in srgb, var(--text) 45%,
  transparent)`, z 190); page content is never dimmed by its own opacity.

## The Plate — the card

The card is a printed plate in the book, and the one object on the table
that keeps its own light in both modes: art palettes and everything
painted on them are **literal, never adaptive** — an adaptive scrim turns
parchment in light mode and erases the name (`visual-system.spec.ts` pins
this).

**Anatomy** — print grain over the art, varnish light above and a vignette
below, an engraved rule fading across the name plate, the name re-set (not
scaled) per size and clamped to two lines, power/toughness in a serif oval
— an object on the card, not a UI chip. Small cards earn only a name;
larger sizes add the rule and the motif line.

**States are things that happen to printed matter:**

| State | Treatment |
|-------|-----------|
| Tapped | *Turned and re-set*: landscape footprint, plate re-typeset on the long edge — boards never read sideways; art one step dimmer |
| Summoning sick | *The ink hasn't dried*: engraver's diagonal hatching |
| Damaged | *Tallied*: red strokes above the oval, one per point (capped at five; the accessible name carries the number) |
| Targeted / focused | *The printer's marks*: four red registration brackets, breathing gently, still under reduced motion |
| Graveyard / exile | *An etching of itself*: monochrome, unvarnished, a fine hatch letting the paper through |
| +1/+1 counters | *Counted in brass*: a brass bead, and a brass hairline on the oval; remove the counter and the brass goes with it |

Badges state facts or do not exist: an effective 0/0 permanent (a land)
shows no oval. Every visual state is also stated in the card's accessible
name.

**Card art is versioned content, not runtime fetches.**
`frontend/scripts/fetch-card-art.mjs` pulls each curated-pack identity's
art crop from Scryfall exactly once per machine (identified User-Agent,
150ms pacing, cache-first, never at runtime) into `src/lib/card-art/` (bundler-known, so a missing file is a lookup miss, never a request);
CardImage layers it over the procedural treatment, which remains the
automatic fallback for anything missing. Images stay untracked; rights sit
in `src/lib/card-art/NOTICE.md` under the WotC Fan Content Policy, as do
the mana symbols in `static/mana/`.

## The Score — Replay and Study

The score is the spine; the board is the current page of the book.

- Every decision is a **line**: actor rubric, game text, a register rule
  and wash on its left edge, grouped under serif **turn rubrics**.
- Clicking a line turns the board to that moment; the scrub rail (red
  thumb) and frame steps walk it. The active line is ringed in red.
- The **margin takes marginalia**: bronze italic notes marked with a
  fleuron, pinned per trace, edited inline. The margin is also where the
  study apparatus will live (see Reserved).
- Registers in the score are currently inferred from game vocabulary — a
  presentation-level heuristic until the engine annotates its log.

---

## Reserved

Named in the contract so they land where they belong, not yet shipped:

- **The bot's mind in the margin** — manabot's weighed alternatives as
  quiet tabular numerals and hairline meters beside the score's lines,
  with one italic gloss under its preferred line; Island register. Waits
  on policy data reaching the frontend.
- **Identity pips** — a small mana-colored glass point on the plate's
  motif line. Waits on color identity in the pack manifest.
- **Held cards, tipped in** — parchment photo corners mounting the hero's
  hand to the page.
- **The illumination beat** — presentation events (spells resolving,
  combat) as framed fantasia flashes: the illuminator's hand answering
  the scribe's quiet beat panels, within the Frame Law.

---

## Fundamentals

**Typography** — three pillars, bundled locally: Cormorant Garamond
(display, rubrics, marginalia, plate names at readable sizes), Lato (body,
controls, small plate names), JetBrains Mono (notation: tempo markings,
rubrics, actor labels). Headings read in bronze; they do not shout.

**Spacing** — 4pt grid (`xxs` 2 … `xxxl` 32). **Radii** — `sm` 4 / `md` 8 /
`lg` 12 / `xl` 16 / `full`. **Hit targets** — 32px desktop, 44px touch
(≤640px). **Z-layers** — dropdown 100, scrim 190, modal 200, toast 300,
tooltip 400.

**Elevation** — depth has one mechanism per layer: the sheet rests on the
desk with `shadow-sheet`; only what floats above the sheet (dialogs, the
hover preview, presentation beats) casts `shadow-raised`. The one
exemption is physical: tipped-in plates sit in slight relief (a contact
shadow, which flattens when tapped), and print objects on them carry
their own micro-shadows. Nothing else has a shadow, and shadows are umber
by day, black by lamplight — never gray.

**Spacing** — steps have jobs: 8 between related elements, 12 within
control clusters, 16–24 panel padding, 24 between staves, 32–48 between
regions, 48–64 page rhythm (`--space-4xl/5xl`). Layout spacing sits on
the 4-grid; 2px and 6px are permitted only as micro-spacing inside a
single control.

**Interaction states** — every interactive element defines rest, hover
(`action-hover`, or a border turning `action`), **pressed**
(`accent-pressed`, or a step down to `panel-muted`), focus (the global
Mountain-red ring), and disabled (reduced ink with borders retained, never
whole-panel opacity). Motion has two speeds: 100ms ease-out for color and
border (`--motion-fast`); 180ms decelerate (`--motion-move`) for anything
that moves or appears. All motion collapses under
`prefers-reduced-motion`.

**Accessibility** — contrast is enforced by tooling, not by a table in
this file: `npm run validate:contrast` checks every documented pair (37 at
last count) straight from `app.css` and fails under AA. Status is never
color-only; state-bearing objects carry their state in the accessible
name; scrollable regions are keyboard-reachable; the scrubber is labeled.

---

## Verification

Before merging UI changes:

- [ ] Components speak only the semantic vocabulary
- [ ] `npm run validate:contrast` passes
- [ ] `npm run check` and `npm test` pass; the e2e suites pass
- [ ] Overlays on card art use literal colors, not adaptive tokens
- [ ] New colored elements are assigned by register, not by taste
- [ ] Dense boards hold: `DENSE_BOARD=1 npx playwright test e2e/dense-board.spec.ts` and read the captures
- [ ] Both modes checked by eye — parchment and lamplight

---

## Known gaps

Acknowledged debts, distinct from Reserved features:

- Hero visual weight: the contract says battlefield-first but does not yet
  enforce an Arena-style ~60/40 split or larger hero cards.
- Score registers are a text heuristic; the engine does not yet annotate
  its log.
- The contrast validator does not yet lint the voice laws (leading
  multiples, no inline font sizes) — they are greppable but unenforced.
- The experience-proof baseline and release visual references still pin the
  pre-redesign board and need regeneration on Linux CI.

---

## Lineage

Sepia Etude replaced the shared Loopflow/kata cream-and-burgundy palette
on 2026-07-16, chosen from a ten-direction exploration seeded by WUBRG and
four reference cards (Lotus Cobra ZNR showcase, Growth Spiral, Deathsprout,
Time Warp Mystical Archive). The Sheet, the Plate, and the Score landed
the same day from a prototype round reviewed by three simulated design
panels; the founding thesis — the active decision burns Mountain red — was
diluted once (prompts briefly spoke Island) and restored by the Sheet. The
token architecture and typography descend from the Loopflow contract.
