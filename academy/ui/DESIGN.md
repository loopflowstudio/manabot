# Academy design system

Academy's visual language is Sepia Etude: an academic, quiet library page,
where colour belongs to the cards. It has a light theme and a dark theme,
chosen by the system's colour scheme. `src/app.css` holds every value; this
file says which values exist and what each is for. `src/lib/design.test.ts`
reads the permitted list below and fails on any literal outside it in
`app.css` and in the board's Svelte styles (`src/lib/board/`).

```permitted
families: Spectral, DM Sans
fallbacks: Georgia, serif, system-ui, sans-serif, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace
sizes: 12/16, 14/20, 16/24, 20/28, 28/32, 40/44
weights: 400, 500, 600
radii: --radius-sm, --radius-md, --radius-lg, --radius-xl, --radius-full, --radius-card, --radius-card-turned
shadows: --lift, --lift-selected, --lift-attacking, --shadow-raised, --shadow-sheet, --shadow-surface
```

## Type

Two families, each with one job, bundled under `static/fonts/` (both under
the SIL Open Font License):

- **Spectral**, the game's face (`--font-serif`, `font-serif`): card and
  player names, life totals, the inspected card's name, page titles, and in
  the rail the moves and the legal choices (a choice's words, not its key or
  probability).
- **DM Sans**, the interface's face (`--font-sans`, the body default):
  prompts, buttons, labels, counts, the stop bar and key hints.

Monospace (`--font-mono`, the system's) is for code a person copies, never
for labels, and never on the board.

Six sizes, each with its line height: 12/16, 14/20, 16/24, 20/28, 28/32 and
40/44. Nothing is smaller than 12 px, on cards included. Three weights: 400,
500 for headings and labels, 600 for names and emphasis. Labels are sentence
case ("Your decision", "Hand · 7"): no capitals by transform, no tracking. Numbers use tabular figures (the body sets
`tabular-nums`). The voices in `app.css` name the common roles:

| Voice | Face | Size | Job |
|---|---|---|---|
| `type-brand` | Spectral 600 | 28/32 | The brand mark |
| `type-display` | Spectral 600 | 20/28 | Page titles |
| `type-title` | Spectral 600 | 16/24 | A player's or a thing's name |
| `type-prompt` | DM Sans 400 | 16/24 | What the game asks |
| `type-heading` | DM Sans 500 | 14/20 | A region's name: Stack, Moves |
| `type-body` | DM Sans 400 | 14/20 | Running text (the body default) |
| `type-label` | DM Sans 500 | 12/16 | Small labels |
| `type-caption` | DM Sans 400 | 12/16 | Counts, hints, empties |

A mana symbol's letter scales with its pip (`calc(var(--pip) * k)`): it is a
symbol, not text, and is the one size outside the scale.

## Colour

Components use tokens, never colour literals. The grounds and inks:

| Token | Utility | Role |
|---|---|---|
| `--bg` | `ground` | The page |
| `--bg-surface` | `panel` | Bars, the rail, floating panels |
| `--bg-muted` | `panel-muted` | The board |
| `--bg-field` | `field` | The fill inside bordered controls |
| `--border`, `--border-strong` | `line`, `line-strong` | Edges |
| `--text`, `--text-secondary`, `--text-tertiary` | `ink`, `ink-2`, `ink-3` | Reading text, in three steps |
| `--accent` (+ `-hover`, `-pressed`, `-soft`), `--on-accent` | `action`, `on-action` | What you act with: the primary button, the selected card's edge |

Fixed roles, each meaning one thing everywhere:

- `--damage`: damage, life lost, a target's dashed edge.
- `--gain`: power or toughness above printed, life gained.
- `--planned`, `--planned-text`: the automatic payer's planned taps.
- `--pip-*`: mana symbols; `--card-*` and `--card-*-ink`: a card's name bar
  in its printed colours (pale tints in light, solid bars in dark).
- `--success`, `--warning`, `--error`, `--info`, `--neutral`, each with a
  `-text` companion that is readable on the grounds and a `-wash` tint.

Text tokens hold at least 4.5:1 on the grounds they sit on, in both themes;
the design test checks the pairs. No status that says nothing: a connection
shows only when it has a problem, and whose turn it is reads in words.
Gradients appear only where colours blend on a multicoloured card's name
bar and in a data bar's fill.

## Shape and depth

- Radii: `--radius-sm` (4 px) for controls and chips, `--radius-md` (8 px)
  for floating panels (menus, the inspect popover, trays), `--radius-lg` and
  `--radius-xl` for the other pages' panels, `--radius-full`
  or `50%` for dots and pips, and a printed card's corner (`--radius-card`,
  `--radius-card-turned` for a card on its side). Regions are square.
- Shadows only for what floats or is lifted: `--lift` (what you can act
  on), `--lift-selected`, `--lift-attacking`, and `--shadow-raised` for
  popovers and menus (`--shadow-sheet`, `--shadow-surface` on pages). An
  edge or ring may be drawn with a zero-blur `box-shadow` in a token colour.
- Floating panels are opaque: the result card, trays, the inspect popover,
  menus and the keys sheet use a solid `--bg-surface`, never a translucent
  mix. One layering order holds on the board, lowest first: cards, the
  lines between cards and targets, floating panels, sheets. It is written once
  in `Board.svelte`'s styles; do not add z-indexes outside it.

## Space and motion

A 4 px grid (`--space-*`). Related things sit 4–8 px apart, groups 16–24.
Group with space and one divider, not boxes. Only what changes moves:
`--motion-fast` (100 ms) for colour, `--motion-move` (180 ms) for movement,
and reduced motion keeps every state change without the travel.

## Checks

- `npm --prefix academy/ui test -- src/lib/design.test.ts`: the permitted
  list and the contrast pairs.
- Both themes checked by eye at 1280×800 before handing a change over.

The lint covers the board and `app.css`. The other pages still carry older
literals (the banner's weave and its colours, page-level sizes); a change to
one of them brings that file under the lint.
