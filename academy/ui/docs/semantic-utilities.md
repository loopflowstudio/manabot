# Semantic color utilities

`src/app.css` defines the Sepia Etude palette as semantic variables
(`--bg-field`, `--text-secondary`, `--accent`, ...), but components
historically consumed them through a `@theme` shim that mapped Tailwind's
stock names onto them (`--color-slate-900: var(--bg-field)`). A component
reading `text-slate-400 hover:bg-blue-500` tells a new teammate nothing.
This spike adds first-class semantic utilities and proves the migration on
one component (`ActionPanel.svelte`) with pixel-identical output.

## Vocabulary

One name per token; the name says what the color *does*, not what it is.

| Utility root | Token | Role |
|---|---|---|
| `ink` | `--text` | Primary reading text ("ink on paper" fits the sepia manuscript register) |
| `ink-2` | `--text-secondary` | Secondary text; numeric suffix marks hierarchy, not hue |
| `ground` | `--bg` | The page floor |
| `panel` | `--bg-surface` | Raised surfaces (cards, asides) |
| `panel-muted` | `--bg-muted` | Muted wells inside panels (badges, insets) |
| `field` | `--bg-field` | The paper-white fill inside bordered controls |
| `line` | `--border` | Hairline borders |
| `line-strong` | `--border-strong` | Emphasized borders |
| `action` / `action-hover` | `--accent` / `--accent-hover` | Interactive accent: fills, hover borders |
| `action-soft` | `--accent-soft` | Quiet accent wash |
| `display` | `--accent-text` | Bronze display text (headings, panel titles) |

The five status families carry Magic's land registers — the game's own
color language, which every teammate on this project already speaks, and
the registers the palette itself assigns (Plains the turn structure,
Island the knowledge, Swamp the hidden and spent, Mountain the actions
and errors, Forest the living board):

| Base (fill) | `-ink` (contrast-safe text) | Tokens |
|---|---|---|
| `forest` | `forest-ink` | `--success` / `--success-text` |
| `plains` | `plains-ink` | `--warning` / `--warning-text` |
| `island` | `island-ink` | `--info` / `--info-text` |
| `mountain` | `mountain-ink` | `--error` / `--error-text` |
| `swamp` | `swamp-ink` | `--neutral` / `--neutral-text` |

The `-ink` companions matter: pairing `text-island` on a light ground is a
contrast bug; `text-island-ink` is always the readable member of the family.

## Conversion recipe (ActionPanel, complete mapping)

Look up what the stock alias resolves to in the shim, then use the semantic
utility bound to the same variable. Opacity modifiers carry over unchanged.

| Before | After | Resolves to |
|---|---|---|
| `border-slate-700` | `border-panel-muted` | `--bg-muted` |
| `bg-slate-800` | `bg-panel` | `--bg-surface` |
| `bg-slate-900` | `bg-field` | `--bg-field` |
| `border-slate-600` | `border-line-strong` | `--border-strong` |
| `text-slate-300`, `text-slate-400` | `text-ink-2` | `--text-secondary` |
| `text-slate-200`, `text-violet-200` | `text-ink` | `--text` |
| `text-accent-text` | `text-display` | `--accent-text` |
| `bg-purple-900/20` | `bg-swamp/20` | `--neutral` at 20% |
| `bg-sky-600/20` | `bg-island/20` | `--info` at 20% |
| `border-violet-500/40`, `bg-violet-900/20` | `border-island/40`, `bg-island/20` | `--info` |
| `bg-emerald-600/20` | `bg-forest/20` | `--success` at 20% |
| `border-amber-300` | `border-plains-ink` | `--warning-text` |
| `hover:border-blue-400`, `focus-visible:outline-blue-400` | `...-action` | `--accent` |

Note the drift the shim had already accumulated: `border-slate-700` is a
*background* token used as a border, and `violet-*` classes are Island
blue, not violet — decision prompts read as `border-violet-500/40` while
actually resolving to `--info`. The semantic names surface both facts
immediately.

## Cost observations

- One ~140-line component: 11 class-attribute edits, roughly 20 minutes
  including verification; the only thinking required is the shim lookup.
- The lookup is *not* mechanical across palette revisions: when the shim
  retargeted (`slate-900` moved from `--bg` to `--bg-field`, `violet-*`
  from `--accent` to `--info`), a conversion done against the old shim
  silently changed meaning. Convert against the shim you ship.
- Zero logic changes; `npm run check`, `npm test` (61 tests), and
  `npm run build` all clean. Pixel identity verified by inspecting the
  built CSS: each new utility emits the exact declaration the old alias
  did (e.g. `.bg-field{background-color:var(--bg-field)}`).
- `npm run validate:contrast` gates the VISUAL_DESIGN.md matrix — the
  seven documented pairs plus every pie ink on ground/panel/muted in both
  modes, 37 pairs total — parsing live hex from `app.css` so it cannot
  drift. It immediately paid for itself: light `--success-text` on
  `--bg-muted` measured 4.46:1, just under AA; darkening forest ink one
  step (`#3a6531` to `#396230`) clears the whole matrix (worst pair now
  4.64:1, everything else comfortably AA or AAA).

## Full codemod

The shim inverts to a pure find-and-replace table (~35 alias → utility
entries covering every `--color-*` alias in `app.css`), applied to
`class=`/`class:` strings and template-literal class fragments across
`src/lib` and `src/routes`. Two judgment calls a script cannot make:

1. Aliases that were already used against the wrong register (like
   `border-slate-700` above) — mechanically translate first, then decide
   per call site whether the *intent* was `border-line`.
2. Aliases with several stock spellings for one token (slate-300/400/500
   all → `ink-2`) collapse safely; run the mapping, grep for leftover
   `slate-|blue-|sky-|violet-|purple-|emerald-|amber-|indigo-|cyan-|rose-`,
   and delete the shim from `@theme` once the grep is empty — the delete
   is the enforcement mechanism.

Estimated cost at ActionPanel's density: an afternoon for the whole
frontend, mostly verification.
