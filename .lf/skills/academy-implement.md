---
extends: implement
---
## Academy

**Preserve the shared decision surface.** Make the acting agent's viewer-safe
observations and complete legal action space clear without hiding strategic
choices, silently selecting actions, revealing unavailable information or
introducing human-only game semantics. Preserve Command binding, stale
rejection, actual pending decisions on reconnect, replay/fork identity and
keyboard accessibility. Keep live combat drafting as sequential per-creature
Commands, reserve preview space so hover never shifts layout, frame policy
comparisons neutrally, send original JSON source spans on game-file import
(avoid JS u64 rounding), and scope resume credentials with the started-launch
marker in `sessionStorage`.

**Respect package boundaries and visual language.** `managym.academy` never
imports `manabot` or Torch—use `managym.bots` / `managym.academy.bots` entry
points. Follow the "Sepia Etude" visual language in `academy/ui/DESIGN.md`.

**Check narrowly per slice.** Run `npm --prefix academy/ui run check` and the
touched `vitest` / `tests/academy/` files; leave full Playwright e2e runs to
`qa` and CI unless the slice directly edits e2e specs.
