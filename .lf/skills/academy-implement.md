---
extends: implement
---
## Academy

**Preserve the shared decision surface.** Follow `AGENTS.md`'s Academy surface
rules and the "Sepia Etude" visual language in `academy/ui/DESIGN.md`.
`managym.academy` never imports `manabot` or Torch—use `managym.bots` /
`managym.academy.bots` entry points.

**Verify shared board modes.** When changing shared `Board.svelte` layout,
chrome, or overlays, check every mode that mounts it (Play, Watch, Replay) at
1280×800 so one mode's change does not leave an overlay open or spill a primary
view past one screen.

**Check narrowly per slice.** Run `npm --prefix academy/ui run check` and the
touched `vitest` / `tests/academy/` files; leave full Playwright e2e runs to
`qa` and CI unless the slice directly edits e2e specs.
