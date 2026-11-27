---
extends: implement
---
## Academy

**Preserve the shared decision surface.** Follow `AGENTS.md`'s Academy surface
rules and the "Sepia Etude" visual language in `academy/ui/DESIGN.md`.
`managym.academy` never imports `manabot` or Torch—use `managym.bots` /
`managym.academy.bots` entry points.

**A board design is for every mode.** A layout, chrome or overlay chosen for
`Board.svelte` is built in every mode that mounts it (Play, Watch, Replay,
Study) unless the design names an exception; the mode a prototype showed is
not its scope. Screenshot each at 1280×800 with a game in progress and look:
nothing covers a hand, nothing is shown twice, no overlay is left open and the
primary view fits one screen.

**Check narrowly per slice.** Run `npm --prefix academy/ui run check` and the
touched `vitest` / `tests/academy/` files; leave full Playwright e2e runs to
`qa` and CI unless the slice directly edits e2e specs.
