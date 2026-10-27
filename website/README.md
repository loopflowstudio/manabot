# Manabot website

The landing site for [manabot.gg](https://manabot.gg) — a FastHTML app in the
Sepia Etude visual language, deployed on Fly.io. It mirrors the
loopflow/website pattern: copy in `content.yaml`, one `main.py`, static
assets bundled locally.

`manabot.gg` is canonical; `etude.gg`, `etudefantasia.com`, `etudefantasia.gg`,
and all `www.` variants 301 onto it, preserving encoded paths and queries. DNS lives in Cloudflare; certificates are
issued by Fly (see `deploy/academy-fly.sh` at the repo root for operations).

Design tokens are hand-copied from `academy/ui/src/app.css` (the source of
truth — see `VISUAL_DESIGN.md`); if the two drift, this site is the bug.
Fonts and mana symbols are copied from `academy/ui/static/`; the board
screenshot is `academy/ui/e2e/visual-references/v3/board-developed.png`.

## Develop

```bash
cd website
uv run --extra test pytest         # smoke tests
uv run python main.py              # serve on :5002
```

## Deploy

```bash
deploy/academy-fly.sh deploy         # from the repo root
```
