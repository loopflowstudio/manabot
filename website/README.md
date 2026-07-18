# Etude Fantasia website

The landing site for [etude.gg](https://etude.gg) — a FastHTML app in the
Sepia Etude visual language, deployed on Fly.io. It mirrors the
loopflow/website pattern: copy in `content.yaml`, one `main.py`, static
assets bundled locally.

`etude.gg` is canonical; `etudefantasia.com` and `etudefantasia.gg` (and
`www.` variants) 301 onto it. DNS lives in Cloudflare; certificates are
issued by Fly (see `deploy/etude-fly.sh` at the repo root for operations).

Design tokens are hand-copied from `frontend/src/app.css` (the source of
truth — see `VISUAL_DESIGN.md`); if the two drift, this site is the bug.
Fonts and mana symbols are copied from `frontend/static/`; the board
screenshot is `frontend/e2e/visual-references/v3/board-developed.png`.

## Develop

```bash
cd website
uv run --extra test pytest         # smoke tests
uv run python main.py              # serve on :5002
```

## Deploy

```bash
deploy/etude-fly.sh deploy         # from the repo root
```
