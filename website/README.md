# Manabot website

The site for [manabot.gg](https://manabot.gg): a FastHTML app, deployed on
Fly.io, that serves the static essays in `blog/` (`/` redirects to the current
one), `robots.txt`, `sitemap.xml` and `/healthz`.

`manabot.gg` is canonical; `etude.gg`, `etudefantasia.com`, `etudefantasia.gg`,
and all `www.` variants 301 onto it, preserving encoded paths and queries. DNS lives in Cloudflare; certificates are
issued by Fly (see `deploy/academy-fly.sh` at the repo root for operations).

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
