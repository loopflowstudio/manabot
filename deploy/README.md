# Fly deploy

Manabot's public surfaces run on Fly.io behind Cloudflare DNS,
following the kata/loopflow deployment pattern:

- **fantasia-website** — the landing site (`website/`), a FastHTML app.
  Serves `etude.gg` (canonical); `www.etude.gg`, `etudefantasia.com`,
  `etudefantasia.gg`, and their `www.` variants 301 onto it.
- **academy-play** — hosted play at `play.etude.gg`: the experience server
  (`academy/server.py` + the managym engine) serving the built Svelte
  frontend as a same-origin SPA (`deploy/play/`). One machine, because
  game state lives in process memory.

```bash
deploy/academy-fly.sh deploy                        # landing site
ACADEMY_FLY_APP=academy-play deploy/academy-fly.sh deploy   # hosted play
deploy/academy-fly.sh dns                           # sync Cloudflare records
deploy/academy-fly.sh health                        # check every domain
```

Both Fly apps belong to the `loopflow-studio` organization. Jack Heart approved
the live service rename on 2026-10-14: `academy-play` serves the web app and
`fantasia-website` serves the project's landing site. The `etude.gg` and
`etudefantasia` hostnames stay unchanged.

## DNS and certificates

Cloudflare is the DNS authority; certificates are issued by Fly. Each
hostname carries three records (the loopflow.studio pattern): a proxied
CNAME to the app's unique Fly alias, a DNS-only `_acme-challenge` CNAME so
Fly can renew certificates behind the Cloudflare proxy, and a
`_fly-ownership` TXT record. `deploy/academy-dns.py` holds the desired state
and syncs it idempotently; the hostname → app/alias table at the top of
that file is the contract. New hostname: `flyctl certs add`, copy the
alias `flyctl certs setup` prints into the table, run
`deploy/academy-fly.sh dns`.

## Secrets

The Cloudflare API token and zone IDs live in Doppler under `fantasia/prd`
(`doppler secrets --project fantasia --config prd --only-names`). Nothing
secret is stored in this repository, and the Fly apps currently need no
runtime secrets.

`fantasia/prd` contains only the four Cloudflare entries needed for these
domains. The separate `etude` project still owns experiment-system credentials;
do not rename it or copy those credentials into the deployment project. Frozen
S3 receipts and wandb history retain their existing names.

## Cost

Both apps ride shared-cpu machines with `auto_stop_machines` — idle cost
is minimal. Keep total spend inside the studio's existing $100/month
automation budget (see kata/deploy/COSTS.md).
