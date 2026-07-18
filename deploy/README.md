# Fly deploy

Etude Fantasia's public surfaces run on Fly.io behind Cloudflare DNS,
following the kata/loopflow deployment pattern:

- **etude-website** — the landing site (`website/`), a FastHTML app.
  Serves `etude.gg` (canonical); `www.etude.gg`, `etudefantasia.com`,
  `etudefantasia.gg`, and their `www.` variants 301 onto it.
- **etude-play** — hosted play at `play.etude.gg`: the experience server
  (`etude/server.py` + the managym engine) serving the built Svelte
  frontend as a same-origin SPA (`deploy/play/`). One machine, because
  game state lives in process memory.

```bash
deploy/etude-fly.sh deploy                        # landing site
ETUDE_FLY_APP=etude-play deploy/etude-fly.sh deploy   # hosted play
deploy/etude-fly.sh dns                           # sync Cloudflare records
deploy/etude-fly.sh health                        # check every domain
```

## DNS and certificates

Cloudflare is the DNS authority; certificates are issued by Fly. Each
hostname carries three records (the loopflow.studio pattern): a proxied
CNAME to the app's unique Fly alias, a DNS-only `_acme-challenge` CNAME so
Fly can renew certificates behind the Cloudflare proxy, and a
`_fly-ownership` TXT record. `deploy/etude-dns.py` holds the desired state
and syncs it idempotently; the hostname → app/alias table at the top of
that file is the contract. New hostname: `flyctl certs add`, copy the
alias `flyctl certs setup` prints into the table, run
`deploy/etude-fly.sh dns`.

## Secrets

The Cloudflare API token and zone IDs live in Doppler under `etude/prd`
(`doppler secrets --project etude --config prd --only-names`). Nothing
secret is stored in this repository, and the Fly apps currently need no
runtime secrets.

## Cost

Both apps ride shared-cpu machines with `auto_stop_machines` — idle cost
is minimal. Keep total spend inside the studio's existing $100/month
automation budget (see kata/deploy/COSTS.md).
