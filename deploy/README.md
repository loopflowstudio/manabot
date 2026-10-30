# Fly deploy

Manabot's public surfaces run on Fly.io behind Cloudflare DNS,
following the kata/loopflow deployment pattern:

- **fantasia-website** — the manabot.gg site (`website/`), a FastHTML app serving the blog.
  Serves `manabot.gg` (canonical); `etude.gg`, `etudefantasia.com`,
  `etudefantasia.gg`, and all `www.` variants 301 onto it.
- **academy-play** — hosted play at `play.manabot.gg` and `academy.manabot.gg`:
  the experience server (`academy/server.py` + the managym engine) serving the built Svelte
  frontend as a same-origin SPA (`deploy/play/`). One machine, because
  game state lives in process memory.

```bash
deploy/academy-fly.sh deploy                        # manabot.gg site
ACADEMY_FLY_APP=academy-play deploy/academy-fly.sh deploy   # hosted play
deploy/academy-fly.sh dns                           # sync Cloudflare records
deploy/academy-fly.sh health                        # check every domain
```

Both Fly apps belong to the `loopflow-studio` organization. Jack Heart approved
the live service rename on 2026-10-14: `academy-play` serves the web app and
`fantasia-website` serves the project site (manabot.gg). Jack Heart subsequently
selected Manabot and `manabot.gg` on the same day. Existing Fly app identifiers remain operational names; the Doppler project
was renamed to `manabot` at Jack Heart's request on 2026-10-14. Legacy website hosts redirect to
`manabot.gg`; `play.etude.gg` redirects HTTP to `play.manabot.gg` with status 308
to preserve request methods. Both preserve encoded paths and query strings.
Existing legacy WebSocket clients remain supported.

Jack Heart selected `play.manabot.gg` and `academy.manabot.gg` as equivalent
addresses on 2026-10-14. Both serve the same SPA, API and WebSocket endpoints
directly from the single `academy-play` machine. Browser storage remains
separate for each origin.

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

The Cloudflare API token and zone IDs live in Doppler under `manabot/prd`
(`doppler secrets --project manabot --config prd --only-names`). Nothing
secret is stored in this repository, and the Fly apps currently need no
runtime secrets.

`manabot/prd` contains the five Cloudflare entries needed for these domains
and `NAMECHEAP_API_USER` / `NAMECHEAP_API_KEY` for registrar DNS management. The separate `etude` project still owns experiment-system credentials;
do not rename it or copy those credentials into the deployment project. Frozen
S3 receipts and wandb history retain their existing names.

## Cost

Both apps ride shared-cpu machines with `auto_stop_machines` — idle cost
is minimal. Keep total spend inside the studio's existing $100/month
automation budget (see kata/deploy/COSTS.md).

## Domain cutover

Namecheap remains the registrar; Cloudflare hosts DNS. Add each purchased domain
to the same Cloudflare account, preserve any existing DNS records (including
mail and verification records), then set Namecheap Custom DNS to the exact
nameservers Cloudflare assigns that zone. Do not use Namecheap URL forwarding.
For `manabot.gg`, store `CLOUDFLARE_ZONE_ID_MANABOT_GG` in `manabot/prd`.

Add Fly certificates and publish the new DNS records before deploying the
redirects. Verify Cloudflare reports an active zone, HTTPS works on the new
website and play hosts, and Fly certificates are ready. Only then deploy both
apps and verify legacy nested URLs and query strings, plus gameplay on the new
host. Keep the old domains and their certificates for redirects.
