#!/usr/bin/env bash
# Manage the Fly-hosted Etude Fantasia apps (mirrors kata/deploy/kata-fly.sh):
# the landing site (etude-website) and hosted play (etude-play).
# DNS lives in Cloudflare; the API token comes from Doppler (etude/prd).

set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: etude-fly.sh <create|deploy|certs|dns|status|logs|health|open>

Commands:
  create   Create the Fly app if it does not exist
  deploy   Deploy the app image to Fly
  certs    Add/list Fly certificates for the app's domains
  dns      Sync Cloudflare DNS for every domain (requires Doppler etude/prd)
  status   Show Fly app status
  logs     Tail Fly logs
  health   Check every public domain
  open     Open the Fly dashboard

Environment:
  ETUDE_FLY_APP=etude-website   # or etude-play
  ETUDE_FLY_ORG=personal
  ETUDE_FLY_REGION=sjc
USAGE
}

command="${1:-}"
if [[ -z "$command" || "$command" == "-h" || "$command" == "--help" ]]; then
  usage
  exit 0
fi
shift || true

app="${ETUDE_FLY_APP:-etude-website}"
org="${ETUDE_FLY_ORG:-personal}"
region="${ETUDE_FLY_REGION:-sjc}"
repo="$(git rev-parse --show-toplevel)"
website_domains=(etude.gg www.etude.gg etudefantasia.com www.etudefantasia.com etudefantasia.gg www.etudefantasia.gg)
play_domains=(play.etude.gg)

case "$app" in
  etude-play)
    config="$repo/deploy/play/fly.toml"
    context="$repo"
    deploy_flags=(--ha=false) # game state is in-process; exactly one machine
    domains=("${play_domains[@]}")
    ;;
  *)
    config="$repo/website/fly.toml"
    context="$repo/website"
    deploy_flags=()
    domains=("${website_domains[@]}")
    ;;
esac

require_tool() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "$1 is required" >&2
    exit 1
  }
}

require_tool flyctl

case "$command" in
  create)
    if flyctl apps list --json | grep -q "\"$app\""; then
      echo "app $app already exists"
    else
      flyctl apps create "$app" --org "$org"
    fi
    ;;
  deploy)
    # ${arr[@]+...} keeps `set -u` happy on bash 3.2 when the array is empty
    flyctl deploy "$context" --config "$config" --app "$app" --regions "$region" ${deploy_flags[@]+"${deploy_flags[@]}"}
    ;;
  certs)
    for d in "${domains[@]}"; do
      flyctl certs add "$d" -a "$app" || true
    done
    flyctl certs list -a "$app"
    ;;
  dns)
    require_tool doppler
    doppler run --project etude --config prd -- \
      uv run --no-project python "$repo/deploy/etude-dns.py"
    ;;
  status)
    flyctl status -a "$app"
    ;;
  logs)
    flyctl logs -a "$app"
    ;;
  health)
    for d in "${website_domains[@]}" "${play_domains[@]}"; do
      printf '%-24s ' "$d"
      curl -s -o /dev/null -w '%{http_code} -> %{redirect_url}\n' "https://$d/" || echo unreachable
    done
    ;;
  open)
    flyctl dashboard -a "$app"
    ;;
  *)
    usage
    exit 1
    ;;
esac
