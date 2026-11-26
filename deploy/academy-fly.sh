#!/usr/bin/env bash
# Manage the Fly-hosted Manabot website (mirrors kata/deploy/kata-fly.sh):
# the manabot.gg site (fantasia-website).
# DNS lives in Cloudflare; the API token comes from Doppler (manabot/prd).

set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: academy-fly.sh <create|deploy|certs|dns|status|logs|health|open>

Commands:
  create   Create the Fly app if it does not exist
  deploy   Deploy the app image to Fly
  certs    Add/list Fly certificates for the app's domains
  dns      Sync Cloudflare DNS for every domain (requires Doppler manabot/prd)
  status   Show Fly app status
  logs     Tail Fly logs
  health   Check every public domain
  open     Open the Fly dashboard

Environment:
  ACADEMY_FLY_APP=fantasia-website
  ACADEMY_FLY_ORG=loopflow-studio
  ACADEMY_FLY_REGION=sjc
USAGE
}

command="${1:-}"
if [[ -z "$command" || "$command" == "-h" || "$command" == "--help" ]]; then
  usage
  exit 0
fi
shift || true

app="${ACADEMY_FLY_APP:-fantasia-website}"
org="${ACADEMY_FLY_ORG:-loopflow-studio}"
region="${ACADEMY_FLY_REGION:-sjc}"
repo="$(git rev-parse --show-toplevel)"
website_domains=(manabot.gg www.manabot.gg etude.gg www.etude.gg etudefantasia.com www.etudefantasia.com etudefantasia.gg www.etudefantasia.gg)

config="$repo/website/fly.toml"
context="$repo/website"
domains=("${website_domains[@]}")

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
    flyctl deploy "$context" --config "$config" --app "$app" --regions "$region"
    ;;
  certs)
    for d in "${domains[@]}"; do
      flyctl certs add "$d" -a "$app" || true
    done
    flyctl certs list -a "$app"
    ;;
  dns)
    require_tool doppler
    doppler run --project manabot --config prd -- \
      uv run --no-project python "$repo/deploy/academy-dns.py"
    ;;
  status)
    flyctl status -a "$app"
    ;;
  logs)
    flyctl logs -a "$app"
    ;;
  health)
    for d in "${website_domains[@]}"; do
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
