"""Point Cloudflare DNS for the Etude domains at the Fly-hosted website.

Mirrors the loopflow.studio zone pattern: a proxied CNAME per hostname to the
app's unique Fly alias, a DNS-only _acme-challenge CNAME so Fly can issue
certificates behind the Cloudflare proxy, and a _fly-ownership TXT record
(required when traffic routes through a proxy).

Run through the wrapper so the token comes from Doppler (etude/prd):

    deploy/etude-fly.sh dns
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

API = "https://api.cloudflare.com/client/v4"

ZONES = {
    "etude.gg": os.environ["CLOUDFLARE_ZONE_ID_ETUDE_GG"],
    "etudefantasia.com": os.environ["CLOUDFLARE_ZONE_ID_ETUDEFANTASIA_COM"],
    "etudefantasia.gg": os.environ["CLOUDFLARE_ZONE_ID_ETUDEFANTASIA_GG"],
}

# hostname -> (fly app, DNS alias prefix printed by `flyctl certs setup`).
# Every zone's apex and www point at the website; play is its own app.
WEBSITE = ("etude-website", "pqm95g3")
HOSTS = {
    **{
        host: WEBSITE
        for zone in ZONES
        for host in (zone, f"www.{zone}")
    },
    "play.etude.gg": ("etude-play", "261enw9"),
}


def api(method: str, path: str, body: dict | None = None) -> dict:
    request = urllib.request.Request(
        f"{API}{path}",
        method=method,
        headers={
            "Authorization": f"Bearer {os.environ['CLOUDFLARE_API_TOKEN']}",
            "Content-Type": "application/json",
        },
        data=json.dumps(body).encode() if body is not None else None,
    )
    with urllib.request.urlopen(request) as response:
        payload = json.load(response)
    if not payload.get("success"):
        raise RuntimeError(f"{method} {path}: {payload.get('errors')}")
    return payload


def desired_records(zone: str) -> list[dict]:
    records = []
    for host, (app, prefix) in HOSTS.items():
        if not (host == zone or host.endswith(f".{zone}")):
            continue
        records += [
            {
                "type": "CNAME",
                "name": host,
                "content": f"{prefix}.{app}.fly.dev",
                "proxied": True,
            },
            {
                "type": "CNAME",
                "name": f"_acme-challenge.{host}",
                "content": f"{host}.{prefix}.flydns.net",
                "proxied": False,
            },
            {
                "type": "TXT",
                "name": f"_fly-ownership.{host}",
                "content": f"app-{prefix}",
                "proxied": False,
            },
        ]
    return records


def sync_zone(zone: str, zone_id: str) -> None:
    existing = api("GET", f"/zones/{zone_id}/dns_records?per_page=200")["result"]
    by_name = {}
    for record in existing:
        by_name.setdefault(record["name"], []).append(record)

    for want in desired_records(zone):
        current = by_name.get(want["name"], [])
        match = next((r for r in current if r["type"] == want["type"]), None)
        # A managed name carries exactly one record; placeholders of another
        # type (the parked A records) are replaced, unmanaged names untouched.
        for stale in current:
            if stale is not match:
                api("DELETE", f"/zones/{zone_id}/dns_records/{stale['id']}")
                print(f"  deleted {stale['type']} {stale['name']} -> {stale['content']}")
        body = {**want, "ttl": 1}
        if match:
            same = (
                match["content"].rstrip(".") == want["content"].rstrip(".")
                and bool(match.get("proxied")) == want["proxied"]
            )
            if same:
                print(f"  ok      {want['type']} {want['name']}")
                continue
            api("PUT", f"/zones/{zone_id}/dns_records/{match['id']}", body)
            print(f"  updated {want['type']} {want['name']} -> {want['content']}")
        else:
            api("POST", f"/zones/{zone_id}/dns_records", body)
            print(f"  created {want['type']} {want['name']} -> {want['content']}")


def main() -> None:
    argparse.ArgumentParser().parse_args()
    for zone, zone_id in ZONES.items():
        print(f"{zone}:")
        sync_zone(zone, zone_id)


if __name__ == "__main__":
    sys.exit(main())
