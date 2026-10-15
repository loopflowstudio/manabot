"""Mini-side Discord publisher with local Keychain secrets and durable receipts.

A pending receipt blocks retries after an ambiguous network failure: Discord has
no idempotency key. Reconcile that attempt before clearing it manually. The
webhook URL and response bodies never enter logs or receipts.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from manabot.verify.learnability.schema import Result


def payload(result: Result, evidence_url: str) -> bytes:
    parsed = urlsplit(evidence_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("An actual HTTPS evidence URL is required")
    failures = sorted({s.family for s in result.scores if not s.passed})
    content = (
        f"manabot nightly: **{result.status}**\n"
        f"`{result.source[:12]}` / {result.world or 'world unavailable'} · "
        f"{result.wall_seconds:.1f}s CPU-only wall\n"
        f"Seeds {result.config.seeds}; failed families: {', '.join(failures) or 'none recorded'}\n"
        f"Errors: {len(result.errors)}; explicit skips: {len(result.skips)}\n"
        f"{evidence_url}\nCapacity and toy PPO diagnostics; no strength claim."
    )
    return json.dumps({"content": content, "allowed_mentions": {"parse": []}}).encode()


def _webhook(service: str) -> str:
    command = subprocess.run(
        ["security", "find-generic-password", "-s", service, "-w"],
        capture_output=True,
        text=True,
        check=False,
    )
    if command.returncode:
        raise RuntimeError("Local Keychain webhook unavailable")
    url = command.stdout.strip()
    parts = urlsplit(url)
    if (
        parts.scheme != "https"
        or parts.hostname != "discord.com"
        or not parts.path.startswith("/api/webhooks/")
        or parts.query
        or parts.fragment
    ):
        raise ValueError("Keychain item is not a Discord webhook URL")
    return url


def publish(report: Path, state: Path, evidence_url: str, service: str) -> bool:
    result = Result.model_validate_json(report.read_text())
    if result.status == "running":
        raise ValueError("Cannot publish an unfinished run")
    body = payload(result, evidence_url)
    # One destination/day, derived from run start in UTC, including failures.
    day = (
        datetime.fromisoformat(result.started_at)
        .astimezone(timezone.utc)
        .date()
        .isoformat()
    )
    key = hashlib.sha256(service.encode()).hexdigest()[:16] + "-" + day
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    receipt = state / (key + ".json")
    with (state / "publish.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt.exists():
            previous = json.loads(receipt.read_text())
            if previous["status"] == "sent":
                return False
            raise RuntimeError(
                "Pending publication requires reconciliation; refusing duplicate"
            )
        url = _webhook(service)
        data = {
            "status": "pending",
            "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            "day": day,
        }
        with receipt.open("x") as handle:
            json.dump(data, handle)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            with urlopen(
                Request(
                    url + "?wait=true",
                    data=body,
                    headers={"Content-Type": "application/json"},
                ),
                timeout=20,
            ) as response:
                message = json.load(response)
        except (HTTPError, URLError, TimeoutError):
            raise RuntimeError(
                "Discord delivery unconfirmed; reconcile pending receipt"
            ) from None
        message_id = message.get("id")
        if not isinstance(message_id, str) or not message_id.isdigit():
            raise RuntimeError(
                "Discord receipt missing message ID; reconcile pending receipt"
            )
        data.update(status="sent", message_id=message_id)
        temporary = receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps(data))
        temporary.replace(receipt)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--evidence-url", required=True)
    parser.add_argument("--keychain-service", default="manabot-nightly-discord")
    args = parser.parse_args()
    print(
        "posted"
        if publish(args.report, args.state, args.evidence_url, args.keychain_service)
        else "already posted"
    )


if __name__ == "__main__":
    main()
