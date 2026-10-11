"""One tenant-side selection/session entry point; only the public client reaches Etude."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import UUID

from etude_client.api import Client
from etude_client.collector import HTTPReadSource
from etude_client.errors import EtudeError
from etude_client.models import RegistrationReceipt

from .fantasia import FantasiaImport
from .live_fantasia import NativeSelection, refresh_native
from .research import import_citations
from .transports import S3Source


def import_collection(
    client: Client, root: Path, collection: str, revision: str | None = None
) -> RegistrationReceipt:
    imported = FantasiaImport(root, collection, revision=revision)
    receipt = client.register_record(imported.registration())
    for batch in imported.batches(receipt.experiment_id):
        client.ingest(batch)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    retained = modes.add_parser("retained")
    retained.add_argument("root", type=Path)
    retained.add_argument("collection", help="retained collection or research-evidence")
    retained.add_argument("--revision", help="read exact Git commit, not working files")
    pinned = modes.add_parser("native")
    pinned.add_argument("experiment_id", type=UUID)
    pinned.add_argument("selection", type=Path)
    pinned.add_argument("objects", type=Path)
    current = modes.add_parser("s3")
    current.add_argument("experiment_id", type=UUID)
    current.add_argument("destination", help="exact native jobs S3 prefix")
    current.add_argument("jobs", nargs="+")
    args = parser.parse_args()
    try:
        token = Path(os.environ["ETUDE_TOKEN_FILE"]).read_text().strip()
        with Client(os.environ["ETUDE_URL"], token) as client:
            if args.mode == "retained":
                if args.collection == "research-evidence" and args.revision is not None:
                    raise EtudeError("citations_already_pin_revisions", 422)
                receipts = (
                    import_citations(client, args.root)
                    if args.collection == "research-evidence"
                    else (import_collection(client, args.root, args.collection, args.revision),)
                )
                print(json.dumps([r.model_dump(mode="json") for r in receipts]))
            else:
                source = (
                    S3Source(args.destination, tuple(args.jobs))
                    if args.mode == "s3"
                    else HTTPReadSource.from_json(args.objects.read_bytes())
                )
                selection = (
                    None
                    if args.mode == "s3"
                    else NativeSelection.model_validate_json(args.selection.read_bytes())
                )
                refresh_native(client, source, args.experiment_id, selection)
    except (EtudeError, ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired):
        print("fantasia_import_failed", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
