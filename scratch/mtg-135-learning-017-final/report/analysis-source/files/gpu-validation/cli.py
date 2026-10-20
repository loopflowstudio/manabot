"""Use the configured Etude issuer in process; never persist credential values."""

import hashlib
import json
import os
from pathlib import Path

import boto3
from botocore.config import Config
try:
    from etude.cli import main
except ImportError:
    from etude.launch.cli import main
from etude.launch.storage.s3 import Bucket


def configure() -> None:
    try:
        issuer = json.loads(os.environ["ETUDE_RUNPOD_ISSUER_CREDENTIALS"])
        values = {
            "AWS_ACCESS_KEY_ID": issuer["AccessKeyId"],
            "AWS_SECRET_ACCESS_KEY": issuer["SecretAccessKey"],
            "AWS_DEFAULT_REGION": issuer["Region"],
        }
        if not all(isinstance(value, str) and value for value in values.values()):
            raise ValueError
    except (KeyError, ValueError, TypeError):
        raise SystemExit(
            "Etude issuer credentials are unavailable or invalid"
        ) from None
    for key in list(os.environ):
        if key.startswith(("AWS_", "ETUDE_", "MANABOT_")):
            del os.environ[key]
    os.environ.update(values)
    os.environ.update(
        ETUDE_HOME=str(Path(__file__).resolve().parent / "launcher"),
        AWS_CONFIG_FILE="/dev/null",
        AWS_SHARED_CREDENTIALS_FILE="/dev/null",
        AWS_EC2_METADATA_DISABLED="true",
    )


def use_prepared_source(receipt_path: Path | None = None) -> None:
    """Copy only a fully verified identical source object, without a second upload."""
    receipt_path = receipt_path or Path(__file__).parent / "prepared-source.json"
    if not receipt_path.exists():
        return
    receipt = json.loads(receipt_path.read_text())
    if receipt["status"] != "verified":
        raise RuntimeError("prepared source is not verified")
    original = Bucket.put
    client = boto3.client(
        "s3",
        region_name="us-west-2",
        config=Config(connect_timeout=10, read_timeout=60, retries={"max_attempts": 1}),
    )

    def put(self: Bucket, key: str, body: bytes, *, once: bool = False) -> None:
        if (
            not once
            and self.name == receipt["bucket"]
            and key.startswith("etude/fantasia-gpu-validation-20261014/")
            and key.endswith(f"/sources/{receipt['commit']}.tar")
        ):
            if (
                len(body) != receipt.get("source_tar_bytes", receipt["bytes"])
                or hashlib.sha256(body).hexdigest() != receipt.get("source_tar_sha256", receipt["sha256"])
            ):
                raise RuntimeError(
                    "source bytes differ from the verified prepared object"
                )
            client.copy_object(
                Bucket=self.name,
                Key=key,
                CopySource={"Bucket": self.name, "Key": receipt["key"]},
            )
            metadata = client.head_object(Bucket=self.name, Key=key)
            if (
                metadata["ContentLength"] != receipt["bytes"]
                or metadata["Metadata"].get("sha256") != receipt["sha256"]
            ):
                raise RuntimeError("copied source receipt differs")
            (Path(__file__).parent / "source-copy.json").write_text(
                json.dumps(
                    {
                        "key": key,
                        "sha256": receipt["sha256"],
                        "bytes": receipt["bytes"],
                    },
                    indent=2,
                )
                + "\n"
            )
            return
        original(self, key, body, once=once)

    Bucket.put = put


if __name__ == "__main__":
    configure()
    use_prepared_source()
    raise SystemExit(main())
