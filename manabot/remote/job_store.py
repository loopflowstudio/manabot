"""Private S3 job control and least-privilege, expiring worker credentials.

Conditional create claims are permanent fences, not renewable leases. A timed-out
PUT may have succeeded; callers reconcile reads and never repeat provider create.
Only the worker's runtime prefix is writable with the delegated STS session.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
import time
from typing import Any, Protocol

from manabot.infra.artifacts import split_s3_uri

from .jobs import Cancellation, Job


@dataclass(frozen=True)
class StoredValue:
    data: bytes
    etag: str


class JobStore(Protocol):
    def read(self, key: str) -> StoredValue | None: ...
    def create(self, key: str, data: bytes) -> bool: ...
    def replace(self, key: str, data: bytes, etag: str) -> bool: ...


def cancellation_requested(store: JobStore) -> float | None:
    """Read the shared mailbox; an empty or absent request means no cancellation."""
    value = store.read("cancel.json")
    if value is None:
        return None
    return Cancellation.model_validate_json(value.data).requested_at


class S3JobStore:
    def __init__(self, prefix: str) -> None:
        import boto3
        from botocore.config import Config

        self.bucket, self.prefix = split_s3_uri(prefix)
        # SDK responses are narrowed at this adapter. Bound every network call;
        # the independent guardian is not subject to these timeouts.
        self.client: Any = boto3.client(
            "s3",
            config=Config(
                connect_timeout=5,
                read_timeout=10,
                retries={"max_attempts": 1},
            ),
        )

    def _key(self, key: str) -> str:
        if key.startswith("/") or ".." in key.split("/"):
            raise ValueError("invalid job control key")
        return f"{self.prefix.rstrip('/')}/{key}"

    def read(self, key: str) -> StoredValue | None:
        from botocore.exceptions import ClientError

        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
        except ClientError as error:
            if error.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
                return None
            raise RuntimeError("job storage read unavailable") from None
        body = response["Body"]
        try:
            data = body.read(16 * 1024**2 + 1)
            if len(data) > 16 * 1024**2:
                raise ValueError("job control object exceeds limit")
            return StoredValue(data, str(response["ETag"]))
        finally:
            body.close()

    def _put(self, key: str, data: bytes, condition: dict[str, str]) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=self._key(key),
                Body=data,
                ContentType="application/json",
                **condition,
            )
            return True
        except ClientError as error:
            if error.response["ResponseMetadata"]["HTTPStatusCode"] == 412:
                return False
            raise RuntimeError(
                "job storage write uncertain; reconcile by job ID"
            ) from None

    def create(self, key: str, data: bytes) -> bool:
        return self._put(key, data, {"IfNoneMatch": "*"})

    def replace(self, key: str, data: bytes, etag: str) -> bool:
        return self._put(key, data, {"IfMatch": etag})


WORKER_ROLE = "manabot-remote-jobs"


def worker_policy(prefix_uri: str, deadline: float | None = None) -> dict[str, object]:
    """Only control reads and runtime evidence writes; no delete or account access."""
    bucket, prefix = split_s3_uri(prefix_uri)
    arn = f"arn:aws:s3:::{bucket}/{prefix.rstrip('/')}"
    statements: list[dict[str, object]] = [
        {
            "Effect": "Allow",
            "Action": ["s3:GetObject"],
            "Resource": [
                f"{arn}/spec.json",
                f"{arn}/cancel.json",
                f"{arn}/training.json",
            ],
        },
        {
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:GetObjectVersion", "s3:PutObject"],
            "Resource": [f"{arn}/runtime/*"],
        },
    ]

    if deadline is not None:
        # Explicit deny also bounds resource-policy grants to this session. STS
        # has a 900-second minimum and may issue tokens beyond a short allocation.
        statements.append(
            {
                "Effect": "Deny",
                "Action": "*",
                "Resource": "*",
                "Condition": {
                    "DateGreaterThanEquals": {
                        "aws:CurrentTime": datetime.fromtimestamp(
                            deadline, timezone.utc
                        ).isoformat()
                    }
                },
            }
        )
    return {"Version": "2012-10-17", "Statement": statements}


def configure_worker_role(destination: str) -> None:
    """Explicit one-time setup using the current AWS identity, including SSO.

    Creates one role restricted to this private job prefix and trusted only by
    the current caller's IAM principal. Existing roles must match exactly; this
    command never broadens an existing trust/policy or changes the storage bucket.
    """
    import boto3
    from botocore.exceptions import ClientError

    session = boto3.Session()
    iam, sts = session.client("iam"), session.client("sts")
    identity = sts.get_caller_identity()
    principal = identity["Arn"]
    if ":assumed-role/" in principal:
        name = principal.split(":assumed-role/", 1)[1].split("/", 1)[0]
        principal = iam.get_role(RoleName=name)["Role"]["Arn"]
    if ":root" in principal:
        raise ValueError("worker setup requires an IAM user or role, not root")
    trust = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": principal},
                "Action": "sts:AssumeRole",
            }
        ],
    }
    policy = worker_policy(f"{destination.rstrip('/')}/*")
    try:
        role = iam.get_role(RoleName=WORKER_ROLE)["Role"]
    except ClientError as error:
        if error.response["Error"]["Code"] != "NoSuchEntity":
            raise RuntimeError("worker role discovery unavailable") from None
        role = iam.create_role(
            RoleName=WORKER_ROLE,
            AssumeRolePolicyDocument=json.dumps(trust),
            MaxSessionDuration=43200,
            Description="Private manabot remote job evidence; no provider account access",
        )["Role"]
    if role["AssumeRolePolicyDocument"] != trust:
        raise ValueError("existing worker trust differs; no authorization was changed")
    try:
        previous = iam.get_role_policy(RoleName=WORKER_ROLE, PolicyName="job-evidence")[
            "PolicyDocument"
        ]
    except ClientError as error:
        if error.response["Error"]["Code"] != "NoSuchEntity":
            raise RuntimeError("worker policy discovery unavailable") from None
        iam.put_role_policy(
            RoleName=WORKER_ROLE,
            PolicyName="job-evidence",
            PolicyDocument=json.dumps(policy),
        )
    else:
        if previous != policy:
            raise ValueError(
                "existing worker storage scope differs; no policy was changed"
            )


def worker_credentials(spec: Job) -> dict[str, str]:
    """Use a job-scoped STS session; account keys never reach a rental.

    SSO/role chaining supports at most one hour. Longer jobs require IAM-user
    federation or a directly assumable configured role; expiry is always admitted.
    """
    import boto3

    # Issuance can use a dedicated IAM principal without replacing the client's
    # ordinary SSO identity for control-plane reads/writes. Its credentials stay
    # local; only the policy-restricted STS result enters the rental environment.
    issuer_profile = os.environ.get("MANABOT_REMOTE_ISSUER_PROFILE")
    role_arn = os.environ.get("MANABOT_REMOTE_ROLE_ARN")
    if issuer_profile and not role_arn:
        raise ValueError("a dedicated issuer requires an explicit worker role ARN")
    try:
        session = (
            boto3.Session(profile_name=issuer_profile)
            if issuer_profile
            else boto3.Session()
        )
        credentials = session.get_credentials()
        if credentials is None:
            raise RuntimeError("missing credentials")
        frozen = credentials.get_frozen_credentials()
    except Exception:
        # credential_process failures may include captured secret-bearing output.
        raise RuntimeError(
            "AWS issuer credentials unavailable; no rental created"
        ) from None
    remaining = spec.deadline - time.time()
    if remaining <= 0:
        raise ValueError("allocation has expired; worker issuance/renewal forbidden")
    duration = max(900, math.ceil(remaining) + 60)
    policy = json.dumps(
        worker_policy(spec.prefix, spec.deadline), separators=(",", ":")
    )
    try:
        sts = session.client("sts")
        if frozen.token is None and duration > 3600:
            principal = sts.get_caller_identity()["Arn"]
            if ":user/" not in principal:
                raise ValueError(
                    "long worker sessions require a dedicated IAM-user issuer"
                )
        if frozen.token is None and role_arn is None:
            result = sts.get_federation_token(
                Name=f"manabot-{spec.identity[:24]}",
                DurationSeconds=duration,
                Policy=policy,
            )
        else:
            # AWS applies the one-hour chaining cap to temporary-role sources.
            # Direct IAM-user AssumeRole supports the role's configured duration;
            # the service validates that limit and we independently admit expiry.
            if frozen.token is not None and duration > 3600:
                raise ValueError(
                    "SSO/role sessions require job wall time below 3540 seconds"
                )
            arn = role_arn
            if arn is None:
                account = sts.get_caller_identity()["Account"]
                arn = f"arn:aws:iam::{account}:role/{WORKER_ROLE}"
            result = sts.assume_role(
                RoleArn=arn,
                RoleSessionName=f"manabot-{spec.identity[:24]}",
                DurationSeconds=duration,
                Policy=policy,
            )
        delegated = result["Credentials"]
        if delegated["Expiration"].timestamp() < spec.deadline:
            raise ValueError("worker credentials expire before billing deadline")
        return {
            "AWS_ACCESS_KEY_ID": str(delegated["AccessKeyId"]),
            "AWS_SECRET_ACCESS_KEY": str(delegated["SecretAccessKey"]),
            "AWS_SESSION_TOKEN": str(delegated["SessionToken"]),
            "AWS_DEFAULT_REGION": session.region_name or "us-west-2",
        }
    except ValueError:
        raise
    except Exception:
        raise RuntimeError(
            "scoped STS session unavailable; run deploy setup-worker or configure MANABOT_REMOTE_ROLE_ARN; no rental created"
        ) from None
