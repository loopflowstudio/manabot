"""Only delegated job-scoped credentials may cross the rental boundary."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json

import boto3
from botocore.credentials import Credentials
import pytest

from manabot.remote import job_store
from manabot.remote.jobs import RemoteJobSpec
from manabot.remote.plan import HardwareMix, compile_plan
from tests.remote.test_compile import ROOT, SOURCE


@dataclass
class Assumption:
    role: str
    duration: int
    policy: dict[str, object]


class STS:
    def __init__(self, expiration: float) -> None:
        self.expiration = expiration
        self.calls: list[Assumption] = []

    def get_caller_identity(self) -> dict[str, str]:
        return {"Arn": "arn:aws:iam::123456789012:user/manabot-issuer"}

    def assume_role(
        self, *, RoleArn: str, RoleSessionName: str, DurationSeconds: int, Policy: str
    ) -> dict[str, dict[str, object]]:
        self.calls.append(Assumption(RoleArn, DurationSeconds, json.loads(Policy)))
        return {
            "Credentials": {
                "AccessKeyId": "delegated-id",
                "SecretAccessKey": "delegated-secret",
                "SessionToken": "delegated-token",
                "Expiration": datetime.fromtimestamp(self.expiration, timezone.utc),
            }
        }


class Session:
    region_name = "us-west-2"

    def __init__(self, sts: STS, token: str | None) -> None:
        self.sts, self.token = sts, token

    def get_credentials(self) -> Credentials:
        return Credentials("issuer-id", "issuer-secret", self.token)

    def client(self, service: str) -> STS:
        assert service == "sts"
        return self.sts


def specification() -> RemoteJobSpec:
    mix = HardwareMix.model_validate_json(
        (ROOT / "ops/mixes/runpod-small.json").read_text()
    )
    mix = HardwareMix.model_validate(
        mix.model_dump()
        | {
            "wall_seconds": 25200,
            "hourly_ceiling": 0.49,
            "dollar_cap": 4.5,
        }
    )
    plan = compile_plan(
        (ROOT / "experiments/regimes/direct-self-play.json").read_text(),
        mix,
        SOURCE,
        10351,
    )
    return RemoteJobSpec(
        job_id="credential-test", plan=plan, created_at=1000, deadline=26200
    )


@pytest.mark.parametrize("token", [None, "sso-session"])
def test_direct_issuer_and_role_chaining_have_distinct_limits(
    monkeypatch: pytest.MonkeyPatch, token: str | None
) -> None:
    spec = specification()
    sts = STS(spec.deadline + 60)
    profiles: list[str | None] = []

    def session(*, profile_name: str | None = None) -> Session:
        profiles.append(profile_name)
        return Session(sts, token)

    monkeypatch.setattr(boto3, "Session", session)
    monkeypatch.setattr(job_store.time, "time", lambda: 1000)
    monkeypatch.setenv("MANABOT_REMOTE_ISSUER_PROFILE", "manabot-issuer")
    monkeypatch.setenv(
        "MANABOT_REMOTE_ROLE_ARN", "arn:aws:iam::123456789012:role/manabot-remote-jobs"
    )
    if token is not None:
        with pytest.raises(ValueError, match="below 3540"):
            job_store.worker_credentials(spec)
        assert not sts.calls
    else:
        result = job_store.worker_credentials(spec)
        assert result["AWS_ACCESS_KEY_ID"] == "delegated-id"
        assert result["AWS_SECRET_ACCESS_KEY"] == "delegated-secret"
        assert result["AWS_SESSION_TOKEN"] == "delegated-token"
        assert sts.calls[0].duration == 25260
        assert sts.calls[0].policy == job_store.worker_policy(spec.prefix)
        assert "issuer-secret" not in json.dumps(result)
    assert profiles == ["manabot-issuer"]


def test_short_expiry_and_implicit_issuer_role_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = specification()
    sts = STS(spec.deadline - 1)
    monkeypatch.setattr(boto3, "Session", lambda **kwargs: Session(sts, None))
    monkeypatch.setattr(job_store.time, "time", lambda: 1000)
    monkeypatch.setenv("MANABOT_REMOTE_ISSUER_PROFILE", "manabot-issuer")
    monkeypatch.delenv("MANABOT_REMOTE_ROLE_ARN", raising=False)
    with pytest.raises(ValueError, match="explicit worker role"):
        job_store.worker_credentials(spec)
    monkeypatch.setenv(
        "MANABOT_REMOTE_ROLE_ARN", "arn:aws:iam::123456789012:role/manabot-remote-jobs"
    )
    with pytest.raises(ValueError, match="expire before"):
        job_store.worker_credentials(spec)


def test_credential_provider_failure_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_session() -> Session:
        raise RuntimeError("secret-bearing credential_process output")

    monkeypatch.delenv("MANABOT_REMOTE_ISSUER_PROFILE", raising=False)
    monkeypatch.setattr(boto3, "Session", failed_session)
    with pytest.raises(
        RuntimeError, match="AWS issuer credentials unavailable"
    ) as caught:
        job_store.worker_credentials(specification())
    assert "secret-bearing" not in str(caught.value)
    assert caught.value.__suppress_context__
