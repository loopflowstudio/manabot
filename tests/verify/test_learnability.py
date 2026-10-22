"""Failure reports and publication receipts must be as reliable as green scores."""

from pathlib import Path
from unittest.mock import Mock
from urllib.error import URLError

import pytest
import torch

from manabot.env import ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.model import ActorCritic
from manabot.verify.action_cases import (
    curriculum,
    decision_case,
    identity_removed,
    reorder_objects,
    unseen_definitions,
)
from manabot.verify.learnability import publish as publisher
from manabot.verify.learnability.__main__ import supervise
from manabot.verify.learnability.report import save
from manabot.verify.learnability.schema import Config, Result


def result() -> Result:
    return Result(
        source="abc",
        source_digest="def",
        config=Config(),
        platform="test",
        started_at="2026-10-15T00:00:00+00:00",
        status="failed",
        errors=["deliberate failure"],
    )


def test_timeout_retains_failure_and_refuses_overwrite(tmp_path: Path) -> None:
    out = tmp_path / "run"
    assert supervise(out, Config(timeout_seconds=0)) == 124
    report = Result.model_validate_json((out / "result.json").read_text())
    assert report.status == "timeout"
    assert report.errors and not report.scores
    assert "timeout" in (out / "report.md").read_text()
    with pytest.raises(FileExistsError):
        supervise(out, Config(timeout_seconds=0))
    assert (out / "result.json").exists()


def test_native_outcomes_and_object_order_invariance() -> None:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        agent = ActorCritic(
            ObservationSpace(), ModelConfig(hidden_dim=8, num_attention_heads=2)
        )
        agent.eval()
        cases = curriculum()
        assert len(cases) == 48
        for case in cases:
            if case.native:
                assert len(case.outcomes) == 2 and case.outcomes[0] != case.outcomes[1]
            else:
                assert case.family == "ability" and not case.outcomes
            with torch.no_grad():
                expected, _ = agent(case.observation)
                actual, _ = agent(reorder_objects(case.observation))
                torch.testing.assert_close(
                    actual.softmax(-1), expected.softmax(-1), atol=1e-6, rtol=1e-5
                )
                removed, _ = agent(identity_removed(case))
                first, second = case.candidates[:2]
                torch.testing.assert_close(
                    removed[0, first], removed[0, second], atol=1e-7, rtol=0
                )
    finally:
        torch.set_num_threads(previous)


def test_held_out_cases_only_show_fit_definitions() -> None:
    # curriculum() asserts the rule; the native seed-112 variant must trip it.
    stale = [c for c in curriculum() if c.family == "learn" and c.split == "fit"]
    stale.append(decision_case("learn", variant=1, split="held_out"))
    assert unseen_definitions(stale) == {"learn/1/0": {3}}


def test_discord_failure_receipt_and_duplicate_prevention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save(tmp_path, result())
    report = tmp_path / "result.json"
    state = tmp_path / "receipts"
    monkeypatch.setattr(
        publisher,
        "_webhook",
        lambda service: "https://discord.com/api/webhooks/123/secret",
    )
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.read.return_value = b'{"id":"987654321"}'
    send = Mock(return_value=response)
    monkeypatch.setattr(publisher, "urlopen", send)
    assert publisher.publish(report, state, "https://example.org/real-artifact", "test")
    assert not publisher.publish(
        report, state, "https://example.org/real-artifact", "test"
    )
    assert send.call_count == 1
    receipt = next(state.glob("*.json")).read_text()
    assert '"status": "sent"' in receipt
    assert "secret" not in receipt
    # Ambiguous network delivery must never be blindly retried the next day/job.
    send.side_effect = URLError("secret URL must not escape")
    with pytest.raises(RuntimeError, match="delivery unconfirmed") as error:
        publisher.publish(
            report, tmp_path / "pending", "https://example.org/real-artifact", "test"
        )
    assert "secret" not in str(error.value)
    with pytest.raises(RuntimeError, match="requires reconciliation"):
        publisher.publish(
            report, tmp_path / "pending", "https://example.org/real-artifact", "test"
        )
    assert send.call_count == 2


def test_discord_payload_includes_failed_status_without_mentions() -> None:
    body = publisher.payload(result(), "https://example.org/real-artifact").decode()
    assert "failed" in body and '"parse": []' in body
