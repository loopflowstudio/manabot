"""Hosted requests retain the shared selection protocol and honest billing."""

import asyncio
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import httpx
import pytest
import torch

from manabot.env import ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.infra.identity import file_sha256
from manabot.llm.budget import BudgetConfig, BudgetLedger
from manabot.llm.openrouter import OpenRouterConfig, OpenRouterResult, OpenRouterSession
from manabot.llm.schema import DecisionInput
from manabot.llm.selection import REPAIR, decision_messages
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import save_checkpoint
from manabot.verify.llm_game import CheckpointOpponent, GameConfig, play
import managym


def decision() -> DecisionInput:
    return DecisionInput(
        text="Complete visible state.",
        legal_references=("A0", "A1"),
        representation_id="test-v1",
        source_identity={"capture_digest": "test"},
    )


def reply(text: str | None = "A1", finish: str = "stop") -> dict[str, object]:
    return {
        "id": "gen-test",
        "model": "test/model",
        "provider": "fixture",
        "choices": [{"message": {"content": text}, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 30,
            "total_tokens": 130,
            "cost": 0.001,
            "prompt_tokens_details": {"cached_tokens": 80, "cache_write_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 27},
        },
    }


@pytest.fixture(autouse=True)
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-secret")


@pytest.fixture
def budget_ledger(tmp_path: Path) -> BudgetLedger:
    return BudgetLedger(
        tmp_path / "spend.sqlite",
        BudgetConfig(
            ceiling_usd=Decimal("1"),
            request_bound_usd=Decimal("0.01"),
            overhead_usd=Decimal(0),
            charge_multiplier=Decimal(1),
            bound_evidence="fixture",
        ),
    )


def test_exact_request_and_usage_without_model_memory() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://openrouter.ai/api/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer fixture-secret"
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=reply())

    with OpenRouterSession(
        OpenRouterConfig(model_id="openai/gpt-6-sol"),
        transport=httpx.MockTransport(handler),
    ) as session:
        first = session.decide(decision())
        second = session.decide(decision())
    assert first.origin == second.origin == "initial"
    assert first.selected_reference == "A1"
    assert requests[0] == requests[1]
    assert requests[0]["messages"] == [
        m.model_dump() for m in decision_messages(decision())
    ]
    assert requests[0]["provider"] == {
        "require_parameters": True,
        "allow_fallbacks": False,
    }
    assert requests[0]["prompt_cache_options"] == {"mode": "explicit"}
    assert requests[0]["transforms"] == []
    assert first.attempts[0].completion.usage.cost == 0.001
    assert first.attempts[0].output_tokens == 30
    assert "fixture-secret" not in first.model_dump_json()
    assert OpenRouterResult.model_validate_json(first.model_dump_json()) == first


def test_winning_objective_is_explicit_and_retained_through_repair() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=reply("bad" if len(requests) == 1 else "A1"))

    with OpenRouterSession(
        OpenRouterConfig(model_id="test/model", objective="win-game"),
        transport=httpx.MockTransport(handler),
    ) as session:
        result = session.decide(decision())
    first = requests[0]["messages"][0]["content"]
    assert first.startswith("You are playing Magic: The Gathering")
    assert "maximizes your probability of winning" in first
    assert decision().text in first
    assert requests[0]["messages"][0] == requests[1]["messages"][0]
    assert result.config.objective == "win-game"
    assert result.origin == "retry"


@pytest.mark.parametrize(
    "finish,text", [("stop", "A99"), ("length", "A1"), ("length", None)]
)
def test_one_repair_keeps_identical_cached_prefix(
    finish: str, text: str | None
) -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200, json=reply(text, finish) if len(requests) == 1 else reply()
        )

    with OpenRouterSession(
        OpenRouterConfig(model_id="test/model", cache_prompt=True),
        transport=httpx.MockTransport(handler),
    ) as session:
        result = session.decide(decision())
    assert result.origin == "retry"
    assert requests[0]["messages"][0] == requests[1]["messages"][0]
    assert requests[1]["messages"][1:] == [
        {"role": "assistant", "content": text or ""},
        {"role": "user", "content": REPAIR},
    ]
    assert requests[0]["messages"][0]["content"][0]["cache_control"] == {
        "type": "ephemeral"
    }
    assert sum(a.completion.usage.cost for a in result.attempts) == 0.002


def test_two_invalid_responses_fall_back_deterministically() -> None:
    def run() -> OpenRouterResult:
        with OpenRouterSession(
            OpenRouterConfig(model_id="test/model"),
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json=reply("bad"))
            ),
        ) as session:
            return session.decide(decision())

    first, second = run(), run()
    assert first.origin == "fallback" and first.failure is None
    assert first.fallback == second.fallback
    assert first.selected_reference in decision().legal_references
    assert len(first.attempts) == 2


@pytest.mark.parametrize("status", [400, 401, 402, 429, 500, 503])
def test_http_failures_are_not_retried_or_fallbacks(status: int) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, json={"error": {"message": "fixture-secret"}})

    with OpenRouterSession(
        OpenRouterConfig(model_id="test/model"), transport=httpx.MockTransport(handler)
    ) as session:
        result = session.decide(decision())
        assert session.closed
    assert calls == 1
    assert result.failure.kind == "runtime" and result.origin is None
    assert str(status) in result.failure.message
    assert "fixture-secret" not in result.model_dump_json()


@pytest.mark.parametrize(
    "payload",
    [
        {"error": {"code": 503}},
        {"choices": []},
        reply("A1", "error"),
        reply("A1", "content_filter"),
    ],
)
def test_bad_envelopes_and_provider_failures_never_select(
    payload: dict[str, object],
) -> None:
    with OpenRouterSession(
        OpenRouterConfig(model_id="test/model"),
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)),
    ) as session:
        result = session.decide(decision())
    assert result.failure.kind == "runtime"
    assert result.selected_reference is None and result.fallback is None
    assert len(result.attempts) == 1


def test_total_deadline_cancels_pending_request_and_retains_progress() -> None:
    cancelled = False

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal cancelled
        try:
            await asyncio.Event().wait()
        finally:
            cancelled = True
        raise AssertionError("unreachable")

    snapshots: list[OpenRouterResult] = []
    with OpenRouterSession(
        OpenRouterConfig(model_id="test/model", deadline_seconds=0.05),
        transport=httpx.MockTransport(handler),
    ) as session:
        result = session.decide(decision(), progress=snapshots.append)
    assert cancelled
    assert result.failure.kind == "deadline" and result.selected_reference is None
    assert snapshots[0].attempts[0].completion is None
    assert result.attempts[0].completion is None


def test_missing_key_and_byte_overflow_do_not_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not send")

    transport = httpx.MockTransport(handler)
    with OpenRouterSession(
        OpenRouterConfig(model_id="test/model", max_prompt_bytes=1), transport=transport
    ) as session:
        assert session.decide(decision()).failure.kind == "overflow"
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with OpenRouterSession(
        OpenRouterConfig(model_id="test/model"), transport=transport
    ) as session:
        assert "OPENROUTER_API_KEY" in session.decide(decision()).failure.message


def test_missing_usage_is_unknown_not_zero() -> None:
    payload = reply()
    del payload["usage"]
    with OpenRouterSession(
        OpenRouterConfig(model_id="test/model"),
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)),
    ) as session:
        result = session.decide(decision())
    assert result.origin == "initial"
    assert result.attempts[0].prompt_tokens is None
    assert result.attempts[0].completion.usage is None


def test_failed_repair_retains_first_charge_and_stops() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return (
            httpx.Response(200, json=reply("bad"))
            if calls == 1
            else httpx.Response(429)
        )

    with OpenRouterSession(
        OpenRouterConfig(model_id="test/model"), transport=httpx.MockTransport(handler)
    ) as session:
        result = session.decide(decision())
    assert result.origin is None and result.failure.kind == "runtime"
    assert calls == 2
    assert result.attempts[0].completion.usage.cost == 0.001
    assert result.attempts[1].completion is None


def test_provider_refusal_cannot_select_a_legal_looking_answer() -> None:
    payload = reply("A1")
    payload["choices"] = [
        {"message": {"content": "A1", "refusal": "refused"}, "finish_reason": "stop"}
    ]
    with OpenRouterSession(
        OpenRouterConfig(model_id="test/model"),
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)),
    ) as session:
        result = session.decide(decision())
    assert result.selected_reference is None and result.failure.kind == "runtime"
    assert result.attempts[0].completion.usage.cost == 0.001


def test_hosted_game_replays_and_accounts_for_every_attempt(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        messages = json.loads(request.content)["messages"]
        refs = (
            messages[0]["content"]
            .split("Legal action references:\n")[1]
            .split("\n\n")[0]
            .splitlines()
        )
        return httpx.Response(200, json=reply(refs[-1]))

    with OpenRouterSession(
        OpenRouterConfig(model_id="test/model"), transport=httpx.MockTransport(handler)
    ) as session:
        summary = play(GameConfig(max_decisions=8), tmp_path / "game", session)
    assert summary.replay_matches and summary.decisions == 8
    assert summary.model_decisions > 0
    assert summary.reported_cost_usd == pytest.approx(0.001 * summary.model_decisions)
    assert summary.prompt_tokens == 100 * summary.model_decisions
    assert summary.reasoning_tokens == 27 * summary.model_decisions
    assert summary.unpriced_attempts == 0
    assert len(summary.model_latencies) == summary.model_decisions
    assert summary.observations["llm_wait_and_dispatch"] >= 0
    assert summary.llm_phases == {}


def test_preloaded_campaign_opponent_keeps_timing_and_model_seat_control(
    tmp_path: Path,
) -> None:
    config = GameConfig(max_decisions=8)
    path = tmp_path / "policy.pt"
    space = ObservationSpace(ObservationConfig(policy_input_version=1))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(91)
        agent = ActorCritic(space, ModelConfig(hidden_dim=16, attention_layers=1))
    save_checkpoint(
        agent,
        space,
        path,
        player_configs=[
            managym.PlayerConfig(str(seat), config.deck) for seat in (0, 1)
        ],
    )
    opponent = CheckpointOpponent(path, file_sha256(path), path.stat().st_size)
    summary = play(config, tmp_path / "game", None, opponent=opponent)
    assert summary.replay_matches and summary.decisions == 8
    assert summary.mode == "random-v-checkpoint-control"
    assert summary.origins["checkpoint-stochastic-cpu"] > 0
    assert summary.origins["random-control"] > 0
    assert (
        len(summary.opponent_latencies) == summary.origins["checkpoint-stochastic-cpu"]
    )
    assert summary.observations["policy_inference"] >= 0


def test_checkpoint_opponent_preserves_legacy_inputs_and_replays(
    tmp_path: Path,
) -> None:
    config = GameConfig(max_decisions=8, opponent_checkpoint=tmp_path / "policy.pt")
    space = ObservationSpace(ObservationConfig(policy_input_version=1))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(91)
        agent = ActorCritic(space, ModelConfig(hidden_dim=16, attention_layers=1))
    save_checkpoint(
        agent,
        space,
        config.opponent_checkpoint,
        player_configs=[
            managym.PlayerConfig(str(seat), config.deck) for seat in (0, 1)
        ],
    )
    summary = play(config, tmp_path / "game", None)
    assert summary.replay_matches and summary.decisions == 8
    assert summary.mode == "random-v-checkpoint-control"
    assert summary.origins["checkpoint-opponent"] > 0
    identity = json.loads((tmp_path / "game/opponent.json").read_text())
    assert identity["policy_input_version"] == 1
    assert identity["policy_history_version"] == 0
    assert identity["sha256"]


def test_game_stops_before_next_hosted_decision_at_spend_threshold(
    tmp_path: Path,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=reply("A0"))

    with OpenRouterSession(
        OpenRouterConfig(model_id="test/model"), transport=httpx.MockTransport(handler)
    ) as session:
        summary = play(
            GameConfig(max_reported_cost_usd=0.0005), tmp_path / "game", session
        )
    assert calls == 1
    assert summary.failure == "RuntimeError: reported spend threshold reached"
    assert summary.reported_cost_usd == 0.001
    assert summary.replay_matches


def test_game_stops_on_missing_cost_before_another_request(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        payload = reply("A0")
        del payload["usage"]
        return httpx.Response(200, json=payload)

    with OpenRouterSession(
        OpenRouterConfig(model_id="test/model"), transport=httpx.MockTransport(handler)
    ) as session:
        summary = play(GameConfig(max_reported_cost_usd=2), tmp_path / "game", session)
    assert calls == 1 and summary.unpriced_attempts == 1
    assert "missing costs" in summary.failure
    assert summary.replay_matches


def test_checkpoint_admission_failure_retains_a_summary(tmp_path: Path) -> None:
    out = tmp_path / "game"
    summary = play(GameConfig(opponent_checkpoint=tmp_path / "missing.pt"), out, None)
    assert not summary.completed and summary.decisions == 0
    assert "FileNotFoundError" in summary.failure
    assert summary.final_digest is None
    assert json.loads((out / "summary.json").read_text()) == summary.model_dump()


def typed_decision(count: int = 2) -> DecisionInput:
    refs = tuple(f"A{i}" for i in range(count))
    return DecisionInput(
        text="Complete viewer-safe observation, with all actions.",
        legal_references=refs,
        action_descriptions={r: f"Action {r}" for r in refs},
        representation_id="fixture",
        source_identity={"capture_digest": "fixture"},
    )


def typed_reply(refs: tuple[str, ...] = ("A0", "A1")) -> dict[str, object]:
    return {
        "id": "gen-dec-fixture",
        "model": "typesafe/jev-1.14-dated",
        "provider": "TypeSafe",
        "answers": {
            "action": {
                "type": "choice",
                "choice": refs[-1],
                "confidence": 0.1,
                # Deliberately different argmax: only the returned choice selects.
                "probabilities": {ref: float(i == 0) for i, ref in enumerate(refs)},
            }
        },
        "usage": {"input_tokens": 400, "output_tokens": 31, "cost": 0.0000168},
    }


def typed_config() -> OpenRouterConfig:
    return OpenRouterConfig(
        model_id="typesafe/jev-1.14",
        transport="decisions",
        objective="win-game",
    )


def test_decisions_full_choice_coverage_and_native_receipt() -> None:
    supplied = typed_decision(255)

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://openrouter.ai/api/alpha/decisions"
        body = json.loads(request.content)
        assert body["state"] == supplied.text
        question = body["questions"]["action"]
        assert set(question["criteria"]) == set(supplied.legal_references)
        assert question["criteria"]["A254"] == "A254: Action A254"
        assert "winning this game" in question["instructions"]
        assert "messages" not in body
        return httpx.Response(200, json=typed_reply(supplied.legal_references))

    with OpenRouterSession(
        typed_config(), transport=httpx.MockTransport(handler)
    ) as session:
        result = session.decide(supplied)
    assert result.selected_reference == "A254"
    assert result.attempts[0].completion is None
    assert result.attempts[0].decision_reply.model == "typesafe/jev-1.14-dated"
    assert result.attempts[0].decision_reply.provider == "TypeSafe"
    assert result.attempts[0].prompt_tokens == 400
    assert result.runtime.runtime == "openrouter-decisions-v1"
    assert OpenRouterResult.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("supplied", [typed_decision(256), decision()])
def test_decisions_reject_incomplete_descriptions_or_excess_choices_before_send(
    supplied: DecisionInput,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("preflight failure cannot dispatch")

    with OpenRouterSession(
        typed_config(), transport=httpx.MockTransport(handler)
    ) as session:
        result = session.decide(supplied)
    assert result.failure and result.selected_reference is None
    assert not result.attempts


@pytest.mark.parametrize(
    "payload",
    [
        {"error": "fixture-secret"},
        {**typed_reply(), "answers": {}},
        {
            **typed_reply(),
            "answers": {
                "action": {
                    "type": "choice",
                    "choice": "A9",
                    "confidence": 1,
                    "probabilities": {"A9": 1},
                }
            },
        },
    ],
)
def test_decisions_bad_answers_stop_without_repair_and_retain_charges(
    payload: dict[str, object], budget_ledger: BudgetLedger
) -> None:
    with OpenRouterSession(
        typed_config(),
        budget=budget_ledger,
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)),
    ) as session:
        result = session.decide(typed_decision())
    assert result.failure and result.selected_reference is None
    assert len(result.attempts) == 1 and result.fallback is None
    assert "fixture-secret" not in result.model_dump_json()
    spend = budget_ledger.snapshot()
    # A rejected action is still billable; a malformed envelope stays uncertain.
    if "usage" in payload:
        assert spend.reported_inclusive_usd == Decimal("0.0000168")
        assert spend.pending_usd == 0
    else:
        assert spend.reported_inclusive_usd == 0
        assert spend.pending_usd == Decimal("0.01")


def test_decisions_native_game_usage_and_replay(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        refs = tuple(json.loads(request.content)["questions"]["action"]["criteria"])
        return httpx.Response(200, json=typed_reply(refs))

    with OpenRouterSession(
        typed_config(), transport=httpx.MockTransport(handler)
    ) as session:
        summary = play(GameConfig(max_decisions=8), tmp_path / "game", session)
    assert summary.replay_matches and summary.decisions == 8
    assert summary.mode == "openrouter-decisions-v-random"
    assert summary.prompt_tokens == 400 * summary.model_decisions
    assert summary.completion_tokens == 31 * summary.model_decisions
    assert summary.reported_cost_usd == pytest.approx(
        0.0000168 * summary.model_decisions
    )
    assert summary.unpriced_attempts == 0
    assert summary.reasoning_tokens == 0


def test_chat_repair_cannot_dispatch_past_durable_budget(tmp_path: Path) -> None:
    ledger = BudgetLedger(
        tmp_path / "spend.sqlite",
        BudgetConfig(
            ceiling_usd=Decimal("0.0015"),
            request_bound_usd=Decimal("0.001"),
            overhead_usd=Decimal(0),
            charge_multiplier=Decimal(1),
            bound_evidence="fixture",
        ),
    )
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, json=reply("bad"))

    with OpenRouterSession(
        OpenRouterConfig(model_id="fixture/model"),
        budget=ledger,
        transport=httpx.MockTransport(handler),
    ) as session:
        result = session.decide(decision())
    assert requests == 1
    assert result.failure and "budget exhausted" in result.failure.message
    assert result.attempts[0].dispatched and not result.attempts[1].dispatched
    assert result.fallback is None
    assert ledger.snapshot().accounted_usd == Decimal("0.001")


@pytest.mark.parametrize("typed", [False, True])
def test_budget_digest_identifies_exact_dispatched_payload(
    budget_ledger: BudgetLedger, typed: bool
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        (charge,) = budget_ledger.snapshot().requests
        assert charge.request_sha256 == hashlib.sha256(request.content).hexdigest()
        assert request.headers["Content-Type"] == "application/json"
        assert "雪" in request.content.decode()
        return httpx.Response(200, json=typed_reply() if typed else reply())

    supplied = typed_decision().model_copy(update={"text": "Visible state: 雪"})
    config = typed_config() if typed else OpenRouterConfig(model_id="fixture/model")
    with OpenRouterSession(
        config, budget=budget_ledger, transport=httpx.MockTransport(handler)
    ) as session:
        result = session.decide(supplied)
    assert result.failure is None and result.selected_reference == "A1"
    assert budget_ledger.snapshot().pending_usd == 0


@pytest.mark.parametrize("failure", ["missing", "deadline", "http"])
def test_decisions_uncertain_charge_stops_and_keeps_reservation(
    failure: str, budget_ledger: BudgetLedger
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert budget_ledger.snapshot().pending_usd == Decimal("0.01")
        if failure == "deadline":
            await asyncio.Event().wait()
        if failure == "http":
            return httpx.Response(503)
        return httpx.Response(200, json={**typed_reply(), "usage": None})

    config = typed_config().model_copy(update={"deadline_seconds": 0.05})
    with OpenRouterSession(
        config, budget=budget_ledger, transport=httpx.MockTransport(handler)
    ) as session:
        result = session.decide(typed_decision())
    assert result.failure is not None and result.selected_reference is None
    assert result.attempts[0].reservation_id
    assert budget_ledger.snapshot().pending_usd == Decimal("0.01")
    with pytest.raises(RuntimeError, match="unsettled"):
        budget_ledger.reserve("next", "digest")


@pytest.mark.parametrize("failures", [1, 2])
def test_decisions_resend_only_unsent_requests_and_keep_their_bounds(
    failures: int, budget_ledger: BudgetLedger, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def no_wait(seconds: float) -> None:
        pass

    monkeypatch.setattr(asyncio, "sleep", no_wait)
    sent = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sent
        sent += 1
        if sent <= failures:
            raise httpx.ConnectError("unreachable", request=request)
        return httpx.Response(200, json=typed_reply())

    config = typed_config().model_copy(update={"connect_retries": 1})
    with OpenRouterSession(
        config, budget=budget_ledger, transport=httpx.MockTransport(handler)
    ) as session:
        result = session.decide(typed_decision())
    spend = budget_ledger.snapshot()
    expected = [True] * failures + [False] * (2 - failures)
    assert [a.connect_failed for a in result.attempts] == expected
    assert [r.abandoned for r in spend.requests] == expected
    if failures == 1:
        assert result.selected_reference == "A1" and result.failure is None
        assert spend.accounted_usd == Decimal("0.0100168")
        budget_ledger.reserve("next", "digest")
    else:
        assert result.failure and "ConnectError" in result.failure.message
        assert spend.accounted_usd == Decimal("0.02")
