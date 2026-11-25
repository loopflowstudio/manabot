"""Read-only payment explanations and deliberate taps share the live authority."""

import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

import managym
from managym.academy import server, trace as trace_store
from managym.experience import ExperienceFrame
from managym.experience_projection import describe_actions
from managym.payment import PaymentPreview, preview_payment
from tests.academy.socket_messages import settled_message


def _keep_opening_hands(env: managym.Env) -> None:
    for _ in range(2):
        env.step(0)


def _root(game: server.GameSession) -> ExperienceFrame:
    assert game.env is not None
    frame = ExperienceFrame.model_validate(game._wire_message()["frame"])
    while keep := next((o for o in frame.offers if o.label == "Keep hand"), None):
        assert frame.prompt is not None
        outcome = game.hero_command(
            {
                "command_id": f"keep-{frame.prompt.id}",
                "match_id": frame.match_id,
                "expected_revision": frame.revision,
                "prompt_id": frame.prompt.id,
                "offer_id": keep.id,
                "answers": [],
            }
        )
        assert outcome["status"] == "accepted"
        frame = ExperienceFrame.model_validate(outcome["update"]["frame"])
    game.env.scenario_clear_hand(0)
    game.env.scenario_force_card_in_hand(0, "Ancestral Recall")
    game.env.scenario_force_battlefield(0, "Island", ready=True)
    game.env.scenario_force_battlefield(0, "Tranquil Cove", ready=True)
    game.obs = game.env.scenario_refresh()
    game.published_prompt = None
    return ExperienceFrame.model_validate(game._wire_message()["frame"])


def test_payment_preview_is_authenticated_read_only_and_stale_after_tap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(trace_store, "TRACES_DIR", tmp_path)
    server.SESSION_REGISTRY.clear()
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/play") as websocket:
            websocket.send_json(
                {
                    "type": "new_game",
                    "config": {
                        "hero_deck": {
                            "Island": 20,
                            "Tranquil Cove": 4,
                            "Ancestral Recall": 8,
                            "Prophetic Prism": 8,
                        },
                        "villain_deck": {"Plains": 40},
                        "villain_type": "passive",
                        "seed": 165,
                        "auto_pass": False,
                    },
                }
            )
            initial = settled_message(websocket)
            record = server.SESSION_REGISTRY[initial["session_id"]]
            frame = _root(record.game)
            assert frame.prompt is not None
            cast = next(
                offer
                for offer in frame.offers
                if offer.label == "Cast Ancestral Recall"
            )
            endpoint = f"/api/sessions/{record.session_id}/payment-preview"
            request = {
                "match_id": frame.match_id,
                "revision": frame.revision,
                "prompt_id": frame.prompt.id,
                "frame_hash": frame.frame_hash,
                "offer_id": cast.id,
            }
            headers = {"x-etude-participant-token": initial["resume_token"]}
            assert client.post(endpoint, json=request).status_code == 403
            assert (
                client.post(
                    endpoint,
                    json=request,
                    headers={"x-etude-participant-token": "wrong"},
                ).status_code
                == 403
            )
            record.participants["watcher"] = server.ParticipantLease(
                viewer_id="watcher",
                resume_token="watcher-test-token",
                role=server.ViewerRole.WATCHER,
            )
            assert (
                client.post(
                    endpoint,
                    json=request,
                    headers={"x-etude-participant-token": "watcher-test-token"},
                ).status_code
                == 403
            )
            del record.participants["watcher"]
            assert record.game.env is not None
            before = record.game.env.search_witness_json()
            prompt_count = record.game._next_prompt_id
            decisions = len(record.game.canonical_decisions)
            accepted = len(record.game.accepted_commands)
            response = client.post(endpoint, json=request, headers=headers)
            assert response.status_code == 200, response.text
            preview = response.json()["preview"]
            assert preview["state"] == "exact"
            assert preview["cost"] == "{U}"
            assert preview["sources"]
            assert before == record.game.env.search_witness_json()
            assert record.game._next_prompt_id == prompt_count
            assert len(record.game.canonical_decisions) == decisions
            assert len(record.game.accepted_commands) == accepted
            for change in (
                {"match_id": "other"},
                {"revision": frame.revision + 1},
                {"prompt_id": frame.prompt.id + 1},
                {"frame_hash": "other"},
                {"offer_id": -1},
            ):
                assert (
                    client.post(
                        endpoint, json={**request, **change}, headers=headers
                    ).status_code
                    == 409
                )
            proposed = {
                source["source"]["id"]["entity"] for source in preview["sources"]
            }

            def choose(current: ExperienceFrame, offer_id: int) -> ExperienceFrame:
                assert current.prompt is not None
                outcome = record.game.hero_command(
                    {
                        "command_id": f"pay-{current.prompt.id}",
                        "match_id": current.match_id,
                        "expected_revision": current.revision,
                        "prompt_id": current.prompt.id,
                        "offer_id": offer_id,
                        "answers": [],
                    }
                )
                assert outcome["status"] == "accepted"
                return ExperienceFrame.model_validate(outcome["update"]["frame"])

            payment = choose(frame, cast.id)
            pay = next(o for o in payment.offers if o.action_type == "PAY_COST")
            assert payment.prompt is not None
            address = {
                **request,
                "revision": payment.revision,
                "prompt_id": payment.prompt.id,
                "frame_hash": payment.frame_hash,
                "offer_id": pay.id,
            }
            plan = client.post(endpoint, json=address, headers=headers).json()[
                "preview"
            ]
            assert proposed == {s["source"]["id"]["entity"] for s in plan["sources"]}
            paid = choose(payment, pay.id)
            assert {
                p.id for p in paid.projection.agent.battlefield if p.tapped
            } == proposed
            assert (
                client.post(endpoint, json=address, headers=headers).status_code == 409
            )


def _native_preview(env: managym.Env, index: int) -> PaymentPreview:
    address = json.loads(env.semantic_decision_frame_json())
    return preview_payment(
        env,
        viewer=address["actor"],
        revision=address["revision"],
        fingerprint=address["fingerprint"],
        action_index=index,
    )


def test_native_preview_names_unannounced_costs_and_refreshes_for_exact_target() -> (
    None
):
    env = managym.Env(seed=165)
    env.reset(
        [
            managym.PlayerConfig(
                "payer",
                {
                    "Island": 16,
                    "Swamp": 8,
                    "Mountain": 8,
                    "Wan Shi Tong, Librarian": 2,
                    "Firebending Lesson": 2,
                    "Swampsnare Trap": 4,
                },
            ),
            managym.PlayerConfig(
                "other", {"Forest": 32, "Spectral Sailor": 4, "Llanowar Elves": 4}
            ),
        ]
    )
    _keep_opening_hands(env)
    env.scenario_clear_hand(0)
    for card in ("Wan Shi Tong, Librarian", "Firebending Lesson", "Swampsnare Trap"):
        env.scenario_force_card_in_hand(0, card)
    for card in ("Island", "Island", "Island", "Island", "Swamp", "Mountain"):
        env.scenario_force_battlefield(0, card, ready=True)
    for card in ("Spectral Sailor", "Llanowar Elves"):
        env.scenario_force_battlefield(1, card, ready=True)
    obs = env.scenario_refresh()
    labels = describe_actions(obs)
    expected = {
        "Wan Shi Tong, Librarian": "X",
        "Firebending Lesson": "optional",
        "Swampsnare Trap": "target-dependent",
    }
    for card, explanation in expected.items():
        action = next(row for row in labels if row["description"] == f"Cast {card}")
        result = _native_preview(env, int(action["index"]))
        assert result.state == "needs_choice"
        assert explanation in result.explanation
        assert result.sources == []
    cast = next(row for row in labels if row["description"] == "Cast Swampsnare Trap")
    obs, *_ = env.step(int(cast["index"]))
    targets = describe_actions(obs)
    assert len(targets) == 2
    costs: set[str | None] = set()
    before = env.search_witness_json()
    for row in targets:
        result = _native_preview(env, int(row["index"]))
        assert result.state == "exact"
        costs.add(result.cost)
    assert costs == {"{1}{B}", "{2}{B}"}
    assert env.search_witness_json() == before


def test_unaffordable_kicker_does_not_hide_the_exact_base_payment() -> None:
    env = managym.Env(seed=165)
    env.reset(
        [
            managym.PlayerConfig("payer", {"Mountain": 32, "Firebending Lesson": 8}),
            managym.PlayerConfig("other", {"Forest": 32, "Llanowar Elves": 8}),
        ]
    )
    _keep_opening_hands(env)
    env.scenario_clear_hand(0)
    env.scenario_force_card_in_hand(0, "Firebending Lesson")
    env.scenario_force_battlefield(0, "Mountain", ready=True)
    env.scenario_force_battlefield(1, "Llanowar Elves", ready=True)
    obs = env.scenario_refresh()
    cast = next(
        row
        for row in describe_actions(obs)
        if row["description"] == "Cast Firebending Lesson"
    )
    result = _native_preview(env, int(cast["index"]))
    assert result.state == "exact"
    assert result.cost == "{R}"
    assert len(result.sources) == 1


@pytest.mark.parametrize(
    ("label", "hand", "lands", "expected"),
    [
        (
            "Cast Prophetic Prism",
            ["Prophetic Prism", "Lightning Bolt"],
            ["Mountain", "Island", "Island"],
            ["Island", "Island"],
        ),
        (
            "Activate ability: Abandoned Air Temple",
            [],
            ["Abandoned Air Temple", "Plains", "Island", "Island", "Island"],
            ["Plains", "Island", "Island", "Island"],
        ),
    ],
)
def test_preview_preserves_followups_and_reserves_activation_source(
    label: str,
    hand: list[str],
    lands: list[str],
    expected: list[str],
) -> None:
    env = managym.Env(seed=165)
    env.reset(
        [
            managym.PlayerConfig("payer", dict.fromkeys(hand + lands, 8)),
            managym.PlayerConfig("other", {"Island": 32, "Spectral Sailor": 8}),
        ]
    )
    _keep_opening_hands(env)
    # A real response on the stack keeps forced passes from reaching untap.
    env.scenario_force_card_in_hand(1, "Spectral Sailor")
    env.scenario_force_battlefield(1, "Island", ready=True)
    env.scenario_clear_hand(0)
    for name in hand:
        env.scenario_force_card_in_hand(0, name)
    for name in lands:
        env.scenario_force_battlefield(0, name, ready=True)
    obs = env.scenario_refresh()
    index = next(
        int(row["index"])
        for row in describe_actions(obs)
        if row["description"] == label
    )
    proposal = _native_preview(env, index)
    assert [s.label for s in proposal.sources] == expected
    assert proposal.pool_after_spending is not None
    assert not any(proposal.pool_after_spending.amounts)
    predicted = {
        s.source.id.entity for s in proposal.sources if s.source.kind == "object"
    }
    if label.startswith("Activate"):
        # The ability's own tap cost is separate from mana production.
        predicted.add(int(describe_actions(obs)[index]["focus"][0]))
    address = json.loads(env.semantic_decision_frame_json())
    with pytest.raises(managym.AgentError):
        preview_payment(
            env,
            viewer=address["actor"],
            revision=address["revision"],
            fingerprint="different",
            action_index=index,
        )
    obs, *_ = env.step(index)
    with pytest.raises(managym.AgentError):
        preview_payment(
            env,
            viewer=address["actor"],
            revision=address["revision"],
            fingerprint=address["fingerprint"],
            action_index=index,
        )
    if label.startswith("Cast"):
        pay = next(row for row in describe_actions(obs) if row["type"] == "PAY_COST")
        obs, *_ = env.step(int(pay["index"]))
    assert {
        p.id for p in [*obs.agent_permanents, *obs.opponent_permanents] if p.tapped
    } == predicted
