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


def _root(game: server.GameSession) -> ExperienceFrame:
    assert game.env is not None
    game.env.scenario_clear_hand(0)
    game.env.scenario_force_card_in_hand(0, "Ancestral Recall")
    for _ in range(2):
        game.env.scenario_force_battlefield(0, "Island", ready=True)
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
                            "Island": 24,
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
            initial = websocket.receive_json()
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
                "request_id": "preview-one",
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
            response = client.post(endpoint, json=request, headers=headers)
            assert response.status_code == 200, response.text
            preview = response.json()["preview"]
            assert preview["state"] == "exact"
            assert preview["cost"] == "{U}"
            assert preview["sources"]
            assert before == record.game.env.search_witness_json()
            assert record.game._next_prompt_id == prompt_count
            assert len(record.game.canonical_decisions) == decisions
            assert not record.game.accepted_commands
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
            tap = next(
                offer
                for offer in frame.offers
                if offer.label.startswith("Tap Island for")
                and offer.source is not None
                and offer.source.kind == "object"
                and offer.source.id.entity not in proposed
            )
            command = {
                "command_id": "manual-island",
                "match_id": frame.match_id,
                "expected_revision": frame.revision,
                "prompt_id": frame.prompt.id,
                "offer_id": tap.id,
                "answers": [],
            }
            outcome = record.game.hero_command(command)
            assert outcome["status"] == "accepted"
            recovered = ExperienceFrame.model_validate(outcome["update"]["frame"])
            assert sum(recovered.projection.agent.mana or []) == 1
            assert (
                client.post(endpoint, json=request, headers=headers).status_code == 409
            )
            duplicate = record.game.hero_command(command)
            assert duplicate["status"] == "duplicate"
            assert sum(record.game.obs.agent.mana) == 1


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


def test_manual_fixed_source_can_produce_before_its_reserved_sacrifice() -> None:
    env = managym.Env(seed=165)
    env.reset(
        [
            managym.PlayerConfig(
                "payer", {"Swamp": 28, "Sol Ring": 4, "Deadly Dispute": 8}
            ),
            managym.PlayerConfig("other", {"Plains": 40}),
        ]
    )
    env.scenario_clear_hand(0)
    env.scenario_force_card_in_hand(0, "Deadly Dispute")
    for card in ("Swamp", "Sol Ring"):
        env.scenario_force_battlefield(0, card, ready=True)
    obs = env.scenario_refresh()
    cast = next(
        row
        for row in describe_actions(obs)
        if row["description"] == "Cast Deadly Dispute"
    )
    obs, *_ = env.step(int(cast["index"]))
    tap = next(
        row
        for row in describe_actions(obs)
        if str(row["description"]).startswith("Tap Sol Ring for")
    )
    obs, *_ = env.step(int(tap["index"]))
    assert obs.agent.mana[5] == 2
    pay = next(
        row for row in describe_actions(obs) if row["description"] == "Pay the cost"
    )
    obs, *_ = env.step(int(pay["index"]))
    # C/C and B can leave a unit-choice decision. Spend through the offered
    # native options, then the reserved source is sacrificed exactly once.
    for _ in range(3):
        choices = describe_actions(obs)
        spend = next(
            (row for row in choices if str(row["description"]).startswith("Spend ")),
            None,
        )
        if spend is None:
            break
        obs, *_ = env.step(int(spend["index"]))
    assert not any(
        card.name == "Sol Ring" and int(card.zone) == int(managym.ZoneEnum.BATTLEFIELD)
        for card in obs.agent_cards
    )


def test_unaffordable_kicker_does_not_hide_the_exact_base_payment() -> None:
    env = managym.Env(seed=165)
    env.reset(
        [
            managym.PlayerConfig("payer", {"Mountain": 32, "Firebending Lesson": 8}),
            managym.PlayerConfig("other", {"Forest": 32, "Llanowar Elves": 8}),
        ]
    )
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
