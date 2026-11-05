"""Controlled capture receipts; no games, policy inference or training runs."""

import json
from pathlib import Path

import pytest

from experiments.growing_pool import BINDING
from experiments.growing_pool_exposure import read_training_exposure, training_exposure
from experiments.growing_pool_protocol import Protocol, World
from manabot.env.failures import (
    CapturedVector,
    CaptureScope,
    NativeTrainingGame,
    TrainingExposureReceipt,
    capture_scope,
)
from manabot.infra.identity import canonical_sha256
import managym


@pytest.fixture
def protocol(monkeypatch: pytest.MonkeyPatch) -> Protocol:
    protocol = Protocol.model_validate_json(BINDING.protocol_path.read_bytes())
    # This compatible-world fixture is only for synthetic reader evidence.
    monkeypatch.setattr(World, "current", classmethod(lambda cls: protocol.world))
    return protocol


def receipt(protocol: Protocol, *, terminal: bool = True) -> TrainingExposureReceipt:
    source = protocol.manifest.source()
    assert source.entry_ids is not None
    names = sorted(
        {name for deck in source.decks for name in (*deck.main, *deck.sideboard)}
    )
    definitions = {name: i + 1 for i, name in enumerate(names)}
    initial: list[dict[str, object]] = []
    opening: list[list[int]] = [[], []]
    players: list[dict[str, object]] = []
    for seat in (0, 1):
        deck = source.decks[seat]
        players.append(
            dict(
                name=source.entry_ids[seat],
                decklist=deck.main,
                sideboard=deck.sideboard,
                world="world",
            )
        )
        for name, count in deck.main.items():
            for _ in range(count):
                copy = len(initial)
                hand = len(opening[seat]) < 7
                if hand:
                    opening[seat].append(copy)
                initial.append(
                    dict(
                        copy=copy,
                        owner=seat,
                        definition=definitions[name],
                        name=name,
                        incarnation=0,
                        zone="Hand" if hand else "Library",
                        origin="deck",
                    )
                )
        for name, count in deck.sideboard.items():
            for _ in range(count):
                initial.append(
                    dict(
                        copy=len(initial),
                        owner=seat,
                        definition=definitions[name],
                        name=name,
                        incarnation=0,
                        zone=None,
                        origin="outside",
                    )
                )
    card = initial[opening[0][0]]
    choices = [
        dict(
            revision=1,
            actor=0,
            kind="Priority",
            selected=1,
            options=[
                dict(action_type="PriorityCastSpell", objects=[card]),
                dict(action_type="PriorityPassPriority", objects=[]),
            ],
        )
    ]
    cards = dict(
        capture=dict(
            version=1,
            content_digest=protocol.world.card_digest,
            initial=initial,
            opening=opening,
            moves=[],
            exposure=dict(choices=choices, resolutions=[]),
        ),
        final_cards=initial,
        turn=1,
    )
    return TrainingExposureReceipt.model_validate(
        dict(
            schema="training-exposure-v1",
            stream=0,
            game=dict(
                seed=77,
                terminal=terminal,
                trace=dict(actions=[1], player_configs=players),
                cards=cards,
            ),
            skip_trivial=True,
            identities={
                "world": protocol.world.world_version,
                "input_buffers": {
                    name: {"shape": list(shape), "dtype": "float32"}
                    for name, shape in protocol.observation_shapes.items()
                },
                "engine_extension_sha256": "e" * 64,
            },
            context=dict(
                run="learner",
                seed=protocol.budget.seeds[0],
                regime_digest=canonical_sha256(
                    protocol.regime().model_dump(mode="json")
                ),
                identities={"engine_extension_sha256": "e" * 64},
            ),
            seconds=1,
        )
    )


def test_declined_opportunity_observed_zero_and_missing_are_distinct(
    protocol: Protocol,
) -> None:
    observed = receipt(protocol)
    report = training_exposure(
        [observed],
        seed=protocol.budget.seeds[0],
        run_id="learner",
        identities={"engine_extension_sha256": "e" * 64},
        protocol=protocol,
    )
    assert report.terminal_games == 1 and not report.complete
    assert report.decisions_by_seat == (1, 0)
    assert sum(c.opportunity_decisions for c in report.cards) == 1
    assert sum(c.chosen_decisions for c in report.cards) == 0
    assert sum(c.completed_frames for c in report.cards) == 0
    assert (
        next(
            a for a in report.actions if a.action_type == "PriorityPassPriority"
        ).chosen_decisions
        == 1
    )
    assert report.cards and all(c.drawn_copies == 0 for c in report.cards)
    missing = training_exposure(
        [],
        seed=protocol.budget.seeds[0],
        run_id="learner",
        identities={"engine_extension_sha256": "e" * 64},
        protocol=protocol,
    )
    assert not missing.cards and not missing.receipts and missing.terminal_games == 0
    interrupted = training_exposure(
        [receipt(protocol, terminal=False)],
        seed=protocol.budget.seeds[0],
        run_id="learner",
        identities={"engine_extension_sha256": "e" * 64},
        protocol=protocol,
    )
    assert interrupted.interrupted_games == 1 and not interrupted.cards


def test_complete_budget_and_changed_identity(protocol: Protocol) -> None:
    original = receipt(protocol)
    receipts = []
    expected = protocol.budget.training_games // len(protocol.budget.seeds)
    for seed in range(expected):
        game = original.game.model_copy(
            update={
                "seed": seed,
            }
        )
        receipts.append(
            original.model_copy(
                update={"game": game, "stream": seed % protocol.budget.streams}
            )
        )
    report = training_exposure(
        receipts,
        seed=protocol.budget.seeds[0],
        run_id="learner",
        identities={"engine_extension_sha256": "e" * 64},
        protocol=protocol,
    )
    assert report.complete and report.terminal_games == expected
    with pytest.raises(ValueError, match="another run"):
        training_exposure(
            receipts,
            seed=protocol.budget.seeds[0],
            run_id="other",
            identities={"engine_extension_sha256": "e" * 64},
            protocol=protocol,
        )
    with pytest.raises(ValueError, match="duplicate"):
        training_exposure(
            [original, original],
            seed=protocol.budget.seeds[0],
            run_id="learner",
            identities={"engine_extension_sha256": "e" * 64},
            protocol=protocol,
        )
    with pytest.raises(ValueError, match="world/input/source"):
        training_exposure(
            [original],
            seed=protocol.budget.seeds[0],
            run_id="learner",
            identities={"world": "wrong"},
            protocol=protocol,
        )


def test_retention_binds_replay_and_interrupted_prefix(
    protocol: Protocol, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = receipt(protocol)

    class NativeFixture:
        """Inject a terminal before auto-reset and a partial current root."""

        def __init__(self, **kwargs: object) -> None:
            self.pending = [[original.game.model_dump_json(by_alias=True)]]

        def enable_exposure_capture(self) -> None:
            pass

        def take_exposure(self) -> list[list[str]]:
            rows, self.pending = self.pending, [[]]
            return rows

        def current_exposure(self) -> list[str]:
            game = original.game.model_dump(mode="json", by_alias=True)
            game["seed"] = 78
            game["terminal"] = False
            return [
                NativeTrainingGame.model_validate(game).model_dump_json(by_alias=True)
            ]

    monkeypatch.setattr(managym, "VectorEnv", NativeFixture)
    scope = CaptureScope(
        tmp_path / "failures", lambda: original.context, training_exposure=True
    )
    with capture_scope(scope):
        vector = CapturedVector(num_envs=1, seed=77)
        vector.games[0].players = original.game.trace.player_configs
        vector.close()
        vector.close()
    retained = read_training_exposure([tmp_path / "exposure"])
    assert len(retained) == 2 and sorted(r.game.terminal for r in retained) == [
        False,
        True,
    ]
    terminal = next(r for r in retained if r.game.terminal)
    replay = terminal.replay_inputs()
    assert replay.games[0].seed == 77 and replay.games[0].actions == [1]
    assert replay.games[0].players == original.game.trace.player_configs
    path = next((tmp_path / "exposure").glob("*.jsonl"))
    body = json.loads(path.read_text())
    body["stream"] = 5
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="digest"):
        read_training_exposure([tmp_path / "exposure"])


def test_native_training_capture_round_trip_keeps_seed_and_copy_identity() -> None:
    # One constructed land decision, not a played game or learnability probe.
    native = managym.VectorEnv(
        num_envs=1, seed=245, skip_trivial=True, opponent_policy="none"
    )
    native.enable_exposure_capture()
    players = [
        managym.PlayerConfig(name=seat, decklist={"Mountain": 20})
        for seat in ("a", "b")
    ]
    native.reset_all(players)
    native.step([0])
    native.reset_all(players)
    rows = native.take_exposure()
    captured = NativeTrainingGame.model_validate_json(rows[0][0])
    assert captured.seed == 245 and captured.trace.actions == [0]
    assert captured.cards.capture.exposure is not None
    assert captured.cards.capture.exposure.choices[0].selected == 0
    assert not captured.terminal
