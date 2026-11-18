"""Etude receives prepared host rows and returns ordered, masked policy logits."""

from pathlib import Path

from etude.checkpoints import CheckpointRef
from etude.eval.records import Code, checkpoint_player
from etude.records import ArtifactRef
import numpy as np
import pytest
import torch

from manabot.backends.torch.checkpoint import save_checkpoint
from manabot.backends.torch.policy import ActorCritic
from manabot.env.env import Env
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.eval.admission import register_anchor
from manabot.eval.agent_loader import load
from manabot.eval.agents import Sampler, build, game_players, registration
import manabot.eval.game as game
from manabot.eval.match import play_cell
from manabot.infra.config import ModelConfig, ObservationConfig, RewardConfig
from manabot.infra.identity import file_sha256
from manabot.policy.prepare import prepare
from manabot.sim.card_evidence import EvidenceRef
from manabot.sim.matchup import DeckMatchup
from manabot.sim.replay import replay_games
import managym
from managym.recording import read_records
from managym.replay import walk
from managym.statistics import Reduction
from tests.eval.arena_fixture import arena_fixture
from tests.fixtures.evaluation_callbacks import PanicException


@pytest.mark.parametrize(
    "error_type",
    [managym.AgentError, PanicException, RuntimeError, ValueError, KeyError],
)
def test_replay_only_retains_native_failures(
    monkeypatch: pytest.MonkeyPatch, error_type: type[BaseException]
) -> None:
    def failed_replay(
        trace: dict[str, object],
        directory: Path | None,
        reduction: Reduction | None = None,
    ) -> tuple[dict[str, object], EvidenceRef | None]:
        raise error_type("controlled replay failure")

    monkeypatch.setattr(game, "verify_game", failed_replay)
    if error_type in (managym.AgentError, PanicException):
        outcome = game.replay_in_process([{}], None)
        assert outcome.error == f"{error_type.__name__}: controlled replay failure"
        assert outcome.receipts is not None
        assert outcome.receipts[0]["passed"] is False
    else:
        with pytest.raises(error_type, match="controlled replay failure"):
            game.replay_in_process([{}], None)


def test_loader_batches_prepared_rows_without_sampling(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(71)
    space = ObservationSpace()
    model = ActorCritic(space, ModelConfig(hidden_dim=8, num_attention_heads=2))
    match = Match()
    path = tmp_path / "policy.safetensors"
    save_checkpoint(model, space, path, player_configs=match.to_rust())
    ref = CheckpointRef(
        address="fixture/policy",
        artifact=ArtifactRef(
            uri=path.as_uri(), sha256=file_sha256(path), size_bytes=path.stat().st_size
        ),
    )
    forward = load(ref)
    sampler = build(checkpoint_player(ref), {ref.key: forward})
    assert isinstance(sampler, Sampler)
    # Etude passes a config-less checkpoint seat its bare forward, never built.
    [seated] = game_players([checkpoint_player(ref)], [forward])
    assert isinstance(seated, Sampler)
    env = Env(match, space, Reward(RewardConfig()), seed=71)
    try:
        obs, _ = env.reset()
        batch = {key: value[None] for key, value in obs.items()}
        row = prepare(batch, sampler.spec.tables, binding=sampler.spec.binding)
        [single] = forward([row])
        joined = forward([row, row])
        assert len(joined) == 2
        for output in joined:
            np.testing.assert_allclose(output, single, atol=1e-6, rtol=1e-5)
            assert np.isneginf(output[~row.legal[0]]).all()
        assert sampler.choose(env, obs, 123) == sampler.choose(env, obs, 123)
        assert seated.choose(env, obs, 123) == sampler.choose(env, obs, 123)
    finally:
        env.close()


@pytest.mark.parametrize("case", ["agent", "panic", "timeout", "commands", "retry"])
def test_native_failures_remain_rows_across_etudes_game_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    from etude.eval.query import statsdb
    from etude.eval.records import Benchmark, Environment, Seating
    from etude.eval.tournament import evaluating

    from manabot.eval.agents import AGENTS
    from manabot.eval.card_evaluation import verification_spec
    from tests.fixtures.evaluation_callbacks import play_native_case

    spec = verification_spec(8)
    tournament = Benchmark(
        name=f"native-{case}-agents-v1",
        environment=Environment(name="native-fixture", files="a" * 64),
        seatings=(
            Seating(
                name="one-group",
                players=tuple(
                    Code(
                        name=p.player_id,
                        config={"registration": p.model_dump(mode="json")},
                    )
                    for p in spec.players
                ),
                config={"case": case},
                games=1,
            ),
        ),
    )
    (tmp_path / "evidence").mkdir()
    monkeypatch.setenv("ETUDE_OUTPUT_DIR", str(tmp_path / "evidence"))
    with evaluating(
        tournament, experiment="native-agents", directory=tmp_path / "track"
    ) as run:
        if case == "retry":
            from etude.errors import RetryRequested

            with pytest.raises(
                RetryRequested, match="controlled agent transport failure"
            ):
                run.play(play_native_case, agents=AGENTS)
            assert list((tmp_path / "evidence").rglob("failures/*.jsonl"))
            return
        result = run.play(play_native_case, agents=AGENTS)
    assert result.played == 1
    with statsdb("native-agents", tmp_path / "track") as db:
        columns = {r[0] for r in db.execute("describe scores").fetchall()}
        # Etude omits a scalar column when every value is null; missing score
        # remains unavailable, never zero credit for an invalid game.
        score = "score" if "score" in columns else "NULL as score"
        rows = db.execute(
            f"select game, {score}, failure, termination from scores order by game"
        ).fetchall()
    assert len(rows) == 8
    if case in {"agent", "panic"}:
        assert rows[0][1] is None
        assert ("AgentError" if case == "agent" else "PanicException") in rows[0][2]
        assert all(row[2] is None for row in rows[1:])
        assert list((tmp_path / "evidence").rglob("failures/*.jsonl"))
    else:
        assert {row[3] for row in rows} == {
            "timeout" if case == "timeout" else "command_cap"
        }
        assert all(row[1] is None for row in rows)


def test_loader_keeps_effect_and_program_routes_across_distinct_rows(
    tmp_path: Path,
) -> None:
    from manabot.verify.representation import kellan_upgrade_root, shared_effect_root
    import managym

    torch.set_num_threads(1)
    torch.manual_seed(72)
    space = ObservationSpace()
    model = ActorCritic(
        space,
        ModelConfig(hidden_dim=8, num_attention_heads=2, world=managym.WORLD_SETUP),
    )
    path = tmp_path / "effects.safetensors"
    save_checkpoint(model, space, path, player_configs=None)
    ref = CheckpointRef(
        address="fixture/effects",
        artifact=ArtifactRef(
            uri=path.as_uri(), sha256=file_sha256(path), size_bytes=path.stat().st_size
        ),
    )
    forward = load(ref)
    sampler = build(checkpoint_player(ref), {ref.key: forward})
    assert isinstance(sampler, Sampler)
    rows = []
    for env in (shared_effect_root(1), kellan_upgrade_root()):
        obs = space.encode(env.observation_for_player(env.current_agent_index()))
        rows.append(
            prepare(
                {k: v[None] for k, v in obs.items()},
                sampler.spec.tables,
                binding=sampler.spec.binding,
            )
        )
    expected = [forward([row])[0] for row in rows]
    actual = forward(rows)
    assert len(actual) == len(expected)
    for single, joined in zip(expected, actual, strict=True):
        np.testing.assert_allclose(joined, single, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("mode", [True, "usable"])
def test_checkpoint_priority_taps_follow_seats_through_replay(
    tmp_path: Path, mode: managym.ManaTaps
) -> None:
    torch.manual_seed(385)
    matchup = DeckMatchup(
        population_sha256="a" * 64,
        entries=("forest", "mountain"),
        lists=(managym.Deck({"Forest": 12}), managym.Deck({"Mountain": 12})),
    )
    key = arena_fixture().key.model_copy(
        update={
            "world": managym.WORLD_VERSION,
            "content_suite": matchup.content_suite,
            "viewer_boundary": "acting-viewer",
        }
    )
    registrations = {}
    agents = {}
    for name, taps in (("tap-on", mode), ("tap-off", False)):
        space = ObservationSpace(ObservationConfig(mana_taps=taps))
        model = ActorCritic(
            space, ModelConfig(world="world", hidden_dim=8, num_attention_heads=2)
        )
        path = tmp_path / f"{name}.safetensors"
        save_checkpoint(model, space, path, player_configs=None)
        ref = CheckpointRef(
            address=f"fixture/{name}",
            artifact=ArtifactRef(
                uri=path.as_uri(),
                sha256=file_sha256(path),
                size_bytes=path.stat().st_size,
            ),
        )
        player = checkpoint_player(ref)
        registrations[name] = registration(
            player,
            name=name,
            match=matchup.match(),
            content_suite=matchup.content_suite,
            role="challenger",
        )
        agents[name] = build(player, {ref.key: load(ref)})
    registrations["heuristic"] = register_anchor(
        registrations["tap-on"], "heuristic", player_id="heuristic"
    )
    agents["heuristic"] = build(Code(name="heuristic"), {})
    for opponent in ("tap-off", "heuristic"):
        rows, trace, receipt = play_cell(
            key=key,
            player_a=registrations["tap-on"],
            player_b=registrations[opponent],
            authored_matchup=matchup,
            deal_seeds=(385,),
            legs=(0, 1),
            agents=agents,
            out_dir=tmp_path / opponent,
            card_evidence_dir=tmp_path / opponent / "cards",
        )
        assert receipt["passed"] and len(rows) == 2
        for record in read_records(Path(trace["path"])):
            metadata = record.context["evaluation"]
            seat = metadata["seat_players"].index("tap-on")
            pair = (mode if seat == 0 else False, mode if seat == 1 else False)
            assert tuple(metadata["match_hypers"]["mana_taps"]) == pair
            assert record.games[0].setup().mana_taps == pair
            replay = walk(record)
            offered = {
                choice.observation.agent.player_index
                for choice in replay
                if any(
                    action.action_type == managym.ActionEnum.TAP_FOR_COST
                    for action in choice.observation.action_space.actions
                )
            }
            # Basic-land decks have no payments; every offered tap is at priority.
            assert offered == ({seat} if mode is True else set())
            assert replay.stop is not None and replay.stop.status == "complete"

            # The native commitments still replay if only the outer match metadata
            # is changed. Verification must catch the conflicting seat assignment.
            metadata["match_hypers"]["mana_taps"] = list(reversed(pair))
            assert not replay_games(
                [record.model_dump(mode="json", by_alias=True)]
            ).passed
