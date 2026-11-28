"""Etude receives prepared host rows and returns ordered, masked policy logits."""

import asyncio
from collections.abc import Sequence
from pathlib import Path

from etude.checkpoints import CheckpointRef
from etude.eval.records import Code, Match as EtudeMatch, checkpoint_player
from etude.records import ArtifactRef
import numpy as np
import pytest
import torch

from manabot.backends.torch.checkpoint import save_checkpoint
from manabot.backends.torch.policy import ActorCritic
from manabot.env.env import Env
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace, encode_observation
from manabot.eval import agent_loader
from manabot.eval.admission import register_anchor
from manabot.eval.agent_loader import load, load_bf16, load_tf32, select_device
from manabot.eval.agents import (
    AGENTS,
    AGENTS_BF16,
    AGENTS_TF32,
    Sampler,
    agents_for,
    build,
    checkpoint_reader,
    etude_registration,
    game_players,
    local_forward,
    local_scalar_sampler,
    play_scalar,
    registration,
)
import manabot.eval.game as game
from manabot.eval.match import play_cell
from manabot.infra.config import ModelConfig, ObservationConfig, RewardConfig
from manabot.infra.identity import file_sha256
from manabot.policy.batching import concatenate
from manabot.policy.decision_memory import MemoryFold
from manabot.policy.prepare import Prepared, prepare
from manabot.sim.agent import GamePlayer
from manabot.sim.card_evidence import EvidenceRef
from manabot.sim.heuristic import HEURISTIC_VERSION
from manabot.sim.matchup import DeckMatchup
from manabot.sim.registration import (
    ETUDE_COMPUTE_CLASS,
    PlayerRegistration,
    etude_compute_class,
)
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


def test_loader_batches_prepared_rows_without_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(71)
    space = ObservationSpace()
    model = ActorCritic(space, ModelConfig(hidden_dim=8, num_attention_heads=2))
    match = Match()
    monkeypatch.setattr(
        "manabot.policy.checkpoint.checkpoint_source",
        lambda: {"commit": "a" * 40},
    )
    path = tmp_path / "policy.safetensors"
    save_checkpoint(model, space, path, player_configs=match.to_rust())
    ref = CheckpointRef(
        address="fixture/policy",
        artifact=ArtifactRef(
            uri=path.as_uri(), sha256=file_sha256(path), size_bytes=path.stat().st_size
        ),
    )
    participant = registration(
        checkpoint_player(ref),
        name="test-policy",
        match=match.config,
        content_suite="fixture",
        role="challenger",
    )
    assert participant.writer_revision == "a" * 40
    assert participant.stored_programs == 0
    forward = load(ref)
    spec = checkpoint_reader(1)(ref).spec
    # Etude passes a config-less checkpoint seat its bare forward, never built.
    [seated] = game_players([checkpoint_player(ref)], [local_forward(forward)])
    assert isinstance(seated, Sampler)
    assert seated.checkpoint_key == ref.key
    env = Env(match, space, Reward(RewardConfig()), seed=71)
    try:
        obs, _ = env.reset()
        batch = {key: value[None] for key, value in obs.items()}
        row = prepare(batch, spec.tables, binding=spec.binding)
        [single] = forward([row])
        assert single.shape == row.legal.shape and single.dtype == np.float32
        joined = forward([row, row])
        assert len(joined) == 2
        for output in joined:
            assert output.shape == row.legal.shape and output.dtype == np.float32
            np.testing.assert_allclose(output, single, atol=1e-6, rtol=1e-5)
            assert np.isneginf(output[~row.legal]).all()
        batched = prepare(
            {
                key: np.concatenate([value, value], axis=0)
                for key, value in batch.items()
            },
            spec.tables,
            binding=spec.binding,
        )
        [combined] = forward([batched])
        assert combined.shape == batched.legal.shape and combined.dtype == np.float32
        for index in range(2):
            np.testing.assert_allclose(
                combined[index : index + 1], single, atol=1e-6, rtol=1e-5
            )
        mixed = forward([row, batched])
        assert [out.shape for out in mixed] == [row.legal.shape, batched.legal.shape]
        np.testing.assert_allclose(mixed[0], single, atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(mixed[1], combined, atol=1e-6, rtol=1e-5)
        with pytest.raises(TypeError, match="nonempty host Prepared"):
            forward([object()])
        player = checkpoint_player(ref)
        managed_match = EtudeMatch(players=(player, player))

        def choose(match: EtudeMatch, agents: Sequence[GamePlayer]) -> tuple[int, int]:
            a, b = agents
            return a.open_game(123)(env, obs, 123), b.open_game(123)(env, obs, 123)

        managed = asyncio.run(
            play_scalar(choose, managed_match, [seated, local_forward(forward)])
        )
        local = local_scalar_sampler(player, forward, checkpoint_reader(1))
        expected = local.choose(env, obs, 123)
        assert managed == (expected, expected)
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
    spec = checkpoint_reader(1)(ref).spec
    rows = []
    for env in (shared_effect_root(1), kellan_upgrade_root()):
        obs = encode_observation(
            space, env.observation_for_player(env.current_agent_index())
        )
        rows.append(
            prepare(
                {k: v[None] for k, v in obs.items()},
                spec.tables,
                binding=spec.binding,
            )
        )
    expected = [forward([row])[0] for row in rows]
    actual = forward(rows)
    assert len(actual) == len(expected)
    for single, joined in zip(expected, actual, strict=True):
        assert joined.shape == single.shape
        np.testing.assert_allclose(joined, single, atol=1e-6, rtol=1e-5)


def test_select_device_order_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MANABOT_EVAL_DEVICE", raising=False)
    monkeypatch.setattr(agent_loader.platform, "system", lambda: "Linux")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(agent_loader, "mlx_available", lambda: True)
    assert select_device() == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device() == "mlx"

    monkeypatch.setattr(agent_loader, "mlx_available", lambda: False)
    assert select_device() == "mps"

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert select_device() == "cpu"

    monkeypatch.setenv("MANABOT_EVAL_DEVICE", "cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert select_device() == "cpu"

    monkeypatch.setenv("MANABOT_EVAL_DEVICE", "mlx")
    monkeypatch.setattr(agent_loader, "mlx_available", lambda: True)
    assert select_device() == "mlx"
    monkeypatch.setattr(agent_loader, "mlx_available", lambda: False)
    with pytest.raises(ValueError, match=r"MLX is unavailable.*uv sync --extra mlx"):
        select_device()

    monkeypatch.setattr(agent_loader, "mlx_available", lambda: True)
    monkeypatch.setenv("MANABOT_EVAL_DEVICE", "mps")
    assert select_device() == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(ValueError, match="MPS is unavailable"):
        select_device()

    monkeypatch.setenv("MANABOT_EVAL_DEVICE", "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA is unavailable"):
        select_device()

    monkeypatch.setenv("MANABOT_EVAL_DEVICE", "tpu")
    with pytest.raises(ValueError, match=r"cuda, mlx, mps, cpu"):
        select_device()


def test_apple_mlx_selection_does_not_import_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from builtins import __import__ as real_import

    def import_module(name: str, *args: object, **kwargs: object) -> object:
        assert name != "torch"
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.delenv("MANABOT_EVAL_DEVICE", raising=False)
    monkeypatch.setattr(agent_loader.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(agent_loader, "mlx_available", lambda: True)
    monkeypatch.setattr("builtins.__import__", import_module)
    assert select_device() == "mlx"


def test_missing_mlx_extra_falls_back_to_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_extra(name: str) -> None:
        assert name == "mlx"
        raise ImportError("mlx extra is unavailable")

    monkeypatch.delenv("MANABOT_EVAL_DEVICE", raising=False)
    monkeypatch.setattr(agent_loader.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(agent_loader.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(agent_loader, "resolve", missing_extra)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert select_device() == "mps"


def test_declared_precision_compute_classes_and_loader_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert etude_compute_class() == ETUDE_COMPUTE_CLASS == "etude-policy-f32-auto-v1"
    assert etude_compute_class("tf32") == "etude-policy-tf32-auto-v1"
    assert etude_compute_class("bf16") == "etude-policy-bf16-auto-v1"
    with pytest.raises(ValueError, match="unsupported eval precision"):
        etude_compute_class("fp16")  # type: ignore[arg-type]

    assert agents_for("float32") == AGENTS
    assert agents_for("tf32") == AGENTS_TF32
    assert agents_for("bf16") == AGENTS_BF16
    assert (AGENTS.load, AGENTS_TF32.load, AGENTS_BF16.load) == (
        load,
        load_tf32,
        load_bf16,
    )
    with pytest.raises(ValueError, match="unsupported eval precision"):
        agents_for("fp16")  # type: ignore[arg-type]

    space = ObservationSpace()
    model = ActorCritic(space, ModelConfig(hidden_dim=8, num_attention_heads=2))
    match = Match()
    path = tmp_path / "precision.safetensors"
    save_checkpoint(model, space, path, player_configs=match.to_rust())
    ref = CheckpointRef(
        address="fixture/precision",
        artifact=ArtifactRef(
            uri=path.as_uri(), sha256=file_sha256(path), size_bytes=path.stat().st_size
        ),
    )
    for precision in ("float32", "tf32", "bf16"):
        reg = registration(
            checkpoint_player(ref),
            name="cand",
            match=match.config,
            content_suite="fixture",
            role="challenger",
            precision=precision,
        )
        assert reg.compute_class_id == etude_compute_class(precision)
        copied = etude_registration(reg, precision=precision)
        assert copied.compute_class_id == etude_compute_class(precision)
    with pytest.raises(ValueError, match="explicit sampling contract"):
        PlayerRegistration.model_validate(
            {**reg.model_dump(), "compute_class_id": "etude-policy-fp8-auto-v1"}
        )

    monkeypatch.setenv("MANABOT_EVAL_DEVICE", "cpu")
    with pytest.raises(ValueError, match="precision tf32 requires CUDA"):
        load_tf32(ref)
    with pytest.raises(ValueError, match="precision bf16 requires CUDA or MPS"):
        load_bf16(ref)


@pytest.mark.parametrize("encoder", ["gru", "transformer"])
def test_concatenate_joins_decision_memory_rows_and_rejects_folds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    encoder: str,
) -> None:
    from dataclasses import replace

    monkeypatch.setenv("MANABOT_EVAL_DEVICE", "cpu")
    torch.set_num_threads(1)
    torch.manual_seed(73)
    space = ObservationSpace(
        ObservationConfig(
            max_decision_records=64,
            max_decision_reference_details=2048,
            max_decision_references=64,
            max_decision_members=512,
        )
    )
    model = ActorCritic(
        space,
        ModelConfig(
            world=managym.WORLD_SETUP,
            hidden_dim=8,
            num_attention_heads=2,
            attention_on=True,
            attention_layers=1,
            policy_film=True,
            value_aggregation="masked_mean",
            decision_memory=encoder,  # type: ignore[arg-type]
            decision_memory_summaries=4,
        ),
    )
    match = Match()
    path = tmp_path / f"memory-{encoder}.safetensors"
    save_checkpoint(model, space, path, player_configs=match.to_rust())
    ref = CheckpointRef(
        address=f"fixture/memory-{encoder}",
        artifact=ArtifactRef(
            uri=path.as_uri(), sha256=file_sha256(path), size_bytes=path.stat().st_size
        ),
    )
    forward = load(ref)
    spec = checkpoint_reader(1)(ref).spec

    # Collect one empty-memory turn-0 row and two distinct nonempty histories.
    rows: list[Prepared[np.ndarray]] = []
    env = Env(match, space, Reward(RewardConfig()), seed=73)
    try:
        obs, _ = env.reset(seed=73)
        raw_obs = [obs]
        for step_idx in range(18):
            legal_actions = np.flatnonzero(obs["actions_valid"] > 0)
            obs, _, terminated, truncated, _ = env.step(int(legal_actions[0]))
            if terminated or truncated:
                break
            if step_idx in (6, 14):
                raw_obs.append(obs)
    finally:
        env.close()
    assert len(raw_obs) == 3

    for obs in raw_obs:
        batch = {key: value[None] for key, value in obs.items()}
        rows.append(
            prepare(
                batch,
                spec.tables,
                binding=spec.binding,
                memory_summaries=spec.model.decision_memory_summaries,
            )
        )
    lengths = [row.memory.lengths[0] for row in rows if row.memory is not None]
    assert lengths[0] == 0 and lengths[1] > 0 and lengths[2] > lengths[1]

    expected = [forward([row])[0] for row in rows]
    joined = forward(rows)
    assert len(joined) == len(expected)
    for single, batched in zip(expected, joined, strict=True):
        assert batched.shape == single.shape and batched.dtype == np.float32
        np.testing.assert_allclose(batched, single, atol=1e-5, rtol=1e-5)

    # Multi-row Prepared inputs also join cleanly with single-row inputs.
    two_rows = concatenate(rows[:2])
    [first_two, third] = forward([two_rows, rows[2]])
    assert first_two.shape == two_rows.legal.shape
    np.testing.assert_allclose(first_two[0:1], expected[0], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(first_two[1:2], expected[1], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(third, expected[2], atol=1e-5, rtol=1e-5)

    # Folded decision memory and mixed memory/non-memory layouts are refused.
    assert rows[1].memory is not None
    folded = replace(
        rows[1],
        memory=replace(
            rows[1].memory,
            fold=MemoryFold(levels=(), tails=np.zeros(1, np.int64), node_count=0),
        ),
    )
    with pytest.raises(ValueError, match="folded decision memory"):
        concatenate([rows[0], folded])
    with pytest.raises(ValueError, match="optional layouts differ"):
        concatenate([rows[0], replace(rows[1], memory=None)])


@pytest.mark.parametrize("device", ["mps", "mlx"])
@pytest.mark.parametrize("memory", ["none", "gru"])
def test_loader_apple_gpu_matches_cpu_logits_and_supports_bf16(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    memory: str,
    device: str,
) -> None:
    if device == "mlx" and not agent_loader.mlx_available():
        pytest.skip("requires Apple Silicon and the MLX extra")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires Apple MPS")
    torch.set_num_threads(1)
    torch.manual_seed(74)
    space = ObservationSpace(
        ObservationConfig(
            max_decision_records=64,
            max_decision_reference_details=2048,
            max_decision_references=64,
            max_decision_members=512,
        )
        if memory != "none"
        else ObservationConfig()
    )
    model = ActorCritic(
        space,
        ModelConfig(
            world=managym.WORLD_SETUP,
            hidden_dim=16,
            num_attention_heads=2,
            attention_on=True,
            attention_layers=1,
            policy_film=True,
            value_aggregation="masked_mean",
            decision_memory=memory,  # type: ignore[arg-type]
        ),
    )
    match = Match()
    path = tmp_path / f"mps-{memory}.safetensors"
    save_checkpoint(model, space, path, player_configs=match.to_rust())
    ref = CheckpointRef(
        address=f"fixture/mps-{memory}",
        artifact=ArtifactRef(
            uri=path.as_uri(), sha256=file_sha256(path), size_bytes=path.stat().st_size
        ),
    )
    monkeypatch.setenv("MANABOT_EVAL_DEVICE", "cpu")
    cpu_forward = load(ref)
    spec = checkpoint_reader(1)(ref).spec

    env = Env(match, space, Reward(RewardConfig()), seed=74)
    rows: list[Prepared[np.ndarray]] = []
    try:
        obs, _ = env.reset(seed=74)
        for step_idx in range(10):
            if step_idx in (0, 8):
                rows.append(
                    prepare(
                        {key: value[None] for key, value in obs.items()},
                        spec.tables,
                        binding=spec.binding,
                        memory_summaries=spec.model.decision_memory_summaries
                        if memory != "none"
                        else 0,
                    )
                )
            legal_actions = np.flatnonzero(obs["actions_valid"] > 0)
            obs, _, terminated, truncated, _ = env.step(int(legal_actions[0]))
            if terminated or truncated:
                break
    finally:
        env.close()
    assert len(rows) == 2

    combined = concatenate(rows)
    [cpu_logits] = cpu_forward([combined])

    monkeypatch.setenv("MANABOT_EVAL_DEVICE", device)
    forward = load(ref)
    split_logits = forward(list(rows))
    [logits] = forward([combined])
    assert len(split_logits) == len(rows)
    np.testing.assert_array_equal(np.concatenate(split_logits), logits)
    assert logits.shape == combined.legal.shape and logits.dtype == np.float32
    assert np.isneginf(logits[~combined.legal]).all()
    np.testing.assert_allclose(
        logits[combined.legal], cpu_logits[combined.legal], atol=1e-4, rtol=1e-4
    )

    with pytest.raises(ValueError, match="precision tf32 requires CUDA"):
        load_tf32(ref)

    if device == "mlx" or torch.backends.mps.is_macos_or_newer(14, 0):
        bf16_forward = load_bf16(ref)
        [bf16_logits] = bf16_forward([combined])
        assert (
            bf16_logits.shape == combined.legal.shape
            and bf16_logits.dtype == np.float32
        )
        assert np.isfinite(bf16_logits[combined.legal]).all()
        assert np.isneginf(bf16_logits[~combined.legal]).all()


@pytest.mark.parametrize("mode", ["all", "usable"])
def test_checkpoint_priority_taps_follow_seats_through_replay(
    tmp_path: Path, mode: managym.ManaFloats
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
    for name, taps in (("tap-on", mode), ("tap-off", "off")):
        space = ObservationSpace(
            ObservationConfig(
                mana_floats=taps, mana_payment="auto" if name == "tap-on" else "manual"
            )
        )
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
        agents[name] = local_scalar_sampler(player, load(ref), checkpoint_reader(1))
    registrations["heuristic"] = register_anchor(
        registrations["tap-on"], "heuristic", player_id="heuristic"
    )
    agents["heuristic"] = build(
        Code(name="heuristic", config={"version": HEURISTIC_VERSION}), {}
    )
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
            pair = (mode if seat == 0 else "off", mode if seat == 1 else "off")
            assert tuple(metadata["match_hypers"]["mana_floats"]) == pair
            assert record.games[0].setup().mana_floats == pair
            assert record.games[0].setup().mana_payment[seat] == "auto"
            assert record.games[0].setup().mana_payment[1 - seat] == "manual"
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
            assert offered == ({seat} if mode == "all" else set())
            assert replay.stop is not None and replay.stop.status == "complete"

            # The native commitments still replay if only the outer match metadata
            # is changed. Verification must catch the conflicting seat assignment.
            metadata["match_hypers"]["mana_floats"] = list(reversed(pair))
            assert not replay_games(
                [record.model_dump(mode="json", by_alias=True)]
            ).passed
