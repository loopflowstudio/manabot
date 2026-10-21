"""Exact ten-deck admission through native, Academy and ordinary policy inputs."""

from collections import Counter
from copy import deepcopy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from academy.curated_pack import ELEMENTAL_WORLD_PACK, curated_pack_for_matchup
from academy.server import GameSession, _parse_game_config
from manabot.env import Env, Match, ObservationSpace, Reward, VectorEnv
from manabot.env.observation import ObservationEncoder
from manabot.infra import Experiment, ExperimentHypers, TrainHypers
from manabot.infra.hypers import (
    AgentSpec,
    MatchHypers,
    ObservationSpaceHypers,
    RewardHypers,
)
from manabot.model import Trainer
from manabot.model.agent import Agent
from manabot.model.catalog import inspect_checkpoint
from manabot.model.world import (
    checkpoint_world,
    validate_agent_setup,
    validate_checkpoint_world,
)
from manabot.semantic.compiler import compile_paths, pretty_json
from manabot.semantic.learning import BoundSemanticPack
from manabot.sim.distill import save_bc_checkpoint
from manabot.sim.flat_mc import load_checkpoint_agent
import managym
from managym.experience import LegacyPermanentView
from managym.experience_projection import describe_actions, serialize_observation

PACK = "elemental-world-v1"
BASE = Path("content/semantic/v1")


def test_exact_accepted_lists_inventory_and_compiled_closure() -> None:
    ir, coverage = compile_paths(
        BASE / "elemental_world.source.json", BASE / "elemental_world.fixtures.json"
    )
    assert (BASE / "generated/elemental_world.ir.json").read_text() == pretty_json(ir)
    assert (
        BASE / "generated/elemental_world.coverage.json"
    ).read_text() == pretty_json(coverage)
    assert coverage["definition_count"] == 100
    assert coverage["admission_closure_complete"]
    accepted = json.loads(Path("docs/rules/elemental/decks.json").read_text())
    with Path("docs/rules/elemental/cube-snapshot.csv").open(newline="") as stream:
        inventory = Counter(row["name"] for row in csv.DictReader(stream))
    basics = {"Plains", "Island", "Swamp", "Mountain", "Forest"}
    assert len(accepted["decks"]) == 10
    assert set(managym.authored_deck_keys(PACK)) == set(ELEMENTAL_WORLD_PACK.deck_ids)
    names: set[str] = set()
    for deck in accepted["decks"]:
        setup = managym.authored_deck_setup(PACK, "elemental_" + deck["key"].lower())
        assert setup.decklist == deck["main"]
        assert setup.sideboard == deck["lesson_pool"]
        assert sum(setup.decklist.values()) == 40
        for name, count in (Counter(setup.decklist) + Counter(setup.sideboard)).items():
            names.add(name)
            assert name in basics or count <= min(4, inventory[name])
    assert len(names - basics) == 89


@pytest.mark.parametrize("left", ELEMENTAL_WORLD_PACK.deck_ids)
def test_both_seats_and_mirrors_preserve_complete_academy_setups(left: str) -> None:
    for right in ELEMENTAL_WORLD_PACK.deck_ids:
        config = _parse_game_config(
            {
                "hero_deck": left,
                "villain_deck": right,
                "villain_type": "random",
                "seed": 164,
            }
        )
        assert curated_pack_for_matchup(left, right) is ELEMENTAL_WORLD_PACK
        configs = config.to_rust()
        for key, actual in zip((left, right), configs, strict=True):
            expected = managym.authored_deck_setup(PACK, key)
            assert actual.decklist == expected.decklist
            assert actual.sideboard == expected.sideboard
            assert actual.content_pack == PACK
        engine = managym.Env(seed=164)
        engine.reset(configs)
        assert engine.content_pack_manifest()["compiled_semantics"]["pack_key"] == PACK


def test_full_roster_checkpoint_admits_every_pair_and_rejects_drift() -> None:
    space = ObservationSpace()
    pair = Match(MatchHypers.authored(PACK, "elemental_ug", "elemental_bg"))
    binding = checkpoint_world(pair.to_rust(), space)
    assert len(binding["setups"]) == 10
    assert validate_checkpoint_world({"world_binding": binding}, space) == binding
    agent = SimpleNamespace(world_binding=binding)
    for key in managym.authored_deck_keys(PACK):
        configs = Match(MatchHypers.authored(PACK, key, key)).to_rust()
        validate_agent_setup(agent, configs)
        assert (
            validate_checkpoint_world({"world_binding": binding}, space, configs)
            == binding
        )
    bad = pair.to_rust()
    bad[0] = managym.PlayerConfig("wrong pool", bad[0].decklist, {}, PACK)
    with pytest.raises(ValueError, match="outside checkpoint"):
        validate_agent_setup(agent, bad)
    missing_pack = [
        managym.PlayerConfig(p.name, p.decklist, p.sideboard) for p in pair.to_rust()
    ]
    with pytest.raises(ValueError, match="explicit content pack"):
        validate_agent_setup(agent, missing_pack)
    drifted = deepcopy(binding)
    drifted["setups"][0]["sideboard"] = {}
    with pytest.raises(ValueError, match="invalid setup|binding differs"):
        validate_checkpoint_world({"world_binding": drifted}, space)
    stale = deepcopy(binding)
    stale["world"] = "w9"
    with pytest.raises(ValueError, match="compatible w10"):
        validate_checkpoint_world({"world_binding": stale}, space)


def test_ordinary_policy_has_complete_programs_and_trains_on_a_real_offer(
    tmp_path: Path,
) -> None:
    torch.set_num_threads(1)
    space = ObservationSpace()
    env = Env(
        Match(MatchHypers.authored(PACK, "elemental_ug", "elemental_bg")),
        space,
        Reward(RewardHypers()),
        seed=164,
    )
    observation, _ = env.reset(seed=164)
    agent = Agent(space, AgentSpec(hidden_dim=16, semantic_pack=PACK, policy_film=True))
    native = managym.Env(seed=164)
    native.reset(
        Match(MatchHypers.authored(PACK, "elemental_ug", "elemental_bg")).to_rust()
    )
    pack = BoundSemanticPack.from_env(native)
    assert len(pack.ir.definitions) == 100
    assert pack.schema.raw["schema_version"] == 6
    pack.pad_catalog()
    tensors = {
        key: torch.as_tensor(value).unsqueeze(0) for key, value in observation.items()
    }
    policy, value = agent(tensors)
    legal = tensors["actions_valid"].bool()
    assert torch.isfinite(policy[legal]).all()
    assert torch.isfinite(value).all()
    loss = value.sum() - policy[legal].mean()
    loss.backward()
    assert any(
        p.grad is not None and bool((p.grad != 0).any()) for p in agent.parameters()
    )
    actions = np.flatnonzero(observation["actions_valid"])
    _, _, _, truncated, info = env.step(int(actions[0]))
    assert not truncated
    assert not info["action_space_truncated"]
    path = tmp_path / "elemental.pt"
    save_bc_checkpoint(agent, space, path, player_configs=env.match.to_rust())
    summary = inspect_checkpoint(path)
    assert summary.compatible
    assert len(summary.setups) == 10
    restored, _ = load_checkpoint_agent(str(path))
    restored_policy, restored_value = restored(tensors)
    torch.testing.assert_close(restored_policy, policy)
    torch.testing.assert_close(restored_value, value)


def test_elemental_names_cannot_fall_back_to_an_uncompiled_mixed_world() -> None:
    with pytest.raises(ValueError, match="Elemental roster"):
        _parse_game_config(
            {
                "hero_deck": "elemental_ug",
                "villain_deck": "ur_lessons",
                "villain_type": "random",
            }
        )


def test_selected_world_runs_ordinary_ppo_and_reloads_its_roster(
    tmp_path: Path,
) -> None:

    torch.set_num_threads(1)
    space = ObservationSpace()
    match = Match(MatchHypers.authored(PACK, "elemental_ug", "elemental_bg"))
    experiment = Experiment(
        ExperimentHypers(
            exp_name="elemental-smoke",
            seed=164,
            wandb=False,
            device="cpu",
            runs_dir=tmp_path,
        )
    )
    env = VectorEnv(
        num_envs=2,
        match=match,
        observation_space=space,
        reward=Reward(RewardHypers()),
        device="cpu",
        opponent_policy="passive",
    )
    agent = Agent(space, AgentSpec(hidden_dim=16, semantic_pack=PACK, policy_film=True))
    before = {name: value.detach().clone() for name, value in agent.named_parameters()}
    trainer = Trainer(
        agent,
        experiment,
        env,
        TrainHypers(
            num_envs=2,
            num_steps=8,
            num_minibatches=2,
            update_epochs=1,
            total_timesteps=32,
            opponent_policy="passive",
        ),
    )
    try:
        trainer.train()
        assert trainer.global_step == 32
        assert any(
            not torch.equal(before[name], value)
            for name, value in agent.named_parameters()
        )
        assert all(torch.isfinite(value).all() for value in agent.parameters())
        trainer.save()
        restored, _ = load_checkpoint_agent(str(experiment.runs_dir / "step_32.pt"))
        assert len(restored.world_binding["setups"]) == 10
        validate_agent_setup(
            restored,
            Match(MatchHypers.authored(PACK, "elemental_wr", "elemental_wr")).to_rust(),
        )
    finally:
        env.close()
        experiment.close()


def test_avatar_grant_reaches_native_python_and_academy_views() -> None:

    engine = managym.Env(seed=164, skip_trivial=False)
    obs, _ = engine.reset(
        [
            managym.authored_deck_setup(PACK, key)
            for key in ("elemental_gw", "elemental_ug")
        ]
    )
    for _ in range(20):
        if int(obs.turn.step) == 3:
            break
        action = next(
            row
            for row in describe_actions(obs)
            if row["type"] == "PRIORITY_PASS_PRIORITY"
        )
        obs = engine.step(action["index"])[0]
    engine.scenario_clear_hand(0)
    engine.scenario_clear_hand(1)
    engine.scenario_force_card_in_hand(0, "Enter the Avatar State")
    engine.scenario_force_battlefield(0, "Compassionate Healer", True)
    engine.scenario_force_battlefield(0, "Plains", True)
    obs = engine.scenario_refresh()
    cast = next(
        row
        for row in describe_actions(obs)
        if row["description"] == "Cast Enter the Avatar State"
    )
    obs = engine.step(cast["index"])[0]
    for _ in range(20):
        view = engine.observation_for_player(0)
        if any(row.temp_subtypes for row in view.agent_permanents):
            break
        actions = describe_actions(obs)
        action = next(
            (
                row
                for row in actions
                if row["description"] == "Target Compassionate Healer"
            ),
            None,
        )
        if action is None:
            action = next(
                row for row in actions if row["type"] == "PRIORITY_PASS_PRIORITY"
            )
        obs = engine.step(action["index"])[0]
    else:
        pytest.fail("Avatar grant did not resolve")
    expected = ObservationEncoder(ObservationSpaceHypers()).encode(view)
    actual = engine.encode_observation(view)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    assert actual["agent_permanents"][0, 26] == 1
    payload = serialize_observation(view)
    permanent = next(
        row
        for row in payload["agent"]["battlefield"]
        if row["name"] == "Compassionate Healer"
    )
    assert LegacyPermanentView.model_validate(permanent).temp_subtypes == ["Avatar"]


def test_academy_records_a_real_elemental_command_with_the_exact_pack(
    tmp_path: Path,
) -> None:

    session = GameSession(tmp_path)
    update = session.new_game(
        {
            "hero_deck": "elemental_ug",
            "villain_deck": "elemental_bg",
            "villain_type": "passive",
            "seed": 164,
            "auto_pass": False,
        }
    )
    frame = update["frame"]
    assert frame["asset_pack"] == ELEMENTAL_WORLD_PACK.reference
    result = session.hero_command(
        {
            "command_id": "elemental-human",
            "match_id": frame["match_id"],
            "expected_revision": frame["revision"],
            "prompt_id": frame["prompt"]["id"],
            "offer_id": frame["offers"][0]["id"],
            "answers": [],
        }
    )
    assert result["status"] == "accepted"
    assert session.env is not None
    assert session.env.content_pack_manifest()["compiled_semantics"]["pack_key"] == PACK
    assert session.trace is not None
    assert (
        session.trace.config.hero_sideboard
        == managym.authored_deck_setup(PACK, "elemental_ug").sideboard
    )
    assert (
        session.trace.config.villain_sideboard
        == managym.authored_deck_setup(PACK, "elemental_bg").sideboard
    )
    session.close(end_reason="declared-validation")


@pytest.mark.parametrize("left,right", [("ug", "bg"), ("bg", "ug")])
def test_elemental_complete_games_use_the_ordinary_training_encoder(
    left: str, right: str
) -> None:
    env = Env(
        Match(MatchHypers.authored(PACK, "elemental_" + left, "elemental_" + right)),
        ObservationSpace(),
        Reward(RewardHypers()),
        seed=164,
    )
    observation, _ = env.reset(seed=164)
    rng = np.random.default_rng(164)
    for _ in range(5000):
        actions = np.flatnonzero(observation["actions_valid"])
        observation, _, terminal, truncated, info = env.step(int(rng.choice(actions)))
        assert not truncated
        assert not info["action_space_truncated"]
        assert not info["card_space_truncated"]
        assert not info["permanent_space_truncated"]
        if terminal:
            return
    pytest.fail(
        f"{left}/{right} did not terminate within the 5000-action encoder smoke cap"
    )
