"""Saved compound-policy search fixtures; no fitting or scientific evaluation."""

from dataclasses import replace
from itertools import product
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from etude.advice import BeliefNormalizationReceipt
from etude.local_advice import local_update_scenario
from manabot.belief.likelihood import file_sha256
from manabot.belief.state import ViewerHistory
from manabot.env import Env, Match, ObservationSpace
from manabot.infra.hypers import AgentSpec
from manabot.model.agent import Agent
from manabot.sim.distill import (
    LOCAL_RECEIPT_KEY,
    LOCAL_TARGET_KEY,
    generate_selfplay_shard,
    load_shards,
    save_bc_checkpoint,
)
from manabot.sim.local_compound import advance_compound, project_compound
from manabot.sim.local_update import (
    LocalSearchConfig,
    LocalUpdateReceipt,
    LocalUpdateTeacher,
)
from manabot.sim.search_supervised import _validate_dataset
from manabot.training.models import CollectLocalUpdate, TrainCompound, TrainingRegime
import managym
from managym.decision import Observation
from tests.sim import test_local_update as local_fixtures
from tests.sim.test_local_update import small_match
from tests.sim.test_local_update_sampling import saved_sampler
from tests.training.test_compound import _agent, _block_root, _root, _waterbend_root


def check() -> None:
    pass


@pytest.mark.parametrize("kind", ["attack", "cast", "blockers"])
def test_native_prefix_projection(kind: str) -> None:
    env, _ = _block_root(2, 2) if kind == "blockers" else _root(kind, 3)
    agent = _agent()
    cursor = None
    for step in range(10):
        projection = project_compound(agent, env, cursor, check)
        assert projection.probabilities.sum() == pytest.approx(1)
        assert np.all(projection.probabilities > 0)
        action = 0
        if kind == "cast" and step == 0:
            action = next(
                index
                for index, row in enumerate(projection.cursor.root.offers.offers)
                if row["verb"] == "cast"
            )
        command = projection.choices[action].command
        cursor = advance_compound(agent, projection, action, check)
        env.execute_semantic_command_json(command.to_json())
        if cursor is None:
            break
    else:
        pytest.fail("compound declaration did not finish")
    assert step >= 1


@pytest.mark.parametrize("lands", [0, 2, 5])
def test_payment_joint_parity(lands: int) -> None:
    torch.set_num_threads(1)
    env, _ = _waterbend_root(6, lands=lands)
    agent = _agent()
    root = project_compound(agent, env, None, check).cursor
    total = 0.0
    saw_zero = False
    for bits in product((0, 1), repeat=6):
        if not max(0, 5 - lands) <= sum(bits) <= 5:
            continue
        with torch.inference_mode():
            output = agent.compound(
                root.root.observation, root.root.offers, tokens=(0, *bits)
            )
        commands = json.loads(
            env.compound_commands_json(
                env.compound_offers(), output.submission.to_json()
            )
        )
        branch = env.clone_env()
        cursor = None
        probability = 1.0
        for row in commands:
            projection = project_compound(agent, branch, cursor, check)
            assert projection.probabilities.sum() == pytest.approx(1, abs=1e-6)
            for i in np.flatnonzero(projection.probabilities == 0):
                saw_zero = True
                with pytest.raises(ValueError, match="outside retained"):
                    advance_compound(agent, projection, int(i), check)
            action = next(
                i
                for i, choice in enumerate(projection.choices)
                if choice.command.offer_id == row["offer_id"]
            )
            probability *= projection.choices[action].probability
            cursor = advance_compound(agent, projection, action, check)
            branch.execute_semantic_command_json(json.dumps(row))
        assert cursor is None
        assert probability == pytest.approx(float(output.log_prob.exp()), rel=1e-5)
        total += probability
    assert total == pytest.approx(1, abs=1e-6)
    assert saw_zero


@pytest.mark.parametrize("compound", [False, True])
@pytest.mark.parametrize("learned", [False, True])
def test_supported_root_rollout_reaching_payment_replays(
    tmp_path: Path, learned: bool, compound: bool
) -> None:
    """An admitted priority root can roll through payment on either sampler."""
    torch.set_num_threads(1)
    configs = [
        managym.PlayerConfig("a", {"Water Tribe Rallier": 80, "Forest": 20}),
        managym.PlayerConfig("b", {"Gray Ogre": 20, "Mountain": 20}),
    ]
    env = managym.Env(seed=81, skip_trivial=False)
    env.reset(configs)
    env.scenario_clear_hand(0)
    env.scenario_clear_hand(1)
    for _ in range(6):
        env.scenario_force_battlefield(0, "Water Tribe Rallier", ready=True)
    raw = env.scenario_refresh()
    space = ObservationSpace()
    agent = (
        _agent()
        if compound
        else Agent(space, AgentSpec(hidden_dim=8, num_attention_heads=2))
    )
    path = tmp_path / "compound.pt"
    save_bc_checkpoint(
        agent,
        space,
        path,
        player_configs=configs,
        extra={"value_semantic": "signed_outcome"},
    )
    config = LocalSearchConfig(depth=2, decision_seconds=20)
    teacher = LocalUpdateTeacher(path, file_sha256(path), config)
    if learned:
        artifact = saved_sampler(teacher, tmp_path / "sampler.pt")
        teacher = LocalUpdateTeacher(
            path,
            file_sha256(path),
            config.model_copy(update={"sampling": "learned"}),
            sampler=artifact,
        )
    action = next(
        i
        for i, row in enumerate(raw.action_space.actions)
        if row.action_type == managym.ActionEnum.PRIORITY_ACTIVATE_ABILITY
    )
    branch = env.clone_env()
    if compound:
        root = project_compound(teacher.agent, env, None, check)
        assert root.choices[action].probability > 0
        branch.execute_semantic_command_json(root.choices[action].command.to_json())
    else:
        assert teacher.predict(env)[0][action] > 0
        branch.step(action)
    assert json.loads(branch.compound_offers().projection_json())["kind"] == "waterbend"
    history = ViewerHistory.from_observation(
        Observation.from_json(
            env.semantic_observation_json(int(env.current_agent_index()))
        )
    )
    before = env.state_digest()
    constructions = env.possible_world_space_construction_count()
    receipt = teacher.search(env, history, seed=19)
    teacher.verify_replay(env, history, receipt)
    LocalUpdateReceipt.from_json(receipt.to_json())
    assert (
        LocalUpdateReceipt.from_json(
            replace(receipt, schema="regularized-local-update/v2").to_json()
        ).target
        == receipt.target
    )
    assert env.state_digest() == before
    assert env.possible_world_space_construction_count() == constructions
    if not compound:
        return

    # Skip the first tap and take the second. The still-legal first tap now has
    # zero conditional mass, while the original root and recurrent prefix survive.
    payment = project_compound(teacher.agent, branch, None, check)
    action = next(
        i for i, choice in enumerate(payment.choices) if choice.prefix == (0, 0, 1)
    )
    cursor = advance_compound(teacher.agent, payment, action, check)
    assert cursor is not None
    branch.execute_semantic_command_json(payment.choices[action].command.to_json())
    viewer = int(branch.current_agent_index())
    history = ViewerHistory.from_observation(
        Observation.from_json(branch.semantic_observation_json(viewer))
    )
    receipt = teacher.search(branch, history, seed=23, compound_cursor=cursor)
    teacher.verify_replay(branch, history, receipt, compound_cursor=cursor)
    assert LocalUpdateReceipt.from_json(receipt.to_json()) == receipt
    inactive = np.asarray(receipt.base) == 0
    assert inactive.any()
    assert all((v is None) == zero for v, zero in zip(receipt.values, inactive))
    assert np.all(np.asarray(receipt.target)[inactive] == 0)
    assert np.all(np.asarray(receipt.allocation_counts)[inactive] == 0)
    assert all(np.isfinite(v) for v in receipt.mixing_diagnostics().values())
    with pytest.raises(TimeoutError):
        teacher.search(branch, history, seed=23, compound_cursor=cursor, deadline=1)
    for schema in ("regularized-local-update/v1", "regularized-local-update/v2"):
        with pytest.raises(ValueError, match="historical"):
            LocalUpdateReceipt.from_json(replace(receipt, schema=schema).to_json())
    bad_values = tuple(0.0 if v is None else v for v in receipt.values)
    with pytest.raises(ValueError, match="exactly on policy support"):
        LocalUpdateReceipt.from_json(replace(receipt, values=bad_values).to_json())
    bad_target = np.asarray(receipt.target)
    bad_target[np.flatnonzero(inactive)[0]] = 1e-12
    with pytest.raises(ValueError, match="target"):
        LocalUpdateReceipt.from_json(
            replace(receipt, target=tuple(bad_target)).to_json()
        )

    encoded = space.encode(branch.observation_for_player(viewer))
    dataset = {key: np.expand_dims(value, 0) for key, value in encoded.items()}
    count = len(receipt.target)
    dataset.update(
        {
            "action": np.array([int(np.argmax(receipt.target))]),
            "game_index": np.array([0]),
            "num_valid": np.array([count]),
            "seat": np.array([viewer]),
            "winner": np.array([-1]),
            LOCAL_RECEIPT_KEY: np.array([receipt.to_json()]),
            LOCAL_TARGET_KEY: np.zeros_like(dataset["actions_valid"]),
        }
    )
    dataset[LOCAL_TARGET_KEY][0, :count] = receipt.target
    saved = tmp_path / "payment-target.npz"
    np.savez(saved, **dataset)
    dataset = load_shards([saved])
    evidence = local_update_scenario(
        receipt,
        BeliefNormalizationReceipt(
            scenario_id="baseline",
            space_identity=receipt.world_identity,
            belief_model_id=receipt.belief_model,
            distribution_sha256=receipt.belief_digest,
            normalized_belief_sha256=receipt.belief_digest,
            positive_support=1,
            normalization_error=0,
            provenance_kind="model_inferred",
            provenance_identity=receipt.policy_sha256,
        ),
        labels=["payment"] * count,
    )
    assert evidence.sampled_worlds == 1
    for action, zero in zip(evidence.actions, inactive):
        assert (action.q.status == "unavailable") == zero
    assert evidence.root_value.status == "available"
    assert np.isfinite(evidence.root_value.value)
    for kind in ("local_soft", "local_argmax", "local_allocation"):
        targets = _validate_dataset(
            dataset, policy_target_kind=kind, value_target_kind="terminal_outcome"
        )
        assert targets is not None
        assert targets.sum() == pytest.approx(1)
        assert np.all(targets[0, :count][inactive] == 0)

    dataset[LOCAL_TARGET_KEY][0, np.flatnonzero(inactive)[0]] = 1e-12
    with pytest.raises(ValueError, match="target differs"):
        _validate_dataset(
            dataset,
            policy_target_kind="local_soft",
            value_target_kind="terminal_outcome",
        )

    # Direct count materialization changes hidden truth, never the viewer policy.
    constraints = branch.hidden_hand_constraints_json(viewer)
    hand = receipt.rollouts[0].sampled_hand
    assert hand is not None
    swapped = branch.materialize_sampled_hand(
        viewer, constraints, dict(hand.counts), 912
    )
    swapped_projection = project_compound(teacher.agent, swapped, cursor, check)
    np.testing.assert_array_equal(swapped_projection.probabilities, receipt.base)
    assert branch.possible_world_space_construction_count() == constructions


@pytest.mark.parametrize("kind", ["attack", "blockers"])
def test_joint_parity(kind: str) -> None:
    env, _ = _root("attack", 3) if kind == "attack" else _block_root(2, 2)
    agent = _agent()
    root = project_compound(agent, env, None, check).cursor
    tapes = (
        list(product((0, 1), repeat=3))
        if kind == "attack"
        else [
            tuple(bit for role in roles for bit in role)
            for roles in product(((0, 0), (1, 0), (0, 1)), repeat=2)
        ]
    )
    for bits in tapes:
        with torch.inference_mode():
            output = agent.compound(
                root.root.observation, root.root.offers, tokens=(0, *bits)
            )
        commands = json.loads(
            env.compound_commands_json(
                env.compound_offers(), output.submission.to_json()
            )
        )
        branch = env.clone_env()
        cursor = None
        probability = 1.0
        for row in commands:
            projected = project_compound(agent, branch, cursor, check)
            action = next(
                index
                for index, choice in enumerate(projected.choices)
                if choice.command.offer_id == row["offer_id"]
            )
            probability *= projected.choices[action].probability
            cursor = advance_compound(agent, projected, action, check)
            branch.execute_semantic_command_json(json.dumps(row))
        assert cursor is None
        assert probability == pytest.approx(float(output.log_prob.exp()), rel=1e-5)


@pytest.mark.parametrize("kind", ["attack", "cast", "blockers"])
@pytest.mark.parametrize("learned", [False, True])
def test_saved_compound_teacher_replays(
    tmp_path: Path, kind: str, learned: bool
) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(3)
    space = ObservationSpace()
    agent = Agent(
        space, AgentSpec(compound_decisions=True, hidden_dim=8, num_attention_heads=2)
    )
    config = [
        managym.PlayerConfig(
            "a",
            {"Lightning Bolt": 4, "Mountain": 36}
            if kind == "cast"
            else {"Gray Ogre": 10 if kind == "blockers" else 11, "Mountain": 4},
        ),
        managym.PlayerConfig("b", {"Gray Ogre": 36, "Mountain": 4}),
    ]
    path = tmp_path / "compound.pt"
    save_bc_checkpoint(agent, space, path, player_configs=config)
    teacher = LocalUpdateTeacher(
        path, file_sha256(path), LocalSearchConfig(depth=2, decision_seconds=20)
    )
    if learned:
        artifact = saved_sampler(teacher, tmp_path / "sampler.pt")
        teacher = LocalUpdateTeacher(
            path,
            file_sha256(path),
            teacher.config.model_copy(update={"sampling": "learned"}),
            sampler=artifact,
        )
    env, _ = _block_root(2, 2) if kind == "blockers" else _root(kind, 3)
    viewer = int(env.current_agent_index())
    history = ViewerHistory.from_observation(
        Observation.from_json(env.semantic_observation_json(viewer))
    )
    before = env.possible_world_space_construction_count()
    receipt = teacher.search(env, history, seed=19)
    teacher.verify_replay(env, history, receipt)
    assert receipt.compound_prefix is not None
    assert all(row.decoder_factors > 0 for row in receipt.rollouts)
    assert env.possible_world_space_construction_count() == before
    assert (
        LocalUpdateReceipt.from_json(receipt.to_json()).replay_identity()
        == receipt.replay_identity()
    )


@pytest.mark.parametrize("learned", [False, True])
def test_compound_complete_game_targets_without_fitting(
    tmp_path: Path, learned: bool
) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(73)
    match = Match(small_match())
    space = ObservationSpace()
    agent = Agent(
        space, AgentSpec(compound_decisions=True, hidden_dim=8, num_attention_heads=2)
    )
    checkpoint = tmp_path / "compound.pt"
    save_bc_checkpoint(agent, space, checkpoint, player_configs=match.to_rust())
    teacher = LocalUpdateTeacher(
        checkpoint,
        file_sha256(checkpoint),
        LocalSearchConfig(depth=1, decision_seconds=20),
    )
    artifact = saved_sampler(teacher, tmp_path / "sampler.pt") if learned else None
    spec = {
        "kind": "local_update",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "config": teacher.config.model_copy(
            update={"sampling": "learned" if learned else "compatible_prior"}
        ).model_dump(),
        **(
            {"sampler": artifact.model_dump(mode="json")}
            if artifact is not None
            else {}
        ),
    }
    shard = tmp_path / "compound-targets.npz"
    generate_selfplay_shard(
        num_games=1,
        seed=7,
        out_path=shard,
        max_steps_per_game=1000,
        teacher_spec=spec,
        match_hypers=small_match(),
    )
    dataset = load_shards([shard])
    labels = _validate_dataset(
        dataset, policy_target_kind="local_soft", value_target_kind="terminal_outcome"
    )
    np.testing.assert_array_equal(labels, dataset[LOCAL_TARGET_KEY])
    receipts = [
        LocalUpdateReceipt.from_json(str(row)) for row in dataset[LOCAL_RECEIPT_KEY]
    ]
    assert all(receipt.compound_prefix is not None for receipt in receipts)
    assert any(
        receipt.compound_prefix.commands
        for receipt in receipts
        if receipt.compound_prefix is not None
    )
    assert {receipt.viewer for receipt in receipts} == {0, 1}


def test_prefix_completion_preserves_complete_tape_contract() -> None:
    env, _ = _root("attack", 3)
    agent = _agent()
    root = project_compound(agent, env, None, check).cursor.root
    with torch.inference_mode():
        output = agent.compound(
            root.observation,
            root.offers,
            prefix=(0, 1),
            generator=torch.Generator().manual_seed(5),
        )
        forced = agent.compound(root.observation, root.offers, tokens=output.tokens)
    assert output.tokens[:2] == (0, 1)
    torch.testing.assert_close(output.log_probs, forced.log_probs)
    torch.testing.assert_close(output.end_value, forced.end_value)
    with pytest.raises(ValueError, match="interrupted"):
        agent.compound(root.observation, root.offers, tokens=(0, 1))
    with pytest.raises(ValueError, match="trailing"):
        agent.compound(root.observation, root.offers, prefix=(0, 1, 0, 1, 1))


def test_interrupted_prefix_fails_closed() -> None:
    env, _ = _root("attack", 3)
    agent = _agent()
    projected = project_compound(agent, env, None, check)
    cursor = advance_compound(agent, projected, 0, check)
    assert cursor is not None
    env.execute_semantic_command_json(projected.choices[1].command.to_json())
    with pytest.raises(ValueError, match="another viewer state"):
        project_compound(agent, env, cursor, check)


def test_compound_regime_admits_direct_collection() -> None:
    regime = TrainingRegime(
        id="saved-compound-collection-contract",
        world=managym.WORLD_VERSION,
        match=small_match(),
        agent=AgentSpec(compound_decisions=True),
        stages=[
            TrainCompound(id="policy", operation="train_compound"),
            CollectLocalUpdate(
                id="labels", operation="collect_local_update", policy="policy", games=2
            ),
        ],
    )
    assert isinstance(regime.stages[-1], CollectLocalUpdate)


@pytest.mark.parametrize("kind,count", [("attack", 65), ("cast", 33), ("payment", 65)])
def test_wide_projection_has_no_declaration_enumeration(kind: str, count: int) -> None:
    env, _ = (
        _waterbend_root(count, lands=2) if kind == "payment" else _root(kind, count)
    )
    agent = _agent(wide=True)
    projection = project_compound(agent, env, None, check)
    assert len(projection.choices) == len(
        env.observation_for_player(env.current_agent_index()).action_space.actions
    )
    assert projection.probabilities.sum() == pytest.approx(1, abs=1e-6)


def test_saved_compound_rejects_exact_reference(tmp_path: Path) -> None:
    agent = _agent()
    checkpoint = tmp_path / "compound.pt"
    save_bc_checkpoint(
        agent,
        agent.observation_space,
        checkpoint,
        player_configs=Match(small_match()).to_rust(),
    )
    with pytest.raises(ValueError, match="exact-history"):
        LocalUpdateTeacher(
            checkpoint, file_sha256(checkpoint), LocalSearchConfig(sampling="belief")
        )


def test_compound_arena_executes_and_replays_saved_policies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_make = local_fixtures.make_teacher

    def make(directory: Path) -> tuple[LocalUpdateTeacher, Env]:
        original, env = original_make(directory)
        agent = Agent(
            original.space,
            AgentSpec(compound_decisions=True, hidden_dim=8, num_attention_heads=2),
        )
        checkpoint = directory / "compound.pt"
        save_bc_checkpoint(
            agent, original.space, checkpoint, player_configs=env.match.to_rust()
        )
        return LocalUpdateTeacher(
            checkpoint,
            file_sha256(checkpoint),
            LocalSearchConfig(depth=1, decision_seconds=20),
        ), env

    def attach(teacher: LocalUpdateTeacher, directory: Path) -> LocalUpdateTeacher:
        artifact = saved_sampler(teacher, directory / "saved-sampler.pt")
        return LocalUpdateTeacher(
            teacher.likelihood.checkpoint,
            teacher.likelihood.checkpoint_sha256,
            teacher.config.model_copy(update={"sampling": "learned"}),
            sampler=artifact,
        )

    monkeypatch.setattr(local_fixtures, "make_teacher", make)
    monkeypatch.setattr(local_fixtures, "attach_learned_sampler", attach)
    local_fixtures.test_arena_uses_exact_range_lifecycle_and_replays(
        tmp_path, learned=True
    )
