"""Analytic mixing, exact branch replay and current-world collection proofs."""

from dataclasses import replace
from pathlib import Path

from managym._managym import AgentError
import numpy as np
import pytest
import torch

from manabot.backends.torch.checkpoint import save_checkpoint
from manabot.backends.torch.policy import ActorCritic
from manabot.belief.sampling_data import collect_frozen_policy
from manabot.belief.sampling_fit import fit_belief_sampler, save_belief_sampler
from manabot.belief.state import ViewerHistory
from manabot.belief.tracker import BeliefTracker
from manabot.env.env import Env
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import (
    MatchConfig,
    ModelConfig,
    ObservationConfig,
    RewardConfig,
)
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.sim.native_game import execute_game
from manabot.sim.registration import PlayerRegistration, SearchSemantics
from manabot.sim.replay import replay_games
from manabot.sim.search import local_update
from manabot.sim.search.local_update import (
    LocalSearchConfig,
    LocalUpdateReceipt,
    LocalUpdateTeacher,
    SamplerArtifact,
    regularized_update,
)
import managym
from managym.decision import Observation
from managym.possible_worlds import PossibleWorldSpace, WorldQuery


def small_match() -> MatchConfig:
    return MatchConfig(
        hero_deck={"Mountain": 8, "Gray Ogre": 8},
        villain_deck={"Forest": 8, "Llanowar Elves": 8},
    )


def make_teacher(
    tmp_path: Path,
    *,
    config: LocalSearchConfig | None = None,
    observation: ObservationConfig | None = None,
    native_encoder: bool = True,
) -> tuple[LocalUpdateTeacher, Env]:
    torch.set_num_threads(1)
    torch.manual_seed(15)
    space = ObservationSpace(observation)
    history = space.encoder.config.policy_history_version == 1
    agent = ActorCritic(
        space,
        ModelConfig(
            hidden_dim=8,
            num_attention_heads=2,
            world=managym.WORLD_SETUP if history else None,
            recent_events=history,
        ),
    )
    match = Match(small_match())
    path = tmp_path / "policy.pt"
    save_checkpoint(
        agent,
        space,
        path,
        player_configs=match.to_rust(),
        extra={"value_semantic": "signed_outcome"},
    )
    teacher = LocalUpdateTeacher(
        path,
        file_sha256(path),
        config
        or LocalSearchConfig(worlds=2, depth=3, decision_seconds=10, sampling="belief"),
        native_encoder=native_encoder,
    )
    env = Env(match, space, Reward(RewardConfig()), enable_profiler=False)
    env.reset(seed=42)
    return teacher, env


@pytest.mark.parametrize(
    "owner",
    [
        "manabot/backends/torch/_facts.py",
        "manabot/sim/search/rollout.py",
        "manabot/sim/search/rollout_receipt.py",
    ],
)
def test_teacher_identity_tracks_implementation_owners(
    monkeypatch: pytest.MonkeyPatch, owner: str
) -> None:
    original = local_update.local_search_source_sha256()
    revision = "implementation-revision-one"

    def source_hash(path: Path) -> str:
        if path.as_posix().endswith("/" + owner):
            return canonical_sha256(revision)
        return file_sha256(path)

    monkeypatch.setattr(local_update, "file_sha256", source_hash)
    first = local_update.local_search_source_sha256()
    assert first != original
    assert local_update.local_search_source_sha256() == first
    revision = "implementation-revision-two"
    assert local_update.local_search_source_sha256() != first


def test_two_kl_optimum_and_shift_invariance() -> None:
    base = np.array([0.2, 0.8])
    reference = np.array([0.5, 0.5])
    values = np.array([-1.0, 1.0])
    result = regularized_update(base, values, reference, alpha=0.3, beta=2)
    # First-order simplex stationarity independently checks the optimizer.
    gradient = (
        values
        - 0.3 * (np.log(result / reference) + 1)
        - 2 * (np.log(result / base) + 1)
    )
    assert gradient[0] == pytest.approx(gradient[1])
    assert np.all(result > 0)
    np.testing.assert_allclose(
        result, regularized_update(base, values + 8, reference, alpha=0.3, beta=2)
    )
    np.testing.assert_array_equal(
        regularized_update(
            np.array([0.0, 1.0]), np.array([np.nan, 1.0]), reference, alpha=0.3, beta=2
        ),
        [0.0, 1.0],
    )


def test_zero_support_update_is_the_restricted_simplex_optimum() -> None:
    base = np.array([0.2, 0.0, 0.8])
    values = np.array([-0.7, np.nan, 0.4])
    reference = np.full(3, 1 / 3)
    target = regularized_update(base, values, reference, alpha=0.3, beta=2)
    active = base > 0
    gradient = (
        values[active]
        - 0.3 * (np.log(target[active] / reference[active]) + 1)
        - 2 * (np.log(target[active] / base[active]) + 1)
    )
    assert gradient[0] == pytest.approx(gradient[1])
    assert target[1] == 0
    assert target.sum() == pytest.approx(1)
    for inaccessible in (-1e10, 1e10, np.nan):
        values[1] = inaccessible
        np.testing.assert_array_equal(
            target, regularized_update(base, values, reference, alpha=0.3, beta=2)
        )


def test_real_root_replay_and_hidden_swap(tmp_path: Path) -> None:
    teacher, env = make_teacher(tmp_path)
    viewer = env._engine.current_agent_index()
    tracker = BeliefTracker.from_engine(
        env._engine, viewer=viewer, likelihood=teacher.likelihood, epsilon=0
    )
    before = env._engine.search_witness_json()
    receipt = teacher.search(env._engine, tracker.posterior, seed=7)
    replay = teacher.search(env._engine, tracker.posterior, seed=7)
    assert receipt.replay_identity() == replay.replay_identity()
    teacher.verify_replay(env._engine, tracker.posterior, receipt)
    assert before == env._engine.search_witness_json()
    assert receipt.rollouts and all(row.branch_audit_json for row in receipt.rollouts)
    assert len(receipt.offer_ids) == len(receipt.target)
    assert sum(receipt.target) == pytest.approx(1)
    for index in (0, tracker.space.support_size - 1):
        swapped = tracker.space.materialize(index, seed=95)
        swapped_tracker = BeliefTracker.from_engine(
            swapped, viewer=viewer, likelihood=teacher.likelihood, epsilon=0
        )
        result = teacher.search(swapped, swapped_tracker.posterior, seed=7)
        np.testing.assert_array_equal(receipt.target, result.target)
    with pytest.raises(TimeoutError):
        teacher.search(env._engine, tracker.posterior, seed=7, deadline=1)
    with pytest.raises(ValueError):
        teacher.search(
            env._engine,
            tracker.posterior,
            seed=7,
            query=WorldQuery.has("Lightning Bolt"),
        )


@pytest.mark.parametrize("learned", [False, True])
def test_arena_uses_exact_range_lifecycle_and_replays(
    tmp_path: Path, learned: bool
) -> None:

    teacher, _ = make_teacher(tmp_path)
    teacher = LocalUpdateTeacher(
        teacher.likelihood.checkpoint,
        teacher.likelihood.checkpoint_sha256,
        teacher.config.model_copy(
            update={"sampling": "compatible_prior", "worlds": 1, "depth": 1}
        ),
    )
    if learned:
        teacher = attach_learned_sampler(teacher, tmp_path)
    identity = "0" * 64
    common = dict(
        role="challenger",
        compute_class_id="bounded-fixture",
        information_boundary="acting-viewer-history-only-v1",
        world="w4",
        content_suite="tiny-fixture",
        observation_abi_sha256=identity,
        action_abi_sha256=identity,
        matchup_sha256=identity,
        player_seed_derivation_id="fixture-seed",
    )
    candidate = PlayerRegistration(
        **common,
        player_id="local-fixture",
        display_name="Local fixture",
        runner_kind="checkpoint",
        checkpoint_sha256=teacher.likelihood.checkpoint_sha256,
        checkpoint_bytes=teacher.likelihood.checkpoint.stat().st_size,
        parameter_count=sum(p.numel() for p in teacher.agent.parameters()),
        training_seed=15,
        artifact_id="untrained-fixture-15",
        evidence_class="fixture",
        player_spec={
            "kind": "local_update",
            "config": teacher.config.model_dump(),
            **(
                {"sampler": teacher.sampler_artifact.model_dump(mode="json")}
                if teacher.sampler_artifact is not None
                else {}
            ),
            "implementation_source_sha256": local_update.local_search_source_sha256(),
        },
        search_call_seed_derivation_id="local-policy-seed-times-1000003-plus-call-mod-2pow63/v1",
        search_semantics=SearchSemantics(
            branch_audit=teacher.config.rollout_evidence == "full",
            root_prior="two-kl-local-update/v1",
            leaf_evaluator="frozen-viewer-policy-signed-value/v1",
        ),
    )
    opponent = PlayerRegistration(
        **common,
        player_id="random-fixture",
        display_name="Random",
        runner_kind="code",
        source_sha256=identity,
        player_spec={"kind": "random"},
    )
    games = []
    for leg in range(2):
        registrations = [candidate, opponent] if leg == 0 else [opponent, candidate]
        game = dict(
            match_id=f"fixture-{leg}",
            deal_seed=93,
            leg=leg,
            match_hypers=small_match().model_dump(),
            seat_players=[p.player_id for p in registrations],
            player_seeds={p.player_id: 7 for p in registrations},
            decisions=[],
            integrity=dict(
                private_exposures=0, root_mutations=0, illegal_actions=0, truncations=0
            ),
        )
        messages: list[tuple[str, object]] = []
        execute_game(
            game,
            registrations,
            {candidate.player_id: str(teacher.likelihood.checkpoint)},
            1000,
            messages.append,
        )
        assert game.get("terminated"), game.get("failure")
        assert any("local_update_receipt" in decision for decision in game["decisions"])
        game["game_trace_sha256"] = canonical_sha256(game)
        games.append(game)
    assert replay_games(games).passed


def attach_learned_sampler(
    teacher: LocalUpdateTeacher, tmp_path: Path
) -> LocalUpdateTeacher:

    dataset = collect_frozen_policy(
        checkpoint=teacher.likelihood.checkpoint,
        match_config=small_match(),
        games=3,
        seed=719,
        max_steps=1000,
    )
    result = fit_belief_sampler(
        dataset, steps=1, batch_size=4, hidden_size=8, evaluation_samples=2, seed=18
    )
    path = tmp_path / "sampler.pt"
    digest = save_belief_sampler(path, result, dataset)
    return LocalUpdateTeacher(
        teacher.likelihood.checkpoint,
        teacher.likelihood.checkpoint_sha256,
        teacher.config.model_copy(update={"sampling": "learned"}),
        sampler=SamplerArtifact(
            path=path,
            sha256=digest,
            dataset_identity=dataset.identity,
            schema_identity=dataset.schema.identity,
            policy_identity=dataset.policy_identity,
            world_identity=dataset.world_identity,
        ),
    )


def test_learned_search_direct_worlds_replay_and_rejection(tmp_path: Path) -> None:

    original, env = make_teacher(tmp_path)
    teacher = attach_learned_sampler(original, tmp_path)
    viewer = env._engine.current_agent_index()
    history = ViewerHistory.from_observation(
        Observation.from_json(env._engine.semantic_observation_json(viewer))
    )
    count_before = env._engine.possible_world_space_construction_count()
    receipt = teacher.search(env._engine, history, seed=17)
    teacher.verify_replay(env._engine, history, receipt)
    assert env._engine.possible_world_space_construction_count() == count_before
    assert receipt.learned_belief is not None
    assert not receipt.sampling_probabilities
    assert all(
        row.world_index is None and row.sampled_hand is not None
        for row in receipt.rollouts
    )
    assert (
        LocalUpdateReceipt.from_json(receipt.to_json()).replay_identity()
        == receipt.replay_identity()
    )
    # Enumeration is used only by this oracle comparison, never learned search.
    space = PossibleWorldSpace.from_engine(env._engine, viewer)
    swapped = space.materialize(space.support_size - 1, seed=55)
    changed = teacher.search(swapped, history, seed=17)
    assert changed.replay_identity() == receipt.replay_identity()
    with pytest.raises(ValueError, match="history"):
        teacher.search(
            env._engine,
            replace(history, current_revision=history.current_revision + 1),
            seed=17,
        )
    with pytest.raises(ValueError, match="query mass"):
        teacher.search(env._engine, history, seed=17, query=WorldQuery.has("Mountain"))
    constraints = receipt.learned_belief.constraints_json
    with pytest.raises(AgentError):
        env._engine.materialize_sampled_hand(viewer, constraints, {"Mountain": 999}, 1)
    with pytest.raises(AgentError, match="stale"):
        env._engine.materialize_sampled_hand(viewer, "{}", {}, 1)
    artifact = teacher.sampler_artifact
    assert artifact is not None
    with pytest.raises(ValueError, match="generating policy"):
        LocalUpdateTeacher(
            original.likelihood.checkpoint,
            original.likelihood.checkpoint_sha256,
            teacher.config,
            sampler=artifact.model_copy(update={"policy_identity": "wrong"}),
        )


def _searched_roots(env: Env, steps: int) -> list[managym.Env]:
    """The opening root plus later roots reached by first-offer Commands."""
    roots = [env._engine]
    for _ in range(steps):
        root = roots[-1].clone_env()
        root.step(0)
        if root.is_game_over():
            break
        roots.append(root)
    return roots


def test_compact_native_defaults_match_full_evidence_and_row_encoding(
    tmp_path: Path,
) -> None:
    compact_config = LocalSearchConfig(
        worlds=2, depth=6, decision_seconds=60, sampling="compatible_prior"
    )
    assert compact_config.rollout_evidence == "compact"
    full_config = compact_config.model_copy(update={"rollout_evidence": "full"})
    full, env = make_teacher(tmp_path, config=full_config, native_encoder=False)
    checkpoint = full.likelihood.checkpoint
    digest = full.likelihood.checkpoint_sha256
    arms = {
        "compact": LocalUpdateTeacher(
            checkpoint, digest, compact_config, native_encoder=False
        ),
        "native": LocalUpdateTeacher(checkpoint, digest, full_config),
        "defaults": LocalUpdateTeacher(checkpoint, digest, compact_config),
    }
    assert arms["defaults"].native_encoder
    for root in _searched_roots(env, 4):
        viewer = root.current_agent_index()
        history = ViewerHistory.from_observation(
            Observation.from_json(root.semantic_observation_json(viewer))
        )
        reference = full.search(root, history, seed=11)
        assert reference.schema == "regularized-local-update/v3"
        assert "rollout_evidence" not in reference.config_json
        # v3 receipts omit the field and still read as full evidence.
        assert (
            LocalUpdateReceipt.from_json(reference.to_json()).replay_identity()
            == reference.replay_identity()
        )
        for name, teacher in arms.items():
            receipt = teacher.search(root, history, seed=11)
            compact = teacher.config.rollout_evidence == "compact"
            assert receipt.schema == (
                "regularized-local-update/v4" if compact else reference.schema
            ), name
            for field in ("base", "values", "target", "allocation_counts"):
                assert getattr(receipt, field) == getattr(reference, field), name
            for row, expected in zip(receipt.rollouts, reference.rollouts, strict=True):
                assert row.canonical_commands == expected.canonical_commands
                assert row.signed_value == expected.signed_value
                assert row.terminal == expected.terminal
                assert row.rollout_seed == expected.rollout_seed
                if compact:
                    assert row.actor_observation_hashes == ()
                    assert '"audit": false' in row.branch_audit_json
                else:
                    assert row == expected
            assert (
                LocalUpdateReceipt.from_json(receipt.to_json()).replay_identity()
                == receipt.replay_identity()
            )
            teacher.verify_replay(root, history, receipt)

    receipt = arms["compact"].search(root, history, seed=11)
    shortened = replace(
        receipt,
        rollouts=(
            replace(
                receipt.rollouts[0],
                canonical_commands=receipt.rollouts[0].canonical_commands - 1 or 2,
            ),
            *receipt.rollouts[1:],
        ),
    )
    with pytest.raises(ValueError, match="Command"):
        LocalUpdateReceipt.from_json(shortened.to_json())
    relabeled = replace(receipt, schema="regularized-local-update/v3")
    with pytest.raises(ValueError, match="v4"):
        LocalUpdateReceipt.from_json(relabeled.to_json())


def test_native_frontier_encoding_writes_the_row_encoder_tensors(
    tmp_path: Path,
) -> None:
    teacher, env = make_teacher(tmp_path, native_encoder=False)
    roots = _searched_roots(env, 6)
    rows, counts = teacher._encode_rows(roots)
    buffers = teacher.space.encoder.allocate(len(roots) + 2)
    for value in buffers.values():
        value.fill(7)
    assert managym.encode_observations_into(roots, buffers, len(roots) + 2) == counts
    for key, expected in rows.items():
        np.testing.assert_array_equal(buffers[key][: len(roots)], expected, key)
        assert (buffers[key][len(roots) :] == 7).all(), key
    # A smaller frontier rewrites only its own rows of a reused buffer.
    assert (
        managym.encode_observations_into(roots[:1], buffers, len(roots) + 2)
        == (counts[:1])
    )
    for key, expected in rows.items():
        np.testing.assert_array_equal(buffers[key][: len(roots)], expected, key)
    with pytest.raises(ValueError, match="capacity"):
        managym.encode_observations_into(roots, buffers, 1)
    narrow = teacher.space.encoder.allocate(len(roots))
    narrow["actions"] = narrow["actions"][..., :-1].copy()
    with pytest.raises(ValueError):
        managym.encode_observations_into(roots, narrow, len(roots))
    frozen = teacher.space.encoder.allocate(len(roots))
    frozen["agent_cards"].flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        managym.encode_observations_into(roots, frozen, len(roots))

    native = LocalUpdateTeacher(
        teacher.likelihood.checkpoint,
        teacher.likelihood.checkpoint_sha256,
        teacher.config,
    )
    for expected, actual in zip(
        teacher.predict_batch(roots), native.predict_batch(roots), strict=True
    ):
        np.testing.assert_array_equal(expected[0], actual[0])
        assert expected[1] == actual[1]


def test_compact_registration_declares_no_branch_audit() -> None:
    identity = "0" * 64

    def register(config: dict[str, object], branch_audit: bool) -> PlayerRegistration:
        return PlayerRegistration(
            role="challenger",
            compute_class_id="bounded-fixture",
            information_boundary="acting-viewer-history-only-v1",
            world="w4",
            content_suite="tiny-fixture",
            observation_abi_sha256=identity,
            action_abi_sha256=identity,
            matchup_sha256=identity,
            player_seed_derivation_id="fixture-seed",
            player_id="local-fixture",
            display_name="Local fixture",
            runner_kind="checkpoint",
            checkpoint_sha256=identity,
            checkpoint_bytes=1,
            parameter_count=1,
            training_seed=15,
            artifact_id="untrained-fixture-15",
            evidence_class="fixture",
            player_spec={
                "kind": "local_update",
                "config": config,
                "implementation_source_sha256": identity,
            },
            search_call_seed_derivation_id="local-policy-seed-times-1000003-plus-call-mod-2pow63/v1",
            search_semantics=SearchSemantics(
                branch_audit=branch_audit,
                root_prior="two-kl-local-update/v1",
                leaf_evaluator="frozen-viewer-policy-signed-value/v1",
            ),
        )

    full = LocalSearchConfig(rollout_evidence="full").model_dump()
    historical = {k: v for k, v in full.items() if k != "rollout_evidence"}
    register(historical, True)
    register(full, True)
    compact = LocalSearchConfig().model_dump()
    register(compact, False)
    with pytest.raises(ValueError, match="search and seed semantics"):
        register(compact, True)
    with pytest.raises(ValueError, match="search and seed semantics"):
        register(historical, False)
