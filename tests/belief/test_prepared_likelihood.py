from __future__ import annotations

import gc
import json
import weakref

import numpy as np
import pytest
import torch

from manabot.belief.likelihood import FrozenPolicyLikelihood, _matching_offer_indexes
from manabot.belief.range import BeliefState
from manabot.sim.teacher1_evidence import _fresh_env
import managym
from managym.decision import SEMANTIC_DECISION_VERSION, DecisionFrame
from managym.possible_worlds import PossibleWorld, PossibleWorldSpace


def test_prepared_python_handle_preserves_scalar_parity_and_live_root() -> None:
    engine = _fresh_env(79)._engine
    space = PossibleWorldSpace.from_engine(engine, 0)
    scalar = [space.materialize(index, seed=100 + index) for index in (0, 1, 2)]
    with pytest.raises(Exception, match="space identity mismatch"):
        engine.prepare_possible_world_materializer(0, "0" * 64, 2)
    prepared = engine.prepare_possible_world_materializer(0, space.identity, 2)

    assert prepared.viewer == 0
    assert prepared.space_identity == space.identity
    assert prepared.support_size == space.support_size
    assert prepared.construction_count == 1

    first = prepared.materialize_indexes([2, 0], [102, 100])
    assert [branch.state_digest() for branch in first] == [
        scalar[2].state_digest(),
        scalar[0].state_digest(),
    ]

    with pytest.raises(Exception, match="preceding batch are still live"):
        prepared.materialize_indexes([1], [101])
    leased_clone = first[0].clone_env()
    del first
    gc.collect()
    with pytest.raises(Exception, match="preceding batch are still live"):
        prepared.materialize_indexes([1], [101])
    del leased_clone
    gc.collect()

    second = prepared.materialize_indexes([1], [101])
    assert second[0].state_digest() == scalar[1].state_digest()

    root_before = engine.state_digest()
    second[0].step(0)
    assert engine.state_digest() == root_before
    del second
    gc.collect()

    with pytest.raises(Exception, match="must not be empty"):
        prepared.materialize_indexes([], [])
    with pytest.raises(Exception, match="equal length"):
        prepared.materialize_indexes([0], [])
    with pytest.raises(Exception, match="exceeds maximum"):
        prepared.materialize_indexes([0, 1, 2], [1, 2, 3])
    with pytest.raises(Exception, match="outside support size"):
        prepared.materialize_indexes([space.support_size], [1])

    engine.step(0)
    with pytest.raises(Exception, match="source identity is stale"):
        prepared.materialize_indexes([0], [100])
    diagnostics = json.loads(prepared.diagnostics_json())
    assert diagnostics == {
        "space_constructions": 1,
        "batch_calls": 7,
        "requested_rows": 3,
        "materialized_rows": 3,
        "rejected_calls": 7,
        "maximum_returned_batch": 2,
        "current_live_prepared_branches": 0,
        "maximum_live_prepared_branches": 3,
    }


class _FakeHypothesis:
    def __init__(self, row: int) -> None:
        self.row = row

    def current_agent_index(self) -> int:
        return 1

    def observation_for_player(self, player: int) -> int:
        assert player == 1
        return self.row

    def semantic_decision_frame_json(self) -> str:
        return json.dumps(
            {
                "schema_version": SEMANTIC_DECISION_VERSION,
                "revision": 7,
                "actor": 1,
                "fingerprint": "prepared-test",
                "offers": [
                    {"id": 0, "public_commitment": {"kind": "play_land", "card": "A"}},
                    {"id": 1, "public_commitment": {"kind": "pass_priority"}},
                ],
                "object_candidates": [],
            }
        )


class _FakePrepared:
    viewer = 0
    space_identity = "prepared-space"
    support_size = 5
    max_batch_size = 2
    construction_count = 1

    def __init__(self) -> None:
        self.calls: list[tuple[list[int], list[int], bool]] = []
        self._previous: list[weakref.ReferenceType[_FakeHypothesis]] = []

    def materialize_indexes(
        self, indexes: list[int], seeds: list[int], refresh: bool
    ) -> list[_FakeHypothesis]:
        gc.collect()
        assert all(reference() is None for reference in self._previous)
        assert len(indexes) <= self.max_batch_size
        branches = [_FakeHypothesis(index) for index in indexes]
        self._previous = [weakref.ref(branch) for branch in branches]
        self.calls.append((list(indexes), list(seeds), refresh))
        return branches

    def diagnostics_json(self) -> str:
        materialized = sum(len(indexes) for indexes, _, _ in self.calls)
        return json.dumps(
            {
                "space_constructions": 1,
                "batch_calls": len(self.calls),
                "requested_rows": materialized,
                "materialized_rows": materialized,
                "rejected_calls": 0,
                "maximum_returned_batch": max(
                    (len(indexes) for indexes, _, _ in self.calls), default=0
                ),
            }
        )


class _FakeRoot:
    def __init__(self, prepared: _FakePrepared) -> None:
        self.prepared = prepared
        self.prepare_calls: list[tuple[int, str, int]] = []

    def prepare_possible_world_materializer(
        self, viewer: int, identity: str, max_batch_size: int
    ) -> _FakePrepared:
        self.prepare_calls.append((viewer, identity, max_batch_size))
        return self.prepared


class _FakeObservationSpace:
    def __init__(self) -> None:
        self.encoded: list[weakref.ReferenceType[np.ndarray]] = []

    def encode(self, row: int) -> dict[str, np.ndarray]:
        encoded = np.asarray([row], dtype=np.float32)
        self.encoded.append(weakref.ref(encoded))
        return {"row": encoded}

    def encoded_batches_released(self) -> bool:
        gc.collect()
        return all(reference() is None for reference in self.encoded)


class _FakeAgent:
    def forward(
        self, tensors: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row = tensors["row"].squeeze(-1)
        logits = torch.stack((torch.zeros_like(row), row + 1.0), dim=-1)
        return logits, torch.zeros_like(row)


def test_frozen_likelihood_uses_one_prepared_handle_and_releases_each_batch() -> None:
    prepared = _FakePrepared()
    root = _FakeRoot(prepared)
    space = PossibleWorldSpace(
        identity="prepared-space",
        viewer=0,
        opponent=1,
        source_revision=7,
        source_viewer_state_hash="viewer-hash",
        hand_size=1,
        pool=(("A", 5),),
        total_weight=5,
        worlds=tuple(PossibleWorld(index, (("A", 1),), 1) for index in range(5)),
        _engine=root,
    )
    belief = BeliefState.compatible_prior(space)
    progress = []
    likelihood = object.__new__(FrozenPolicyLikelihood)
    likelihood.batch_size = 2
    likelihood.device = torch.device("cpu")
    likelihood.counterfactual_seed = 907
    likelihood.agent = _FakeAgent()
    observation_space = _FakeObservationSpace()
    likelihood.obs_space = observation_space

    def observe(event) -> None:
        assert observation_space.encoded_batches_released()
        progress.append(event)

    likelihood.batch_observer = observe

    result = likelihood.evaluate(
        root,
        viewer=0,
        commitment={"kind": "pass_priority"},
        belief=belief,
    )

    assert root.prepare_calls == [(0, "prepared-space", 2)]
    assert prepared.calls == [
        ([0, 1], [907, 907], True),
        ([2, 3], [907, 907], True),
        ([4], [907], True),
    ]
    assert [event.completed_rows for event in progress] == [2, 4, 5]
    assert [event.batch_size for event in progress] == [2, 2, 1]
    assert result.batches == 3
    assert result.space_constructions == 1
    assert result.max_batch_size == 2
    assert np.all(result.legal_action_counts == 2)
    assert np.all(result.matching_action_counts == 1)
    assert np.all((result.likelihoods > 0.5) & (result.likelihoods < 1.0))


def test_batch_observer_aborts_before_the_next_provider_call() -> None:
    prepared = _FakePrepared()
    root = _FakeRoot(prepared)
    space = PossibleWorldSpace(
        identity="prepared-space",
        viewer=0,
        opponent=1,
        source_revision=7,
        source_viewer_state_hash="viewer-hash",
        hand_size=1,
        pool=(("A", 5),),
        total_weight=5,
        worlds=tuple(PossibleWorld(index, (("A", 1),), 1) for index in range(5)),
        _engine=root,
    )
    likelihood = object.__new__(FrozenPolicyLikelihood)
    likelihood.batch_size = 2
    likelihood.device = torch.device("cpu")
    likelihood.counterfactual_seed = 907
    likelihood.agent = _FakeAgent()
    likelihood.obs_space = _FakeObservationSpace()

    def stop_after_first_batch(progress) -> None:
        assert progress.completed_rows == 2
        raise RuntimeError("resource cap exceeded")

    likelihood.batch_observer = stop_after_first_batch
    with pytest.raises(RuntimeError, match="resource cap exceeded"):
        likelihood.evaluate(
            root,
            viewer=0,
            commitment={"kind": "pass_priority"},
            belief=BeliefState.compatible_prior(space),
        )

    assert prepared.calls == [([0, 1], [907, 907], True)]


def test_real_prepared_likelihood_matches_scalar_reference(monkeypatch) -> None:
    from manabot.env import ObservationSpace
    from manabot.infra.hypers import AgentHypers
    from manabot.model.agent import Agent

    root = managym.Env(seed=157)
    deck = {"Mountain": 4, "Raging Goblin": 4}
    root.reset([managym.PlayerConfig("a", deck), managym.PlayerConfig("b", deck)])
    viewer = 1 - root.current_agent_index()
    space = PossibleWorldSpace.from_engine(root, viewer)
    belief = BeliefState.compatible_prior(space)
    assert space.support_size > 1
    likelihood = object.__new__(FrozenPolicyLikelihood)
    likelihood.batch_size = 1
    likelihood.device = torch.device("cpu")
    likelihood.counterfactual_seed = 907
    likelihood.obs_space = ObservationSpace()
    torch.manual_seed(19)
    likelihood.agent = Agent(likelihood.obs_space, AgentHypers()).eval()
    progress = []
    likelihood.batch_observer = progress.append
    commitment = {"kind": "pass_priority"}
    expected, legal, matching = [], [], []
    for world in space.worlds:
        branch = space.materialize(
            world.index, seed=907, refresh_opponent_commitment=True
        )
        frame = DecisionFrame.from_json(branch.semantic_decision_frame_json())
        indexes, count = _matching_offer_indexes(frame, commitment)
        encoded = likelihood.obs_space.encode(branch.observation_for_player(1 - viewer))
        tensors = {key: torch.from_numpy(value[None]) for key, value in encoded.items()}
        with torch.inference_mode():
            logits, _ = likelihood.agent.forward(tensors)
            expected.append(float(torch.softmax(logits, dim=-1)[0, indexes].sum()))
        legal.append(count)
        matching.append(len(indexes))
    root_digest = root.state_digest()

    def forbid_projection(*args, **kwargs):
        raise AssertionError("evaluate must not build a second Python world projection")

    monkeypatch.setattr(PossibleWorldSpace, "from_engine", forbid_projection)
    result = likelihood.evaluate(
        root, viewer=viewer, commitment=commitment, belief=belief
    )
    np.testing.assert_allclose(result.likelihoods, expected, rtol=1e-7)
    np.testing.assert_array_equal(result.legal_action_counts, legal)
    np.testing.assert_array_equal(result.matching_action_counts, matching)
    assert root.state_digest() == root_digest
    assert result.batches == space.support_size
    assert all(row.prepared_diagnostics["space_constructions"] == 1 for row in progress)
    assert all(
        row.prepared_diagnostics["current_live_prepared_branches"] == 0
        for row in progress
    )


def test_handle_keeps_its_root_alive_after_python_owner_is_released() -> None:
    engine = _fresh_env(79)._engine
    space = PossibleWorldSpace.from_engine(engine, 0)
    expected = space.materialize(0, seed=100).state_digest()
    prepared = engine.prepare_possible_world_materializer(0, space.identity, 1)
    del engine, space
    gc.collect()
    assert prepared.materialize_indexes([0], [100])[0].state_digest() == expected
