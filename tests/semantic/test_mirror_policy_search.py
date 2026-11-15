"""Independent view-only policies search native games without training imports."""

from dataclasses import replace
import random

import numpy as np
import pytest

import managym
from managym import mirror_search
from managym.mirror import regularized_update
from managym.mirror_search import (
    SearchError,
    SearchSpec,
    mirror_decision,
    searched_descriptor,
)
from managym.policy import (
    PolicyDecision,
    PolicyDescriptor,
    PolicyEvaluation,
    PolicyRunner,
    PolicyView,
    ValueEstimate,
)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def perf_counter(self) -> float:
        return self.now


class IndependentSession:
    def __init__(
        self,
        *,
        seed: int = 0,
        children: list["IndependentSession"] | None = None,
        invalid: str | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.children = [] if children is None else children
        self.invalid = invalid
        self.clock = clock
        self.evaluations: list[PolicyView] = []
        self.closed = False

    def observe(self, view: PolicyView) -> None:
        pass

    def choose(self, view: PolicyView) -> PolicyDecision:
        raise AssertionError("Search must not choose with the live session")

    def evaluate(self, view: PolicyView) -> PolicyEvaluation:
        assert not self.closed
        assert view.seat == int(view.observation.agent.player_index)
        assert not hasattr(view, "env")
        self.evaluations.append(view)
        self.rng.random()
        if self.clock:
            self.clock.now = 31
        count = len(view.choices)
        probabilities = [1 / count] * count
        if count > 2:
            probabilities[1] = 0
            probabilities[0] += 1 / count
        if self.invalid == "alignment":
            probabilities.append(0)
        if self.invalid == "sum":
            probabilities = [0] * count
        value = ValueEstimate(
            0.25,
            1 - view.seat if self.invalid == "perspective" else view.seat,
            "logit" if self.invalid == "semantic" else "signed_outcome",
        )
        return PolicyEvaluation(tuple(probabilities), value)

    def fork(self, *, seed: int) -> "IndependentSession":
        if self.invalid == "fork":
            return self
        child = IndependentSession(
            seed=seed, children=self.children, invalid=self.invalid, clock=self.clock
        )
        self.children.append(child)
        return child

    def close(self) -> None:
        assert not self.closed
        self.closed = True


def _game(*, closed: bool = False) -> tuple[managym.Env, managym.Observation]:
    engine = managym.Env(seed=42)
    observation, _ = engine.reset(
        [
            managym.PlayerConfig(
                "a",
                {"Mountain": 12, "Gray Ogre": 12},
                disclosure="closed" if closed else "open",
            ),
            managym.PlayerConfig(
                "b",
                {"Forest": 12, "Llanowar Elves": 12},
                disclosure="closed" if closed else "open",
            ),
        ]
    )
    return engine, observation


def _runner(session: IndependentSession, seat: int) -> PolicyRunner:
    return PolicyRunner(
        session,
        PolicyDescriptor("Independent", "test", "fixed-runner", supports_search=True),
        seat,
    )


def test_independent_mirror_search_preserves_live_game_rng_and_zero_support() -> None:
    engine, observation = _game()
    live = IndependentSession()
    runner = _runner(live, int(engine.current_agent_index()))
    original = engine.state_digest()
    rng = live.rng.getstate()
    spec = SearchSpec(worlds=2, depth=2)
    decision = mirror_decision(runner, engine, observation, spec, seed=7)
    assert engine.state_digest() == original
    assert live.rng.getstate() == rng
    assert live.evaluations == [] and not live.closed
    assert all(child.closed for child in live.children)
    assert decision.search is not None and decision.belief is not None
    assert decision.search.worlds == decision.belief.sampled_worlds == 2
    assert decision.search.rollouts == 2 * sum(p > 0 for p in decision.search.prior)
    assert decision.belief.model == "physical-compatible-hand/v1"
    prior = np.array(decision.search.prior)
    np.testing.assert_allclose(
        decision.probabilities,
        regularized_update(
            prior,
            np.array(decision.search_scores, dtype=np.float64),
            np.full(len(prior), 1 / len(prior)),
            alpha=spec.alpha,
            beta=spec.beta,
        ),
    )
    assert np.array_equal(np.array(decision.probabilities) > 0, prior > 0)
    assert decision.probabilities[decision.action] > 0
    repeated = mirror_decision(runner, engine, observation, spec, seed=7)
    assert repeated.action == decision.action
    assert repeated.search is not None
    assert repeated.search.receipt_sha256 == decision.search.receipt_sha256


def test_search_leaf_values_follow_the_rollout_actor_perspective() -> None:
    engine, observation = _game()
    viewer = int(engine.current_agent_index())
    live = IndependentSession()
    decision = mirror_decision(
        _runner(live, viewer),
        engine,
        observation,
        SearchSpec(worlds=1, depth=1),
        seed=19,
    )
    assert decision.search is not None
    root = live.children[0]
    assert len(root.evaluations) == 1
    leaves = iter(live.children[1:])
    for prior, score in zip(decision.search.prior, decision.search_scores, strict=True):
        if prior == 0:
            assert score is None
            continue
        leaf = next(leaves)
        assert len(leaf.evaluations) == 1
        actor = leaf.evaluations[0].seat
        assert score == pytest.approx(0.25 if actor == viewer else -0.25)


def test_search_is_independent_of_the_actual_hidden_deal() -> None:
    engine, observation = _game()
    seat = int(engine.current_agent_index())
    alternative = engine.clone_env()
    alternative.determinize(seed=1007, perspective=seat)
    assert engine.semantic_viewer_state_hash(
        seat
    ) == alternative.semantic_viewer_state_hash(seat)
    spec = SearchSpec(worlds=2, depth=2)
    first = mirror_decision(
        _runner(IndependentSession(), seat), engine, observation, spec, seed=18
    )
    second = mirror_decision(
        _runner(IndependentSession(), seat),
        alternative,
        observation,
        spec,
        seed=18,
    )
    assert first.search is not None and second.search is not None
    assert first.search.receipt_sha256 == second.search.receipt_sha256
    assert first.probabilities == second.probabilities


def test_search_evaluates_the_live_root_projection_with_recent_events() -> None:
    engine, observation = _game()
    action = next(
        index
        for index, choice in enumerate(observation.action_space.actions)
        if int(choice.action_type) == int(managym.ActionEnum.PRIORITY_PLAY_LAND)
    )
    observation, *_ = engine.step(action)
    assert observation.recent_events
    assert not engine.observation_for_player(
        int(engine.current_agent_index())
    ).recent_events
    session = IndependentSession()
    mirror_decision(
        _runner(session, int(engine.current_agent_index())),
        engine,
        observation,
        SearchSpec(worlds=1, depth=1),
        seed=8,
    )
    received = session.children[0].evaluations[0].observation
    assert received.toJSON() == observation.toJSON()


def test_runner_retains_search_evidence_and_complete_composition_identity() -> None:
    engine, observation = _game()
    runner = _runner(IndependentSession(), int(engine.current_agent_index()))
    base = runner.descriptor
    spec = SearchSpec(worlds=1, depth=1)
    runner.configure_search(spec, seed=7)
    assert runner.descriptor.id != base.id
    assert runner.descriptor.config["base_policy"] == base.id
    assert runner.descriptor.base_policy == base
    first_id = runner.descriptor.id
    runner.configure_search(SearchSpec(worlds=2, depth=1), seed=7)
    assert runner.descriptor.id != first_id
    original = engine.state_digest()
    action = runner(engine, observation)
    assert runner.last_decision is not None and runner.last_decision.search is not None
    assert action == runner.last_decision.action
    assert runner.last_decision.search.worlds == 2
    assert engine.state_digest() == original
    with pytest.raises(ValueError, match="cannot change"):
        runner.configure_search(None, seed=7)


def test_search_identity_binds_execution_and_keeps_display_metadata_separate() -> None:
    base = PolicyDescriptor(
        "First label", "independent", "test-runner", supports_search=True
    )
    spec = SearchSpec(worlds=1, depth=1)
    original = searched_descriptor(base, spec)
    renamed = searched_descriptor(
        replace(base, name="Better label", group="Training run"), spec
    )
    assert original.id == renamed.id
    assert (
        original.id
        != searched_descriptor(replace(base, runner="changed-runner"), spec).id
    )


def test_closed_decklists_refuse_before_policy_evaluation() -> None:
    engine, observation = _game(closed=True)
    live = IndependentSession()
    original = engine.state_digest()
    with pytest.raises(SearchError, match="decklist closed") as failure:
        mirror_decision(
            _runner(live, int(engine.current_agent_index())),
            engine,
            observation,
            SearchSpec(worlds=1, depth=1),
            seed=0,
        )
    assert failure.value.reason == "search_world_unavailable"
    assert live.children == []
    assert engine.state_digest() == original


@pytest.mark.parametrize(
    "invalid", ["alignment", "sum", "perspective", "semantic", "fork"]
)
def test_invalid_search_capabilities_commit_nothing_and_release_forks(
    invalid: str,
) -> None:
    engine, observation = _game()
    original = engine.state_digest()
    live = IndependentSession(invalid=invalid)
    runner = _runner(live, int(engine.current_agent_index()))
    with pytest.raises(SearchError) as failure:
        mirror_decision(
            runner, engine, observation, SearchSpec(worlds=1, depth=1), seed=3
        )
    assert failure.value.reason == (
        "search_branch_isolation" if invalid == "fork" else "search_invalid_evaluation"
    )
    assert engine.state_digest() == original
    assert all(child.closed for child in live.children)
    assert not live.closed


@pytest.mark.parametrize("recurrent", [False, True])
def test_unsupported_or_recurrent_policy_cannot_search(recurrent: bool) -> None:
    engine, observation = _game()
    live = IndependentSession()
    runner = _runner(live, int(engine.current_agent_index()))
    runner = PolicyRunner(
        live,
        replace(runner.base_descriptor, supports_search=recurrent, recurrent=recurrent),
        runner.seat,
    )
    with pytest.raises(SearchError, match="current view|does not support"):
        mirror_decision(runner, engine, observation, SearchSpec(), seed=0)
    assert live.children == []


def test_search_timeout_preserves_pending_decision_and_allows_explicit_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, observation = _game()
    original = engine.state_digest()
    clock = Clock()
    monkeypatch.setattr(mirror_search, "time", clock)
    live = IndependentSession(clock=clock)
    runner = _runner(live, int(engine.current_agent_index()))
    with pytest.raises(SearchError) as failure:
        mirror_decision(
            runner, engine, observation, SearchSpec(worlds=1, depth=1), seed=4
        )
    assert failure.value.reason == "search_deadline_exceeded"
    assert engine.state_digest() == original
    assert all(child.closed for child in live.children)
    live.clock = None
    decision = mirror_decision(
        runner, engine, observation, SearchSpec(worlds=1, depth=1), seed=4
    )
    assert decision.probabilities[decision.action] > 0
    assert engine.state_digest() == original
