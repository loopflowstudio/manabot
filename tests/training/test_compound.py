"""Joint probability, native Command parity, and trained checkpoint acceptance."""

from dataclasses import replace
from itertools import product
import json
from pathlib import Path
from typing import Literal

import pytest
import torch

from manabot.env import Match, ObservationSpace
from manabot.infra.hypers import AgentSpec, MatchHypers, ObservationSpaceHypers
from manabot.model.agent import Agent
from manabot.model.compound import CompoundDecoder
from manabot.sim.compound import CompoundPolicy, sample_compound
from manabot.sim.flat_mc import load_checkpoint_agent, make_player
from manabot.sim.structured_policy import StructuredPolicyError, flatten_projection
from manabot.training.compound import (
    collect_game,
    episode_credit,
    optimize_games,
    replay_game,
)
from manabot.training.execution import execute_regime
from manabot.training.models import Learning, TrainingRegime
from manabot.verify.store import VerifyStore
import managym
from managym.choice import OfferProjection


def _projection(count: int, minimum: int, maximum: int) -> dict[str, object]:
    return {
        "schema_version": 7,
        "factorization_version": 1,
        "revision": 0,
        "actor": 0,
        "kind": "declare_attackers",
        "offers": [
            {
                "id": 4,
                "actor": 0,
                "details": {
                    key: None
                    for key in (
                        "subject",
                        "target",
                        "outside_candidate",
                        "program",
                        "requirement",
                        "attack",
                        "mana",
                    )
                },
                "verb": "declare_attackers",
                "label": "Attack",
                "source": None,
                "help": None,
                "confirm_label": "Attack",
                "choices": [
                    {
                        "kind": "select",
                        "context": {"kind": "selection"},
                        "role": 1,
                        "label": "Attackers",
                        "min": minimum,
                        "max": maximum,
                        "distinct": True,
                        "ordered": False,
                        "candidates": {
                            "id": 0,
                            "depends_on": [],
                            "initial": [
                                {
                                    "id": index,
                                    "label": f"Creature {index}",
                                    "help": None,
                                    "preview": None,
                                    "value": {
                                        "kind": "subject",
                                        "subject": {
                                            "kind": "object",
                                            "id": {"entity": index, "incarnation": 0},
                                        },
                                    },
                                }
                                for index in range(count)
                            ],
                        },
                    }
                ],
            }
        ],
    }


def _agent(
    *, wide: bool = False, features: Literal["labels", "objects"] = "labels"
) -> Agent:
    torch.set_num_threads(1)
    return Agent(
        ObservationSpace(
            ObservationSpaceHypers(
                max_cards_per_player=100 if wide else 60,
                max_permanents_per_player=80 if wide else 40,
                max_actions=128 if wide else 64,
            )
        ),
        AgentSpec(
            compound_decisions=True,
            compound_features=features,
            hidden_dim=16,
            num_attention_heads=2,
        ),
    )


def test_normalized_joint_and_score_gradient() -> None:
    torch.manual_seed(1)
    decoder = CompoundDecoder(8).double()
    root = torch.randn(8, dtype=torch.double, requires_grad=True)
    batch = flatten_projection(_projection(3, 1, 2))
    outputs = [
        decoder(root, batch, tokens=(0, *bits))
        for bits in product((0, 1), repeat=3)
        if 1 <= sum(bits) <= 2
    ]
    probabilities = torch.stack([output.log_prob.exp() for output in outputs])
    torch.testing.assert_close(
        probabilities.sum(), torch.tensor(1.0, dtype=torch.double)
    )
    for output in outputs:
        torch.testing.assert_close(output.log_prob, output.log_probs.sum())
        for conditional in output.probabilities:
            torch.testing.assert_close(conditional.sum(), conditional.new_tensor(1))
    # Exact enumeration verifies the score-function gradient, including GRU prefix
    # dependence, against the direct expected-return gradient.
    rewards = torch.arange(len(outputs), dtype=torch.double)
    direct = torch.autograd.grad(
        (probabilities * rewards).sum(), root, retain_graph=True
    )[0]
    score = torch.autograd.grad(
        sum(
            p.detach() * reward * output.log_prob
            for p, reward, output in zip(probabilities, rewards, outputs, strict=True)
        ),
        root,
    )[0]
    torch.testing.assert_close(direct, score)
    assert direct.abs().sum() > 0
    assert torch.autograd.gradcheck(
        lambda x: decoder(x, batch, tokens=(0, 1, 0, 0)).log_prob, (root,)
    )


def test_prefix_changes_conditionals_and_rejects_incomplete_illegal_tapes() -> None:
    decoder = CompoundDecoder(8)
    batch = flatten_projection(_projection(3, 0, 3))
    context = torch.zeros(8)
    first = decoder(context, batch, tokens=(0, 0, 0, 0))
    second = decoder(context, batch, tokens=(0, 1, 0, 0))
    assert not torch.equal(first.probabilities[2], second.probabilities[2])
    with pytest.raises(StructuredPolicyError, match="interrupted"):
        decoder(context, batch, tokens=(0, 1))
    with pytest.raises(StructuredPolicyError, match="trailing"):
        decoder(context, batch, tokens=(0, 1, 0, 0, 0))
    with pytest.raises(StructuredPolicyError, match="illegal"):
        decoder(context, flatten_projection(_projection(1, 1, 1)), tokens=(0, 0))
    output = decoder(context, flatten_projection(_projection(65, 0, 65)))
    assert len(output.tokens) == 66
    assert torch.isfinite(output.log_prob)


def _root(
    kind: str, count: int, *, blockers: int = 1
) -> tuple[managym.Env, managym.Observation]:
    env = managym.Env(seed=81, skip_trivial=False)
    obs, _ = env.reset(
        [
            managym.PlayerConfig(
                "a",
                {"Lightning Bolt": 4, "Mountain": 36}
                if kind == "cast"
                else {"Gray Ogre": count + 8, "Mountain": 4},
            ),
            managym.PlayerConfig(
                "b", {"Gray Ogre": max(36, blockers + 8), "Mountain": 4}
            ),
        ]
    )
    env.scenario_clear_hand(0)
    env.scenario_clear_hand(1)
    if kind == "cast":
        env.scenario_force_card_in_hand(0, "Lightning Bolt")
        env.scenario_force_battlefield(0, "Mountain")
    for _ in range(count):
        env.scenario_force_battlefield(
            1 if kind == "cast" else 0, "Gray Ogre", ready=True
        )
    if kind == "attack":
        for _ in range(blockers):
            env.scenario_force_battlefield(1, "Gray Ogre")
    obs = env.scenario_refresh()
    if kind == "attack":
        for _ in range(30):
            if (
                obs.action_space.action_space_type
                == managym.ActionSpaceEnum.DECLARE_ATTACKER
            ):
                break
            index = next(
                i
                for i, action in enumerate(obs.action_space.actions)
                if action.action_type == managym.ActionEnum.PRIORITY_PASS_PRIORITY
            )
            obs, _, _, _, _ = env.step(index)
        assert (
            obs.action_space.action_space_type
            == managym.ActionSpaceEnum.DECLARE_ATTACKER
        )
    return env, obs


@pytest.mark.parametrize("kind,count", [("cast", 33), ("attack", 65)])
def test_native_wide_compound_commands_exact_atomic_parity(
    kind: str, count: int
) -> None:
    env, obs = _root(kind, count)
    offers = env.compound_offers()
    batch = flatten_projection(json.loads(offers.projection_json()))
    agent = _agent(wide=True)
    tokens = (
        next(
            i
            for i, offer in enumerate(batch.offers)
            if offer["verb"] == ("cast" if kind == "cast" else "declare_attackers")
        ),
    )
    candidates = 35 if kind == "cast" else count
    tokens += tuple(
        int(index == candidates - 1) if kind == "cast" else index % 2
        for index in range(candidates)
    )
    raw = {
        key: torch.as_tensor(value).unsqueeze(0)
        for key, value in agent.observation_space.encode(obs).items()
    }
    output = agent.compound(raw, batch, tokens=tokens)
    before = env.state_digest()
    tape = json.loads(env.compound_commands_json(offers, output.submission.to_json()))
    assert env.state_digest() == before
    atomic = env.clone_env()
    atomic.step_structured(offers, output.submission.to_json())
    for command in tape:
        env.execute_semantic_command_json(json.dumps(command))
    assert env.state_digest() == atomic.state_digest()
    assert len(tape) == (2 if kind == "cast" else count)
    with pytest.raises(managym.AgentError, match="no longer"):
        env.compound_commands_json(offers, output.submission.to_json())


def _check() -> None:
    return None


def _match() -> Match:
    return Match(
        MatchHypers(
            hero_deck={"Mountain": 6, "Gray Ogre": 6},
            villain_deck={"Mountain": 6, "Gray Ogre": 6},
        )
    )


def test_terminal_credit_and_replay_and_interrupted_attempt(tmp_path: Path) -> None:
    agent = _agent()
    game = collect_game(
        agent,
        _match(),
        85,
        torch.Generator().manual_seed(2),
        tmp_path / "game.jsonl",
        max_commands=600,
        check=_check,
    )
    assert replay_game(tmp_path / "game.jsonl") == game.microchoices
    for grouped in (False, True):
        outcome = episode_credit(
            game, grouped=grouped, estimator="outcome", learning=Learning(gamma=1)
        )
        boot = episode_credit(
            game,
            grouped=grouped,
            estimator="bootstrapped",
            learning=Learning(gamma=1, policy_lambda=0, value_lambda=0),
        )
        for row, credit in zip(game.decisions, outcome, strict=True):
            reward = (
                0 if game.winner is None else (1 if game.winner == row.actor else -1)
            )
            torch.testing.assert_close(
                credit.returns, torch.full_like(credit.returns, reward)
            )
            assert not credit.returns.requires_grad
        assert any(
            not torch.allclose(a.returns, b.returns)
            for a, b in zip(outcome, boot, strict=True)
        )
        for actor in (0, 1):
            last = max(i for i, row in enumerate(game.decisions) if row.actor == actor)
            torch.testing.assert_close(
                outcome[last].returns[-1], boot[last].returns[-1]
            )
    with pytest.raises(RuntimeError, match="cap"):
        collect_game(
            agent,
            _match(),
            85,
            torch.Generator().manual_seed(2),
            tmp_path / "partial.jsonl",
            max_commands=1,
            check=_check,
        )
    with pytest.raises(ValueError, match="incomplete"):
        replay_game(tmp_path / "partial.jsonl")


@pytest.mark.parametrize("features", ["labels", "objects"])
def test_regime_trains_reloads_and_ordinary_player_uses_compound(
    tmp_path: Path,
    features: Literal["labels", "objects"],
) -> None:
    torch.set_num_threads(1)
    regime = TrainingRegime.model_validate(
        {
            "id": "compound-test",
            "world": managym.WORLD_VERSION,
            "match": _match().hypers.model_dump(),
            "agent": {
                "compound_decisions": True,
                "compound_features": features,
                "hidden_dim": 16,
                "num_attention_heads": 2,
            },
            "stages": [
                {
                    "id": "fit",
                    "operation": "train_compound",
                    "games_per_update": 1,
                    "learning": {"epochs": 1, "minibatches": 4, "retained_fraction": 1},
                }
            ],
        }
    )
    with VerifyStore(tmp_path / "store.sqlite") as store:
        run = execute_regime(regime, 91, tmp_path / "run", store)
    assert run.status == "completed"
    assert run.stages[0].optimizer_exposures > 0
    path = run.stages[0].artifacts["raw"]["path"]
    loaded, _ = load_checkpoint_agent(path)
    torch.manual_seed(91)
    original = Agent(loaded.observation_space, regime.agent)
    assert not torch.equal(
        original.compound_decoder.include_score.weight,
        loaded.compound_decoder.include_score.weight,
    )
    player, _ = make_player({"kind": "checkpoint", "path": path}, seed=1)
    assert player.compound is not None
    from etude.villain import CheckpointVillain

    villain = CheckpointVillain(path, player_configs=_match().to_rust())
    env = managym.Env(seed=9)
    raw, _ = env.reset(_match().to_rust())
    action = villain(env, raw)
    assert 0 <= action < len(raw.action_space.actions)
    with pytest.raises(ValueError, match="authoritative offers"):
        loaded({})


@pytest.mark.parametrize("kind", ["attack", "blockers", "payment"])
def test_serving_interrupt_discards_suffix(kind: str) -> None:
    if kind == "blockers":
        env, raw = _block_root(2, 6)
    elif kind == "payment":
        env, raw = _waterbend_root(6)
    else:
        env, raw = _root("attack", 6)
    player = CompoundPolicy(_agent())
    player.act(env, raw)
    assert player.pending
    with pytest.raises(ValueError, match="interrupted"):
        player.act(env, raw)  # caller did not execute the first command
    assert not player.pending


def test_payment_is_a_new_observation_not_a_cached_suffix() -> None:
    env = managym.Env(seed=17, skip_trivial=False)
    raw, _ = env.reset(
        [
            managym.PlayerConfig("a", {"Mountain": 30, "Firebending Lesson": 10}),
            managym.PlayerConfig("b", {"Mountain": 30, "Gray Ogre": 10}),
        ]
    )
    env.scenario_clear_hand(0)
    env.scenario_force_card_in_hand(0, "Firebending Lesson")
    env.scenario_force_battlefield(1, "Gray Ogre")
    for _ in range(6):
        env.scenario_force_battlefield(0, "Mountain")
    raw = env.scenario_refresh()
    for _ in range(30):
        projection = json.loads(env.compound_offers().projection_json())
        casts = [offer for offer in projection["offers"] if offer["verb"] == "cast"]
        if casts:
            break
        index = next(
            i
            for i, action in enumerate(raw.action_space.actions)
            if action.action_type == managym.ActionEnum.PRIORITY_PASS_PRIORITY
        )
        raw, _, _, _, _ = env.step(index)
    else:
        pytest.fail("fixture did not reach sorcery cast")
    assert not casts[0]["choices"]  # kicker cannot be crossed by target grouping
    offers = env.compound_offers()
    tape = json.loads(
        env.compound_commands_json(
            offers, json.dumps({"offer_id": casts[0]["id"], "answers": []})
        )
    )
    assert len(tape) == 1
    env.execute_semantic_command_json(json.dumps(tape[0]))
    assert json.loads(env.compound_offers().projection_json())["kind"] == "pay_or_not"
    raw = env.observation_for_player(0)
    with torch.no_grad():
        payment = sample_compound(_agent(), env, raw)
    assert len(payment.commands) == 1
    env.execute_semantic_command_json(payment.commands[0].to_json())


def _block_root(
    attackers: int, blockers: int
) -> tuple[managym.Env, managym.Observation]:
    env, raw = _root("attack", attackers, blockers=blockers)
    offers = env.compound_offers()
    projection = json.loads(offers.projection_json())
    offer = projection["offers"][0]
    choice = offer["choices"][0]
    submission = {
        "offer_id": offer["id"],
        "answers": [
            {
                "kind": "candidates",
                "role": choice["role"],
                "candidates": [row["id"] for row in choice["candidates"]["initial"]],
            }
        ],
    }
    for command in json.loads(
        env.compound_commands_json(offers, json.dumps(submission))
    ):
        env.execute_semantic_command_json(json.dumps(command))
    for _ in range(30):
        actor = env.current_agent_index()
        assert actor is not None
        raw = env.observation_for_player(actor)
        if (
            raw.action_space.action_space_type
            == managym.ActionSpaceEnum.DECLARE_BLOCKER
        ):
            break
        index = next(
            i
            for i, action in enumerate(raw.action_space.actions)
            if action.action_type == managym.ActionEnum.PRIORITY_PASS_PRIORITY
        )
        env.step(index)
    else:
        pytest.fail("fixture did not reach blockers")
    return env, raw


def _assert_authoritative_replay(
    env: managym.Env, submission: str
) -> list[dict[str, object]]:
    """Compare canonical replay with native submission and repeated receipts."""
    offers = env.compound_offers()
    before = env.state_digest()
    tape = json.loads(env.compound_commands_json(offers, submission))
    assert env.state_digest() == before
    atomic = env.clone_env()
    atomic.step_structured(offers, submission)
    replay = env.clone_env()
    for command in tape:
        text = json.dumps(command)
        assert env.execute_semantic_command_json(
            text
        ) == replay.execute_semantic_command_json(text)
    assert env.state_digest() == atomic.state_digest() == replay.state_digest()
    return tape


def test_blocker_joint_roles_normalize_and_differentiate() -> None:
    env, _ = _block_root(2, 2)
    batch = flatten_projection(json.loads(env.compound_offers().projection_json()))
    assert len(batch.choices) == 2
    assert all(row.minimum == 0 and row.maximum == 1 for row in batch.choices)
    decoder = CompoundDecoder(8).double()
    context = torch.randn(8, dtype=torch.double, requires_grad=True)
    assignments = ((0, 0), (1, 0), (0, 1))
    outputs = [
        decoder(context, batch, tokens=(0, *first, *second))
        for first, second in product(assignments, repeat=2)
    ]
    probs = torch.stack([output.log_prob.exp() for output in outputs])
    torch.testing.assert_close(probs.sum(), probs.new_tensor(1))
    rewards = torch.arange(len(outputs), dtype=torch.double)
    direct = torch.autograd.grad((probs * rewards).sum(), context, retain_graph=True)[0]
    score = torch.autograd.grad(
        sum(
            p.detach() * reward * output.log_prob
            for p, reward, output in zip(probs, rewards, outputs, strict=True)
        ),
        context,
    )[0]
    torch.testing.assert_close(direct, score)
    assert direct.abs().sum() > 0
    assert torch.autograd.gradcheck(
        lambda x: decoder(x, batch, tokens=(0, 1, 0, 0, 1)).log_prob, (context,)
    )
    for output in outputs:
        _assert_authoritative_replay(env.clone_env(), output.submission.to_json())


def test_wide_blocker_declaration_replays_and_stops_at_priority() -> None:
    env, raw = _block_root(35, 65)
    with torch.no_grad():
        decision = sample_compound(_agent(wide=True), env, raw, deterministic=True)
    assert decision.offers.offers[0]["verb"] == "declare_blockers"
    assert len(decision.offers.choices) == 65
    assert decision.offers.max_candidate_count == 35
    assert len(decision.commands) == 65
    _assert_authoritative_replay(env, decision.output.submission.to_json())
    assert json.loads(env.compound_offers().projection_json())["kind"] == "priority"


def _waterbend_root(
    count: int, *, lands: int = 0, bonus_mana: bool = False
) -> tuple[managym.Env, managym.Observation]:
    env = managym.Env(seed=81, skip_trivial=False)
    env.reset(
        [
            managym.PlayerConfig(
                "a", {"Water Tribe Rallier": 80, "Forest": 20, "Badgermole Cub": 4}
            ),
            managym.PlayerConfig("b", {"Gray Ogre": 20, "Mountain": 20}),
        ]
    )
    env.scenario_clear_hand(0)
    env.scenario_clear_hand(1)
    for _ in range(count):
        env.scenario_force_battlefield(0, "Water Tribe Rallier", ready=True)
    for _ in range(lands):
        env.scenario_force_battlefield(0, "Forest", ready=True)
    if bonus_mana:
        env.scenario_force_battlefield(0, "Badgermole Cub", ready=True)
    raw = env.scenario_refresh()
    index = next(
        i
        for i, action in enumerate(raw.action_space.actions)
        if action.action_type == managym.ActionEnum.PRIORITY_ACTIVATE_ABILITY
    )
    raw, _, _, _, _ = env.step(index)
    assert json.loads(env.compound_offers().projection_json())["kind"] == "waterbend"
    return env, raw


@pytest.mark.parametrize("lands,taps", [(0, 5), (2, 3), (5, 0)])
def test_wide_payment_subset_replays_and_stops_before_reveal(
    lands: int, taps: int
) -> None:
    env, raw = _waterbend_root(65, lands=lands)
    agent = _agent(wide=True)
    batch = flatten_projection(json.loads(env.compound_offers().projection_json()))
    assert batch.offers[0]["verb"] == "pay_waterbend"
    assert batch.max_candidate_count == 65
    row = batch.choices[0]
    assert (row.minimum, row.maximum) == (max(0, 5 - lands), 5)
    observation = {
        key: torch.as_tensor(value).unsqueeze(0)
        for key, value in agent.observation_space.encode(raw).items()
    }
    output = agent.compound(
        observation, batch, tokens=(0, *(int(i < taps) for i in range(65)))
    )
    assert torch.isfinite(output.log_prob)
    tape = _assert_authoritative_replay(env, output.submission.to_json())
    assert len(tape) == taps + int(taps < 5)
    assert json.loads(env.compound_offers().projection_json())["kind"] == "priority"
    # The library look happens only after both players pass: it is never part
    # of the payment policy's sampled suffix.
    for _ in range(2):
        actor = env.current_agent_index()
        assert actor is not None
        raw = env.observation_for_player(actor)
        index = next(
            i
            for i, action in enumerate(raw.action_space.actions)
            if action.action_type == managym.ActionEnum.PRIORITY_PASS_PRIORITY
        )
        env.step(index)
    assert (
        json.loads(env.compound_offers().projection_json())["kind"] == "look_and_select"
    )


def test_triggered_mana_payment_remains_sequential() -> None:
    env, raw = _waterbend_root(5, bonus_mana=True)
    with torch.no_grad():
        decision = sample_compound(_agent(), env, raw)
    assert all(not offer["choices"] for offer in decision.offers.offers)
    assert len(decision.commands) == 1
    _assert_authoritative_replay(env, decision.output.submission.to_json())


@pytest.mark.parametrize("kind", ["priority", "blockers", "payment"])
@pytest.mark.parametrize("features", ["labels", "objects"])
def test_hidden_world_swap_cannot_change_root_policy(
    kind: str, features: Literal["labels", "objects"]
) -> None:
    if kind == "blockers":
        env, raw = _block_root(2, 2)
    elif kind == "payment":
        env, raw = _waterbend_root(6)
    else:
        env = managym.Env(seed=94)
        raw, _ = env.reset(_match().to_rust())
    viewer = env.current_agent_index()
    assert viewer is not None
    raw = env.observation_for_player(viewer)
    fork = env.clone_env()
    fork.determinize(981, perspective=viewer)
    other = fork.observation_for_player(viewer)
    assert raw.toJSON() == other.toJSON()
    agent = _agent(features=features)
    with torch.no_grad():
        original = sample_compound(agent, env, raw, deterministic=True)
        swapped = sample_compound(agent, fork, other, deterministic=True)
    assert original.output.tokens == swapped.output.tokens
    for left, right in zip(
        original.output.probabilities, swapped.output.probabilities, strict=True
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_legacy_recipe_serialization_does_not_gain_compound_identity() -> None:
    assert "compound_decisions" not in AgentSpec().model_dump()
    assert AgentSpec(compound_decisions=True).model_dump()["compound_decisions"]


def test_failed_regime_registers_incomplete_attempt_without_checkpoint(
    tmp_path: Path,
) -> None:
    regime = TrainingRegime.model_validate(
        {
            "id": "partial-compound",
            "world": managym.WORLD_VERSION,
            "match": _match().hypers.model_dump(),
            "agent": {
                "compound_decisions": True,
                "hidden_dim": 16,
                "num_attention_heads": 2,
            },
            "stages": [
                {
                    "id": "fit",
                    "operation": "train_compound",
                    "max_commands": 1,
                    "games_per_update": 1,
                }
            ],
        }
    )
    with VerifyStore(tmp_path / "store.sqlite") as store:
        with pytest.raises(RuntimeError, match="cap"):
            execute_regime(regime, 91, tmp_path / "run", store)
    receipt = json.loads((tmp_path / "run/run.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["selected_artifact"] is None
    stage = receipt["stages"][0]
    assert stage["diagnostics"][0]["attempted_games"] == 1
    assert stage["diagnostics"][0]["failed_games"] == 1
    assert stage["diagnostics"][0]["interrupted_microchoices"] == 1
    assert stage["rejected_artifacts"]["game-0"]["bytes"] > 0
    assert stage["collection_seconds"] > 0


def test_optionless_auto_resolution_has_separate_accounting(tmp_path: Path) -> None:
    agent = _agent()
    observed: dict[bool, tuple[int, int]] = {}
    for collapse in (False, True):
        path = tmp_path / f"collapse-{collapse}.jsonl"
        game = collect_game(
            agent,
            _match(),
            85,
            torch.Generator().manual_seed(2),
            path,
            max_commands=1000,
            check=_check,
            skip_trivial=collapse,
        )
        observed[collapse] = (game.microchoices, game.auto_resolved)
        assert replay_game(path) == game.microchoices
    assert observed[False][1] == 0
    assert observed[True][1] > 0
    # The RNG tape consumes forced prompts differently; these are accounting
    # checks, not a paired outcome or speed claim.


def test_terminal_game_without_choices_does_not_invent_optimizer_rows(
    tmp_path: Path,
) -> None:
    agent = _agent()
    match = Match(
        MatchHypers(hero_deck={"Craw Wurm": 7}, villain_deck={"Craw Wurm": 7})
    )
    game = collect_game(
        agent,
        match,
        95,
        torch.Generator().manual_seed(2),
        tmp_path / "forced.jsonl",
        max_commands=100,
        check=_check,
    )
    assert not game.decisions
    update = optimize_games(
        agent,
        torch.optim.Adam(agent.parameters()),
        [game],
        Learning(),
        grouped=True,
        estimator="outcome",
        progress=0,
        generator=torch.Generator().manual_seed(3),
        check=_check,
    )
    assert update.optimizer_exposures == 0
    assert not update.losses
    assert replay_game(tmp_path / "forced.jsonl") == 0


def test_native_prefix_support_rejects_stale_and_bad_order() -> None:
    env, raw = _root("attack", 3)
    offers = env.compound_offers()
    assert env.compound_prefix_support(offers, []) == [True]
    assert env.compound_prefix_support(offers, [0]) == [True, True]
    with pytest.raises(managym.AgentError, match="illegal compound token"):
        env.compound_prefix_support(offers, [0, 2])
    projection = json.loads(offers.projection_json())
    projection["factorization_version"] = 9
    with pytest.raises(ValueError, match="factorization"):
        managym.ChoiceSupport(json.dumps(projection))
    env.step(0)
    with pytest.raises(Exception, match="root changed"):
        env.compound_prefix_support(offers, [0])


def test_object_decoder_joins_same_name_blockers_and_rescores() -> None:
    env, raw = _block_root(3, 2)
    agent = _agent(features="objects")
    decision = sample_compound(agent, env, raw)
    roles = decision.offers.role_inputs
    assert len(roles) == 2 and roles[0].objects != roles[1].objects
    rescored = agent.compound(
        decision.observation, decision.offers, tape=decision.output
    )
    torch.testing.assert_close(
        decision.output.log_probs, rescored.log_probs, rtol=0, atol=0
    )
    (-rescored.log_prob + rescored.values.square().mean()).backward()
    assert all(
        torch.isfinite(p.grad).all() for p in agent.parameters() if p.grad is not None
    )
    assert agent.compound_decoder is not None
    assert agent.compound_decoder.parameters_projection.weight.grad is not None
    for command in decision.commands:
        env.execute_semantic_command_json(command.to_json())


@pytest.mark.parametrize("kind", ["attack", "blockers", "payment"])
def test_wide_object_decoder_has_complete_native_support(kind: str) -> None:
    if kind == "blockers":
        env, raw = _block_root(35, 65)
    elif kind == "payment":
        env, raw = _waterbend_root(65)
    else:
        env, raw = _root("attack", 65)
    with torch.no_grad():
        decision = sample_compound(
            _agent(wide=True, features="objects"), env, raw, deterministic=True
        )
    assert torch.isfinite(decision.output.log_prob)
    assert decision.offers.max_candidate_count >= 35
    for command in decision.commands:
        env.execute_semantic_command_json(command.to_json())


def test_native_offers_do_not_survive_same_seed_reset() -> None:
    env = managym.Env(seed=91)
    env.reset(_match().to_rust())
    offers = env.compound_offers()
    env.reset(_match().to_rust())
    before = env.state_digest()
    with pytest.raises(managym.AgentError, match="previous episode"):
        env.compound_prefix_support(offers, [])
    assert env.state_digest() == before


def test_rescoring_rejects_another_viewer_root() -> None:
    env, raw = _block_root(2, 2)
    agent = _agent(features="objects")
    decision = sample_compound(agent, env, raw)
    changed = replace(decision.offers, fingerprint="another-root")
    with pytest.raises(StructuredPolicyError, match="another root"):
        agent.compound(decision.observation, changed, tape=decision.output)


def test_compound_rejects_a_nonacting_viewer() -> None:
    env, _ = _root("attack", 2)
    actor = env.current_agent_index()
    assert actor is not None
    raw = env.observation_for_player(1 - actor)
    with pytest.raises(ValueError, match="acting viewer"):
        sample_compound(_agent(features="objects"), env, raw)


def test_shared_payment_meaning_identifies_native_source_and_units() -> None:
    env, _ = _waterbend_root(6, lands=2)
    projection = OfferProjection.from_json(env.compound_offers().projection_json())
    offer = projection.offers[0]
    assert offer.source is not None
    assert offer.details.program is not None
    assert len(offer.details.program.digest) == 64
    context = offer.choices[0].context
    assert context.kind == "payment"
    assert context.tap_generic_units == 1
    assert sum(context.mana) == 5


def test_object_decoder_distinguishes_verbs_without_presentation_labels() -> None:
    torch.manual_seed(107)
    decoder = CompoundDecoder(8, object_features=True)
    projection = _projection(0, 0, 0)
    # Two actions on the same source need different semantics even with no
    # candidate factors. Labels may be localized without changing probabilities.
    offers = json.loads(json.dumps(projection["offers"]))
    offers[0].update(verb="scry_keep", choices=[])
    offers.append({**offers[0], "id": 5, "verb": "scry_bottom"})
    projection.update(kind="scry", offers=offers)
    context = torch.randn(8)
    objects = torch.zeros(1, 8)
    original = decoder(
        context,
        flatten_projection(projection, object_rows={}),
        objects=objects,
        tokens=(0,),
    )
    assert original.probabilities[0][0] != original.probabilities[0][1]
    for offer in offers:
        offer["label"] = "Localized presentation"
    localized = decoder(
        context,
        flatten_projection(projection, object_rows={}),
        objects=objects,
        tokens=(0,),
    )
    torch.testing.assert_close(original.log_probs, localized.log_probs, rtol=0, atol=0)
