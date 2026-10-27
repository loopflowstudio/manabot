"""Elemental admission through the compiler, live rules and learning readers."""

from copy import deepcopy
import json
from pathlib import Path

from managym.decision import Command, DecisionFrame, SemanticTransition
from managym.experience_projection import describe_actions
import numpy as np
import pytest
import torch

from manabot.belief.tracker import BeliefTracker
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.model.actor_critic import ActorCritic
from manabot.semantic.compiler import (
    SemanticCompileError,
    compile_paths,
    compile_source,
    pretty_json,
)
from manabot.semantic.learning import BoundSemanticPack, ContentPackBindingError
import managym

BASE = Path("content/semantic/v1")
PACK = "elemental-combat-v1"


def test_elemental_artifacts_are_canonical_and_close_all_42_cards() -> None:
    ir, coverage = compile_paths(
        BASE / "elemental.source.json", BASE / "elemental.fixtures.json"
    )
    assert (BASE / "generated/elemental.ir.json").read_text() == pretty_json(ir)
    assert (BASE / "generated/elemental.coverage.json").read_text() == pretty_json(
        coverage
    )
    assert coverage["definition_count"] == 51
    assert coverage["admission_closure_complete"]
    assert coverage["no_card_name_dispatch"]
    assert (
        len(
            [d for d in ir["definitions"] if d["semantic_key"].startswith("elemental.")]
        )
        == 42
    )


@pytest.mark.parametrize("bad", ["quantity", "mode", "role", "tap", "binding"])
def test_elemental_compiler_rejects_invalid_typed_programs(bad: str) -> None:
    source = json.loads((BASE / "elemental.source.json").read_text())
    programs = [p for d in source["definitions"] for p in d["programs"]]
    if bad == "quantity":
        program = next(p for p in programs if p["ops"][0]["op"] == "drain_life_value")
        program["ops"][0]["amount"] = {"kind": "oracle_text"}
    elif bad == "mode":
        program = next(p for p in programs if p["ops"][0]["op"] == "modal")
        program["ops"][0]["modes"][0] = [{"op": "destroy", "target": "missing"}]
    elif bad == "role":
        program = next(p for p in programs if p["ops"][0]["op"] == "target_clause")
        program["ops"][0]["role"] = "missing"
    elif bad == "tap":
        program = next(p for p in programs if p["kind"] == "activated")
        program["cost"]["tap_source"] = 1
    else:
        programs[0]["target_binding"] = "latest"
    with pytest.raises(SemanticCompileError):
        compile_source(source)


def _engine() -> tuple[managym.Env, managym.Observation]:
    engine = managym.Env(seed=161, skip_trivial=False)
    obs, _ = engine.reset(
        [
            managym.authored_deck_setup(PACK, key)
            for key in ("elemental_right", "elemental_left")
        ]
    )
    for _ in range(20):
        if int(obs.turn.step) == 3:
            break
        action = next(
            a for a in describe_actions(obs) if a["type"] == "PRIORITY_PASS_PRIORITY"
        )
        obs, *_ = engine.step(action["index"])
    else:
        pytest.fail("main phase was not reached")
    for player in (0, 1):
        engine.scenario_clear_hand(player)
    engine.scenario_force_card_in_hand(0, "Shared Roots")
    engine.scenario_force_card_in_hand(1, "Lightning Strike")
    for name in ("Forest", "Forest"):
        engine.scenario_force_battlefield(0, name, True)
    return engine, engine.scenario_refresh()


def test_compiled_search_reaches_belief_and_public_history_consumers() -> None:
    engine, obs = _engine()
    pack = BoundSemanticPack.from_env(engine)
    assert pack.schema.raw["schema_version"] == 3
    assert len(pack.definition_row_by_card_def_id) == 51
    pack.pad_catalog()  # All programs fit the declared budget without truncation.
    wrong = deepcopy(engine.content_pack_manifest())
    wrong["compiled_semantics"]["ir_hash"] = "0" * 64
    with pytest.raises(ContentPackBindingError):
        BoundSemanticPack.bind(wrong)
    cast = next(
        a for a in describe_actions(obs) if a["description"] == "Cast Shared Roots"
    )
    obs, *_ = engine.step(cast["index"])
    for _ in range(10):
        if any(a["type"] == "SELECT_CARD" for a in describe_actions(obs)):
            break
        action = next(
            a for a in describe_actions(obs) if a["type"] == "PRIORITY_PASS_PRIORITY"
        )
        obs, *_ = engine.step(action["index"])
    else:
        pytest.fail("search was not offered")
    viewer = int(obs.agent.player_index)
    assert viewer == 0
    tracker = BeliefTracker.from_engine(
        engine, viewer=viewer, likelihood=None, epsilon=0.0
    )
    frame = DecisionFrame.from_json(engine.semantic_decision_frame_json())
    root = engine.state_digest()
    command = Command("elemental-search", frame.revision, int(frame.offers[0]["id"]))
    result = SemanticTransition.from_json(
        engine.execute_semantic_command_json(command.to_json())
    )
    tracker.observe(engine, acting=viewer, transition=result)
    assert tracker.stats.updates == 1
    assert tracker.posterior.normalization_error < 1e-12
    assert engine.state_digest() != root
    history_space = ObservationSpace(ObservationConfig(policy_history_version=1))
    obs = engine.observation_for_player(0)
    encoded = history_space.encode(obs)
    assert np.asarray(encoded["events_valid"]).any()
    assert np.asarray(encoded["events"])[..., [2, 7]].max() > 0
    torch.set_num_threads(1)
    agent = ActorCritic(
        history_space,
        ModelConfig(hidden_dim=16, semantic_pack=PACK, recent_events=True),
    )
    tensors = {
        key: torch.as_tensor(value).unsqueeze(0) for key, value in encoded.items()
    }
    policy, value = agent(tensors)
    assert torch.isfinite(policy[tensors["actions_valid"].bool()]).all()
    assert torch.isfinite(value).all()
    assert agent.recent_events is not None
    gradients = torch.autograd.grad(
        value.sum(), tuple(agent.recent_events.parameters())
    )
    assert sum(float(g.abs().sum()) for g in gradients) > 0
    # Determinization must not alter the viewer's private search or its public history.
    sibling = engine.clone_env()
    sibling.determinize(16101, perspective=0)
    sibling_encoded = history_space.encode(sibling.observation_for_player(0))
    np.testing.assert_array_equal(sibling_encoded["events"], encoded["events"])
    np.testing.assert_array_equal(
        sibling_encoded["events_valid"], encoded["events_valid"]
    )
