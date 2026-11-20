"""Choice audits resolve native aliases without labelling choices by storage id."""

import json

import torch

from manabot.verify.action_cases import encode
from manabot.verify.choice_corpus import _signatures, action_inputs
import managym

# Seed 82000 reaches waterbend payment, whose source uses the card identity of
# a battlefield permanent. At prefix 19 it also has a spell on the stack.
PAYMENT_TRACE = (0, 1, 3, 1, 0, 0, 0, 0, 1, 1, 0, 4, 0, 3, 0, 1, 1, 1, 2, 1, 1, 0, 2)


def replay_prefix(length: int) -> tuple[managym.Env, managym.Observation]:
    env = managym.Env(seed=82000)
    raw, _ = env.reset(
        [
            managym.authored_deck_setup("ur-lessons-vs-gw-allies", key)
            for key in ("ur_lessons", "gw_allies")
        ]
    )
    for index in PAYMENT_TRACE[:length]:
        raw, _, done, truncated, _ = env.step(index)
        assert not done and not truncated
    return env, raw


def test_payment_source_names_the_battlefield_object() -> None:
    env, raw = replay_prefix(len(PAYMENT_TRACE))
    signatures = _signatures(raw)
    action = raw.action_space.actions[-1]
    permanent = next(
        item for item in raw.agent_permanents if item.object_id == action.focus[0]
    )
    assert permanent.id != permanent.object_id
    descriptor = json.loads(signatures[-1])[-1][0]
    assert descriptor[:3] == ["permanent", "self", permanent.registry_key]
    inputs = action_inputs(encode(raw), len(signatures), "clean")
    assert inputs[-1].focus[0] is not None
    assert inputs[-1].focus[0].kind == "permanents"


def test_stack_aliases_share_a_signature_and_reach_collision_inputs() -> None:
    env, raw = replay_prefix(19)
    spell = raw.stack_objects[0]
    assert spell.object_id is not None
    # Project the same diagnostic choice onto each native alias. Both aliases
    # must resolve to the same stack row, including in the collision checker.
    space = raw.action_space
    action = space.actions[0]
    signatures = []
    inputs = []
    for identity in (spell.stack_object_id, spell.object_id):
        action.focus = [identity]
        space.actions = [action]
        raw.action_space = space
        signatures.append(_signatures(raw))
        obs = encode(raw)
        inputs.append(action_inputs(obs, 1, "clean"))
        focus = inputs[-1][0].focus[0]
        assert focus is not None and focus.kind == "stack"
        torch.testing.assert_close(
            torch.tensor(focus.features), obs["stack"][0, 0], rtol=0, atol=0
        )
    assert signatures[0] == signatures[1]
    assert inputs[0] == inputs[1]
