"""Ordinary native choices must support either preference, without strategy labels.

These probes widen the attack/decline regression across action verbs, two-object
focus, player targets, revealed cards and hand selections. Two shared fits use
opposite labels on the same fixed states: this proves capacity, not good play or
generalization. Native encoding and action execution remain in the loop.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
import torch

from manabot.env.observation import (
    ACTION_DECLARED_INDEX,
    ACTION_TYPE_DIM,
    PERMANENT_DEFINITION_INDEX,
    ObservationSpace,
)
from manabot.infra.config import ModelConfig
from manabot.model.actor_critic import ActorCritic
from manabot.verify.action_cases import encode, initial, reach, reorder_objects
from manabot.verify.trigger_order import ordering_root
import managym

ChoiceName = Literal[
    "land_pass",
    "cast_pass",
    "block_decline",
    "block_target",
    "scry",
    "pay_decline",
    "player_target",
    "look_select",
    "learn_discard",
    "waterbend",
    "trigger_order",
]
CHOICES: tuple[ChoiceName, ...] = (
    "land_pass",
    "cast_pass",
    "block_decline",
    "block_target",
    "scry",
    "pay_decline",
    "player_target",
    "look_select",
    "learn_discard",
    "waterbend",
    "trigger_order",
)


@dataclass
class NativeChoice:
    name: ChoiceName
    env: managym.Env
    raw: managym.Observation
    groups: tuple[tuple[int, ...], tuple[int, ...]]

    @property
    def candidates(self) -> tuple[int, int]:
        return self.groups[0][0], self.groups[1][0]


def _indices(raw: managym.Observation, kind: managym.ActionEnum) -> list[int]:
    return [
        i
        for i, action in enumerate(raw.action_space.actions)
        if action.action_type == kind
    ]


def _choice(name: ChoiceName) -> NativeChoice:
    if name == "trigger_order":
        env, raw = ordering_root()
        candidates = (0, 1)
    elif name in {"block_decline", "block_target"}:
        env = initial(
            authored=False,
            deck={"Gray Ogre": 16, "Water Tribe Rallier": 16, "Mountain": 8},
        )
        env.scenario_force_battlefield(0, "Gray Ogre", ready=True)
        if name == "block_target":
            env.scenario_force_battlefield(0, "Water Tribe Rallier", ready=True)
        env.scenario_force_battlefield(1, "Gray Ogre", ready=True)
        raw = env.scenario_refresh()
        for _ in range(100):
            if (
                raw.action_space.action_space_type
                == managym.ActionSpaceEnum.DECLARE_BLOCKER
            ):
                break
            index = next(
                (
                    i
                    for i, a in enumerate(raw.action_space.actions)
                    if a.declared is True
                ),
                None,
            )
            if index is None:
                index = _indices(raw, managym.ActionEnum.PRIORITY_PASS_PRIORITY)[0]
            raw, _, done, _, _ = env.step(index)
            assert not done
        else:
            raise AssertionError("did not reach blocking")
        yes = [i for i, a in enumerate(raw.action_space.actions) if a.declared is True]
        no = [i for i, a in enumerate(raw.action_space.actions) if a.declared is False]
        candidates = (yes[0], no[0]) if name == "block_decline" else (yes[0], yes[1])
    elif name == "land_pass":
        env = initial(authored=False, deck={"Island": 20, "Mountain": 20})
        env.scenario_force_card_in_hand(0, "Island")
        raw = env.scenario_refresh()
        for _ in range(100):
            if raw.turn.phase == managym.PhaseEnum.POSTCOMBAT_MAIN and _indices(
                raw, managym.ActionEnum.PRIORITY_PLAY_LAND
            ):
                break
            raw, _, done, _, _ = env.step(
                _indices(raw, managym.ActionEnum.PRIORITY_PASS_PRIORITY)[0]
            )
            assert not done
        else:
            raise AssertionError("did not reach a second-main land drop")
        assert raw.turn.turn_number == 1
        candidates = (
            _indices(raw, managym.ActionEnum.PRIORITY_PLAY_LAND)[0],
            _indices(raw, managym.ActionEnum.PRIORITY_PASS_PRIORITY)[0],
        )
    elif name == "waterbend":
        env = initial(
            authored=False,
            deck={"Water Tribe Rallier": 4, "Gray Ogre": 4, "Forest": 32},
        )
        for creature in ("Water Tribe Rallier", "Gray Ogre"):
            env.scenario_force_battlefield(0, creature, ready=True)
        for _ in range(5):
            env.scenario_force_battlefield(0, "Forest", ready=True)
        raw = env.scenario_refresh()
        raw, _, _, _, _ = env.step(
            _indices(raw, managym.ActionEnum.PRIORITY_ACTIVATE_ABILITY)[0]
        )
        assert raw.action_space.action_space_type == managym.ActionSpaceEnum.WATERBEND
        taps = _indices(raw, managym.ActionEnum.TAP_FOR_COST)
        candidates = (taps[0], taps[1])
    else:
        decks: dict[str, dict[str, int]] = {
            "cast_pass": {"Gray Ogre": 4, "Mountain": 36},
            "scry": {"Glider Kids": 4, "Plains": 18, "Island": 18},
            "pay_decline": {"Firebending Lesson": 4, "Gray Ogre": 4, "Mountain": 32},
            "player_target": {"Lightning Bolt": 4, "Mountain": 36},
            "look_select": {
                "Accumulate Wisdom": 4,
                "Island": 12,
                "Mountain": 12,
                "Forest": 12,
            },
            "learn_discard": {"Pop Quiz": 4, "Island": 18, "Mountain": 18},
        }
        spells = {
            "cast_pass": "Gray Ogre",
            "scry": "Glider Kids",
            "pay_decline": "Firebending Lesson",
            "player_target": "Lightning Bolt",
            "look_select": "Accumulate Wisdom",
            "learn_discard": "Pop Quiz",
        }
        kinds = {
            "scry": 5,
            "pay_decline": 7,
            "player_target": 4,
            "look_select": 6,
            "learn_discard": 9,
        }
        env = initial(authored=False, deck=decks[name])
        land = (
            "Plains"
            if name == "scry"
            else "Island"
            if name in {"look_select", "learn_discard"}
            else "Mountain"
        )
        for _ in range(5):
            env.scenario_force_battlefield(0, land, ready=True)
        env.scenario_force_card_in_hand(0, spells[name])
        if name == "pay_decline":
            env.scenario_force_battlefield(1, "Gray Ogre", ready=True)
        if name == "learn_discard":
            env.scenario_force_card_in_hand(0, "Island")
            env.scenario_force_card_in_hand(0, "Mountain")
        raw = env.scenario_refresh()
        if name == "cast_pass":
            for _ in range(100):
                if _indices(raw, managym.ActionEnum.PRIORITY_CAST_SPELL):
                    break
                raw, _, done, _, _ = env.step(
                    _indices(raw, managym.ActionEnum.PRIORITY_PASS_PRIORITY)[0]
                )
                assert not done
            else:
                raise AssertionError("did not reach a cast/pass decision")
            candidates = (
                _indices(raw, managym.ActionEnum.PRIORITY_CAST_SPELL)[0],
                _indices(raw, managym.ActionEnum.PRIORITY_PASS_PRIORITY)[0],
            )
        else:
            raw = reach(env, kinds[name])
            if name == "player_target":
                candidates = (
                    next(
                        i
                        for i, a in enumerate(raw.action_space.actions)
                        if a.focus == [raw.agent.id]
                    ),
                    next(
                        i
                        for i, a in enumerate(raw.action_space.actions)
                        if a.focus == [raw.opponent.id]
                    ),
                )
            elif name in {"look_select", "learn_discard"}:
                kind = (
                    managym.ActionEnum.SELECT_CARD
                    if name == "look_select"
                    else managym.ActionEnum.LEARN_DISCARD
                )
                definitions = {c.id: c.registry_key for c in raw.agent_cards}
                offers = _indices(raw, kind)
                first = offers[0]
                second = next(
                    i
                    for i in offers[1:]
                    if definitions[raw.action_space.actions[i].focus[0]]
                    != definitions[raw.action_space.actions[first].focus[0]]
                )
                candidates = (first, second)
            else:
                assert len(raw.action_space.actions) == 2
                candidates = (0, 1)
    groups = ((candidates[0],), (candidates[1],))
    if name in {"look_select", "learn_discard"}:
        # Physical copies of the same visible card are exchangeable. Fit their
        # combined probability, without teaching arbitrary instance IDs.
        definitions = {c.id: c.registry_key for c in raw.agent_cards}
        actions = raw.action_space.actions

        def copies(index: int) -> tuple[int, ...]:
            selected = actions[index]
            return tuple(
                i
                for i, action in enumerate(actions)
                if action.action_type == selected.action_type
                and definitions[action.focus[0]] == definitions[selected.focus[0]]
            )

        groups = (copies(candidates[0]), copies(candidates[1]))
    return NativeChoice(name, env, raw, groups)


def _remove_discriminant(
    case: NativeChoice, obs: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Reproduce a specific information-loss hypothesis for each choice pair."""
    erased = {key: value.clone() for key, value in obs.items()}
    a, b = case.candidates
    if case.name == "trigger_order":
        erased["actions"][..., -2] = 0
        erased["action_focus"].fill_(-1)
    elif case.name in {"scry", "pay_decline"}:
        erased["actions"][..., :ACTION_TYPE_DIM] = 0
    elif case.name in {"look_select", "learn_discard"}:
        erased["semantic_cards"].zero_()
    elif case.name in {"block_target", "waterbend"}:
        for key in ("agent_permanents", "opponent_permanents"):
            erased[key][..., PERMANENT_DEFINITION_INDEX] = 0
    elif case.name == "block_decline":
        erased["actions"][..., ACTION_DECLARED_INDEX] = 0
        erased["action_focus"][..., 1:] = -1
    elif case.name == "player_target":
        erased["action_focus"][:, b] = erased["action_focus"][:, a]
    else:
        erased["actions"][..., :ACTION_TYPE_DIM] = 0
        erased["action_focus"].fill_(-1)
    return erased


@pytest.fixture(scope="module")
def choices() -> list[NativeChoice]:
    return [_choice(name) for name in CHOICES]


@pytest.fixture(autouse=True)
def one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng():
            yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("name", CHOICES)
def test_native_choice_preserves_focus_and_changes_state(name: ChoiceName) -> None:
    case = _choice(name)
    obs = encode(case.env, case.raw)
    assert int(obs["actions_valid"].sum()) == len(case.raw.action_space.actions)
    # A nonempty native focus must not silently become padding in either encoder.
    for i, action in enumerate(case.raw.action_space.actions):
        assert (obs["action_focus"][0, i, : len(action.focus)] >= 0).all(), (
            name,
            i,
            action.focus,
        )
    root = case.env.state_digest()
    outcomes: list[str] = []
    for index in case.candidates:
        branch = case.env.clone_env()
        branch.step(index)
        outcomes.append(branch.state_digest())
    assert outcomes[0] != outcomes[1], name
    assert case.env.state_digest() == root


@pytest.mark.parametrize("preference", [0, 1])
def test_shared_policy_can_learn_either_choice(
    choices: list[NativeChoice], preference: int, tmp_path: Path
) -> None:
    # Fixed protocol before scoring: two shared fits total, 180 updates each;
    # no early stopping, per-family fitting, or strategically "correct" labels.
    torch.manual_seed(17)
    observations = [encode(case.env, case.raw) for case in choices]
    batch = {
        key: torch.cat([obs[key] for obs in observations]) for key in observations[0]
    }
    selected = torch.zeros_like(batch["actions_valid"], dtype=torch.bool)
    for row, case in enumerate(choices):
        selected[row, list(case.groups[preference])] = True
    spec = ModelConfig(
        hidden_dim=32, num_attention_heads=2, attention_feedforward_dim=64
    )
    agent = ActorCritic(ObservationSpace(), spec)
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.003)
    for _ in range(180):
        logits, _ = agent(batch)
        # NLL of the selected observable choice; identical copies share mass.
        log_probabilities = logits.log_softmax(-1)
        loss = (
            -log_probabilities.masked_fill(~selected, -torch.inf).logsumexp(-1).mean()
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    path = tmp_path / "shared-choices.pt"
    torch.save(agent.state_dict(), path)
    restored = ActorCritic(ObservationSpace(), spec)
    restored.load_state_dict(torch.load(path, weights_only=True))
    restored.eval()
    with torch.no_grad():
        for case, obs in zip(choices, observations, strict=True):
            logits, _ = restored(obs)
            probability = logits.softmax(-1)[0, list(case.groups[preference])].sum()
            assert float(probability) >= 0.95, (case.name, float(probability))
            for group in case.groups:
                for index in group[1:]:
                    torch.testing.assert_close(
                        logits[0, index], logits[0, group[0]], atol=1e-5, rtol=1e-5
                    )
            selected_action, _, _, _ = restored.get_action_and_value(
                obs, deterministic=True
            )
            assert int(selected_action) in case.groups[preference], case.name
            # Reorder the native offers before either encoder, not just a tensor.
            space = case.raw.action_space
            original = space.actions
            space.actions = original[::-1]
            case.raw.action_space = space
            try:
                reordered = encode(case.env, case.raw)
            finally:
                space.actions = original
                case.raw.action_space = space
            permuted, _ = restored(reordered)
            count = len(original)
            torch.testing.assert_close(permuted[0, :count].flip(0), logits[0, :count])
            permuted, _ = restored(reorder_objects(obs))
            torch.testing.assert_close(permuted, logits, atol=1e-5, rtol=1e-5)
            assert not logits.softmax(-1)[obs["actions_valid"] == 0].any()
            # A specific dropped channel must recreate the forced tie. These
            # mutations are hypotheses, not historical-version evidence.
            a, b = case.candidates
            tied, _ = restored(_remove_discriminant(case, obs))
            torch.testing.assert_close(tied[0, a], tied[0, b], atol=0, rtol=0)
