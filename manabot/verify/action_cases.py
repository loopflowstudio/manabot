"""Native action-identity diagnostic cases shared by structural and learning tests.

These constructed decisions test capacity, not game strategy. Ability selectors
are a synthetic projection of a native modal decision; Rust owns their native
legality and resolution regression.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from manabot.env import observation as observation_schema
from manabot.env.observation import ObservationSpace
import managym


def initial(
    *, authored: bool = True, seed: int = 111, deck: dict[str, int] | None = None
) -> managym.Env:
    # Diagnostic exception: `reach` and the outcome probes step constructed
    # positions one native window at a time, including sole-offer ones. These
    # fixed cases (CASE_SET) never play a game for a model or policy.
    env = managym.Env(seed=seed, skip_trivial=False)
    configs = (
        [
            managym.authored_deck_setup("ur-lessons-vs-gw-allies", key)
            for key in ("ur_lessons", "gw_allies")
        ]
        if authored
        else [
            managym.PlayerConfig(
                str(i), deck or {"Island": 36, "Crossroads of Destiny": 4}
            )
            for i in range(2)
        ]
    )
    env.reset(configs)
    for seat in range(2):
        env.scenario_clear_hand(seat)
    return env


def reach(env: managym.Env, kind: int) -> managym.Observation:
    raw = env.scenario_refresh()
    for _ in range(100):
        if int(raw.action_space.action_space_type) == kind:
            return raw
        actions = raw.action_space.actions
        index = next(
            (i for i, a in enumerate(actions) if int(a.action_type) == 1), None
        )
        if index is None:
            index = next(
                (i for i, a in enumerate(actions) if int(a.action_type) == 2), 0
            )
        raw, _, done, _, _ = env.step(index)
        assert not done
    raise AssertionError(f"did not reach decision {kind}")


def encode(env: managym.Env, raw: managym.Observation) -> dict[str, torch.Tensor]:
    native = env.encode_observation(raw)
    scalar = ObservationSpace().encode(raw)
    for key in native:
        np.testing.assert_array_equal(native[key], scalar[key])
    return {key: torch.as_tensor(value).unsqueeze(0) for key, value in native.items()}


Family = Literal["attack", "target", "mode", "ability", "land", "learn"]
FAMILIES: tuple[Family, ...] = ("attack", "target", "mode", "ability", "land", "learn")


@dataclass(frozen=True)
class NativeOutcome:
    life: int
    opponent_life: int
    hand_definitions: list[int]
    permanent_definitions: list[int]
    opponent_permanent_definitions: list[int]


def _resolved_outcomes(
    env: managym.Env, raw: managym.Observation, kind: Family, candidates: list[int]
) -> list[NativeOutcome]:
    outcomes: list[NativeOutcome] = []
    for index in candidates[:2]:
        branch = env.clone_env()
        current, _, done, _, _ = branch.step(index)
        if kind in {"target", "mode", "ability"}:
            for _ in range(2):
                passing = next(
                    i
                    for i, a in enumerate(current.action_space.actions)
                    if int(a.action_type) == 2
                )
                current, _, done, _, _ = branch.step(passing)
            if kind == "target":
                target_id = raw.action_space.actions[index].focus[0]
                assert target_id not in [
                    p.id for p in branch.observation_for_player(0).opponent_permanents
                ]
        elif kind == "attack":
            for _ in range(40):
                if current.turn.turn_number != raw.turn.turn_number or done:
                    break
                passing = next(
                    (
                        i
                        for i, a in enumerate(current.action_space.actions)
                        if int(a.action_type) == 2
                    ),
                    None,
                )
                assert passing is not None, "unexpected attack-resolution decision"
                current, _, done, _, _ = branch.step(passing)
            assert current.turn.turn_number != raw.turn.turn_number or done, (
                "combat did not resolve"
            )
        owner = branch.observation_for_player(0)
        outcomes.append(
            NativeOutcome(
                owner.agent.life,
                owner.opponent.life,
                sorted(c.registry_key for c in owner.agent_cards if int(c.zone) == 1),
                sorted(p.registry_key for p in owner.agent_permanents),
                sorted(p.registry_key for p in owner.opponent_permanents),
            )
        )
    assert outcomes[0] != outcomes[1], (
        f"{kind} choices did not produce distinct native consequences"
    )
    if kind == "attack":
        assert [o.opponent_life for o in outcomes] == [18, 20]
    return outcomes


@dataclass
class DecisionCase:
    family: Family
    observation: dict[str, torch.Tensor]
    candidates: list[int]
    preference: int
    split: str
    variant: int
    native: bool
    outcomes: list[NativeOutcome]

    @property
    def preferred(self) -> int:
        return self.candidates[self.preference]


def decision_case(
    kind: Family,
    *,
    preference: int = 0,
    variant: int = 0,
    split: str = "fit",
    legacy_selectors: bool = False,
    seed: int | None = None,
) -> DecisionCase:
    """Build one decision; `variant` sets the life offset and, unless `seed`
    overrides it, the native shuffle seed (111 + variant)."""
    env = initial(
        authored=kind not in {"mode", "ability", "attack"},
        seed=111 + variant if seed is None else seed,
        deck={"Gray Ogre": 36, "Mountain": 4} if kind == "attack" else None,
    )
    # Public life supplies coherent opposite labels within a single policy.
    # Held-out levels and seeds are real changed native states, not permutations.
    life = (5, 35)[preference] + (variant if preference == 0 else -variant)
    env.scenario_set_life(0, life)
    if kind == "attack":
        env.scenario_force_battlefield(0, "Gray Ogre", ready=True)
        raw = reach(env, 2)
        assert [a.declared for a in raw.action_space.actions] == [True, False]
        candidates = [0, 1]
    elif kind == "target":
        env.scenario_force_battlefield(0, "Mountain", ready=True)
        env.scenario_force_card_in_hand(0, "Firebending Lesson")
        for name in ("Compassionate Healer", "Water Tribe Rallier"):
            env.scenario_force_battlefield(1, name, ready=True)
        raw = reach(env, 4)
        candidates = list(range(len(raw.action_space.actions)))
    elif kind in {"mode", "ability"}:
        env.scenario_force_battlefield(0, "Island", ready=True)
        env.scenario_force_card_in_hand(0, "Crossroads of Destiny")
        raw = reach(env, 8)
        if not legacy_selectors:
            assert [a.mode_index for a in raw.action_space.actions] == [0, 1]
        outcomes = []
        for index in range(2):
            branch = env.clone_env()
            current, _, _, _, _ = branch.step(index)
            # Spell modes are announced before payment and priority; their
            # consequences arrive after both players pass on the stack.
            for _ in range(2):
                passing = next(
                    i
                    for i, a in enumerate(current.action_space.actions)
                    if int(a.action_type) == 2
                )
                current, _, _, _, _ = branch.step(passing)
            owner = branch.observation_for_player(0)
            outcomes.append(
                (owner.agent.life, sum(int(c.zone) == 1 for c in owner.agent_cards))
            )
        assert outcomes == [(life + 3, 0), (life, 1)]
        if kind == "ability":
            # The Rust CR 602 regression proves two-ability legality/resolution.
            # Reuse equal-focus rows here to isolate Python bindings and scoring.
            space = raw.action_space
            actions = space.actions
            for i, action in enumerate(actions):
                action.action_type = managym.ActionEnum.PRIORITY_ACTIVATE_ABILITY
                if not legacy_selectors:
                    action.mode_index = None
                    action.ability_index = i
            space.actions = actions
            raw.action_space = space
        candidates = [0, 1]
    elif kind == "learn":
        for _ in range(3):
            env.scenario_force_battlefield(0, "Island", ready=True)
        env.scenario_force_card_in_hand(0, "Pop Quiz")
        raw = reach(env, 9)
        candidates = [
            i
            for i, a in enumerate(raw.action_space.actions)
            if int(a.action_type) == 14
        ]
    else:
        for name in ("Island", "Mountain"):
            env.scenario_force_card_in_hand(0, name)
        raw = env.scenario_refresh()
        for _ in range(100):
            if sum(int(a.action_type) == 0 for a in raw.action_space.actions) == 2:
                break
            index = next(
                (
                    i
                    for i, a in enumerate(raw.action_space.actions)
                    if int(a.action_type) == 2
                ),
                0,
            )
            raw, _, done, _, _ = env.step(index)
            assert not done
        candidates = [
            i for i, a in enumerate(raw.action_space.actions) if int(a.action_type) == 0
        ]
    assert len(candidates) >= 2
    return DecisionCase(
        kind,
        encode(env, raw),
        candidates,
        preference,
        split,
        variant,
        kind != "ability",
        []
        if legacy_selectors or kind == "ability"
        else _resolved_outcomes(env, raw, kind, candidates),
    )


def identity_removed(case: DecisionCase) -> dict[str, torch.Tensor]:
    """Erase the audited identity channel without changing legal offers."""
    obs = {k: v.clone() for k, v in case.observation.items()}
    columns = {
        "attack": observation_schema.ACTION_DECLARED_INDEX,
        "mode": observation_schema.ACTION_MODE_INDEX,
        "ability": observation_schema.ACTION_ABILITY_INDEX,
    }
    if case.family in columns:
        obs["actions"][..., columns[case.family]] = 0
    elif case.family == "target":
        obs["opponent_permanents"][
            ..., observation_schema.PERMANENT_DEFINITION_INDEX
        ] = 0
    else:
        obs["semantic_cards"].zero_()
    return obs


def reorder_offers(
    obs: dict[str, torch.Tensor], order: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Permute legal offers together with their masks and object references."""
    result = {k: v.clone() for k, v in obs.items()}
    for key in ("actions", "actions_valid", "action_focus"):
        result[key][:, : len(order)] = obs[key][:, order]
    return result


def reorder_objects(obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Reverse object slots with their identity joins, masks and focus references."""
    result = {k: v.clone() for k, v in obs.items()}
    offset = 2  # The two player rows precede cards and permanents.
    for index, key in enumerate(
        ("agent_cards", "opponent_cards", "agent_permanents", "opponent_permanents")
    ):
        count = obs[key].shape[1]
        order = torch.arange(count - 1, -1, -1)
        result[key] = obs[key][:, order]
        result[key + "_valid"] = obs[key + "_valid"][:, order]
        if index < 2:
            result["semantic_cards"][:, index] = obs["semantic_cards"][:, index, order]
        focus = obs["action_focus"]
        mask = (focus >= offset) & (focus < offset + count)
        result["action_focus"][mask] = offset + count - 1 - (focus[mask] - offset)
        if "relations" in obs:
            # Continuation references use object index + 1 (zero is absence).
            for column in (6, 24):
                refs = obs["relations"][..., column]
                mask = (refs > offset) & (refs <= offset + count)
                result["relations"][..., column][mask] = (
                    2 * offset + count + 1 - refs[mask]
                )
        offset += count
    return result


# Version of the case set `curriculum()` builds; bump it when its digest changes.
CASE_SET = "action-cases-2"
# Held-out cases change state only: every definition they show appears in that
# family's fit cases (`curriculum` asserts it). action-cases-2 moved Learn
# held-out variant 1 from seed 112, whose Pop Quiz draws a Mountain no Learn fit
# case shows, to seed 116, which draws an Island.
_NATIVE_SEEDS: dict[tuple[Family, int], int] = {("learn", 1): 116}


def _definitions(obs: dict[str, torch.Tensor]) -> set[int]:
    """Card definitions visible anywhere in one observation (registry key + 1)."""
    found = {int(x) for x in obs["semantic_cards"].flatten() if x > 0}
    found |= {int(x) for x in obs["known_hand"][..., 0].flatten() if x > 0}
    for key in ("agent_permanents", "opponent_permanents"):
        rows = obs[key][obs[key + "_valid"] > 0]
        found |= {
            int(x) for x in rows[:, observation_schema.PERMANENT_DEFINITION_INDEX]
        }
    return found


def unseen_definitions(cases: list[DecisionCase]) -> dict[str, set[int]]:
    """Held-out case key -> definitions its family's fit cases never show."""
    fit: dict[Family, set[int]] = {family: set() for family in FAMILIES}
    for case in cases:
        if case.split == "fit":
            fit[case.family] |= _definitions(case.observation)
    unseen: dict[str, set[int]] = {}
    for case in cases:
        if case.split == "held_out":
            extra = _definitions(case.observation) - fit[case.family]
            if extra:
                unseen[f"{case.family}/{case.variant}/{case.preference}"] = extra
    return unseen


def curriculum() -> list[DecisionCase]:
    cases = [
        decision_case(
            family,
            preference=preference,
            variant=variant,
            split=split,
            seed=_NATIVE_SEEDS.get((family, variant)),
        )
        for split, variants in (("fit", (0, 2)), ("held_out", (1, 3)))
        for variant in variants
        for family in FAMILIES
        for preference in (0, 1)
    ]
    unseen = unseen_definitions(cases)
    assert not unseen, f"held-out cases show definitions absent from fit: {unseen}"
    return cases
