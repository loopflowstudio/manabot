"""Replayable native choices for capacity tests and frozen-policy inspection.

Collect model-independent authored-game prefixes, retain seed/action traces and
both encoders, and select public action classes without arbitrary instance IDs.
This corpus contains no strategic labels and never trains a model.
"""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations, zip_longest
import json
from pathlib import Path
import random
import time
from typing import Literal

import torch

from manabot.env.observation import ACTION_DECLARED_INDEX, PERMANENT_DEFINITION_INDEX
from manabot.verify.action_cases import encode, reorder_offers
import managym

PACK = "ur-lessons-vs-gw-allies"
KEYWORDS = (
    "flying",
    "reach",
    "haste",
    "flash",
    "vigilance",
    "trample",
    "first_strike",
    "double_strike",
    "deathtouch",
    "lifelink",
    "defender",
    "menace",
    "hexproof",
)


@dataclass
class Case:
    name: str
    seed: int
    swapped: bool
    trace: list[int]
    digest: str
    family: int
    groups: tuple[list[int], list[int]]
    signatures: list[str]
    raw_json: str
    obs: dict[str, torch.Tensor]
    successors: tuple[str, str]

    def receipt(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if key != "obs"}


def _keywords(value: managym.Keywords) -> list[bool]:
    return [bool(getattr(value, key)) for key in KEYWORDS]


def _signatures(raw: managym.Observation) -> list[str]:
    """Conservative observable classes; equal copies share probability mass.

    The descriptor is independent of the tensor encoder. Equal descriptors do
    not prove full game-theoretic equivalence; relational distinctions outside
    this descriptor are explicitly outside the search's coverage.
    """
    objects: dict[int, list[object]] = {
        raw.agent.id: ["player", "self"],
        raw.opponent.id: ["player", "opponent"],
    }
    for role, cards, permanents in (
        ("self", raw.agent_cards, raw.agent_permanents),
        ("opponent", raw.opponent_cards, raw.opponent_permanents),
    ):
        for card in cards:
            objects[card.id] = [
                "card",
                role,
                card.registry_key,
                int(card.zone),
                card.power,
                card.toughness,
                card.is_token,
                card.is_ally,
                card.is_lesson,
                card.ward_cost,
                card.kicker_cost,
                _keywords(card.keywords),
            ]
        for permanent in permanents:
            objects[permanent.id] = [
                "permanent",
                role,
                permanent.registry_key,
                permanent.tapped,
                permanent.damage,
                permanent.is_summoning_sick,
                permanent.plus1_counters,
                permanent.cant_be_blocked_this_turn,
                permanent.power,
                permanent.toughness,
                permanent.is_animated,
                permanent.has_exile_link,
                _keywords(permanent.keywords),
            ]
    for card in raw.agent_sideboard:
        objects[card.candidate_id] = ["outside", "self", card.registry_key]
    # Unknown references are errors, not labels derived from arbitrary IDs.
    return [
        json.dumps(
            [
                int(action.action_type),
                action.declared,
                action.mode_index,
                action.ability_index,
                [objects[index] for index in action.focus],
            ],
            separators=(",", ":"),
        )
        for action in raw.action_space.actions
    ]


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _flatten(value: object) -> list[str]:
    if isinstance(value, list):
        return [part for item in value for part in _flatten(item)]
    return [repr(value)]


def _pair(groups: dict[str, list[int]], closest: bool) -> tuple[list[int], list[int]]:
    keys = sorted(groups)
    if not closest:
        return groups[keys[0]], groups[keys[-1]]

    def distance(pair: tuple[str, str]) -> tuple[bool, int]:
        left, right = (_flatten(json.loads(key)) for key in pair)
        # Prefer the same verb, then the fewest differing public fields.
        # Registry keys are categorical identities, never a numeric distance.
        return left[0] != right[0], sum(a != b for a, b in zip_longest(left, right))

    left, right = min(combinations(keys, 2), key=distance)
    return groups[left], groups[right]


def collect_choices(
    out: Path, round_index: int, deadline: float, closest: bool = False
) -> tuple[list[Case], Counter[str], list[dict[str, object]]]:
    rng = random.Random(50_000 + round_index)
    buckets: dict[int, list[Case]] = defaultdict(list)
    counts: Counter[str] = Counter()
    findings: list[dict[str, object]] = []
    # Eight whole-game prefixes per round; both seats are visited naturally.
    for game in range(8):
        seed = 50_000 + round_index * 8 + game
        swapped = bool(game % 2)
        configs = [
            managym.authored_deck_setup(PACK, key)
            for key in ("ur_lessons", "gw_allies")
        ]
        if swapped:
            configs.reverse()
        env = managym.Env(seed=seed, skip_trivial=False)
        raw, _ = env.reset(configs)
        trace: list[int] = []
        for step in range(1200):
            if time.perf_counter() >= deadline:
                break
            actions = raw.action_space.actions
            if not actions or raw.game_over:
                counts["terminal_games"] += 1
                break
            family = int(raw.action_space.action_space_type)
            counts[f"visited_family_{family}"] += 1
            if len(actions) > 1:
                try:
                    signatures = _signatures(raw)
                    groups: dict[str, list[int]] = defaultdict(list)
                    for index, signature in enumerate(signatures):
                        groups[signature].append(index)
                    if len(groups) > 1:
                        obs = encode(env, raw)
                        assert int(obs["actions_valid"].sum()) == len(actions), (
                            "legal mask count"
                        )
                        for index, action in enumerate(actions):
                            assert (
                                obs["action_focus"][0, index, : len(action.focus)] >= 0
                            ).all(), "unresolved focus"
                        # Reverse native offers before running both encoders.
                        space, original = raw.action_space, actions
                        space.actions = original[::-1]
                        raw.action_space = space
                        try:
                            reversed_obs = encode(env, raw)
                        finally:
                            space.actions = original
                            raw.action_space = space
                        expected = reorder_offers(
                            obs, torch.arange(len(actions) - 1, -1, -1)
                        )
                        for key in obs:
                            torch.testing.assert_close(
                                reversed_obs[key], expected[key], rtol=0, atol=0
                            )
                        counts[f"audited_family_{family}"] += 1
                        selected = _pair(groups, closest)
                        # Reservoir sampling within each family avoids priority dominance.
                        seen = counts[f"audited_family_{family}"]
                        slot = rng.randrange(seen)
                        if len(buckets[family]) < 6 or slot < 6:
                            root = env.state_digest()
                            left, right = env.clone_env(), env.clone_env()
                            left.step(selected[0][0])
                            right.step(selected[1][0])
                            assert env.state_digest() == root, "clone mutated root"
                            case = Case(
                                f"{seed}-{step}",
                                seed,
                                swapped,
                                trace.copy(),
                                root,
                                family,
                                selected,
                                signatures,
                                raw.toJSON(),
                                obs,
                                (left.state_digest(), right.state_digest()),
                            )
                            if len(buckets[family]) < 6:
                                buckets[family].append(case)
                            else:
                                buckets[family][slot] = case
                except Exception as error:
                    finding = {
                        "stage": "native_audit",
                        "seed": seed,
                        "swapped": swapped,
                        "trace": trace.copy(),
                        "family": family,
                        "error": f"{type(error).__name__}: {error}",
                        "raw": raw.toJSON(),
                    }
                    findings.append(finding)
                    # Keep the first 20 detailed errors per round; count every occurrence.
                    counts["native_errors"] += 1
                    if len(findings) > 20:
                        findings.pop()
            # Encourage spell/combat exposure without using an expert or hidden state.
            weights = [4 if int(a.action_type) in {0, 1, 6} else 1 for a in actions]
            index = rng.choices(range(len(actions)), weights=weights, k=1)[0]
            trace.append(index)
            try:
                raw, _, done, truncated, _ = env.step(index)
            except Exception as error:
                findings.append(
                    {
                        "stage": "native_step",
                        "seed": seed,
                        "swapped": swapped,
                        "trace": trace.copy(),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                break
            if done or truncated:
                counts["terminal_games"] += int(done)
                break
        else:
            counts["step_capped_games"] += 1
    # Round-robin families so rare decision types get shared-fit slots.
    cases = [
        bucket[index]
        for index in range(6)
        for _, bucket in sorted(buckets.items())
        if index < len(bucket)
    ][:32]
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "cases.json", [case.receipt() for case in cases])
    _write_json(out / "native-findings.json", findings)
    torch.save({case.name: case.obs for case in cases}, out / "observations.pt")
    return cases, counts, findings


Mutation = Literal["clean", "without_declaration", "without_definition"]


@dataclass(frozen=True)
class ObjectInput:
    kind: str
    role: str
    features: tuple[float, ...]


@dataclass(frozen=True)
class ActionInput:
    features: tuple[float, ...]
    focus: tuple[ObjectInput | None, ...]


def action_inputs(
    obs: dict[str, torch.Tensor], count: int, mutation: Mutation
) -> list[ActionInput]:
    rows: list[ObjectInput] = []
    for key in (
        "agent_player",
        "opponent_player",
        "agent_cards",
        "opponent_cards",
        "agent_permanents",
        "opponent_permanents",
    ):
        role, kind = key.split("_")
        for slot, tensor in enumerate(obs[key][0]):
            row: list[float] = tensor.tolist()
            if kind == "cards":
                definition = float(
                    obs["semantic_cards"][0, int(role == "opponent"), slot]
                )
                row.append(0.0 if mutation == "without_definition" else definition)
            elif kind == "permanents" and mutation == "without_definition":
                row[PERMANENT_DEFINITION_INDEX] = 0.0
            rows.append(ObjectInput(kind, role, tuple(row)))
    result: list[ActionInput] = []
    for index in range(count):
        action: list[float] = obs["actions"][0, index, :-1].tolist()
        if mutation == "without_declaration":
            action[ACTION_DECLARED_INDEX] = 0.0
        focus = tuple(
            rows[int(i)] if i >= 0 else None
            for i in obs["action_focus"][0, index].tolist()
        )
        result.append(ActionInput(tuple(action), focus))
    return result


def input_collisions(case: Case) -> list[list[int]]:
    """Different public choices receiving identical modeled inputs are findings."""
    groups: dict[ActionInput, list[int]] = defaultdict(list)
    for index, fingerprint in enumerate(
        action_inputs(case.obs, len(case.signatures), "clean")
    ):
        groups[fingerprint].append(index)
    return [
        group
        for group in groups.values()
        if len({case.signatures[i] for i in group}) > 1
    ]
