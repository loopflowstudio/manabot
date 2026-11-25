"""Replayable native choices for capacity tests and frozen-policy inspection.

Collect model-independent authored-game prefixes, retain seed/action traces and
encoded observations, and select public action classes without arbitrary instance IDs.
This corpus contains no strategic labels and never trains a model.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from itertools import combinations, zip_longest
import json
from pathlib import Path
import random
import time
from typing import Literal

import numpy as np

from manabot.env.observation import (
    ACTION_DECLARED_INDEX,
    ACTION_VALID_INDEX,
    PERMANENT_DEFINITION_INDEX,
    STACK_DEFINITION_INDEX,
    ObservationSpace,
)
from manabot.policy.spans import ObservationSelection
from manabot.verify.choice_inputs import encode, reorder_offers
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
    raw: managym.Observation
    obs: ObservationSelection
    successors: tuple[str, str]

    def receipt(self) -> dict[str, object]:
        return {
            "raw_json": self.raw.toJSON(),
            **{
                field.name: getattr(self, field.name)
                for field in fields(self)
                if field.name not in {"obs", "raw"}
            },
        }


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
                card.characteristics.power,
                card.characteristics.toughness,
                card.is_token,
                card.characteristics.is_ally,
                card.characteristics.is_lesson,
                card.characteristics.ward_cost,
                card.characteristics.kicker_cost,
                _keywords(card.characteristics.keywords),
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
                permanent.characteristics.power,
                permanent.characteristics.toughness,
                permanent.is_animated,
                permanent.has_exile_link,
                _keywords(permanent.characteristics.keywords),
            ]
            if permanent.object_id is not None:
                objects[permanent.object_id] = objects[permanent.id]
    for item in raw.stack_objects:
        # Spell card ids and stack ids name the same public object, as in v11.
        descriptor: list[object] = [
            "stack",
            "self" if item.controller_id == raw.agent.id else "opponent",
            int(item.kind),
            item.source_card_registry_key,
            item.ability_index,
            item.chosen_mode,
            item.announced_x,
            item.x,
            item.kicked,
            item.is_copy,
        ]
        objects[item.stack_object_id] = descriptor
        if item.object_id is not None:
            objects[item.object_id] = descriptor
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

    fields = {key: _flatten(json.loads(key)) for key in keys}

    def distance(pair: tuple[str, str]) -> tuple[bool, int]:
        left, right = fields[pair[0]], fields[pair[1]]
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
    observation_space = ObservationSpace()
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
        env = managym.Env(seed=seed)
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
                        obs = encode(raw, observation_space)
                        assert obs.action_counts.tolist() == [len(actions)], (
                            "legal mask count"
                        )
                        focus = obs.spans.tables["action_focus"].records
                        for row, action in zip(focus, actions, strict=True):
                            assert (row[: len(action.focus)] >= 0).all(), (
                                "unresolved focus"
                            )
                        order = np.arange(len(actions) - 1, -1, -1)
                        with reorder_offers(raw, order):
                            reversed_obs = encode(raw, observation_space)
                        # Compare each packed table independently of the encoder's
                        # join resolution. Only offer-aligned rows may move.
                        aligned = {
                            key for key, _ in observation_space.axis_buffers["actions"]
                        }
                        for key, table in obs.spans.tables.items():
                            expected = (
                                table.records[order]
                                if key in aligned
                                else table.records
                            )
                            np.testing.assert_array_equal(
                                reversed_obs.spans.tables[key].records,
                                expected,
                                strict=True,
                            )
                        counts[f"audited_family_{family}"] += 1
                        # Reservoir sampling within each family avoids priority dominance.
                        seen = counts[f"audited_family_{family}"]
                        slot = rng.randrange(seen)
                        if len(buckets[family]) < 6 or slot < 6:
                            selected = _pair(groups, closest)
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
                                raw,
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
    np.savez(
        out / "observations.npz",
        **{
            case.name: np.frombuffer(case.obs.batch.to_bytes(), np.uint8)
            for case in cases
        },
    )
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
    obs: ObservationSelection, count: int, mutation: Mutation
) -> list[ActionInput]:
    rows: list[ObjectInput] = []
    definitions = obs.project("semantic_cards")[0]
    for key in (
        "agent_player",
        "opponent_player",
        "agent_cards",
        "opponent_cards",
        "agent_permanents",
        "opponent_permanents",
    ):
        role, kind = key.split("_")
        for slot, tensor in enumerate(obs.project(key)[0]):
            row: list[float] = tensor.tolist()
            if kind == "cards":
                definition = float(definitions[int(role == "opponent"), slot])
                row.append(0.0 if mutation == "without_definition" else definition)
            elif kind == "permanents" and mutation == "without_definition":
                row[PERMANENT_DEFINITION_INDEX] = 0.0
            rows.append(ObjectInput(kind, role, tuple(row)))
    for tensor in obs.project("stack")[0]:
        stack_row: list[float] = tensor.tolist()
        if mutation == "without_definition":
            stack_row[STACK_DEFINITION_INDEX] = 0.0
        rows.append(ObjectInput("stack", "shared", tuple(stack_row)))
    result: list[ActionInput] = []
    actions = obs.spans.tables["actions"].features()
    focuses = obs.spans.tables["action_focus"].records
    for index in range(count):
        # Every action column except validity, which is 1 for every offer.
        action: list[float] = np.delete(actions[index], ACTION_VALID_INDEX).tolist()
        if mutation == "without_declaration":
            action[ACTION_DECLARED_INDEX] = 0.0
        focus = tuple(rows[int(i)] if i >= 0 else None for i in focuses[index].tolist())
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
