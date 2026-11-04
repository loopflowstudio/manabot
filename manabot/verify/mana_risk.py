"""Bounded mana policy probes: paired observations and CPU forward timings.

These are constructed positions and forward/backward diagnostics, never an RL
experiment. Pair labels name the only varied mana fact; old-input projection
and native parity remain independent checks. Run with
`uv run python -m manabot.verify.mana_risk --out <file>`.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import TypedDict

import numpy as np
import torch

from manabot.env.observation import (
    FACT_ID_INDEX,
    FACT_KIND_INDEX,
    FACT_MANA_AMOUNT,
    FACT_MANA_LOCATION,
    FACT_VALID_INDEX,
    FACT_VALUE_INDEX,
    RELATION_MANA_TAPS_INDEX,
    ObservationSpace,
)
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.managym.enums import ActionEnum
from manabot.model.actor_critic import ActorCritic
from manabot.verify.text_observation import priority_root
import managym


@dataclass(frozen=True)
class ManaPair:
    name: str
    left: dict[str, np.ndarray]
    right: dict[str, np.ndarray]
    equivalent: bool = False


class ProjectionCost(TypedDict):
    case: str
    facts: int
    fact_capacity: int
    fact_bytes: int
    total_input_bytes: int
    python_microseconds: list[float]
    native_microseconds: list[float]


class InputCost(TypedDict):
    pair: str
    version: int
    round: int
    facts_microseconds: float
    model_microseconds: float


def mana_pairs(space: ObservationSpace) -> list[ManaPair]:
    """Hold cards, actions and pooled total fixed while varying mana joins."""
    env = priority_root(("Gray Ogre", "Hill Giant"), ("Mountain",) * 4)
    payload = json.loads(env.semantic_observation_json(env.current_agent_index()))[
        "viewer_state"
    ]
    payload["agent"]["lesson_mana"] = [0, 3, 0, 0, 0, 0]

    def encoded(classes: list[dict[str, object]]) -> dict[str, np.ndarray]:
        payload["agent"]["restricted_mana"] = classes
        return space.encode(managym.Observation.from_json(json.dumps(payload)))

    def restriction(
        names: list[str], amount: int, purpose: str = "spells"
    ) -> dict[str, object]:
        return {purpose: names, "mana": [0, amount, 0, 0, 0, 0]}

    base = encoded([restriction(["Citizen"], 2), restriction(["Elf"], 1)])
    swapped = encoded([restriction(["Citizen"], 1), restriction(["Elf"], 2)])
    purpose = encoded(
        [restriction(["Citizen"], 2, "abilities"), restriction(["Elf"], 1)]
    )
    combined = encoded([restriction(["Citizen", "Elf"], 3)])
    reordered = {key: value.copy() for key, value in base.items()}
    reordered["facts"] = reordered["facts"][::-1].copy()
    renumbered = {key: value.copy() for key, value in base.items()}
    facts = renumbered["facts"]
    mana = (facts[:, FACT_KIND_INDEX] >= FACT_MANA_AMOUNT) & (
        facts[:, FACT_VALID_INDEX] == 1
    )
    # IDs are joins, not quantity or strategic order. Swap their labels.
    facts[mana, FACT_ID_INDEX + 1] = 3 - facts[mana, FACT_ID_INDEX + 1]
    pairs = [
        ManaPair("amount_assignment", base, swapped),
        ManaPair("purpose", base, purpose),
        ManaPair("combined_classes", base, combined),
        ManaPair("row_order", base, reordered, True),
        ManaPair("class_labels", base, renumbered, True),
    ]
    pending_env = priority_root((), ("Hermitic Herbalist",))
    obs = pending_env.scenario_refresh()
    choice = next(
        i
        for i, action in enumerate(obs.action_space.actions)
        if int(action.action_type) == ActionEnum.PRIORITY_ACTIVATE_ABILITY
        and action.ability_index == 1
    )
    pending, *_ = pending_env.step(choice)
    output = space.encode(pending)
    for name, column in (
        ("output_mode", FACT_VALUE_INDEX + 1),
        ("output_frame", FACT_VALUE_INDEX),
    ):
        altered = {key: value.copy() for key, value in output.items()}
        locations = altered["facts"][:, FACT_KIND_INDEX] == FACT_MANA_LOCATION
        altered["facts"][locations, column] += 1
        pairs.append(ManaPair(name, output, altered))
    tap_env = priority_root((), ("Supportive Parents", "Gray Ogre", "Gray Ogre"))
    obs = tap_env.scenario_refresh()
    choice = next(
        i
        for i, action in enumerate(obs.action_space.actions)
        if int(action.action_type) == ActionEnum.PRIORITY_ACTIVATE_ABILITY
    )
    pending, *_ = tap_env.step(choice)
    taps = space.encode(pending)
    fewer = {key: value.copy() for key, value in taps.items()}
    remaining = fewer["relations"][:, RELATION_MANA_TAPS_INDEX]
    remaining[remaining == 2] = 1
    pairs.append(ManaPair("remaining_taps", taps, fewer))
    return pairs


def tensor_batch(*observations: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        key: torch.from_numpy(np.stack([obs[key] for obs in observations]))
        for key in observations[0]
    }


def empty_fact_objects(model: ActorCritic, batch_size: int) -> torch.Tensor:
    """Isolate the fact adapter using the model's configured object capacities."""
    encoder = model.observation_space.encoder
    count = 2 + 2 * encoder.cards_per_player + 2 * encoder.perms_per_player
    return next(model.parameters()).new_zeros(
        (batch_size, count, model.config.hidden_dim)
    )


def open_fact_adapter(model: ActorCritic) -> None:
    """Probe structural capacity after the intentional zero output gate opens."""
    assert model.facts is not None
    with torch.no_grad():
        torch.nn.init.eye_(model.facts.output.weight)
        model.facts.output.bias.zero_()
        if model.policy_film is not None:
            torch.nn.init.eye_(model.policy_film.weight)
            model.policy_film.bias.zero_()


def projection_costs(version: int) -> list[ProjectionCost]:
    """Time real scalar projections, excluding fixture creation and warm-up."""
    space = ObservationSpace(
        ObservationConfig.model_validate({"policy_input_version": version})
    )
    results: list[ProjectionCost] = []
    for name, board in (
        ("fixed", ("Mountain",) * 3),
        ("shared_taps", ("Supportive Parents", "Gray Ogre", "Gray Ogre")),
        ("prism", ("Prophetic Prism", "Mountain")),
        ("variable", ("White Lotus Tile", "Gray Ogre", "Gray Ogre")),
        ("restricted_outputs", ("Hermitic Herbalist",)),
    ):
        env = priority_root((), board)
        obs = env.scenario_refresh()
        choices = [
            i
            for i, action in enumerate(obs.action_space.actions)
            if int(action.action_type) == ActionEnum.PRIORITY_ACTIVATE_ABILITY
        ]
        if choices:
            obs, *_ = env.step(choices[-1])
        encoded = space.encode(obs)
        native = env.encode_observation(obs, version)
        for key in encoded:
            np.testing.assert_array_equal(encoded[key], native[key])
        python_times: list[float] = []
        native_times: list[float] = []
        for _ in range(10):
            start = perf_counter()
            space.encode(obs)
            python_times.append((perf_counter() - start) * 1e6)
            start = perf_counter()
            env.encode_observation(obs, version)
            native_times.append((perf_counter() - start) * 1e6)
        occupied = int(encoded["facts"][:, FACT_VALID_INDEX].sum())
        results.append(
            {
                "case": name,
                "facts": occupied,
                "fact_capacity": space.encoder.config.fact_rows,
                "fact_bytes": encoded["facts"].nbytes,
                "total_input_bytes": sum(row.nbytes for row in encoded.values()),
                "python_microseconds": python_times,
                "native_microseconds": native_times,
            }
        )
    return results


def compare_input_costs() -> list[InputCost]:
    """Alternate v8/v9 in one process, including the isolated fact adapter cost."""
    models: dict[int, ActorCritic] = {}
    for version in (8, 9):
        torch.manual_seed(242)
        space = ObservationSpace(
            ObservationConfig.model_validate({"policy_input_version": version})
        )
        model = ActorCritic(
            space,
            ModelConfig(hidden_dim=16, policy_film=True, world=managym.WORLD_SETUP),
        ).eval()
        open_fact_adapter(model)
        models[version] = model
    results: list[InputCost] = []
    for pair in mana_pairs(models[8].observation_space):
        if pair.name not in ("amount_assignment", "output_mode", "remaining_taps"):
            continue
        batch = tensor_batch(pair.left, pair.right)
        objects = empty_fact_objects(models[8], 2)
        with torch.no_grad():
            for model in models.values():
                model(batch)
            for round_number in range(10):
                for version in (8, 9) if round_number % 2 == 0 else (9, 8):
                    model = models[version]
                    assert model.facts is not None
                    start = perf_counter()
                    model.facts(batch["facts"], objects)
                    fact_time = (perf_counter() - start) * 1e6
                    start = perf_counter()
                    model(batch)
                    full_time = (perf_counter() - start) * 1e6
                    results.append(
                        {
                            "pair": pair.name,
                            "version": version,
                            "round": round_number,
                            "facts_microseconds": fact_time,
                            "model_microseconds": full_time,
                        }
                    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--version", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(242)
    space = ObservationSpace(
        ObservationConfig.model_validate({"policy_input_version": args.version})
    )
    model = ActorCritic(
        space, ModelConfig(hidden_dim=16, policy_film=True, world=managym.WORLD_SETUP)
    ).eval()
    pairs = mana_pairs(space)
    open_fact_adapter(model)
    rows: list[dict[str, object]] = []
    for pair in pairs:
        batch = tensor_batch(pair.left, pair.right)
        times = []
        with torch.no_grad():
            model(batch)
            for _ in range(10):
                start = perf_counter()
                logits, values = model(batch)
                times.append((perf_counter() - start) * 1e6)
            assert model.facts is not None
            objects = empty_fact_objects(model, 2)
            represented = model.facts(batch["facts"], objects)
        backward_times = []
        for _ in range(3):
            model.zero_grad(set_to_none=True)
            start = perf_counter()
            train_logits, train_values = model(batch)
            (
                train_logits[batch["actions_valid"].bool()].square().mean()
                + train_values.square().mean()
            ).backward()
            backward_times.append((perf_counter() - start) * 1e6)
        valid = batch["actions_valid"][0].bool()
        rows.append(
            {
                "pair": pair.name,
                "equivalent": pair.equivalent,
                "representation_delta": (represented[0] - represented[1])
                .abs()
                .max()
                .item(),
                "logit_delta": (logits[0] - logits[1]).abs().max().item(),
                "value_delta": (values[0] - values[1]).abs().max().item(),
                "log_probability_delta": (
                    logits[0, valid].log_softmax(0) - logits[1, valid].log_softmax(0)
                )
                .abs()
                .max()
                .item(),
                "fact_rows": [
                    int(obs["facts"][:, FACT_VALID_INDEX].sum())
                    for obs in (pair.left, pair.right)
                ],
                "forward_microseconds_batch2": times,
                "forward_backward_microseconds_batch2": backward_times,
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "version": args.version,
                "seed": 242,
                "hidden_dim": 16,
                "threads": 1,
                "policy_film": True,
                "observation_config": space.encoder.config.model_dump(mode="json"),
                "model_config": model.config.model_dump(mode="json"),
                "adapter_probe": "identity output and FiLM gates, no optimizer",
                "pairs": rows,
                "projections": projection_costs(args.version),
                "alternating_costs": compare_input_costs() if args.version == 9 else [],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
