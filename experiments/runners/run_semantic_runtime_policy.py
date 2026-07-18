"""Train and evaluate the INT-11 learned semantic runtime policy.

One bounded CPU command regenerates authoritative decisions, trains matched
semantic controls, executes their structured Commands, plays a paired-seat
micro-matchup, and writes content-addressed replayable artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter, perf_counter_ns, process_time
from typing import Any, Mapping, Sequence, cast

import torch

from etude.experience_protocol import Command
from manabot.semantic import SemanticDecision, SemanticDecisionAdapter
from manabot.semantic.runtime_policy import (
    POLICY_ARMS,
    PolicyArm,
    PolicyTargets,
    RuntimePolicyFeatures,
    RuntimePolicyProjector,
    SemanticRuntimePolicy,
    canonical_json,
    canonical_sha256,
    parameter_count,
    percentile,
    targets_from_submission,
)
from manabot.sim.structured_policy import DecodedSubmission, RaggedPolicyDecoder
import managym

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKLOAD = ROOT / "experiments/workloads/int-11-semantic-runtime-policy-v1.json"
DEFAULT_OUT_DIR = ROOT / ".runs/int-11-semantic-runtime-policy-v1"
RUNNER_SCHEMA_VERSION = 1
ASSET_MANIFEST_HASH = "int-11-no-player-assets"

TRAIN_CREATURES = (
    "Badgermole Cub",
    "Compassionate Healer",
    "First-Time Flyer",
    "Forecasting Fortune Teller",
    "Otter-Penguin",
    "White Lotus Reinforcements",
)
IDENTITY_HOLDOUT = "Fire Nation Cadets"
COMPOSITION_HOLDOUT = "South Pole Voyager"
TARGET_SPELL = "Igneous Inspiration"
ARENA_CREATURE = "Otter-Penguin"
ARENA_COMMAND_CAP = 32


class RuntimeExperimentError(RuntimeError):
    """The checked experiment contract or an authority invariant failed."""


@dataclass(frozen=True)
class RuntimeExample:
    example_id: str
    split: str
    recipe: Mapping[str, Any]
    features: RuntimePolicyFeatures
    targets: PolicyTargets
    oracle_submission: DecodedSubmission
    artifact: Mapping[str, Any]


@dataclass(frozen=True)
class TrainedPolicy:
    arm: PolicyArm
    seed: int
    model: SemanticRuntimePolicy
    checkpoint: Mapping[str, Any]
    training: Mapping[str, Any]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> str:
    payload = (canonical_json(value) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha256_bytes(payload)


def load_workload(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    workload = json.loads(raw)
    validate_workload(workload)
    return workload, sha256_bytes(raw)


def validate_workload(workload: Mapping[str, Any]) -> None:
    if workload.get("schema_version") != 1:
        raise RuntimeExperimentError("workload schema_version must be 1")
    if workload.get("world") != "w2":
        raise RuntimeExperimentError("runtime policy workload must remain in w2")
    if tuple(workload.get("arms", ())) != POLICY_ARMS:
        raise RuntimeExperimentError("workload arms do not match policy implementation")
    seeds = workload.get("model_seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 3
        or len(set(seeds)) != len(seeds)
        or not all(isinstance(seed, int) and seed >= 0 for seed in seeds)
    ):
        raise RuntimeExperimentError("model_seeds must contain >=3 distinct integers")
    dataset = workload.get("dataset")
    if not isinstance(dataset, Mapping):
        raise RuntimeExperimentError("workload dataset must be an object")
    if dataset.get("identity_holdout") != "tla.fire_nation_cadets":
        raise RuntimeExperimentError("identity holdout changed")
    if dataset.get("composition_holdout") != "tla.south_pole_voyager":
        raise RuntimeExperimentError("composition holdout changed")
    if set(dataset.get("composition_known_operations", ())) != {
        "gain_life",
        "branch",
    }:
        raise RuntimeExperimentError("composition holdout must use known operations")
    if int(dataset.get("minimum_target_candidates", 0)) <= 32:
        raise RuntimeExperimentError("target frontier must exceed 32 choices")
    if int(dataset.get("minimum_combat_branches", 0)) <= 32:
        raise RuntimeExperimentError("combat frontier must exceed 32 branches")
    training = workload.get("training")
    if not isinstance(training, Mapping):
        raise RuntimeExperimentError("workload training must be an object")
    for field in ("epochs", "learning_rate", "max_wall_clock_seconds"):
        if not isinstance(training.get(field), (int, float)) or training[field] <= 0:
            raise RuntimeExperimentError(f"training.{field} must be positive")
    gates = workload.get("gates")
    if not isinstance(gates, Mapping) or gates.get("illegal_commands") != 0:
        raise RuntimeExperimentError("zero illegal Commands is a required gate")
    evaluation = workload.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise RuntimeExperimentError("workload evaluation must be an object")
    if evaluation.get("arena_combat_fixture") != "tla.otter_penguin":
        raise RuntimeExperimentError("arena combat fixture changed")
    if evaluation.get("arena_command_cap") != ARENA_COMMAND_CAP:
        raise RuntimeExperimentError("arena Command cap changed")


def _deck(names: Sequence[str], *, minimum: int = 40) -> dict[str, int]:
    counts = Counter(names)
    counts["Mountain"] += max(0, minimum - sum(counts.values()))
    return dict(counts)


def _pass_index(observation: Any) -> int:
    return next(
        (
            index
            for index, action in enumerate(observation.action_space.actions)
            if action.action_type == managym.ActionEnum.PRIORITY_PASS_PRIORITY
        ),
        0,
    )


def _target_root(recipe: Mapping[str, Any]) -> tuple[managym.Env, Any]:
    target_count = int(recipe["target_count"])
    target_name = str(recipe.get("target_name", IDENTITY_HOLDOUT))
    if not 1 <= target_count <= 33:
        raise RuntimeExperimentError("target_count must be 1..33")
    env = managym.Env(seed=int(recipe["seed"]), skip_trivial=False)
    env.reset(
        [
            managym.PlayerConfig("caster", {TARGET_SPELL: 4, "Mountain": 36}),
            managym.PlayerConfig(
                "target", _deck([target_name] * target_count, minimum=40)
            ),
        ]
    )
    env.scenario_clear_hand(0)
    env.scenario_force_card_in_hand(0, TARGET_SPELL)
    for _ in range(3):
        env.scenario_force_battlefield(0, "Mountain")
    for _ in range(target_count):
        env.scenario_force_battlefield(1, target_name)
    observation = env.scenario_refresh()
    for _ in range(100):
        if int(observation.agent.player_index) == 0 and any(
            action.action_type == managym.ActionEnum.PRIORITY_CAST_SPELL
            for action in observation.action_space.actions
        ):
            return env, observation
        observation, _, done, _, _ = env.step(_pass_index(observation))
        if done:
            break
    raise RuntimeExperimentError("target recipe did not reach a castable root")


def _combat_root(recipe: Mapping[str, Any]) -> tuple[managym.Env, Any]:
    creatures = tuple(str(value) for value in recipe["creatures"])
    if not 1 <= len(creatures) <= 6:
        raise RuntimeExperimentError("combat recipe must contain 1..6 creatures")
    env = managym.Env(
        seed=int(recipe["seed"]),
        skip_trivial=bool(recipe.get("skip_trivial", False)),
    )
    observation, _ = env.reset(
        [
            managym.PlayerConfig("attacker", _deck(creatures, minimum=40)),
            managym.PlayerConfig("defender", {"Mountain": 40}),
        ]
    )
    env.scenario_clear_hand(0)
    env.scenario_clear_hand(1)
    for name in creatures:
        env.scenario_force_battlefield(0, name, ready=True)
    if "opponent_life" in recipe:
        env.scenario_set_life(1, int(recipe["opponent_life"]))
    observation = env.scenario_refresh()
    for _ in range(100):
        if (
            observation.action_space.action_space_type
            == managym.ActionSpaceEnum.DECLARE_ATTACKER
        ):
            return env, observation
        observation, _, done, _, _ = env.step(_pass_index(observation))
        if done:
            break
    raise RuntimeExperimentError("combat recipe did not reach declare attackers")


def _pass_root(recipe: Mapping[str, Any]) -> tuple[managym.Env, Any]:
    env = managym.Env(seed=int(recipe["seed"]), skip_trivial=False)
    observation, _ = env.reset(
        [
            managym.PlayerConfig("active", {"Mountain": 40}),
            managym.PlayerConfig("other", {"Forest": 40}),
        ]
    )
    return env, observation


def reconstruct_root(recipe: Mapping[str, Any]) -> tuple[managym.Env, Any]:
    kind = recipe.get("kind")
    if kind == "target":
        return _target_root(recipe)
    if kind == "combat":
        return _combat_root(recipe)
    if kind == "pass":
        return _pass_root(recipe)
    raise RuntimeExperimentError(f"unknown scenario kind {kind!r}")


def bind_recipe(
    recipe: Mapping[str, Any], adapter: SemanticDecisionAdapter
) -> tuple[managym.Env, SemanticDecision]:
    env, observation = reconstruct_root(recipe)
    decision = adapter.bind(
        env,
        observation,
        match_id=str(recipe["example_id"]),
        revision=int(recipe["revision"]),
        content_hash=adapter.pack.content_pack_hash,
        asset_manifest_hash=ASSET_MANIFEST_HASH,
    )
    return env, decision


def _attack_utility(adapter: SemanticDecisionAdapter, definition_row: int) -> int:
    weights = {
        "add_mana": 2,
        "modify_pt": 2,
        "restrict_blocking": 3,
        "branch": 1,
        "set_power_from_count": 2,
        "gain_life": -1,
        "scry": -1,
        "create_token": 0,
    }
    total = 0
    for program_row in adapter.pack.program_rows(definition_row):
        for instruction in adapter.pack.ir.programs[program_row]["instructions"]:
            total += weights.get(str(instruction["op_name"]), 0)
    return total


def oracle_submission(
    decision: SemanticDecision,
    recipe: Mapping[str, Any],
    adapter: SemanticDecisionAdapter,
) -> DecodedSubmission:
    kind = recipe["kind"]
    if kind == "pass":
        offer = next(
            offer for offer in decision.batch.offers if offer["verb"] == "pass_priority"
        )
        return DecodedSubmission(int(offer["id"]), ())

    if kind == "target":
        offer_index = next(
            index
            for index, offer in enumerate(decision.batch.offers)
            if offer["verb"] == "cast"
        )
        rows = decision.batch.choices[
            decision.batch.choice_offsets[offer_index] : decision.batch.choice_offsets[
                offer_index + 1
            ]
        ]
        if len(rows) != 1:
            raise RuntimeExperimentError("target oracle expected one choice role")
        row = rows[0]
        candidates = range(row.candidate_start, row.candidate_stop)
        selected_index: int | None = None
        for candidate_index in candidates:
            binding = decision.candidate_subjects[candidate_index]
            if bool(recipe["target_player"]):
                if binding.player_index == 1:
                    selected_index = candidate_index
                    break
            elif binding.object_row is not None:
                runtime = decision.objects[binding.object_row]
                if runtime.controller == 1 and runtime.zone == "battlefield":
                    selected_index = candidate_index
                    break
        if selected_index is None:
            raise RuntimeExperimentError("target oracle found no requested subject")
        return DecodedSubmission(
            int(decision.batch.offers[offer_index]["id"]),
            (
                {
                    "kind": "candidates",
                    "role": row.role,
                    "candidates": [
                        int(decision.batch.candidates[selected_index]["id"])
                    ],
                },
            ),
        )

    offer_index = next(
        index
        for index, offer in enumerate(decision.batch.offers)
        if offer["verb"] == "declare_attackers"
    )
    rows = decision.batch.choices[
        decision.batch.choice_offsets[offer_index] : decision.batch.choice_offsets[
            offer_index + 1
        ]
    ]
    if len(rows) != 1:
        raise RuntimeExperimentError("combat oracle expected one choice role")
    row = rows[0]
    selected = []
    for candidate_index in range(row.candidate_start, row.candidate_stop):
        binding = decision.candidate_subjects[candidate_index]
        if binding.object_row is None:
            raise RuntimeExperimentError("attacker candidate is not an object")
        definition_row = decision.objects[binding.object_row].definition_row
        if _attack_utility(adapter, definition_row) > 0:
            selected.append(int(decision.batch.candidates[candidate_index]["id"]))
    return DecodedSubmission(
        int(decision.batch.offers[offer_index]["id"]),
        (
            {
                "kind": "candidates",
                "role": row.role,
                "candidates": selected,
            },
        ),
    )


def dataset_recipes(workload: Mapping[str, Any]) -> list[dict[str, Any]]:
    config = workload["dataset"]
    base_seed = int(config["seed"])
    recipes: list[dict[str, Any]] = []

    def append(split: str, kind: str, **values: Any) -> None:
        index = len(recipes)
        recipes.append(
            {
                "example_id": f"int11-{split}-{kind}-{index:03d}",
                "revision": index,
                "seed": base_seed + index,
                "split": split,
                "kind": kind,
                **values,
            }
        )

    repetitions = int(config["train_repetitions"])
    for repetition in range(repetitions):
        target_name = TRAIN_CREATURES[repetition % len(TRAIN_CREATURES)]
        append(
            "train",
            "target",
            target_count=3 + repetition * 3,
            target_name=target_name,
            target_player=bool(repetition % 2),
        )
        first = TRAIN_CREATURES[(repetition * 2) % len(TRAIN_CREATURES)]
        second = TRAIN_CREATURES[(repetition * 2 + 1) % len(TRAIN_CREATURES)]
        append("train", "combat", creatures=[first, second] * 3)
        append("train", "pass")

    for repetition in range(int(config["evaluation_repetitions"])):
        append(
            "evaluation",
            "target",
            target_count=8 + repetition,
            target_name=TRAIN_CREATURES[-1 - repetition],
            target_player=bool((repetition + 1) % 2),
        )
        append(
            "evaluation",
            "combat",
            creatures=[TRAIN_CREATURES[0], TRAIN_CREATURES[1 + repetition]] * 3,
        )
        append("evaluation", "pass")
        append(
            "identity_holdout",
            "combat",
            creatures=[IDENTITY_HOLDOUT, TRAIN_CREATURES[1]] * 3,
        )
        append(
            "composition_holdout",
            "combat",
            creatures=[COMPOSITION_HOLDOUT, TRAIN_CREATURES[0]] * 3,
        )

    append(
        "frontier",
        "target",
        target_count=33,
        target_name=IDENTITY_HOLDOUT,
        target_player=False,
    )
    append(
        "frontier",
        "combat",
        creatures=[IDENTITY_HOLDOUT] * 6,
    )
    return recipes


def _bootstrap_adapter() -> SemanticDecisionAdapter:
    env, _ = _pass_root({"seed": 0})
    return SemanticDecisionAdapter.from_env(env)


def generate_dataset(
    workload: Mapping[str, Any], adapter: SemanticDecisionAdapter
) -> list[RuntimeExample]:
    projector = RuntimePolicyProjector(adapter.pack)
    examples: list[RuntimeExample] = []
    seen_operations: set[str] = set()
    for recipe in dataset_recipes(workload):
        env, decision = bind_recipe(recipe, adapter)
        features = projector.project(decision)
        submission = oracle_submission(decision, recipe, adapter)
        targets = targets_from_submission(decision, submission)
        command = decision.command(
            submission, command_id=f"oracle:{recipe['example_id']}"
        )
        before = env.state_digest()
        decision.step(env, command)
        after = env.state_digest()
        if before == after:
            raise RuntimeExperimentError("oracle Command did not mutate authority")
        if recipe["split"] == "train":
            for definition_row in features.object_definitions.tolist():
                for program_row in adapter.pack.program_rows(int(definition_row)):
                    seen_operations.update(
                        str(instruction["op_name"])
                        for instruction in adapter.pack.ir.programs[program_row][
                            "instructions"
                        ]
                    )
        artifact = {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "example_id": recipe["example_id"],
            "split": recipe["split"],
            "recipe": dict(recipe),
            "source_digest": before,
            "post_command_digest": after,
            "frame": decision.frame.model_dump(mode="json", exclude_unset=True),
            "selected_offer": next(
                dict(offer)
                for offer in decision.batch.offers
                if int(offer["id"]) == submission.offer_id
            ),
            "command": command.model_dump(mode="json"),
            "feature_digest": features.digest,
            "semantic_pack_hash": decision.semantic_pack_hash,
        }
        examples.append(
            RuntimeExample(
                example_id=str(recipe["example_id"]),
                split=str(recipe["split"]),
                recipe=recipe,
                features=features,
                targets=targets,
                oracle_submission=submission,
                artifact=artifact,
            )
        )

    required = set(workload["dataset"]["composition_known_operations"])
    if not required <= seen_operations:
        raise RuntimeExperimentError(
            f"composition primitives absent from training: {sorted(required - seen_operations)}"
        )
    if any(
        key
        in {
            workload["dataset"]["identity_holdout"],
            workload["dataset"]["composition_holdout"],
        }
        for example in examples
        if example.split == "train"
        for key in (
            projector.catalog.definition_keys[index]
            for index in example.features.object_definitions.tolist()
        )
    ):
        raise RuntimeExperimentError("a held-out definition leaked into training")
    max_candidates = max(
        len(example.targets.candidate_selected) for example in examples
    )
    max_branches = 0
    for example in examples:
        if example.recipe["kind"] == "combat":
            max_branches = max(max_branches, 2 ** len(example.recipe["creatures"]))
    if max_candidates < int(workload["dataset"]["minimum_target_candidates"]):
        raise RuntimeExperimentError("dataset missed the target frontier")
    if max_branches < int(workload["dataset"]["minimum_combat_branches"]):
        raise RuntimeExperimentError("dataset missed the combat frontier")
    return examples


def decode_output(
    decision: SemanticDecision,
    output: Any,
) -> DecodedSubmission:
    return RaggedPolicyDecoder().decode(decision.batch, output.scores())


def evaluate_agreement(
    model: SemanticRuntimePolicy,
    examples: Sequence[RuntimeExample],
    adapter: SemanticDecisionAdapter,
) -> float:
    model.eval()
    correct = 0
    with torch.inference_mode():
        definitions = model.encode_definitions()
        for example in examples:
            _, decision = bind_recipe(example.recipe, adapter)
            decoded = decode_output(decision, model(example.features, definitions))
            if targets_from_submission(decision, decoded) == example.targets:
                correct += 1
    return correct / len(examples)


def train_policy(
    arm: PolicyArm,
    seed: int,
    examples: Sequence[RuntimeExample],
    projector: RuntimePolicyProjector,
    workload: Mapping[str, Any],
    out_dir: Path,
    dataset_digest: str,
    adapter: SemanticDecisionAdapter,
) -> TrainedPolicy:
    torch.manual_seed(seed)
    model_config = workload["model"]
    model = SemanticRuntimePolicy(
        projector.catalog,
        arm,
        hidden_dim=int(model_config["hidden_dim"]),
        attention_heads=int(model_config["attention_heads"]),
        transformer_layers=int(model_config["transformer_layers"]),
        dropout=float(model_config["dropout"]),
    )
    training = workload["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    epochs = int(training["epochs"])
    threshold = float(training["sample_efficiency_threshold"])
    losses: list[float] = []
    curve: list[dict[str, Any]] = []
    examples_to_threshold: int | None = None
    started = perf_counter()
    cpu_started = process_time()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        definitions = model.encode_definitions()
        epoch_losses = [
            model.loss(
                model(example.features, definitions),
                example.features,
                example.targets,
            )
            for example in examples
        ]
        loss = torch.stack(epoch_losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if epoch in {1, max(1, epochs // 4), max(1, epochs // 2), epochs}:
            agreement = evaluate_agreement(model, examples, adapter)
            curve.append(
                {
                    "epoch": epoch,
                    "examples_seen": epoch * len(examples),
                    "loss": losses[-1],
                    "exact_agreement": agreement,
                }
            )
            if agreement >= threshold and examples_to_threshold is None:
                examples_to_threshold = epoch * len(examples)
        if perf_counter() - started > float(training["max_wall_clock_seconds"]):
            raise RuntimeExperimentError(f"{arm}/{seed} exceeded the wall-clock cap")

    checkpoint_dir = out_dir / "checkpoints" / arm / str(seed)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    provisional = checkpoint_dir / "checkpoint.pt"
    checkpoint_manifest = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "world": workload["world"],
        "arm": arm,
        "seed": seed,
        "dataset_sha256": dataset_digest,
        "semantic_pack_hash": projector.catalog.semantic_pack_hash,
        "catalog_digest": projector.catalog.digest,
        "architecture": model.architecture,
        "epochs": epochs,
    }
    torch.save(
        {"manifest": checkpoint_manifest, "model_state": model.state_dict()},
        provisional,
    )
    checkpoint_sha = file_sha256(provisional)
    final_path = checkpoint_dir / f"sha256-{checkpoint_sha}.pt"
    provisional.replace(final_path)
    checkpoint = {
        **checkpoint_manifest,
        "path": str(final_path.relative_to(out_dir)),
        "sha256": checkpoint_sha,
        "bytes": final_path.stat().st_size,
    }
    timing = {
        "wall_seconds": perf_counter() - started,
        "cpu_seconds": process_time() - cpu_started,
        "optimizer_steps": epochs,
        "examples_per_epoch": len(examples),
        "examples_seen": epochs * len(examples),
        "examples_to_threshold": examples_to_threshold,
        "final_loss": losses[-1],
        "learning_curve": curve,
        "parameters": parameter_count(model),
    }
    return TrainedPolicy(arm, seed, model, checkpoint, timing)


def load_checkpoint_model(
    checkpoint: Mapping[str, Any],
    out_dir: Path,
    projector: RuntimePolicyProjector,
) -> SemanticRuntimePolicy:
    path = out_dir / str(checkpoint["path"])
    if not path.is_file():
        raise RuntimeExperimentError("checkpoint bytes are missing")
    if path.stat().st_size != int(checkpoint["bytes"]):
        raise RuntimeExperimentError("checkpoint byte count mismatch")
    if file_sha256(path) != checkpoint["sha256"]:
        raise RuntimeExperimentError("checkpoint SHA-256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected_manifest = {
        key: value
        for key, value in checkpoint.items()
        if key not in {"path", "sha256", "bytes"}
    }
    if payload.get("manifest") != expected_manifest:
        raise RuntimeExperimentError("checkpoint manifest mismatch")
    architecture = checkpoint["architecture"]
    model = SemanticRuntimePolicy(
        projector.catalog,
        cast(PolicyArm, checkpoint["arm"]),
        hidden_dim=int(architecture["hidden_dim"]),
        attention_heads=int(architecture["attention_heads"]),
        transformer_layers=int(architecture["transformer_layers"]),
        dropout=0.0,
    )
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model


def evaluate_policy(
    trained: TrainedPolicy,
    examples: Sequence[RuntimeExample],
    adapter: SemanticDecisionAdapter,
    projector: RuntimePolicyProjector,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model = trained.model
    model.eval()
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    competencies: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    replay_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        definitions = model.encode_definitions()
        for example in examples:
            env, decision = bind_recipe(example.recipe, adapter)
            features = projector.project(decision)
            if features.digest != example.features.digest:
                raise RuntimeExperimentError("regenerated viewer features drifted")
            output = model(features, definitions)
            loss = float(model.loss(output, features, example.targets))
            submission = decode_output(decision, output)
            predicted = targets_from_submission(decision, submission)
            before = env.state_digest()
            command = decision.command(
                submission,
                command_id=f"{trained.arm}:{trained.seed}:{example.example_id}",
            )
            try:
                decision.step(env, command)
            except Exception as error:
                raise RuntimeExperimentError(
                    f"illegal learned Command for {example.example_id}"
                ) from error
            after = env.state_digest()
            exact = predicted == example.targets
            offer_correct = predicted.offer_index == example.targets.offer_index
            candidate_exact = (
                predicted.candidate_selected == example.targets.candidate_selected
            )
            bucket = totals[example.split]
            bucket["examples"] += 1
            bucket["loss"] += loss
            bucket["offer_correct"] += float(offer_correct)
            bucket["candidate_exact"] += float(candidate_exact)
            bucket["exact"] += float(exact)
            bucket["accepted_commands"] += 1
            priority = competencies["priority"]
            priority["examples"] += 1
            priority["correct"] += float(offer_correct)
            if example.recipe["kind"] == "target":
                targeting = competencies["targeting"]
                targeting["examples"] += 1
                targeting["correct"] += float(candidate_exact)
            if example.recipe["kind"] == "combat":
                combat = competencies["combat"]
                combat["examples"] += 1
                combat["correct"] += float(candidate_exact)
            replay_rows.append(
                {
                    "schema_version": RUNNER_SCHEMA_VERSION,
                    "arm": trained.arm,
                    "seed": trained.seed,
                    "example_id": example.example_id,
                    "recipe": dict(example.recipe),
                    "source_digest": before,
                    "post_command_digest": after,
                    "frame_hash": decision.frame.frame_hash,
                    "command": command.model_dump(mode="json"),
                    "feature_digest": features.digest,
                    "exact_oracle_agreement": exact,
                }
            )

    splits: dict[str, Any] = {}
    for split, bucket in sorted(totals.items()):
        count = bucket["examples"]
        splits[split] = {
            "examples": int(count),
            "policy_loss": bucket["loss"] / count,
            "offer_accuracy": bucket["offer_correct"] / count,
            "candidate_exact_accuracy": bucket["candidate_exact"] / count,
            "exact_agreement": bucket["exact"] / count,
            "accepted_commands": int(bucket["accepted_commands"]),
            "illegal_commands": 0,
        }
    return {
        "splits": splits,
        "competencies": {
            name: {
                "examples": int(bucket["examples"]),
                "accuracy": bucket["correct"] / bucket["examples"],
            }
            for name, bucket in sorted(competencies.items())
        },
    }, replay_rows


def benchmark_policy(
    trained: TrainedPolicy,
    examples: Sequence[RuntimeExample],
    workload: Mapping[str, Any],
) -> dict[str, Any]:
    model = trained.model
    model.eval()
    config = workload["evaluation"]
    sample = max(examples, key=lambda example: len(example.targets.candidate_selected))
    warmup = int(config["latency_warmup"])
    samples = int(config["latency_samples"])
    repetitions = int(config["throughput_repetitions"])
    batch = tuple(example.features for example in examples)
    with torch.inference_mode():
        definitions = model.encode_definitions()
        for _ in range(warmup):
            model(sample.features, definitions)
        durations = []
        for _ in range(samples):
            started = perf_counter_ns()
            model(sample.features, definitions)
            durations.append((perf_counter_ns() - started) / 1_000_000)
        started = perf_counter()
        for _ in range(repetitions):
            model.forward_many(batch, definitions)
        elapsed = perf_counter() - started
    decisions = repetitions * len(batch)
    return {
        "catalog_cache": "checkpoint_static_definition_representations",
        "latency_ms": {
            "p50": percentile(durations, 0.50),
            "p95": percentile(durations, 0.95),
            "samples": samples,
        },
        "batch_throughput_decisions_per_second": decisions / elapsed,
        "batch_size": len(batch),
        "throughput_decisions": decisions,
        "throughput_seconds": elapsed,
    }


def replay_rows(
    rows: Sequence[Mapping[str, Any]],
    adapter: SemanticDecisionAdapter,
    projector: RuntimePolicyProjector,
) -> dict[str, Any]:
    started = perf_counter()
    for row in rows:
        env, decision = bind_recipe(row["recipe"], adapter)
        if env.state_digest() != row["source_digest"]:
            raise RuntimeExperimentError("replay source digest mismatch")
        features = projector.project(decision)
        if features.digest != row["feature_digest"]:
            raise RuntimeExperimentError("replay feature digest mismatch")
        command = Command.model_validate(row["command"])
        decision.step(env, command)
        if env.state_digest() != row["post_command_digest"]:
            raise RuntimeExperimentError("replay post-command digest mismatch")
    elapsed = perf_counter() - started
    return {
        "evaluation_rows": len(rows),
        "mismatches": 0,
        "seconds": elapsed,
        "environment_sps": len(rows) / elapsed if elapsed else 0.0,
    }


def verify_viewer_private_features(
    examples: Sequence[RuntimeExample],
    adapter: SemanticDecisionAdapter,
    projector: RuntimePolicyProjector,
) -> dict[str, int]:
    example = next(
        example
        for example in examples
        if example.split == "evaluation" and example.recipe["kind"] == "target"
    )
    env, decision = bind_recipe(example.recipe, adapter)
    baseline = projector.project(decision)
    clone = env.clone_env()
    clone.determinize(91_001, perspective=int(decision.frame.prompt.actor))
    observation = clone.scenario_refresh()
    rebound = adapter.bind(
        clone,
        observation,
        match_id=str(example.recipe["example_id"]),
        revision=int(example.recipe["revision"]),
        content_hash=adapter.pack.content_pack_hash,
        asset_manifest_hash=ASSET_MANIFEST_HASH,
    )
    if projector.project(rebound).digest != baseline.digest:
        raise RuntimeExperimentError(
            "opponent-private determinization changed runtime policy features"
        )
    return {"checks": 1, "viewer_private_feature_mismatches": 0}


def replay_arena_games(
    games: Sequence[Mapping[str, Any]],
    adapter: SemanticDecisionAdapter,
) -> dict[str, Any]:
    started = perf_counter()
    command_count = 0
    for game in games:
        recipe = game["recipe"]
        env, observation = reconstruct_root(recipe)
        for row in game["trace"]:
            if env.state_digest() != row["source_digest"]:
                raise RuntimeExperimentError("arena replay source digest mismatch")
            decision = adapter.bind(
                env,
                observation,
                match_id=str(recipe["example_id"]),
                revision=int(row["revision"]),
                content_hash=adapter.pack.content_pack_hash,
                asset_manifest_hash=ASSET_MANIFEST_HASH,
            )
            if decision.frame.frame_hash != row["frame_hash"]:
                raise RuntimeExperimentError("arena replay frame hash mismatch")
            command = Command.model_validate(row["command"])
            observation, _, _, _, _, _ = decision.step(env, command)
            if env.state_digest() != row["post_command_digest"]:
                raise RuntimeExperimentError(
                    "arena replay post-command digest mismatch"
                )
            command_count += 1
        if not observation.game_over or not game["terminal"]:
            raise RuntimeExperimentError("arena replay did not reach terminal state")
        if _winner(observation) != game["winner_seat"]:
            raise RuntimeExperimentError("arena replay winner mismatch")
    elapsed = perf_counter() - started
    return {
        "arena_games": len(games),
        "arena_commands": command_count,
        "mismatches": 0,
        "seconds": elapsed,
        "environment_sps": command_count / elapsed if elapsed else 0.0,
    }


def _winner(observation: Any) -> int | None:
    if not observation.game_over:
        return None
    agent = int(observation.agent.player_index)
    return agent if observation.won else 1 - agent


def play_match(
    seat_policies: Sequence[TrainedPolicy],
    seed: int,
    adapter: SemanticDecisionAdapter,
    projector: RuntimePolicyProjector,
    command_cap: int = ARENA_COMMAND_CAP,
) -> dict[str, Any]:
    recipe = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "example_id": f"arena-{seed}-{seat_policies[0].arm}-{seat_policies[1].arm}",
        "revision": 0,
        "seed": seed,
        "split": "arena",
        "kind": "combat",
        "creatures": [ARENA_CREATURE] * 6,
        "opponent_life": 1,
        "skip_trivial": True,
    }
    env, observation = reconstruct_root(recipe)
    definitions = []
    for policy in seat_policies:
        policy.model.eval()
        with torch.inference_mode():
            definitions.append(policy.model.encode_definitions())
    trace: list[dict[str, Any]] = []
    started = perf_counter()
    for revision in range(command_cap):
        if observation.game_over:
            break
        decision = adapter.bind(
            env,
            observation,
            match_id=str(recipe["example_id"]),
            revision=revision,
            content_hash=adapter.pack.content_pack_hash,
            asset_manifest_hash=ASSET_MANIFEST_HASH,
        )
        features = projector.project(decision)
        actor = int(decision.frame.prompt.actor)
        policy = seat_policies[actor]
        with torch.inference_mode():
            output = policy.model(features, definitions[actor])
        submission = decode_output(decision, output)
        command = decision.command(
            submission,
            command_id=f"arena:{seed}:{revision}:{policy.arm}:{policy.seed}",
        )
        before = env.state_digest()
        observation, _, done, truncated, _, _ = decision.step(env, command)
        trace.append(
            {
                "revision": revision,
                "actor": actor,
                "player": {"arm": policy.arm, "seed": policy.seed},
                "source_digest": before,
                "post_command_digest": env.state_digest(),
                "frame_hash": decision.frame.frame_hash,
                "command": command.model_dump(mode="json"),
            }
        )
        if done or truncated:
            break
    if not observation.game_over:
        raise RuntimeExperimentError(
            f"arena match {recipe['example_id']} did not terminate within "
            f"the {command_cap}-Command cap"
        )
    elapsed = perf_counter() - started
    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "world": "w2",
        "arena_version": "development_paired_arena_v1",
        "promotion_authority": False,
        "seed": seed,
        "recipe": recipe,
        "command_cap": command_cap,
        "seat_registrations": [
            {
                "seat": seat,
                "arm": policy.arm,
                "training_seed": policy.seed,
                "checkpoint_sha256": policy.checkpoint["sha256"],
            }
            for seat, policy in enumerate(seat_policies)
        ],
        "winner_seat": _winner(observation),
        "terminal": bool(observation.game_over),
        "commands": len(trace),
        "seconds": elapsed,
        "environment_sps": len(trace) / elapsed if elapsed else 0.0,
        "trace": trace,
    }


def run_arena(
    trained: Sequence[TrainedPolicy],
    workload: Mapping[str, Any],
    adapter: SemanticDecisionAdapter,
    projector: RuntimePolicyProjector,
) -> dict[str, Any]:
    by_key = {(policy.arm, policy.seed): policy for policy in trained}
    games = []
    payoff: dict[str, dict[str, list[float]]] = {
        left: {right: [] for right in POLICY_ARMS} for left in POLICY_ARMS
    }
    for seed in workload["model_seeds"]:
        for left_index, left in enumerate(POLICY_ARMS):
            for right in POLICY_ARMS[left_index + 1 :]:
                for deal_seed in workload["evaluation"]["deal_seeds"]:
                    paired_seed = int(deal_seed) * 10 + int(seed) % 10
                    for seats in ((left, right), (right, left)):
                        game = play_match(
                            [by_key[(seats[0], seed)], by_key[(seats[1], seed)]],
                            paired_seed,
                            adapter,
                            projector,
                            int(workload["evaluation"]["arena_command_cap"]),
                        )
                        games.append(game)
                        winner = game["winner_seat"]
                        left_score = (
                            0.5 if winner is None else float(seats[winner] == left)
                        )
                        payoff[left][right].append(left_score)
                        payoff[right][left].append(1.0 - left_score)
    matrix = {
        left: {
            right: (
                sum(values) / len(values)
                if values
                else (0.5 if left == right else None)
            )
            for right, values in rows.items()
        }
        for left, rows in payoff.items()
    }
    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "arena_version": "development_paired_arena_v1",
        "promotion_authority": False,
        "world": "w2",
        "information_boundary": workload["information_boundary"],
        "combat_fixture": ARENA_CREATURE,
        "command_cap": int(workload["evaluation"]["arena_command_cap"]),
        "all_games_terminal": all(game["terminal"] for game in games),
        "terminal_games": sum(int(game["terminal"]) for game in games),
        "nonterminal_games": sum(not game["terminal"] for game in games),
        "paired_seat_games": games,
        "payoff_matrix_left_score": matrix,
        "paired_strength": {
            arm: sum(
                value for opponent, value in matrix[arm].items() if opponent != arm
            )
            / (len(POLICY_ARMS) - 1)
            for arm in POLICY_ARMS
        },
    }


def engine_identity() -> dict[str, Any]:
    extension = next(Path(managym.__file__).parent.glob("_managym*.so"), None)
    return {
        "python_package": str(Path(managym.__file__).resolve()),
        "extension": str(extension.resolve()) if extension else None,
        "extension_sha256": file_sha256(extension) if extension else None,
    }


def transfer_outcome(evaluations: Mapping[str, Any]) -> dict[str, Any]:
    splits = ("identity_holdout", "composition_holdout")
    by_arm: dict[str, Any] = {}
    for arm in POLICY_ARMS:
        rows = [
            value["splits"]
            for key, value in evaluations.items()
            if key.startswith(f"{arm}/")
        ]
        by_arm[arm] = {
            split: {
                "mean_exact_agreement": sum(
                    row[split]["exact_agreement"] for row in rows
                )
                / len(rows),
                "mean_policy_loss": sum(row[split]["policy_loss"] for row in rows)
                / len(rows),
                "seed_exact_agreement": [row[split]["exact_agreement"] for row in rows],
            }
            for split in splits
        }
    semantic_identity = by_arm["semantic"]["identity_holdout"]["mean_exact_agreement"]
    identity_identity = by_arm["identity_only"]["identity_holdout"][
        "mean_exact_agreement"
    ]
    shuffled_identity = by_arm["structure_shuffled"]["identity_holdout"][
        "mean_exact_agreement"
    ]
    semantic_composition = by_arm["semantic"]["composition_holdout"][
        "seed_exact_agreement"
    ]
    structure_advantage = (
        semantic_identity > identity_identity
        and semantic_identity > shuffled_identity
        and by_arm["semantic"]["composition_holdout"]["mean_exact_agreement"]
        > max(
            by_arm["identity_only"]["composition_holdout"]["mean_exact_agreement"],
            by_arm["structure_shuffled"]["composition_holdout"]["mean_exact_agreement"],
        )
    )
    return {
        "verdict": (
            "semantic_structure_advantage"
            if structure_advantage
            else "null_or_ambiguous_structure_evidence"
        ),
        "semantic_structure_claim_supported": structure_advantage,
        "by_arm": by_arm,
        "observations": [
            (
                "semantic improves identity-holdout exact agreement over identity-only"
                if semantic_identity > identity_identity
                else "semantic does not improve identity-holdout exact agreement "
                "over identity-only"
            ),
            (
                "semantic does not separate from structure-shuffled on the "
                "identity holdout"
                if semantic_identity <= shuffled_identity
                else "semantic separates from structure-shuffled on the identity "
                "holdout"
            ),
            (
                "semantic composition transfer varies across seeds"
                if min(semantic_composition) != max(semantic_composition)
                else "semantic composition transfer is consistent across seeds"
            ),
        ],
    }


def summarize(
    workload: Mapping[str, Any],
    workload_sha: str,
    examples: Sequence[RuntimeExample],
    trained: Sequence[TrainedPolicy],
    evaluations: Mapping[str, Any],
    benchmarks: Mapping[str, Any],
    arena: Mapping[str, Any],
    replay: Mapping[str, Any],
    privacy: Mapping[str, Any],
    adapter: SemanticDecisionAdapter,
    projector: RuntimePolicyProjector,
) -> dict[str, Any]:
    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "experiment": workload["experiment"],
        "claim": workload["claim"],
        "status": "completed_development_evidence",
        "admission": "not_promotion_authority",
        "identities": {
            "world": workload["world"],
            "content_pack": adapter.pack.content_pack_hash,
            "semantic_pack": adapter.pack.semantic_pack_hash,
            "semantic_ir": adapter.pack.ir.ir_hash,
            "learning_schema": adapter.pack.schema.schema_hash,
            "experience_protocol": 1,
            "offer_contract": "int2-semantic-decision-v1",
            "model_contract": "semantic-runtime-policy-v1",
            "catalog": projector.catalog.digest,
            "workload_sha256": workload_sha,
            "engine": engine_identity(),
        },
        "dataset": {
            "examples": len(examples),
            "splits": dict(Counter(example.split for example in examples)),
            "max_target_candidates": max(
                len(example.targets.candidate_selected) for example in examples
            ),
            "max_combat_branches": max(
                2 ** len(example.recipe["creatures"])
                for example in examples
                if example.recipe["kind"] == "combat"
            ),
            "identity_holdout": workload["dataset"]["identity_holdout"],
            "composition_holdout": workload["dataset"]["composition_holdout"],
            "composition_known_operations": workload["dataset"][
                "composition_known_operations"
            ],
        },
        "checkpoints": [policy.checkpoint for policy in trained],
        "training": [
            {"arm": policy.arm, "seed": policy.seed, **policy.training}
            for policy in trained
        ],
        "evaluation": evaluations,
        "systems": benchmarks,
        "arena": arena,
        "replay": replay,
        "privacy": privacy,
        "outcome": transfer_outcome(evaluations),
        "integrity": {
            "verified_checkpoints": len(trained),
            "all_arena_games_terminal": arena["all_games_terminal"],
            "terminal_arena_games": arena["terminal_games"],
            "nonterminal_arena_games": arena["nonterminal_games"],
            "viewer_private_feature_mismatches": privacy[
                "viewer_private_feature_mismatches"
            ],
        },
        "legality": {
            "accepted_evaluation_commands": sum(
                split["accepted_commands"]
                for policy in evaluations.values()
                for split in policy["splits"].values()
            ),
            "accepted_arena_commands": replay["arena_commands"],
            "accepted_commands": sum(
                split["accepted_commands"]
                for policy in evaluations.values()
                for split in policy["splits"].values()
            )
            + replay["arena_commands"],
            "illegal_commands": 0,
            "replay_mismatches": replay["mismatches"],
        },
        "seeds": list(workload["model_seeds"]),
        "compute": {
            "host": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "declared_wall_clock_cap_seconds_per_checkpoint": workload["training"][
                "max_wall_clock_seconds"
            ],
        },
        "limitations": workload["limitations"],
    }


def run(workload_path: Path, out_dir: Path) -> dict[str, Any]:
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    workload, workload_sha = load_workload(workload_path)
    adapter = _bootstrap_adapter()
    projector = RuntimePolicyProjector(adapter.pack)
    examples = generate_dataset(workload, adapter)
    dataset_payload = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "experiment": workload["experiment"],
        "workload_sha256": workload_sha,
        "catalog_digest": projector.catalog.digest,
        "examples": [example.artifact for example in examples],
    }
    dataset_digest = write_json(out_dir / "dataset.json", dataset_payload)
    train_examples = [example for example in examples if example.split == "train"]
    trained_in_memory = [
        train_policy(
            arm,
            int(seed),
            train_examples,
            projector,
            workload,
            out_dir,
            dataset_digest,
            adapter,
        )
        for arm in POLICY_ARMS
        for seed in workload["model_seeds"]
    ]
    trained = [
        TrainedPolicy(
            policy.arm,
            policy.seed,
            load_checkpoint_model(policy.checkpoint, out_dir, projector),
            policy.checkpoint,
            policy.training,
        )
        for policy in trained_in_memory
    ]
    evaluation_examples = [example for example in examples if example.split != "train"]
    evaluations: dict[str, Any] = {}
    all_replay_rows: list[dict[str, Any]] = []
    benchmarks: dict[str, Any] = {}
    for policy in trained:
        key = f"{policy.arm}/seed-{policy.seed}"
        metrics, rows = evaluate_policy(policy, evaluation_examples, adapter, projector)
        evaluations[key] = metrics
        all_replay_rows.extend(rows)
        benchmarks[key] = benchmark_policy(policy, evaluation_examples, workload)
    evaluation_replay = replay_rows(all_replay_rows, adapter, projector)
    privacy = verify_viewer_private_features(examples, adapter, projector)
    arena = run_arena(trained, workload, adapter, projector)
    if not arena["all_games_terminal"]:
        raise RuntimeExperimentError("arena retained a nonterminal game")
    arena_replay = replay_arena_games(arena["paired_seat_games"], adapter)
    replay_seconds = evaluation_replay["seconds"] + arena_replay["seconds"]
    replay_commands = (
        evaluation_replay["evaluation_rows"] + arena_replay["arena_commands"]
    )
    replay = {
        "evaluation_rows": evaluation_replay["evaluation_rows"],
        "arena_games": arena_replay["arena_games"],
        "arena_commands": arena_replay["arena_commands"],
        "mismatches": 0,
        "seconds": replay_seconds,
        "environment_sps": replay_commands / replay_seconds,
    }
    replays_payload = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "experiment": workload["experiment"],
        "rows": all_replay_rows,
        "arena_games": arena["paired_seat_games"],
    }
    replays_digest = write_json(out_dir / "replays.json", replays_payload)
    result = summarize(
        workload,
        workload_sha,
        examples,
        trained,
        evaluations,
        benchmarks,
        arena,
        replay,
        privacy,
        adapter,
        projector,
    )
    result["artifacts"] = {
        "dataset": {"path": "dataset.json", "sha256": dataset_digest},
        "replays": {"path": "replays.json", "sha256": replays_digest},
    }
    result_digest = write_json(out_dir / "result.json", result)
    manifest = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "experiment": workload["experiment"],
        "result": {"path": "result.json", "sha256": result_digest},
        "dataset": result["artifacts"]["dataset"],
        "replays": result["artifacts"]["replays"],
        "checkpoints": [policy.checkpoint for policy in trained],
    }
    manifest["manifest_digest"] = canonical_sha256(manifest)
    write_json(out_dir / "manifest.json", manifest)
    print(
        canonical_json(
            {
                "experiment": workload["experiment"],
                "status": result["status"],
                "manifest": str(out_dir / "manifest.json"),
                "illegal_commands": 0,
                "replay_mismatches": 0,
                "paired_strength": arena["paired_strength"],
            }
        )
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run(args.workload.resolve(), args.out_dir.resolve())


if __name__ == "__main__":
    main(sys.argv[1:])
