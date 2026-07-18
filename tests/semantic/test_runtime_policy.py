"""End-to-end proofs for the learned semantic runtime policy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.runners.run_semantic_runtime_policy import (
    RuntimeExperimentError,
    _bootstrap_adapter,
    bind_recipe,
    dataset_recipes,
    decode_output,
    generate_dataset,
    load_checkpoint_model,
    load_workload,
    play_match,
    replay_arena_games,
    replay_rows,
    train_policy,
)
from manabot.semantic.runtime_policy import (
    POLICY_ARMS,
    RuntimePolicyError,
    RuntimePolicyProjector,
    SemanticRuntimePolicy,
    parameter_count,
    targets_from_submission,
)

WORKLOAD = Path("experiments/workloads/int-11-semantic-runtime-policy-v1.json")


@pytest.fixture(scope="module")
def workload():
    value, _ = load_workload(WORKLOAD)
    return value


@pytest.fixture(scope="module")
def adapter():
    return _bootstrap_adapter()


@pytest.fixture(scope="module")
def examples(workload, adapter):
    return generate_dataset(workload, adapter)


def test_dataset_is_authoritative_replayable_and_keeps_holdouts_out_of_training(
    workload, adapter, examples
) -> None:
    projector = RuntimePolicyProjector(adapter.pack)
    train = [example for example in examples if example.split == "train"]
    train_definitions = {
        projector.catalog.definition_keys[index]
        for example in train
        for index in example.features.object_definitions.tolist()
    }

    assert workload["dataset"]["identity_holdout"] not in train_definitions
    assert workload["dataset"]["composition_holdout"] not in train_definitions
    assert max(len(example.targets.candidate_selected) for example in examples) == 35
    assert (
        max(
            2 ** len(example.recipe["creatures"])
            for example in examples
            if example.recipe["kind"] == "combat"
        )
        == 64
    )

    result = replay_rows(
        [examples[-2].artifact, examples[-1].artifact], adapter, projector
    )
    assert result["evaluation_rows"] == 2
    assert result["mismatches"] == 0
    assert result["environment_sps"] > 0


def test_all_arms_have_matched_capacity_and_decode_normal_legal_commands(
    workload, adapter, examples
) -> None:
    projector = RuntimePolicyProjector(adapter.pack)
    frontier = next(
        example
        for example in examples
        if example.split == "frontier" and example.recipe["kind"] == "target"
    )
    counts = set()

    for index, arm in enumerate(POLICY_ARMS):
        torch.manual_seed(11_100 + index)
        model = SemanticRuntimePolicy(
            projector.catalog,
            arm,
            hidden_dim=int(workload["model"]["hidden_dim"]),
            attention_heads=int(workload["model"]["attention_heads"]),
            transformer_layers=int(workload["model"]["transformer_layers"]),
            dropout=0.0,
        )
        counts.add(parameter_count(model))
        env, decision = bind_recipe(frontier.recipe, adapter)
        features = projector.project(decision)
        comparison = examples[0].features
        definitions = model.encode_definitions()
        output, comparison_output = model.forward_many(
            (features, comparison), definitions
        )
        sequential_comparison = model(comparison, definitions)
        assert torch.allclose(
            comparison_output.offer_logits, sequential_comparison.offer_logits
        )
        assert torch.allclose(
            comparison_output.candidate_logits,
            sequential_comparison.candidate_logits,
        )
        submission = decode_output(decision, output)
        command = decision.command(submission, command_id=f"test:{arm}")
        before = env.state_digest()
        decision.step(env, command)

        assert len(output.candidate_logits) == 35
        assert env.state_digest() != before

    assert len(counts) == 1


def test_viewer_features_ignore_runtime_ids_and_private_determinization(
    workload, adapter
) -> None:
    recipe = next(
        recipe
        for recipe in dataset_recipes(workload)
        if recipe["split"] == "evaluation" and recipe["kind"] == "target"
    )
    env, decision = bind_recipe(recipe, adapter)
    projector = RuntimePolicyProjector(adapter.pack)
    baseline = projector.project(decision)

    clone = env.clone_env()
    clone.determinize(91_001, perspective=0)
    hidden = clone.scenario_refresh()
    rebound = adapter.bind(
        clone,
        hidden,
        match_id=str(recipe["example_id"]),
        revision=int(recipe["revision"]),
        content_hash=adapter.pack.content_pack_hash,
        asset_manifest_hash="int-11-no-player-assets",
    )

    assert projector.project(rebound).digest == baseline.digest


def test_unknown_prompt_and_misaligned_definition_cache_fail_before_scoring(
    workload, adapter, examples
) -> None:
    projector = RuntimePolicyProjector(adapter.pack)
    example = examples[0]
    model = SemanticRuntimePolicy(projector.catalog, "semantic")

    with pytest.raises(RuntimePolicyError, match="cached definition"):
        model(example.features, torch.zeros(1, model.hidden_dim))

    decision_env, decision = bind_recipe(example.recipe, adapter)
    del decision_env
    invalid = decision.frame.model_copy(
        update={"prompt": decision.frame.prompt.model_copy(update={"kind": "unknown"})}
    )
    with pytest.raises(RuntimePolicyError, match="unknown prompt"):
        projector.project(decision.__class__(**{**decision.__dict__, "frame": invalid}))


def test_small_training_run_learns_and_emits_a_verified_checkpoint(
    tmp_path, workload, adapter, examples
) -> None:
    local = {**workload, "training": {**workload["training"], "epochs": 2}}
    projector = RuntimePolicyProjector(adapter.pack)
    train = [example for example in examples if example.split == "train"]
    trained = train_policy(
        "semantic",
        int(workload["model_seeds"][0]),
        train,
        projector,
        local,
        tmp_path,
        "0" * 64,
        adapter,
    )
    checkpoint = tmp_path / str(trained.checkpoint["path"])

    assert trained.training["optimizer_steps"] == 2
    assert (
        trained.training["final_loss"] < trained.training["learning_curve"][0]["loss"]
    )
    assert checkpoint.is_file()
    assert checkpoint.stat().st_size == trained.checkpoint["bytes"]
    loaded = load_checkpoint_model(trained.checkpoint, tmp_path, projector)
    assert parameter_count(loaded) == parameter_count(trained.model)

    env, decision = bind_recipe(train[0].recipe, adapter)
    submission = trained.model.submission(decision, projector.project(decision))
    predicted = targets_from_submission(decision, submission)
    assert 0 <= predicted.offer_index < len(decision.batch.offers)
    decision.step(env, decision.command(submission, command_id="trained-smoke"))

    candidate_output = trained.model.candidate_head[-1]
    with torch.no_grad():
        candidate_output.weight.zero_()
        candidate_output.bias.fill_(-100.0)
    with pytest.raises(RuntimeExperimentError, match="did not terminate"):
        play_match([trained, trained], 11_201, adapter, projector)

    with torch.no_grad():
        candidate_output.bias.fill_(100.0)
    game = play_match([trained, trained], 11_201, adapter, projector)
    assert 1 <= game["commands"] <= 32
    assert game["terminal"] is True
    assert len(game["trace"]) == game["commands"]
    arena_replay = replay_arena_games([game], adapter)
    assert arena_replay["arena_games"] == 1
    assert arena_replay["mismatches"] == 0


def test_workload_rejects_post_result_arm_or_seed_changes(workload, tmp_path) -> None:
    mutated = {**workload, "model_seeds": [1101, 1101, 1102]}
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(mutated))
    with pytest.raises(RuntimeExperimentError, match="model_seeds"):
        load_workload(path)

    mutated = {**workload, "arms": ["semantic"]}
    path.write_text(json.dumps(mutated))
    with pytest.raises(RuntimeExperimentError, match="arms"):
        load_workload(path)

    mutated = {
        **workload,
        "evaluation": {**workload["evaluation"], "arena_command_cap": 33},
    }
    path.write_text(json.dumps(mutated))
    with pytest.raises(RuntimeExperimentError, match="Command cap"):
        load_workload(path)
