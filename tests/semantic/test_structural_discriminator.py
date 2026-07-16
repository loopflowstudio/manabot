from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from experiments.runners.run_structural_encoder_discriminator import (
    load_discriminator_contract,
    verify_cached_equivalence,
)
from manabot.semantic.structural import (
    build_matched_models,
    build_relation_message_model,
    trainable_parameter_count,
)
from manabot.semantic.structural_katas import (
    KataTensorCatalog,
    load_suite,
    records_to_batch,
)

CONTRACT = Path("experiments/workloads/structural-semantic-katas-v1-discriminator.json")


def test_discriminator_contract_preserves_original_gates_and_budgets() -> None:
    contract, _ = load_discriminator_contract(CONTRACT)

    assert contract["model_seeds"] == [21401, 21402, 21403, 21404, 21405]
    assert contract["stages"]["bag_v1"]["maximum_steps_per_seed"] == 800
    assert (
        contract["stages"]["relational_semantic_encoder_v1_opt4000"][
            "maximum_steps_per_seed"
        ]
        == 4000
    )
    assert contract["gates"]["cached_batch1_p95_ratio_maximum"] == 2.5
    assert contract["gates"]["cached_batch128_throughput_ratio_minimum"] == 0.4
    assert contract["budget"]["maximum_optimizer_steps"] == 44000
    assert contract["budget"]["maximum_presented_examples"] == 2816000
    assert contract["budget"]["maximum_retries"] == 0


def test_message_arm_repairs_relation_reachability_with_192_parameters() -> None:
    suite, _ = load_suite()
    records = [row for row in suite["programs"] if row["split"] == "test"][:8]
    _, optimization = build_matched_models(
        token_count=len(suite["token_vocabulary"]),
        token_kind_count=int(suite["token_kind_count"]),
        seed=21401,
    )
    message = build_relation_message_model(
        token_count=len(suite["token_vocabulary"]),
        token_kind_count=int(suite["token_kind_count"]),
        seed=21401,
    )
    batch = records_to_batch(records, seed=21401, include_relations=True)
    without_relations = replace(batch, relations=torch.zeros_like(batch.relations))

    with torch.no_grad():
        assert torch.equal(
            optimization.encoder(batch), optimization.encoder(without_relations)
        )
        assert not torch.equal(
            message.encoder(batch), message.encoder(without_relations)
        )
    assert (
        trainable_parameter_count(message) - trainable_parameter_count(optimization)
        == 192
    )
    assert trainable_parameter_count(message) == 9030


def test_tensor_catalog_matches_online_projection_for_all_seeds_and_widths() -> None:
    suite, suite_sha256 = load_suite()
    catalog = KataTensorCatalog(suite["programs"], suite_sha256=suite_sha256)
    subsets = (suite["programs"][:1], suite["programs"][20:37], suite["programs"])

    for seed in (21401, 21402, 21403, 21404, 21405):
        for records in subsets:
            for include_relations in (False, True):
                online = records_to_batch(
                    records, seed=seed, include_relations=include_relations
                )
                cached = catalog.batch(
                    records, seed=seed, include_relations=include_relations
                )
                assert all(
                    torch.equal(getattr(online, field), getattr(cached, field))
                    for field in online.__dataclass_fields__
                )


def test_cached_equivalence_checks_tensors_logits_and_metrics() -> None:
    suite, suite_sha256 = load_suite()
    records = suite["programs"][:32]
    catalog = KataTensorCatalog(suite["programs"], suite_sha256=suite_sha256)
    _, model = build_matched_models(
        token_count=len(suite["token_vocabulary"]),
        token_kind_count=int(suite["token_kind_count"]),
        seed=21401,
    )

    receipt = verify_cached_equivalence(
        model,
        records,
        catalog,
        seed=21401,
        include_relations=True,
    )

    assert receipt == {
        "programs": 32,
        "tensor_equality": True,
        "logit_equality": True,
        "metric_equality": True,
    }
