"""Run the pre-registered W2-266 structural encoder discriminator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter_ns
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from manabot.semantic.compiler import pretty_json
from manabot.semantic.structural import (
    KATA_FAMILIES,
    KataBatch,
    build_matched_models,
    build_relation_message_model,
    trainable_parameter_bytes,
    trainable_parameter_count,
)
from manabot.semantic.structural_katas import (
    COMPILER_PATH,
    ORACLE_PATH,
    ROOT,
    SCHEMA_PATH,
    SOURCE_PATH,
    KataTensorCatalog,
    StructuralKataError,
    evaluate_model,
    load_suite,
    paired_symmetry_audit,
    records_to_batch,
    sha256_bytes,
    t_interval,
    train_model,
)

RUNNER_PATH = ROOT / "experiments/runners/run_structural_encoder_discriminator.py"
TEST_PATH = ROOT / "tests/semantic/test_structural_discriminator.py"
STRUCTURAL_PATH = ROOT / "manabot/semantic/structural.py"
KATAS_PATH = ROOT / "manabot/semantic/structural_katas.py"
ARM_BAG = "bag_v1"
ARM_OPT = "relational_semantic_encoder_v1_opt4000"
ARM_MESSAGE = "relational_message_encoder_v1"


def _git(args: Sequence[str]) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True
    ).stdout


def load_discriminator_contract(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    contract = json.loads(raw)
    if (
        contract.get("schema_version") != 1
        or contract.get("id") != "structural-semantic-katas-v1-discriminator"
    ):
        raise StructuralKataError("unknown structural discriminator contract")
    if contract.get("model_seeds") != [21401, 21402, 21403, 21404, 21405]:
        raise StructuralKataError("discriminator must retain the five W2-214 seeds")
    if contract["budget"]["maximum_retries"] != 0:
        raise StructuralKataError("discriminator retries are forbidden")
    return contract, sha256_bytes(raw)


def _verify_preregistration(
    revision: str,
    contract_path: Path,
    suite_path: Path,
    contract: Mapping[str, Any],
) -> str:
    head = _git(["rev-parse", "HEAD"]).decode().strip()
    resolved = _git(["rev-parse", revision]).decode().strip()
    if head != resolved:
        raise StructuralKataError(
            "primary run requires HEAD to equal the preregistration revision"
        )
    required = (
        contract_path,
        suite_path,
        SOURCE_PATH,
        ORACLE_PATH,
        COMPILER_PATH,
        SCHEMA_PATH,
        STRUCTURAL_PATH,
        KATAS_PATH,
        RUNNER_PATH,
        TEST_PATH,
    )
    for path in required:
        relative = str(path.relative_to(ROOT))
        committed = _git(["show", f"{resolved}:{relative}"])
        if committed != path.read_bytes():
            raise StructuralKataError(
                f"{relative} differs from preregistration revision {resolved}"
            )
    authority = contract["authority"]
    checks = {
        COMPILER_PATH: authority["compiler_sha256"],
        SCHEMA_PATH: authority["learning_schema_sha256"],
        ORACLE_PATH: authority["oracle_sha256"],
        SOURCE_PATH: authority["source_sha256"],
        suite_path: authority["suite_sha256"],
    }
    for path, expected in checks.items():
        if sha256_bytes(path.read_bytes()) != expected:
            raise StructuralKataError(f"authority hash mismatch: {path}")
    return resolved


def _batch_equal(left: KataBatch, right: KataBatch) -> bool:
    return all(
        torch.equal(getattr(left, field), getattr(right, field))
        for field in (
            "token_ids",
            "token_kinds",
            "token_mask",
            "depth",
            "relations",
            "families",
            "candidate_orders",
            "labels",
        )
    )


def verify_cached_equivalence(
    model: nn.Module,
    records: Sequence[Mapping[str, Any]],
    catalog: KataTensorCatalog,
    *,
    seed: int,
    include_relations: bool,
) -> dict[str, Any]:
    online = records_to_batch(records, seed=seed, include_relations=include_relations)
    cached = catalog.batch(records, seed=seed, include_relations=include_relations)
    if not _batch_equal(online, cached):
        raise StructuralKataError("cached projection changed input tensors")
    model.eval()
    with torch.no_grad():
        online_logits = model(online)
        cached_logits = model(cached)
    if not torch.equal(online_logits, cached_logits):
        raise StructuralKataError("cached projection changed model logits")
    online_metrics = evaluate_model(model, online)
    cached_metrics = evaluate_model(model, cached)
    if online_metrics != cached_metrics:
        raise StructuralKataError("cached projection changed model metrics")
    return {
        "programs": len(records),
        "tensor_equality": True,
        "logit_equality": True,
        "metric_equality": True,
    }


def _percentiles(values: Sequence[int]) -> dict[str, Any]:
    return {
        "p50": int(np.percentile(values, 50)),
        "p95": int(np.percentile(values, 95)),
        "samples": len(values),
        "raw_ns": list(values),
    }


def _benchmark(
    model: nn.Module,
    single_call: Callable[[], None],
    throughput_call: Callable[[], None],
    *,
    throughput_examples: int,
    throughput_tokens: int,
    performance: Mapping[str, Any],
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        for _ in range(int(performance["batch1_warmups"])):
            single_call()
        single_timings = []
        for _ in range(int(performance["batch1_samples"])):
            started = perf_counter_ns()
            single_call()
            single_timings.append(perf_counter_ns() - started)
        for _ in range(int(performance["throughput_warmups"])):
            throughput_call()
        throughput_timings = []
        for _ in range(int(performance["throughput_samples"])):
            started = perf_counter_ns()
            throughput_call()
            throughput_timings.append(perf_counter_ns() - started)
    elapsed = sum(throughput_timings) / 1_000_000_000
    samples = int(performance["throughput_samples"])
    return {
        "batch1_latency_ns": _percentiles(single_timings),
        "batch128_call_latency_ns": _percentiles(throughput_timings),
        "batch128_examples_per_second": throughput_examples * samples / elapsed,
        "batch128_tokens_per_second": throughput_tokens * samples / elapsed,
    }


def _performance_receipt(
    model: nn.Module,
    records: Sequence[Mapping[str, Any]],
    catalog: KataTensorCatalog,
    *,
    seed: int,
    include_relations: bool,
    performance: Mapping[str, Any],
) -> dict[str, Any]:
    single_records = records[:1]
    throughput_records = records[: int(performance["throughput_batch_size"])]
    single_batch = catalog.batch(
        single_records, seed=seed, include_relations=include_relations
    )
    throughput_batch = catalog.batch(
        throughput_records, seed=seed, include_relations=include_relations
    )
    token_count = sum(int(row["token_length"]) for row in throughput_records)

    model_only = _benchmark(
        model,
        lambda: model(single_batch),
        lambda: model(throughput_batch),
        throughput_examples=len(throughput_records),
        throughput_tokens=token_count,
        performance=performance,
    )
    online = _benchmark(
        model,
        lambda: model(
            records_to_batch(
                single_records, seed=seed, include_relations=include_relations
            )
        ),
        lambda: model(
            records_to_batch(
                throughput_records, seed=seed, include_relations=include_relations
            )
        ),
        throughput_examples=len(throughput_records),
        throughput_tokens=token_count,
        performance=performance,
    )
    cached = _benchmark(
        model,
        lambda: model(
            catalog.batch(
                single_records, seed=seed, include_relations=include_relations
            )
        ),
        lambda: model(
            catalog.batch(
                throughput_records, seed=seed, include_relations=include_relations
            )
        ),
        throughput_examples=len(throughput_records),
        throughput_tokens=token_count,
        performance=performance,
    )
    components = {
        name: sum(parameter.numel() for parameter in child.parameters())
        for name, child in model.named_children()
    }
    return {
        "parameter_count": trainable_parameter_count(model),
        "parameter_bytes": trainable_parameter_bytes(model),
        "component_parameter_count": components,
        "model_only": model_only,
        "online_projector_e2e": online,
        "cached_projector_e2e": cached,
        "attribution": {
            "online_minus_cached_p50_ns": (
                online["batch1_latency_ns"]["p50"] - cached["batch1_latency_ns"]["p50"]
            ),
            "cached_minus_model_p50_ns": (
                cached["batch1_latency_ns"]["p50"]
                - model_only["batch1_latency_ns"]["p50"]
            ),
        },
    }


def _train_arm(
    arm: str,
    models: Sequence[tuple[int, nn.Module]],
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    stage = contract["stages"][arm]
    include_relations = bool(stage["include_relations"])
    selected = []
    for seed, model in models:
        train = records_to_batch(
            records["train"], seed=seed, include_relations=include_relations
        )
        validation = records_to_batch(
            records["validation"], seed=seed, include_relations=include_relations
        )
        training = train_model(
            model,
            train,
            validation,
            seed=seed,
            steps=int(stage["maximum_steps_per_seed"]),
            validation_interval=int(contract["training"]["validation_interval"]),
            learning_rate=float(contract["training"]["learning_rate"]),
        )
        selected.append(
            {"arm": arm, "model_seed": seed, "model": model, "training": training}
        )
    return selected


def _evaluate_arm(
    trained: Sequence[Mapping[str, Any]],
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    all_records: Sequence[Mapping[str, Any]],
    catalog: KataTensorCatalog,
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    runs = []
    for selected in trained:
        arm = str(selected["arm"])
        seed = int(selected["model_seed"])
        model = selected["model"]
        include_relations = bool(contract["stages"][arm]["include_relations"])
        batches = {
            split: records_to_batch(
                rows, seed=seed, include_relations=include_relations
            )
            for split, rows in records.items()
        }
        equivalence = verify_cached_equivalence(
            model,
            all_records,
            catalog,
            seed=seed,
            include_relations=include_relations,
        )
        metrics = {
            split: evaluate_model(model, batch) for split, batch in batches.items()
        }
        symmetry = (
            {
                split: paired_symmetry_audit(model, batches[split], records[split])
                for split in records
            }
            if arm == ARM_BAG
            else None
        )
        runs.append(
            {
                "arm": arm,
                "model_seed": seed,
                "training": selected["training"],
                **metrics,
                "symmetry": symmetry,
                "cached_equivalence": equivalence,
                "performance": _performance_receipt(
                    model,
                    records["test"],
                    catalog,
                    seed=seed,
                    include_relations=include_relations,
                    performance=contract["performance"],
                ),
            }
        )
    return runs


def _aggregate(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        out[split] = {
            metric: t_interval([float(run[split][metric]) for run in runs])
            for metric in ("accuracy", "brier", "ece_5_bin", "nll")
        }
        out[split]["by_family"] = {
            family: {
                metric: t_interval(
                    [float(run[split]["by_family"][family][metric]) for run in runs]
                )
                for metric in ("accuracy", "brier", "ece_5_bin", "nll")
            }
            for family in KATA_FAMILIES
        }
    out["training"] = {
        "maximum_training_accuracy": t_interval(
            [float(run["training"]["maximum_training_accuracy"]) for run in runs]
        ),
        "wall_clock_seconds": t_interval(
            [float(run["training"]["wall_clock_seconds"]) for run in runs]
        ),
    }
    return out


def _uplift(
    candidate_runs: Sequence[Mapping[str, Any]],
    bag_runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "aggregate": t_interval(
            [
                float(candidate["test"]["accuracy"] - bag["test"]["accuracy"])
                for candidate, bag in zip(candidate_runs, bag_runs, strict=True)
            ]
        ),
        "by_family": {
            family: t_interval(
                [
                    float(
                        candidate["test"]["by_family"][family]["accuracy"]
                        - bag["test"]["by_family"][family]["accuracy"]
                    )
                    for candidate, bag in zip(candidate_runs, bag_runs, strict=True)
                ]
            )
            for family in KATA_FAMILIES
        },
    }


def decide(
    runs_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    aggregate: Mapping[str, Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    gates = contract["gates"]
    bag_runs = runs_by_arm[ARM_BAG]
    bag_symmetry = all(
        run[split]["accuracy"] == gates["bag_accuracy_every_family_seed"]
        and all(
            row["accuracy"] == gates["bag_accuracy_every_family_seed"]
            for row in run[split]["by_family"].values()
        )
        and run["symmetry"][split]["paired_prediction_disagreements"] == 0
        and run["symmetry"][split]["maximum_paired_probability_difference"]
        <= gates["bag_maximum_paired_probability_difference"]
        for run in bag_runs
        for split in ("train", "validation", "test")
    )
    if not bag_symmetry:
        return "KILL_REDESIGN instrument_invalid", {"bag_symmetry": False}

    optimization_runs = runs_by_arm[ARM_OPT]
    optimization_fits = all(
        run["training"]["maximum_training_accuracy"]
        >= contract["training"]["capacity_trigger_minimum_each_seed"]
        for run in optimization_runs
    )
    candidate_arm = ARM_OPT if optimization_fits else ARM_MESSAGE
    candidate_runs = runs_by_arm.get(candidate_arm, ())
    if not candidate_runs:
        return "KILL_REDESIGN structural_capacity", {
            "bag_symmetry": True,
            "optimization_fits_training": optimization_fits,
        }
    candidate_fits = all(
        run["training"]["maximum_training_accuracy"]
        >= contract["training"]["capacity_trigger_minimum_each_seed"]
        for run in candidate_runs
    )
    if not candidate_fits:
        return "KILL_REDESIGN structural_capacity", {
            "bag_symmetry": True,
            "optimization_fits_training": False,
            "message_fits_training": False,
        }

    uplift = _uplift(candidate_runs, bag_runs)
    candidate = aggregate[candidate_arm]
    selected_train = all(
        run["train"]["accuracy"] >= gates["structural_train_accuracy_each_seed_minimum"]
        for run in candidate_runs
    )
    selected_validation = all(
        run["validation"]["accuracy"]
        >= gates["structural_validation_accuracy_each_seed_minimum"]
        for run in candidate_runs
    )
    failed_families = [
        family
        for family in KATA_FAMILIES
        if candidate["test"]["by_family"][family]["accuracy"]["mean"]
        < gates["structural_per_family_test_accuracy_minimum"]
        or uplift["by_family"][family]["mean"] < gates["per_family_uplift_minimum"]
    ]
    semantic_accuracy = (
        selected_train
        and selected_validation
        and candidate["test"]["accuracy"]["mean"]
        >= gates["structural_aggregate_test_accuracy_minimum"]
        and not failed_families
        and uplift["aggregate"]["t95_low"]
        > gates["aggregate_uplift_t95_lower_minimum_exclusive"]
    )
    calibration = (
        candidate["test"]["brier"]["mean"] <= gates["structural_brier_maximum"]
        and candidate["test"]["nll"]["mean"] <= gates["structural_nll_maximum"]
    )

    parameter_fractions = []
    cached_latency_ratios = []
    cached_throughput_ratios = []
    model_latency_ratios = []
    model_throughput_ratios = []
    for candidate_run, bag_run in zip(candidate_runs, bag_runs, strict=True):
        candidate_perf = candidate_run["performance"]
        bag_perf = bag_run["performance"]
        parameter_fractions.append(
            abs(candidate_perf["parameter_count"] - bag_perf["parameter_count"])
            / bag_perf["parameter_count"]
        )
        cached_latency_ratios.append(
            candidate_perf["cached_projector_e2e"]["batch1_latency_ns"]["p95"]
            / bag_perf["cached_projector_e2e"]["batch1_latency_ns"]["p95"]
        )
        cached_throughput_ratios.append(
            candidate_perf["cached_projector_e2e"]["batch128_examples_per_second"]
            / bag_perf["cached_projector_e2e"]["batch128_examples_per_second"]
        )
        model_latency_ratios.append(
            candidate_perf["model_only"]["batch1_latency_ns"]["p95"]
            / bag_perf["model_only"]["batch1_latency_ns"]["p95"]
        )
        model_throughput_ratios.append(
            candidate_perf["model_only"]["batch128_examples_per_second"]
            / bag_perf["model_only"]["batch128_examples_per_second"]
        )
    parameters = (
        max(parameter_fractions) <= gates["parameter_difference_fraction_maximum"]
    )
    cached_cost = (
        max(cached_latency_ratios) <= gates["cached_batch1_p95_ratio_maximum"]
        and min(cached_throughput_ratios)
        >= gates["cached_batch128_throughput_ratio_minimum"]
    )
    model_cost = (
        max(model_latency_ratios) <= gates["cached_batch1_p95_ratio_maximum"]
        and min(model_throughput_ratios)
        >= gates["cached_batch128_throughput_ratio_minimum"]
    )
    evidence = {
        "bag_symmetry": True,
        "optimization_fits_training": optimization_fits,
        "candidate_arm": candidate_arm,
        "candidate_fits_training": candidate_fits,
        "selected_train_gate": selected_train,
        "selected_validation_gate": selected_validation,
        "semantic_accuracy": semantic_accuracy,
        "calibration": calibration,
        "parameters": parameters,
        "cached_cost": cached_cost,
        "model_cost_diagnostic": model_cost,
        "failed_families": failed_families,
        "uplift": uplift,
        "parameter_difference_fractions": parameter_fractions,
        "cached_batch1_p95_ratios": cached_latency_ratios,
        "cached_batch128_throughput_ratios": cached_throughput_ratios,
        "model_batch1_p95_ratios": model_latency_ratios,
        "model_batch128_throughput_ratios": model_throughput_ratios,
    }
    if not semantic_accuracy or not calibration:
        return "KILL_REDESIGN structural_generalization", evidence
    if not parameters:
        return "KILL_REDESIGN model_execution_cost", evidence
    if not cached_cost:
        reason = "projector_execution_cost" if model_cost else "model_execution_cost"
        return f"KILL_REDESIGN {reason}", evidence
    return f"NOMINATE_FOR_W2_213 {candidate_arm}", evidence


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_report(result: Mapping[str, Any]) -> str:
    terminal = result["decision"]["terminal"]
    lines = [
        "# W2-266 structural encoder discriminator",
        "",
        f"Status: **{result['status'].upper()}**",
        "",
        f"Decision: **`{terminal}`**",
        "",
        "## Claim boundary",
        "",
        result["claim_boundary"],
        "",
        "The preserved static suite contains no runtime objects or legal offers. A nomination",
        "only makes one encoder eligible for W2-213; this run does not start W2-213 or",
        "authorize gameplay integration.",
        "",
        "## Sequential path",
        "",
        f"- Optimization arm fit every seed: **{result['decision']['evidence'].get('optimization_fits_training')}**.",
        f"- Conditional message arm executed: **{result['stage_c_triggered']}**.",
        f"- Surviving candidate: `{result['decision']['evidence'].get('candidate_arm', 'none')}`.",
        "",
        "## Seed and family receipts",
        "",
        "| Arm | Seed | Split | Family | Accuracy | Brier | NLL | ECE-5 |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for run in result["runs"]:
        for split in ("train", "validation", "test"):
            for family in KATA_FAMILIES:
                row = run[split]["by_family"][family]
                lines.append(
                    f"| `{run['arm']}` | {run['model_seed']} | {split} | `{family}` | "
                    f"{_percent(row['accuracy'])} | {row['brier']:.4f} | "
                    f"{row['nll']:.4f} | {row['ece_5_bin']:.4f} |"
                )
    lines.extend(
        [
            "",
            "## Training and CPU receipts",
            "",
            "| Arm | Seed | First 99% step | Max train | Selected step | Parameters | model p50/p95 µs | online p50/p95 µs | cached p50/p95 µs | cached batch-128/s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run in result["runs"]:
        training = run["training"]
        perf = run["performance"]

        def latency(path: str) -> str:
            row = perf[path]["batch1_latency_ns"]
            return f"{row['p50'] / 1000:.1f}/{row['p95'] / 1000:.1f}"

        lines.append(
            f"| `{run['arm']}` | {run['model_seed']} | "
            f"{training['first_99_train_step']} | "
            f"{_percent(training['maximum_training_accuracy'])} | "
            f"{training['selected_checkpoint_step']} | {perf['parameter_count']} | "
            f"{latency('model_only')} | {latency('online_projector_e2e')} | "
            f"{latency('cached_projector_e2e')} | "
            f"{perf['cached_projector_e2e']['batch128_examples_per_second']:.0f} |"
        )
    evidence = result["decision"]["evidence"]
    lines.extend(
        [
            "",
            "## Gates and diagnosis",
            "",
            f"- Bag exact symmetry and 50% ceiling: **{evidence.get('bag_symmetry')}**.",
            f"- Selected semantic accuracy: **{evidence.get('semantic_accuracy')}**.",
            f"- Calibration: **{evidence.get('calibration')}**.",
            f"- Parameter match: **{evidence.get('parameters')}**.",
            f"- Cached hot-path CPU gate: **{evidence.get('cached_cost')}**.",
            f"- Model-only cost diagnostic: **{evidence.get('model_cost_diagnostic')}**.",
            f"- Cold tensor-catalog build: {result['catalog']['cold_build_ns'] / 1_000_000:.3f} ms (not amortized).",
            "- Cached projection matched online tensors, logits, and metrics exactly for all 800 programs and all executed seeds/arms before timing.",
            "",
            "## Provenance and budget",
            "",
            f"- Preregistration revision: `{result['provenance']['preregistration_revision']}`.",
            f"- Contract SHA-256: `{result['provenance']['contract_sha256']}`.",
            f"- Suite SHA-256: `{result['provenance']['suite_sha256']}`.",
            f"- Optimizer steps: {result['budget']['optimizer_steps']} / {result['budget']['optimizer_steps_cap']}.",
            f"- Presented examples: {result['budget']['presented_examples']} / {result['budget']['presented_examples_cap']}.",
            f"- Wall clock: {result['budget']['wall_clock_seconds']:.2f}s / {result['budget']['wall_clock_seconds_cap']}s.",
            "",
            "## Result-contingent next step",
            "",
            f"`{terminal}`",
            "",
        ]
    )
    if terminal.startswith("NOMINATE_FOR_W2_213"):
        lines.append(
            "The named encoder is eligible for a separately authorized W2-213 run; W2-213 was not started here."
        )
    else:
        lines.append(
            "The named redesign diagnosis is terminal for this bounded experiment; no gate was relaxed and W2-213 remains blocked."
        )
    return "\n".join(lines) + "\n"


def run(
    contract_path: Path,
    suite_path: Path,
    *,
    preregistration_revision: str,
) -> dict[str, Any]:
    if sys.version_info[:2] != (3, 12):
        raise StructuralKataError("primary run requires CPython 3.12")
    contract_path = (
        contract_path if contract_path.is_absolute() else ROOT / contract_path
    ).resolve()
    suite_path = (
        suite_path if suite_path.is_absolute() else ROOT / suite_path
    ).resolve()
    contract, contract_digest = load_discriminator_contract(contract_path)
    suite, suite_digest = load_suite(suite_path)
    preregistration_revision = _verify_preregistration(
        preregistration_revision, contract_path, suite_path, contract
    )
    if suite_digest != contract["authority"]["suite_sha256"]:
        raise StructuralKataError("loaded suite differs from discriminator contract")

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    records = {
        split: [row for row in suite["programs"] if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    all_records = list(suite["programs"])
    started = perf_counter_ns()
    catalog_started = perf_counter_ns()
    catalog = KataTensorCatalog(all_records, suite_sha256=suite_digest)
    catalog_build_ns = perf_counter_ns() - catalog_started

    bag_models = []
    opt_models = []
    for seed_value in contract["model_seeds"]:
        seed = int(seed_value)
        bag, optimization = build_matched_models(
            token_count=len(suite["token_vocabulary"]),
            token_kind_count=int(suite["token_kind_count"]),
            seed=seed,
        )
        if not all(
            torch.equal(left, right)
            for left, right in zip(
                bag.probe.parameters(), optimization.probe.parameters(), strict=True
            )
        ):
            raise StructuralKataError("matched probe heads differ before training")
        bag_models.append((seed, bag))
        opt_models.append((seed, optimization))

    trained_bag = _train_arm(ARM_BAG, bag_models, records, contract)
    trained_opt = _train_arm(ARM_OPT, opt_models, records, contract)
    optimization_fits = all(
        row["training"]["maximum_training_accuracy"]
        >= contract["training"]["capacity_trigger_minimum_each_seed"]
        for row in trained_opt
    )
    trained_message: list[dict[str, Any]] = []
    if not optimization_fits:
        message_models = [
            (
                int(seed),
                build_relation_message_model(
                    token_count=len(suite["token_vocabulary"]),
                    token_kind_count=int(suite["token_kind_count"]),
                    seed=int(seed),
                ),
            )
            for seed in contract["model_seeds"]
        ]
        trained_message = _train_arm(ARM_MESSAGE, message_models, records, contract)

    bag_counts = {
        int(row["model_seed"]): trainable_parameter_count(row["model"])
        for row in trained_bag
    }
    for row in trained_opt + trained_message:
        seed = int(row["model_seed"])
        difference = (
            abs(trainable_parameter_count(row["model"]) - bag_counts[seed])
            / bag_counts[seed]
        )
        if difference > contract["gates"]["parameter_difference_fraction_maximum"]:
            raise StructuralKataError(
                f"parameter match failed before result access: {difference:.6f}"
            )

    runs = []
    for arm_rows in (trained_bag, trained_opt, trained_message):
        if arm_rows:
            runs.extend(
                _evaluate_arm(arm_rows, records, all_records, catalog, contract)
            )
    runs_by_arm = {
        arm: sorted(
            [run for run in runs if run["arm"] == arm],
            key=lambda run: run["model_seed"],
        )
        for arm in (ARM_BAG, ARM_OPT, ARM_MESSAGE)
        if any(run["arm"] == arm for run in runs)
    }
    aggregate = {arm: _aggregate(rows) for arm, rows in runs_by_arm.items()}
    terminal, evidence = decide(runs_by_arm, aggregate, contract)

    total_steps = sum(int(row["training"]["optimizer_steps"]) for row in runs)
    total_examples = sum(int(row["training"]["presented_examples"]) for row in runs)
    elapsed = (perf_counter_ns() - started) / 1_000_000_000
    budget = contract["budget"]
    if (
        total_steps > budget["maximum_optimizer_steps"]
        or total_examples > budget["maximum_presented_examples"]
        or elapsed > budget["maximum_wall_clock_seconds"]
    ):
        terminal = "KILL_REDESIGN budget_exceeded"
        evidence = {**evidence, "budget_exceeded": True}

    return {
        "schema_version": 1,
        "status": "pass" if terminal.startswith("NOMINATE") else "redesign",
        "experiment": contract["id"],
        "claim_boundary": contract["claim_boundary"],
        "stage_c_triggered": bool(trained_message),
        "decision": {"terminal": terminal, "evidence": evidence},
        "provenance": {
            "preregistration_revision": preregistration_revision,
            "measurement_code_revision": _git(["rev-parse", "HEAD"]).decode().strip(),
            "contract_path": str(contract_path.relative_to(ROOT)),
            "contract_sha256": contract_digest,
            "suite_path": str(suite_path.relative_to(ROOT)),
            "suite_sha256": suite_digest,
            "source_sha256": contract["authority"]["source_sha256"],
            "oracle_sha256": contract["authority"]["oracle_sha256"],
            "compiler_sha256": contract["authority"]["compiler_sha256"],
            "learning_schema_sha256": contract["authority"]["learning_schema_sha256"],
            "code_sha256": {
                str(path.relative_to(ROOT)): sha256_bytes(path.read_bytes())
                for path in (STRUCTURAL_PATH, KATAS_PATH, RUNNER_PATH)
            },
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED", "unset"),
            "torch_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "catalog": {
            "suite_sha256": catalog.suite_sha256,
            "cold_build_ns": catalog_build_ns,
            "cold_build_amortized_into_hot_gate": False,
        },
        "budget": {
            "optimizer_steps": total_steps,
            "optimizer_steps_cap": budget["maximum_optimizer_steps"],
            "presented_examples": total_examples,
            "presented_examples_cap": budget["maximum_presented_examples"],
            "wall_clock_seconds": elapsed,
            "wall_clock_seconds_cap": budget["maximum_wall_clock_seconds"],
        },
        "suite_audit": suite["audit"],
        "runs": sorted(runs, key=lambda run: (run["arm"], run["model_seed"])),
        "aggregate": aggregate,
        "contract": contract,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--preregistration-revision", required=True)
    args = parser.parse_args()
    result = run(
        args.contract,
        args.suite,
        preregistration_revision=args.preregistration_revision,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(pretty_json(result), encoding="utf-8")
    args.report.write_text(render_report(result), encoding="utf-8")
    print(result["decision"]["terminal"])


if __name__ == "__main__":
    main()
