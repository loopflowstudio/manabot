"""Run the pre-registered W2-214 structural semantic kata experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter_ns
from typing import Any, Mapping, Sequence

import torch

from manabot.semantic.compiler import pretty_json
from manabot.semantic.structural import (
    KATA_FAMILIES,
    build_matched_models,
    trainable_parameter_count,
)
from manabot.semantic.structural_katas import (
    COMPILER_PATH,
    ORACLE_PATH,
    ROOT,
    SCHEMA_PATH,
    SOURCE_PATH,
    StructuralKataError,
    end_to_end_performance,
    evaluate_model,
    load_contract,
    load_suite,
    paired_symmetry_audit,
    performance_metrics,
    records_to_batch,
    sha256_bytes,
    t_interval,
    train_model,
)


def _git(args: Sequence[str]) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


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
            "primary training requires HEAD to equal the preregistration revision"
        )
    required = (
        contract_path,
        suite_path,
        SOURCE_PATH,
        ORACLE_PATH,
        COMPILER_PATH,
        SCHEMA_PATH,
        ROOT / "manabot/semantic/structural.py",
        ROOT / "manabot/semantic/structural_katas.py",
        ROOT / "experiments/runners/run_structural_semantic_katas.py",
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
        actual = sha256_bytes(path.read_bytes())
        if actual != expected:
            raise StructuralKataError(
                f"{path.relative_to(ROOT)}: expected {expected}, got {actual}"
            )
    return resolved


def _parameter_breakdown(model: torch.nn.Module) -> dict[str, Any]:
    components: dict[str, dict[str, int]] = {}
    for name, parameter in model.named_parameters():
        component = name.split(".")[1] if name.startswith("encoder.") else "five_family_heads"
        row = components.setdefault(component, {"count": 0, "bytes": 0})
        row["count"] += parameter.numel()
        row["bytes"] += parameter.numel() * parameter.element_size()
    return components


def _run_one(
    *,
    arm: str,
    model: torch.nn.Module,
    seed: int,
    records: Mapping[str, list[dict[str, Any]]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    training = contract["training"]
    performance = contract["performance"]
    train_batch = records_to_batch(records["train"], seed=seed)
    validation_batch = records_to_batch(records["validation"], seed=seed)
    test_batch = records_to_batch(records["test"], seed=seed)
    receipt = train_model(
        model,
        train_batch,
        validation_batch,
        seed=seed,
        steps=int(training["maximum_steps_per_run"]),
        validation_interval=int(training["validation_interval"]),
        learning_rate=float(training["learning_rate"]),
    )
    train_metrics = evaluate_model(model, train_batch)
    validation_metrics = evaluate_model(model, validation_batch)
    test_metrics = evaluate_model(model, test_batch)
    single = test_batch.select(torch.tensor([0]))
    throughput = test_batch.select(torch.arange(128))
    model_performance = performance_metrics(
        model,
        single,
        throughput,
        warmups=int(performance["batch1_warmups"]),
        samples=int(performance["batch1_samples"]),
        throughput_warmups=int(performance["throughput_warmups"]),
        throughput_samples=int(performance["throughput_samples"]),
    )
    model_performance.update(
        end_to_end_performance(
            model,
            records["test"][0],
            records["test"][:128],
            seed=seed,
            include_relations=arm == "relational_semantic_encoder_v1",
            warmups=int(performance["batch1_warmups"]),
            samples=int(performance["batch1_samples"]),
            throughput_warmups=int(performance["throughput_warmups"]),
            throughput_samples=int(performance["throughput_samples"]),
        )
    )
    result = {
        "arm": arm,
        "model_seed": seed,
        "training": receipt,
        "train": train_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
        "performance": model_performance,
        "parameter_breakdown": _parameter_breakdown(model),
    }
    if arm == "bag_v1":
        result["symmetry"] = paired_symmetry_audit(
            model, test_batch, records["test"]
        )
    return result


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
                    [
                        float(run[split]["by_family"][family][metric])
                        for run in runs
                    ]
                )
                for metric in ("accuracy", "brier", "ece_5_bin", "nll")
            }
            for family in KATA_FAMILIES
        }
    out["performance"] = {
        metric: t_interval([float(run["performance"][metric]) for run in runs])
        for metric in (
            "batch128_examples_per_second",
            "batch128_tokens_per_second",
            "catalog_projector_plus_model_batch128_examples_per_second",
            "catalog_projector_plus_model_batch128_tokens_per_second",
            "parameter_count",
            "parameter_bytes",
        )
    }
    out["performance"]["model_batch1_p50_ns"] = t_interval(
        [float(run["performance"]["model_batch1_latency_ns"]["p50"]) for run in runs]
    )
    out["performance"]["model_batch1_p95_ns"] = t_interval(
        [float(run["performance"]["model_batch1_latency_ns"]["p95"]) for run in runs]
    )
    out["performance"]["projector_model_batch1_p50_ns"] = t_interval(
        [
            float(
                run["performance"][
                    "catalog_projector_plus_model_batch1_latency_ns"
                ]["p50"]
            )
            for run in runs
        ]
    )
    out["performance"]["projector_model_batch1_p95_ns"] = t_interval(
        [
            float(
                run["performance"][
                    "catalog_projector_plus_model_batch1_latency_ns"
                ]["p95"]
            )
            for run in runs
        ]
    )
    return out


def _decision(
    runs_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    aggregate: Mapping[str, Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    bag_runs = runs_by_arm["bag_v1"]
    structural_runs = runs_by_arm["relational_semantic_encoder_v1"]
    gates = contract["gates"]
    bag_symmetry = all(
        run["test"]["accuracy"] == 0.5
        and all(
            row["accuracy"] == 0.5
            for row in run["test"]["by_family"].values()
        )
        and run["symmetry"]["paired_prediction_disagreements"] == 0
        and run["symmetry"]["maximum_paired_probability_difference"] <= 1e-6
        for run in bag_runs
    )
    trainability = all(
        run["train"]["accuracy"]
        >= gates["structural_train_accuracy_each_seed_minimum"]
        and run["validation"]["accuracy"]
        >= gates["structural_validation_accuracy_each_seed_minimum"]
        for run in structural_runs
    )
    uplift = {
        "aggregate": t_interval(
            [
                float(structural["test"]["accuracy"] - bag["test"]["accuracy"])
                for structural, bag in zip(structural_runs, bag_runs, strict=True)
            ]
        ),
        "by_family": {
            family: t_interval(
                [
                    float(
                        structural["test"]["by_family"][family]["accuracy"]
                        - bag["test"]["by_family"][family]["accuracy"]
                    )
                    for structural, bag in zip(structural_runs, bag_runs, strict=True)
                ]
            )
            for family in KATA_FAMILIES
        },
    }
    structural = aggregate["relational_semantic_encoder_v1"]
    failed_families = [
        family
        for family in KATA_FAMILIES
        if structural["test"]["by_family"][family]["accuracy"]["mean"]
        < gates["structural_per_family_test_accuracy_minimum"]
        or uplift["by_family"][family]["mean"]
        < gates["per_family_uplift_minimum"]
    ]
    semantic_accuracy = (
        structural["test"]["accuracy"]["mean"]
        >= gates["structural_aggregate_test_accuracy_minimum"]
        and not failed_families
        and uplift["aggregate"]["t95_low"]
        > gates["aggregate_uplift_t95_lower_minimum_exclusive"]
    )
    calibration = (
        structural["test"]["brier"]["mean"]
        <= gates["structural_brier_maximum"]
        and structural["test"]["nll"]["mean"]
        <= gates["structural_nll_maximum"]
    )

    parameter_fractions = [
        abs(
            structural_run["performance"]["parameter_count"]
            - bag_run["performance"]["parameter_count"]
        )
        / bag_run["performance"]["parameter_count"]
        for structural_run, bag_run in zip(structural_runs, bag_runs, strict=True)
    ]
    latency_ratios = [
        structural_run["performance"][
            "catalog_projector_plus_model_batch1_latency_ns"
        ]["p95"]
        / bag_run["performance"]["catalog_projector_plus_model_batch1_latency_ns"][
            "p95"
        ]
        for structural_run, bag_run in zip(structural_runs, bag_runs, strict=True)
    ]
    throughput_ratios = [
        structural_run["performance"][
            "catalog_projector_plus_model_batch128_examples_per_second"
        ]
        / bag_run["performance"][
            "catalog_projector_plus_model_batch128_examples_per_second"
        ]
        for structural_run, bag_run in zip(structural_runs, bag_runs, strict=True)
    ]
    cost = (
        max(parameter_fractions)
        <= gates["parameter_difference_fraction_maximum"]
        and max(latency_ratios)
        <= gates["structural_batch1_p95_ratio_maximum"]
        and min(throughput_ratios)
        >= gates["structural_batch128_throughput_ratio_minimum"]
    )
    evidence = {
        "bag_symmetry": bag_symmetry,
        "structural_trainability": trainability,
        "semantic_accuracy": semantic_accuracy,
        "calibration": calibration,
        "cost": cost,
        "failed_families": failed_families,
        "uplift": uplift,
        "parameter_difference_fractions": parameter_fractions,
        "projector_model_batch1_p95_ratios": latency_ratios,
        "projector_model_batch128_throughput_ratios": throughput_ratios,
    }
    if not bag_symmetry:
        return "REDESIGN instrument_invalid", evidence
    if not trainability:
        return "REDESIGN optimization_or_capacity_unresolved", evidence
    if 1 <= len(failed_families) <= 2:
        return "REDESIGN missing_structural_relation", evidence
    if not semantic_accuracy or not calibration:
        return "REDESIGN encoder_redesign", evidence
    if not cost:
        return "REDESIGN cost_redesign", evidence
    return "NOMINATE_FOR_W2_213 relational_semantic_encoder_v1", evidence


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _render_report(result: Mapping[str, Any]) -> str:
    aggregate = result["aggregate"]
    decision = result["decision"]["terminal"]
    lines = [
        "# W2-214 structural semantic katas",
        "",
        f"Status: **{result['status'].upper()}**",
        "",
        f"Decision: **`{decision}`**",
        "",
        "World: **offline static semantic diagnostic; no gameplay ABI or world change**",
        "",
        "## Claim boundary",
        "",
        result["claim_boundary"],
        "",
        "A nomination is only a candidate intake for W2-213. Dynamic binding,",
        "held-out recombination, card transfer, and gameplay integration remain untested.",
        "",
        "## Per-kata result",
        "",
        "| Kata | bag accuracy | structural accuracy | uplift | structural Brier | structural NLL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    bag = aggregate["bag_v1"]
    structural = aggregate["relational_semantic_encoder_v1"]
    for family in KATA_FAMILIES:
        bag_accuracy = bag["test"]["by_family"][family]["accuracy"]
        structural_accuracy = structural["test"]["by_family"][family]["accuracy"]
        uplift = result["decision"]["evidence"]["uplift"]["by_family"][family]
        lines.append(
            f"| `{family}` | {_percent(bag_accuracy['mean'])} "
            f"[{_percent(bag_accuracy['t95_low'])}, {_percent(bag_accuracy['t95_high'])}] | "
            f"{_percent(structural_accuracy['mean'])} "
            f"[{_percent(structural_accuracy['t95_low'])}, {_percent(structural_accuracy['t95_high'])}] | "
            f"{_percent(uplift['mean'])} | "
            f"{structural['test']['by_family'][family]['brier']['mean']:.4f} | "
            f"{structural['test']['by_family'][family]['nll']['mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Training seed is the independent unit; brackets are two-sided 95% t intervals over five seeds.",
            "",
            "## Aggregate result",
            "",
            "| Arm | train accuracy | validation accuracy | test accuracy | Brier | NLL | ECE (5 bins) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in ("bag_v1", "relational_semantic_encoder_v1"):
        row = aggregate[arm]
        lines.append(
            f"| `{arm}` | {_percent(row['train']['accuracy']['mean'])} | "
            f"{_percent(row['validation']['accuracy']['mean'])} | "
            f"{_percent(row['test']['accuracy']['mean'])} | "
            f"{row['test']['brier']['mean']:.4f} | {row['test']['nll']['mean']:.4f} | "
            f"{row['test']['ece_5_bin']['mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Seed receipts",
            "",
            "| Arm | seed | selected step | train | validation | test | parameters | model p50/p95 (µs) | projector+model p50/p95 (µs) | batch-128 examples/s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run in result["runs"]:
        perf = run["performance"]
        model_latency = perf["model_batch1_latency_ns"]
        projected_latency = perf["catalog_projector_plus_model_batch1_latency_ns"]
        lines.append(
            f"| `{run['arm']}` | {run['model_seed']} | "
            f"{run['training']['selected_checkpoint_step']} | "
            f"{_percent(run['train']['accuracy'])} | {_percent(run['validation']['accuracy'])} | "
            f"{_percent(run['test']['accuracy'])} | {perf['parameter_count']} | "
            f"{model_latency['p50'] / 1000:.1f}/{model_latency['p95'] / 1000:.1f} | "
            f"{projected_latency['p50'] / 1000:.1f}/{projected_latency['p95'] / 1000:.1f} | "
            f"{perf['catalog_projector_plus_model_batch128_examples_per_second']:.0f} |"
        )
    evidence = result["decision"]["evidence"]
    audit = result["suite_audit"]
    lines.extend(
        [
            "",
            "## Gates and instrument audit",
            "",
            f"- Bag exact symmetry: **{evidence['bag_symmetry']}**.",
            f"- Structural trainability: **{evidence['structural_trainability']}**.",
            f"- Static semantic accuracy: **{evidence['semantic_accuracy']}**.",
            f"- Calibration: **{evidence['calibration']}**.",
            f"- Matched parameter/CPU cost: **{evidence['cost']}**.",
            f"- Aggregate uplift t95 lower bound: {_percent(evidence['uplift']['aggregate']['t95_low'])}.",
            f"- Maximum parameter difference: {_percent(max(evidence['parameter_difference_fractions']))}.",
            f"- Worst projector+model p95 ratio: {max(evidence['projector_model_batch1_p95_ratios']):.3f}x.",
            f"- Worst projector+model throughput ratio: {min(evidence['projector_model_batch128_throughput_ratios']):.3f}x.",
            "- Normalized-program split overlaps: " + json.dumps(audit["overlap"]["normalized_program_hash"], sort_keys=True) + ".",
            "- Nuisance split overlaps: " + json.dumps(audit["overlap"]["nuisance_signature"], sort_keys=True) + ".",
            "- Pair-template split overlaps: " + json.dumps(audit["overlap"]["pair_template_signature"], sort_keys=True) + ".",
            "- Intentional family-skeleton overlaps: " + json.dumps(audit["overlap"]["skeleton_id"], sort_keys=True) + ".",
            f"- Definition-reference tokens: {audit['definition_references']}; opaque identity fields tensorized: {audit['identity_fields_tensorized']}.",
            "- All checked label-independent field contingencies are exactly balanced; full tables are in the JSON receipt.",
            "",
            "## Provenance and budget",
            "",
            f"- Pre-registration revision: `{result['provenance']['preregistration_revision']}`.",
            f"- Contract SHA-256: `{result['provenance']['contract_sha256']}`.",
            f"- Suite SHA-256: `{result['provenance']['suite_sha256']}`.",
            f"- Compiler SHA-256: `{result['provenance']['compiler_sha256']}`.",
            f"- Learning-schema SHA-256: `{result['provenance']['learning_schema_sha256']}`.",
            f"- Optimizer steps: {result['budget']['optimizer_steps']} / {result['budget']['optimizer_steps_cap']}.",
            f"- Presented examples: {result['budget']['presented_examples']} / {result['budget']['presented_examples_cap']}.",
            f"- Wall clock: {result['budget']['wall_clock_seconds']:.2f}s / {result['budget']['wall_clock_seconds_cap']}s.",
            "",
            "## Result-contingent next step",
            "",
            f"`{decision}`",
            "",
        ]
    )
    if decision.startswith("NOMINATE_FOR_W2_213"):
        lines.extend(
            [
                "W2-213 may now test the nominated static encoder on held-out",
                "compositions, identities, and dynamic bindings. This result does not",
                "authorize gameplay integration; the full Semantic Project and the",
                "independent Teacher Project remain required gates.",
            ]
        )
    else:
        lines.append(
            "The declared redesign branch is the only authorized continuation; no model scaling or gameplay integration follows from this result."
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
    suite_path = (suite_path if suite_path.is_absolute() else ROOT / suite_path).resolve()
    contract, contract_digest = load_contract(contract_path)
    suite, suite_digest = load_suite(suite_path)
    preregistration_revision = _verify_preregistration(
        preregistration_revision, contract_path, suite_path, contract
    )
    if contract["authority"]["suite_sha256"] != suite_digest:
        raise StructuralKataError("loaded suite differs from the contract")

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    records = {
        split: [row for row in suite["programs"] if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    started = perf_counter_ns()
    runs = []
    total_steps = 0
    total_examples = 0
    for seed in contract["model_seeds"]:
        bag, structural = build_matched_models(
            token_count=len(suite["token_vocabulary"]),
            token_kind_count=int(suite["token_kind_count"]),
            seed=int(seed),
        )
        bag_count = trainable_parameter_count(bag)
        structural_count = trainable_parameter_count(structural)
        difference = abs(structural_count - bag_count) / bag_count
        if difference > contract["gates"]["parameter_difference_fraction_maximum"]:
            raise StructuralKataError(
                f"parameter match failed before training: {difference:.6f}"
            )
        if not all(
            torch.equal(left, right)
            for left, right in zip(
                bag.probe.parameters(), structural.probe.parameters(), strict=True
            )
        ):
            raise StructuralKataError("family heads are not byte-identical at start")
        for arm, model in (
            ("bag_v1", bag),
            ("relational_semantic_encoder_v1", structural),
        ):
            run_receipt = _run_one(
                arm=arm,
                model=model,
                seed=int(seed),
                records=records,
                contract=contract,
            )
            runs.append(run_receipt)
            total_steps += int(run_receipt["training"]["optimizer_steps"])
            total_examples += int(run_receipt["training"]["presented_examples"])
            elapsed = (perf_counter_ns() - started) / 1_000_000_000
            if elapsed > contract["budget"]["maximum_wall_clock_seconds"]:
                raise StructuralKataError("wall-clock cap exceeded; partial runs discarded")

    if total_steps > contract["budget"]["maximum_optimizer_steps"]:
        raise StructuralKataError("optimizer-step cap exceeded")
    if total_examples > contract["budget"]["maximum_presented_examples"]:
        raise StructuralKataError("presented-example cap exceeded")

    runs_by_arm = {
        arm: sorted(
            [run for run in runs if run["arm"] == arm],
            key=lambda run: run["model_seed"],
        )
        for arm in ("bag_v1", "relational_semantic_encoder_v1")
    }
    aggregate = {arm: _aggregate(rows) for arm, rows in runs_by_arm.items()}
    terminal, evidence = _decision(runs_by_arm, aggregate, contract)
    elapsed = (perf_counter_ns() - started) / 1_000_000_000
    result = {
        "schema_version": 1,
        "status": "pass" if terminal.startswith("NOMINATE") else "redesign",
        "experiment": contract["id"],
        "claim_boundary": contract["claim_boundary"],
        "decision": {"terminal": terminal, "evidence": evidence},
        "provenance": {
            "preregistration_revision": preregistration_revision,
            "measurement_code_revision": _git(["rev-parse", "HEAD"]).decode().strip(),
            "contract_path": str(contract_path),
            "contract_sha256": contract_digest,
            "suite_path": str(suite_path),
            "suite_sha256": suite_digest,
            "source_sha256": contract["authority"]["source_sha256"],
            "oracle_sha256": contract["authority"]["oracle_sha256"],
            "compiler_sha256": contract["authority"]["compiler_sha256"],
            "learning_schema_sha256": contract["authority"]["learning_schema_sha256"],
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
        "budget": {
            "optimizer_steps": total_steps,
            "optimizer_steps_cap": contract["budget"]["maximum_optimizer_steps"],
            "presented_examples": total_examples,
            "presented_examples_cap": contract["budget"]["maximum_presented_examples"],
            "wall_clock_seconds": elapsed,
            "wall_clock_seconds_cap": contract["budget"]["maximum_wall_clock_seconds"],
        },
        "suite_audit": suite["audit"],
        "runs": sorted(runs, key=lambda run: (run["arm"], run["model_seed"])),
        "aggregate": aggregate,
        "contract": contract,
    }
    return result


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
    args.report.write_text(_render_report(result), encoding="utf-8")
    print(result["decision"]["terminal"])


if __name__ == "__main__":
    main()
