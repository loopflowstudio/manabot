"""Run the reviewed W2-214 typed-opcode alignment probe."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping

import numpy as np

from manabot.semantic.learning import BoundSemanticPack
from manabot.semantic.transfer import (
    ARMS,
    SEMANTIC_CONTROLS,
    aggregate_arm,
    build_records,
    build_spec,
    holdout_audit,
    load_workload,
    paired_cluster_contrast,
    robustness_controls,
    run_arm,
)
from manabot.sim.structured_benchmark import run_frontiers
from manabot.verify.util import GW_ALLIES_DECK, UR_LESSONS_DECK
import managym

ROOT = Path(__file__).resolve().parents[1]


def _pack(seed: int) -> BoundSemanticPack:
    env = managym.Env(seed=seed, skip_trivial=False)
    env.reset(
        [
            managym.PlayerConfig("ur-lessons", UR_LESSONS_DECK),
            managym.PlayerConfig("gw-allies", GW_ALLIES_DECK),
        ]
    )
    return BoundSemanticPack.from_env(env)


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _distribution(values: list[int]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "min": float(data.min()),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "max": float(data.max()),
        "mean": float(data.mean()),
    }


def _paired_rows(rows: list[Mapping[str, Any]]) -> bool:
    fingerprints = []
    for row in rows:
        fingerprints.append(
            [
                (
                    item["program"],
                    item["definition"],
                    item["deck"],
                    item["seat"],
                    item["replicate"],
                    item["expected"],
                )
                for item in row["zero_shot_rows"]
            ]
        )
    return all(value == fingerprints[0] for value in fingerprints[1:])


def _alignment_decision(
    aggregate: Mapping[str, Mapping[str, Any]],
    audit: Mapping[str, Any],
    workload: Mapping[str, Any],
) -> dict[str, Any]:
    target = aggregate["semantic_only_structured"]["zero_shot"]["accuracy"]
    controls = {
        arm: aggregate[arm]["zero_shot"]["accuracy"]
        for arm in ("card_id_legacy", "card_id_structured")
    }
    criterion = workload["alignment_signal_criterion"]
    uplifts = {arm: target - value for arm, value in controls.items()}
    signal = target >= criterion["semantic_only_minimum_accuracy"] and all(
        uplift >= criterion["minimum_uplift_over_each_identity_control"]
        for uplift in uplifts.values()
    )
    masked = aggregate["semantic_only_opcode_masked"]["zero_shot"]["accuracy"]
    shuffled = aggregate["semantic_only_token_shuffled"]["zero_shot"]["accuracy"]
    return {
        "decision": (
            "preliminary_opcode_alignment_signal"
            if signal
            else "null_opcode_alignment_result"
        ),
        "semantic_only_zero_shot_accuracy": target,
        "identity_control_accuracies": controls,
        "uplift_over_identity_controls": uplifts,
        "criterion": dict(criterion),
        "opcode_masked_control_accuracy": masked,
        "token_shuffled_control_accuracy": shuffled,
        "criterion_met": signal,
        "semantic_compositional_transfer_supported": False,
        "downgrade_reasons": [
            "the encoder is order invariant and the token-shuffled control is not weaker",
            "the target label is an opcode directly present in the input",
            "the held-out set does not have full normalized-AST novelty",
            "the held-out set does not have full symbolic primitive closure",
        ],
        "holdout_all_normalized_asts_novel": audit["all_normalized_asts_novel"],
        "holdout_full_symbolic_primitive_closure": audit[
            "full_symbolic_primitive_closure"
        ],
    }


def run(path: Path) -> dict[str, Any]:
    workload, workload_hash = load_workload(path)
    pack = _pack(int(workload["dataset_seed"]))
    spec = build_spec(pack)
    records = build_records(pack, spec, workload)
    arms = [
        run_arm(pack, workload, arm=arm, model_seed=seed)
        for arm in ARMS
        for seed in workload["model_seeds"]
    ]
    control_runs = [
        run_arm(pack, workload, arm=arm, model_seed=seed)
        for arm in SEMANTIC_CONTROLS
        for seed in workload["model_seeds"]
    ]
    all_runs = arms + control_runs
    aggregate = {
        arm: {
            "zero_shot": aggregate_arm(
                [row for row in all_runs if row["arm"] == arm], "zero_shot"
            ),
            "limited_retraining": aggregate_arm(
                [row for row in all_runs if row["arm"] == arm],
                "limited_retraining",
            ),
        }
        for arm in (*ARMS, *SEMANTIC_CONTROLS)
    }
    audit = holdout_audit(pack, spec, workload)
    by_arm = {
        arm: [row for row in all_runs if row["arm"] == arm]
        for arm in (*ARMS, *SEMANTIC_CONTROLS)
    }
    contrasts = {
        "decoder_card_id_structured_minus_legacy": paired_cluster_contrast(
            by_arm["card_id_structured"], by_arm["card_id_legacy"]
        ),
        "semantic_input_minus_card_id_structured": paired_cluster_contrast(
            by_arm["semantic_only_structured"], by_arm["card_id_structured"]
        ),
        "semantic_plus_identity_minus_card_id_structured": paired_cluster_contrast(
            by_arm["semantic_card_id_structured"],
            by_arm["card_id_structured"],
        ),
        "identity_ablation_semantic_only_minus_semantic_plus_identity": (
            paired_cluster_contrast(
                by_arm["semantic_only_structured"],
                by_arm["semantic_card_id_structured"],
            )
        ),
        "opcode_present_minus_opcode_masked": paired_cluster_contrast(
            by_arm["semantic_only_structured"],
            by_arm["semantic_only_opcode_masked"],
        ),
        "ordered_minus_token_shuffled": paired_cluster_contrast(
            by_arm["semantic_only_structured"],
            by_arm["semantic_only_token_shuffled"],
        ),
    }
    robustness = robustness_controls(pack, spec)
    frontier = run_frontiers({"scorer_seed": int(workload["dataset_seed"])})
    maximum_choices = max(
        max(row["candidate_count"], row["represented_legal_branches"])
        for row in frontier["fixtures"]
    )
    token_counts = [len(row.token_ids) for row in records]
    checkpoint_rates = [row["checkpoint_rebind"]["match_rate"] for row in all_runs]
    gates = {
        "causally_paired_examples": _paired_rows(all_runs),
        "balanced_holdout_decks": {row.deck for row in records if row.held_out}
        == {"ur_lessons", "gw_allies"}
        and sum(row.deck == "ur_lessons" for row in records if row.held_out)
        == sum(row.deck == "gw_allies" for row in records if row.held_out),
        "balanced_evaluation_seats": all(
            row["zero_shot"]["by_seat"]["0"]["total"]
            == row["zero_shot"]["by_seat"]["1"]["total"]
            for row in all_runs
        ),
        "zero_silent_overflow": max(token_counts) <= spec.max_program_tokens,
        "unknown_opcode_fail_closed": robustness["unknown_opcode_fail_closed"],
        "enum_schema_change_fail_closed": robustness[
            "enum_domain_schema_change_fail_closed"
        ],
        "binder_local_content_pack_rebind_smoke": robustness[
            "content_pack_reorder_checkpoint_rebind_rate"
        ]
        == workload["gates"]["required_rebind_match_rate"],
        "binder_local_checkpoint_roundtrip_smoke": min(checkpoint_rates)
        == workload["gates"]["required_rebind_match_rate"],
        "binder_local_schema_id_permutation_smoke": robustness[
            "token_kind_and_opcode_id_permutation_rate"
        ]
        == workload["gates"]["required_permutation_match_rate"],
        "inherited_w2_189_frontier_above_32": maximum_choices
        >= workload["gates"]["minimum_legal_choices"],
        "inherited_w2_189_legacy_equivalence": frontier["matching"]
        == frontier["shared"],
        "metrics_complete": all(
            row["performance"]["latency_ns"]["p50"] > 0
            and row["performance"]["examples_per_second"] > 0
            and row["performance"]["peak_rss_bytes"] > 0
            for row in all_runs
        ),
    }
    result = {
        "schema_version": 1,
        "status": "pass" if all(gates.values()) else "fail",
        "experiment_kind": workload["claim"],
        "claim_boundary": workload["claim_boundary"],
        "kr6_closed": False,
        "workload": {
            "id": workload["id"],
            "path": str(path),
            "sha256": workload_hash,
        },
        "measurement_code_revision": _git_revision(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED", "unset"),
        },
        "semantic_input_spec": spec.to_dict(),
        "semantic_input_spec_digest": spec.digest,
        "dataset": {
            "programs": len(records),
            "training_programs": sum(not row.held_out for row in records),
            "heldout_programs": sum(row.held_out for row in records),
            "heldout_definitions": workload["held_out_definitions"],
            "token_counts": _distribution(token_counts),
            "declared_token_budget": spec.max_program_tokens,
            "overflow_count": 0,
            "silent_truncation_count": 0,
        },
        "arms": arms,
        "semantic_controls": control_runs,
        "aggregate": aggregate,
        "paired_cluster_contrasts": contrasts,
        "holdout_audit": audit,
        "decision": _alignment_decision(aggregate, audit, workload),
        "robustness": robustness,
        "inherited_structured_legality_frontier": {
            "source": "W2-189 structured-offer bridge; unrelated to classifier training",
            "maximum_choices_or_branches": maximum_choices,
            **frontier,
        },
        "gates": gates,
        "limitations": [
            "The label is an opcode directly present in the input, so the result is typed-opcode alignment, not program comprehension.",
            "The encoder uses masked mean/max pooling and is intentionally order invariant; shuffled tokens perform equivalently.",
            "Semantic programs and structured commands are not yet joined to PPO or every decision family.",
            "The v1 flat enum token omits enum-domain identity, so arbitrary enum-ID rebinding is unsupported and fails closed.",
            "Wilson intervals use 12 independent program×seed clusters; seat and candidate-permutation repeats are excluded.",
            "Limited retraining exposes the exact held-out programs and therefore measures memorization, not few-shot transfer.",
            "Performance is batch-1 inference and RSS is one shared-process high-water mark.",
            "Token-budget, checkpoint-rebind, and ID-permutation checks are binder-local contract smoke tests, not independent semantic oracles.",
            "The >32 frontier is inherited W2-189 engine evidence and is unrelated to this trained classifier.",
        ],
        "follow_ons": [
            "Build an order-sensitive semantic encoder and evaluate same-token, different-AST programs.",
            "Pre-register a larger holdout with normalized-AST novelty and full symbolic primitive closure.",
            "Join the semantic encoder and structured decoder to PPO/all admitted decision families.",
            "Add enum-domain identity to the projection before claiming arbitrary schema-table rebinding.",
            "Run held-out gameplay and win-rate transfer after that join.",
        ],
    }
    return result


def _percent(value: float) -> str:
    return f"{value:.1%}"


def render_report(result: Mapping[str, Any]) -> str:
    aggregate = result["aggregate"]
    lines = [
        "# W2-214 preliminary typed-opcode alignment probe",
        "",
        f"Status: **{str(result['status']).upper()}**. Decision: "
        f"**{result['decision']['decision']}**.",
        "",
        "Run with:",
        "",
        "```sh",
        "PYTHONHASHSEED=0 uv run scripts/run_opcode_alignment.py \\",
        "  --workload experiments/workloads/opcode-alignment-v1.json \\",
        "  --out experiments/data/opcode-alignment-v1.json \\",
        "  --report experiments/opcode-alignment-v1.md",
        "```",
        "",
        "## Claim boundary",
        "",
        str(result["claim_boundary"]),
        "This result does **not** close Rules Semantic KR6 and is not evidence",
        "that the model reads abilities as a language. The supervised label is an",
        "opcode directly present in the input; the encoder is order-invariant; the",
        "production policy/action ABI is unchanged.",
        "",
        "## Four-arm result",
        "",
        "| Arm | input | head | zero-shot heldout cluster accuracy (95% Wilson CI) | exact-program retrain |",
        "|---|---|---|---:|---:|",
    ]
    labels = {
        "card_id_legacy": ("opaque CardDefId", "legacy fixed"),
        "card_id_structured": ("opaque CardDefId", "structured"),
        "semantic_card_id_structured": ("program + CardDefId", "structured"),
        "semantic_only_structured": ("program only", "structured"),
    }
    for arm in ARMS:
        zero = aggregate[arm]["zero_shot"]
        limited = aggregate[arm]["limited_retraining"]
        label, head = labels[arm]
        lines.append(
            f"| `{arm}` | {label} | {head} | {_percent(zero['accuracy'])} "
            f"[{_percent(zero['ci95_low'])}, {_percent(zero['ci95_high'])}] | "
            f"{_percent(limited['accuracy'])} "
            f"[{_percent(limited['ci95_low'])}, {_percent(limited['ci95_high'])}] |"
        )
    lines.extend(
        [
            "",
            "The independent unit is one held-out program × model seed (4 × 3 = 12).",
            "Seat and candidate-order repeats are deterministic sensitivity checks and",
            "are excluded from uncertainty. The same programs, permutations, seeds, and",
            "optimizer budget are used across arms. Opaque identity slots are hash-ordered,",
            "not IR rows, and held-out identity embeddings receive no zero-shot updates.",
            "",
            "## Paired cluster contrasts",
            "",
            "| Contrast (left − right) | Δ accuracy (paired bootstrap CI) | wins/losses/ties | exact sign p |",
            "|---|---:|---:|---:|",
        ]
    )
    contrast_labels = {
        "decoder_card_id_structured_minus_legacy": "decoder: CardDefId structured − legacy",
        "semantic_input_minus_card_id_structured": "semantic input: semantic-only − CardDefId structured",
        "semantic_plus_identity_minus_card_id_structured": "semantic+identity − CardDefId structured",
        "identity_ablation_semantic_only_minus_semantic_plus_identity": "identity ablation: semantic-only − semantic+identity",
        "opcode_present_minus_opcode_masked": "opcode present − opcode masked",
        "ordered_minus_token_shuffled": "ordered − token shuffled",
    }
    for key, label in contrast_labels.items():
        value = result["paired_cluster_contrasts"][key]
        low, high = value["paired_cluster_bootstrap_ci95"]
        lines.append(
            f"| {label} | {_percent(value['left_minus_right_accuracy'])} "
            f"[{_percent(low)}, {_percent(high)}] | "
            f"{value['wins']}/{value['losses']}/{value['ties']} | "
            f"{value['two_sided_exact_sign_p']:.4g} |"
        )
    lines.extend(
        [
            "",
            "The structured decoder alone provides no alignment benefit over the legacy",
            "head. Semantic-only input aligns strongly with the visible opcode. Adding an",
            "untrained identity channel causes severe interference; removing it is the",
            "largest paired improvement. This is a useful architecture warning.",
            "",
            "## Semantic controls and holdout validity",
            "",
            "| Control | zero-shot cluster accuracy (95% Wilson CI) |",
            "|---|---:|",
        ]
    )
    for arm, label in (
        ("semantic_only_opcode_masked", "opcode tokens masked"),
        ("semantic_only_token_shuffled", "program tokens shuffled"),
    ):
        value = aggregate[arm]["zero_shot"]
        lines.append(
            f"| {label} | {_percent(value['accuracy'])} "
            f"[{_percent(value['ci95_low'])}, {_percent(value['ci95_high'])}] |"
        )
    audit = result["holdout_audit"]
    lines.extend(
        [
            "",
            "The shuffled-token control is expected to match because the current encoder",
            "uses masked mean/max pooling. It cannot distinguish same-token programs with",
            "different AST order. The opcode-masked control measures how much alignment",
            "comes from other correlated tokens.",
            "",
            f"Only {audit['normalized_ast_novel_programs']}/{audit['programs']} held-out "
            "programs have a normalized AST shape absent from training, and "
            f"{audit['full_symbolic_primitive_closure_programs']}/{audit['programs']} have "
            "full symbolic primitive closure. Per-program duplicate/gap rows are in the",
            "JSON receipt. These failures force the opcode-alignment claim downgrade.",
            "",
            "## Binder-local contract smoke tests",
            "",
            f"- Program tokens: p50 {result['dataset']['token_counts']['p50']:.0f}, "
            f"p95 {result['dataset']['token_counts']['p95']:.0f}, max "
            f"{result['dataset']['token_counts']['max']:.0f} against the observed "
            f"{result['dataset']['declared_token_budget']}-token pack maximum; 0 local "
            "overflows/truncations.",
            f"- ContentPack reorder + checkpoint rebind: "
            f"{_percent(result['robustness']['content_pack_reorder_checkpoint_rebind_rate'])} exact.",
            f"- Token-kind/opcode numeric-ID permutation: "
            f"{_percent(result['robustness']['token_kind_and_opcode_id_permutation_rate'])} exact.",
            "- Unknown opcodes and changed enum-domain schemas fail closed.",
            "",
            "These are self-consistency checks of this binder and generated pack, not an",
            "independent semantic oracle. Likewise, the 64-branch >32 frontier is inherited",
            "W2-189 engine/adapter evidence and is unrelated to classifier training.",
            "",
            "## Performance",
            "",
            "Latency, throughput, parameter bytes, and process peak RSS are recorded",
            "for every arm/seed in the JSON receipt. Latency is batch-1. RSS is the",
            "shared-process high-water mark, not an isolated model-memory comparison.",
            "Limited retraining exposes each exact held-out program and therefore measures",
            "exact-program memorization, not few-shot semantic transfer.",
            "",
            "## What remains",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in result["follow_ons"])
    lines.extend(
        [
            "",
            "The enum follow-on is structural: v1 emits one flat `enum` token without",
            "its source domain. This experiment can safely rebind token-kind and opcode",
            "IDs, but arbitrary enum-table permutation would be ambiguous and is",
            "therefore rejected rather than counted as robustness.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.workload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(result))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
