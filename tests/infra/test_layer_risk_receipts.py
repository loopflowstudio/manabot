"""A faster payment must still preserve every legal action and spent resource."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _receipts(directory: Path, mutation: str | None = None) -> None:
    for leg in ("before", "after"):
        (directory / leg).mkdir()
        rows = []
        for funding in ("pool", "lands"):
            for sample in range(8):
                actions = ["activate", "pass"]
                mana = 0
                digest = "paid"
                if leg == "after" and sample == 0:
                    if mutation == "action":
                        actions.pop()
                    elif mutation == "resource":
                        mana = 1
                    elif mutation == "digest":
                        digest = "different"
                row = {
                    "funding": funding,
                    "sample": sample,
                    "before": {"legal_actions": actions, "digest": "root"},
                    "trace": [
                        {
                            "index": 0,
                            "action": "activate",
                            "after": {"mana": mana, "digest": digest},
                        }
                    ],
                    "metrics": {"boards": 2 if leg == "before" else 1},
                    "allocation_count": 200 if leg == "before" else 100,
                    "allocated_bytes": 2000 if leg == "before" else 1000,
                    "counting_allocations": True,
                }
                rows.append(row)
        (directory / leg / "payment.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows)
        )


def _compare(
    directory: Path, *, payment_only: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/measure_layer_risk.py"),
            "compare",
            "--out",
            str(directory),
            *(["--payment-only"] if payment_only else []),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_layer_payment_comparison_retains_improvement_without_changing_semantics(
    tmp_path: Path,
) -> None:
    _receipts(tmp_path)
    result = _compare(tmp_path)
    assert result.returncode == 0, result.stderr
    receipt = json.loads((tmp_path / "comparison.json").read_text())
    assert len(receipt["payment"]) == 16
    assert receipt["payment"][0]["after_allocations"] == 100
    assert receipt["payment"][0]["semantics_equal"]


@pytest.mark.parametrize("mutation", ["action", "resource", "digest"])
def test_layer_payment_comparison_rejects_shortcuts(
    tmp_path: Path, mutation: str
) -> None:
    _receipts(tmp_path, mutation)
    result = _compare(tmp_path)
    assert result.returncode != 0
    assert "Payment pool/0 differs" in result.stderr
    assert not (tmp_path / "comparison.json").exists()


def _search_receipts(
    directory: Path, *, missing_seed: bool = False, fewer_simulations: bool = False
) -> None:
    """A completed steps receipt followed by a deliberately bounded search file."""
    for leg in ("baseline", "before", "after"):
        (directory / leg).mkdir(exist_ok=True)
        steps = [
            {"games": 512, "steps": 1000, "digest": 11, "capped": 0},
            {"step_prefix_samples_s": [0.01] * 128},
        ]
        (directory / leg / "bench_steps.jsonl").write_text(
            "\n".join(json.dumps(row) for row in steps)
        )
        search = [
            {
                "seed": seed,
                "decisions": 40,
                "samples_s": [0.1] * 40,
                "action_counts": [3] * 40,
                "simulations": 7680,
                "cap_hits": 0,
                "digest": seed,
                "score_digest_blake3": str(seed) * 64,
            }
            for seed in ((1, 2) if missing_seed else (1, 2, 3))
        ]
        if leg == "after" and fewer_simulations:
            search[0]["simulations"] = 1
        (directory / leg / "bench_search.jsonl").write_text(
            "\n".join(json.dumps(row) for row in search)
        )


@pytest.mark.parametrize("missing_seed", [False, True])
def test_layer_risk_rejects_fewer_simulations_or_incomplete_search(
    tmp_path: Path, missing_seed: bool
) -> None:
    _receipts(tmp_path)
    _search_receipts(
        tmp_path, missing_seed=missing_seed, fewer_simulations=not missing_seed
    )
    result = _compare(tmp_path, payment_only=False)
    assert result.returncode != 0
    assert (
        "Expected all three original search seeds"
        if missing_seed
        else "Mitigation changed workload semantics: bench_search"
    ) in result.stderr
    assert not (tmp_path / "comparison.json").exists()


def _full_receipts(directory: Path) -> None:
    """Injected measurements exercise the comparison CLI without executing games."""
    _receipts(directory)
    _search_receipts(directory)
    for leg in ("before", "after"):
        dependency = [
            {
                "workload": "dependency_steps",
                "games": 16,
                "digest": "steps",
                "step_prefix_samples_s": [0.01] * 128,
            },
            *(
                {"workload": "dependency_search", "decision": i, "digest": str(i)}
                for i in range(8)
            ),
        ]
        for name in ("layers", "layers-allocations"):
            (directory / leg / f"{name}.jsonl").write_text(
                "\n".join(json.dumps(row) for row in dependency)
            )
        for state in ("ordinary", "shared", "program"):
            for version in (8,) if state == "program" else (6, 7, 8):
                receipt = {
                    "state": state,
                    "version": version,
                    "inventory_only": False,
                    "observation_sha256": "a" * 64,
                    "observation_config": {"policy_input_version": version},
                    "model_config": {"hidden_dim": 64},
                    "batch": 16,
                    "device": "cpu",
                    "dtype": "float32",
                    "threads": 1,
                    "interop_threads": 1,
                    "torch": "2.10.0",
                    "platform": "fixture",
                    "observation_bytes": 200,
                    "batch_observation_bytes": 3200,
                    "parameter_bytes": 1000,
                    "peak_rss_bytes": 2000,
                    "live_details": 21,
                    "live_facts": 11,
                }
                for operation in (
                    "native_encode",
                    "python_encode",
                    "observation",
                    "inference",
                    "forward_backward",
                ):
                    receipt[operation] = {
                        "samples_s": [0.01 if leg == "before" else 0.02] * 8,
                        "sample_count": 8,
                        "warmups": 2,
                    }
                (directory / leg / f"input-v{version}-{state}.json").write_text(
                    json.dumps(receipt)
                )


def test_layer_risk_retains_all_matched_cases_without_approving_their_timings(
    tmp_path: Path,
) -> None:
    _full_receipts(tmp_path)
    result = _compare(tmp_path, payment_only=False)
    assert result.returncode == 0, result.stderr
    receipt = json.loads((tmp_path / "comparison.json").read_text())
    assert len(receipt["workloads"]) == 4
    assert len(receipt["policy"]) == 7
    assert receipt["policy"][0]["after"]["inference"]["samples_s"] == [0.02] * 8


@pytest.mark.parametrize("mutation", ["duplicate", "missing"])
def test_layer_payment_requires_each_funding_case_and_sample(
    tmp_path: Path, mutation: str
) -> None:
    _receipts(tmp_path)
    # Identical corruptions on both sides must not count as semantic agreement.
    for leg in ("before", "after"):
        path = tmp_path / leg / "payment.jsonl"
        lines = path.read_text().splitlines()
        if mutation == "duplicate":
            lines[-1] = lines[0]
        else:
            lines.pop()
        path.write_text("\n".join(lines))
    result = _compare(tmp_path)
    assert result.returncode != 0
    assert "Expected eight complete payments" in result.stderr


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("missing", "input-v8-program.json"),
        ("label", "Mislabeled policy case"),
        ("inventory", "Inventory is not a policy timing receipt"),
        ("samples", "Expected 8 finite positive timing samples"),
        ("nan", "Expected 8 finite positive timing samples"),
        ("warmups", "Unmatched policy sampling"),
        ("storage", "Unmatched policy input"),
        ("memory", "Missing or invalid policy memory"),
    ],
)
def test_layer_risk_rejects_incomplete_or_unmatched_policy_receipts(
    tmp_path: Path, mutation: str, message: str
) -> None:
    _full_receipts(tmp_path)
    path = tmp_path / "after/input-v8-program.json"
    receipt = json.loads(path.read_text())
    if mutation == "label":
        receipt["state"] = "ordinary"
    elif mutation == "inventory":
        receipt["inventory_only"] = True
    elif mutation == "samples":
        receipt["forward_backward"]["samples_s"] = []
    elif mutation == "nan":
        receipt["inference"]["samples_s"][0] = float("nan")
    elif mutation == "warmups":
        receipt["python_encode"]["warmups"] = 0
    elif mutation == "storage":
        receipt["observation_bytes"] += 1
    elif mutation == "memory":
        del receipt["peak_rss_bytes"]
    path.write_text(json.dumps(receipt))
    if mutation == "missing":
        path.rename(path.with_name("input-v8-unrelated.json"))
    result = _compare(tmp_path, payment_only=False)
    assert result.returncode != 0
    assert message in result.stderr
    assert not (tmp_path / "comparison.json").exists()
