"""Plan ETU-80 time/cash envelopes; never launch training or cloud jobs.

uv run --no-project --python 3.12 python scripts/budget_plan.py --help
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = {"10min": 600, "1hr": 3600, "1day": 86400, "1week": 604800}
# Serial phase allowances in seconds; only METRICS phases may be scaled.
PHASES = ("setup", "generation", "training", "export", "evaluation")
ALLOWANCES = {"10min": (120, 180, 120, 60, 120),
              "1hr": (300, 1500, 900, 120, 780),
              "1day": (900, 39600, 18000, 300, 14400),
              "1week": (900, 86400, 72000, 300, 43200)}
METRICS = {"generation": ("accepted_labels", "accepted_training_rows", "complete_games"),
           "training": ("training_exposures",), "evaluation": ("scored_paired_deals",)}
CONSUMERS = {
    "setup": "ETU-79 corrected-world binding and device smoke",
    "generation": "manabot.sim.selected_branchdriver_teacher.run_teacher_game",
    "training": "manabot.sim.search_supervised.train_search_supervised",
    "export": "manabot.sim.distill.save_bc_checkpoint -> etude.villain.CheckpointVillain",
    "evaluation": "manabot.arena.match.play_cell (ETU-79 selected-match binding required)",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def number(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("numbers must be finite and nonnegative")
    return result


@dataclass(frozen=True)
class Inputs:
    local_hardware: str = "M4-Max-128GiB"
    local_hours: float = 48
    electricity_low: float = .03
    electricity_high: float = .09
    local_workers: int = 4
    setup_seconds: float | None = None
    cloud_startup_seconds: float = 900
    transfer_seconds: float = 120
    storage_usd: float = 1
    transfer_usd: float = 1
    tax_rate: float = .10
    inference_mode: str = "policy_only"
    inference_device: str = "cpu"
    inference_p95_ms: float = 100
    inference_rss_gib: float = 1
    inference_search_simulations: int = 0
    world_identity: str = "pending"
    pipeline_identity: str = "pending"
    config_identity: str = "pending"
    preparation: str = "Existing Mac; setup allowance includes environment check. No admitted dataset assumed."


def validate_inputs(inputs):
    for key, default in asdict(Inputs()).items():
        value = getattr(inputs, key)
        if key == "setup_seconds" and value is None:
            continue
        if isinstance(default, str):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{key} must be a nonempty string")
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be a number")
            number(value)
            if key in ("local_workers", "inference_search_simulations") and not isinstance(value, int):
                raise ValueError(f"{key} must be an integer")
    if inputs.electricity_low > inputs.electricity_high:
        raise ValueError("electricity interval reversed")
    if inputs.local_workers < 1 or not inputs.local_hardware:
        raise ValueError("name hardware and at least one local worker")
    if inputs.inference_p95_ms <= 0 or inputs.inference_rss_gib <= 0:
        raise ValueError("deployment latency and memory limits must be positive")
    if inputs.inference_mode not in ("policy_only", "search_assisted"):
        raise ValueError("unknown inference mode")
    if (inputs.inference_mode == "search_assisted") != (inputs.inference_search_simulations > 0):
        raise ValueError("search_assisted requires an explicit positive simulation budget; policy_only requires zero")


def capacities(binding, seconds, evidence):
    """Consume summaries of existing receipts, not a new receipt store."""
    unknown = {metric: None for metric in METRICS[binding["phase"]]}
    if "pending" in (binding[k] for k in ("world_identity", "pipeline_identity", "config_identity")):
        return unknown, ["ETU-79 world/pipeline/config binding pending"]
    matching = [r for r in evidence.get("profiles", []) if r.get("binding") == binding]
    if not matching:
        return unknown, ["missing exact world/config/hardware/concurrency/unit profile"]
    if len(matching) != 1:
        raise ValueError("duplicate matching profile")
    attempts = matching[0]["attempts"]
    rates = {key: [] for key in unknown}
    seen = set()
    failures = 0
    for attempt in attempts:
        receipt = attempt["receipt_sha256"]
        if not isinstance(receipt, str) or not re.fullmatch(r"[0-9a-f]{64}", receipt) or receipt in seen:
            raise ValueError("attempts need distinct source receipt SHA256 digests")
        seen.add(receipt)
        if attempt["status"] == "failed":
            failures += 1
            continue
        if attempt["status"] != "complete":
            raise ValueError("attempt status must be complete or failed")
        elapsed = number(attempt["productive_seconds"])
        if elapsed <= 0:
            raise ValueError("productive_seconds must be positive")
        if "accepted_training_rows" in rates and number(attempt["counts"]["accepted_training_rows"]) > number(attempt["counts"]["accepted_labels"]):
            raise ValueError("training rows cannot exceed accepted labels")
        for key in rates:
            count = number(attempt["counts"][key])
            if not count.is_integer():
                raise ValueError("work counts must be whole units")
            rates[key].append(count / elapsed)
    issues = [f"{failures} failed profile attempts retained; replan before execution"] if failures else []
    if any(len(values) < 3 for values in rates.values()):
        return unknown, issues + ["need at least three complete profiling attempts"]
    return {key: [math.floor(seconds * min(values)), math.floor(seconds * max(values))]
            for key, values in rates.items()}, issues


def schedule(window, deadline, cap, paid, inputs, prices, evidence):
    """Fit one fixed serial schedule; scale work allowances, never startup."""
    cpu_hosts = 2 if window in ("1day", "1week") else 1
    resources = {
        "local": (inputs.local_hardware, 1, inputs.local_workers, inputs.electricity_low, inputs.electricity_high),
        "cpu": ("CCX33", cpu_hosts, 4, prices["cpu_quote"], prices["cpu_ceiling"]),
        "gpu": ("Lambda-A6000", 1, 1, prices["gpu_quote"], prices["gpu_ceiling"]),
    }
    # Release each rental after its phase; bill provisioning and transfer on that host.
    phases = []
    for phase, duration in zip(PHASES, ALLOWANCES[window]):
        if phase == "setup" and inputs.setup_seconds is not None:
            duration = inputs.setup_seconds
        work = phase in METRICS
        resource = ("gpu" if phase == "training" else "cpu") if paid and work else "local"
        if resource != "local":
            phases.append((f"{phase}_startup_transfer", resource,
                           inputs.cloud_startup_seconds + inputs.transfer_seconds, False))
        phases.append((phase, resource, duration, work))
    fixed = sum(sec for _, _, sec, work in phases if not work)
    productive = sum(sec for _, _, sec, work in phases if work)
    fixed_local = sum(sec for _, res, sec, work in phases if not work and res == "local")
    work_local = sum(sec for _, res, sec, work in phases if work and res == "local")
    ancillary = inputs.storage_usd + inputs.transfer_usd if paid else 0
    reserve = cap * .2

    def billed_hours(resource, seconds, work):
        if resource != "local" and work:
            # Startup and work share one lease. Attribute its rounding to work.
            startup = inputs.cloud_startup_seconds + inputs.transfer_seconds
            quantum = prices[f"{resource}_billing_seconds"]
            seconds = math.ceil((startup + seconds) / quantum) * quantum - startup
        return seconds / 3600 * resources[resource][1]

    def upper_cash(scale):
        return ancillary * (1 + inputs.tax_rate) + sum(
            billed_hours(res, sec * scale if work else sec, work) * resources[res][4]
            * (1 if res == "local" else 1 + inputs.tax_rate)
            for _, res, sec, work in phases)

    fixed_cash = upper_cash(0)
    scales = [(deadline - fixed) / productive]
    if work_local:
        scales.append((inputs.local_hours * 3600 - fixed_local) / work_local)
    feasible = fixed < deadline and fixed_local <= inputs.local_hours * 3600 and fixed_cash <= cap - reserve
    scale = max(0, min(1, *scales)) if feasible else 0
    if feasible and upper_cash(scale) > cap - reserve:
        # Billing increments make cash a monotone staircase, not a linear rate.
        lower, upper = 0, scale
        for _ in range(60):
            middle = (lower + upper) / 2
            if upper_cash(middle) <= cap - reserve:
                lower = middle
            else:
                upper = middle
        scale = lower
    feasible = feasible and scale > 0
    rows, offset = [], 0
    volumes = {metric: None for metrics in METRICS.values() for metric in metrics}
    issues = []
    if not feasible:
        issues.append("startup/export, local availability or fixed cash consumes the envelope")
    for phase, resource, allowance, work in phases if feasible else []:
        seconds = allowance * scale if work else allowance
        hardware, hosts, workers, low, high = resources[resource]
        if phase in ("setup", "training", "export") and resource == "local":
            workers = 1
        binding = {"world_identity": inputs.world_identity, "pipeline_identity": inputs.pipeline_identity,
                   "config_identity": inputs.config_identity, "phase": phase, "hardware": hardware,
                   "hosts": hosts, "workers_per_host": workers, "torch_threads_per_worker": 1,
                   "timing_basis": "productive_excluding_setup_transfer_export",
                   "units": list(METRICS.get(phase, ())),
                   "workload": {"recipe": "search_distillation", "hidden_dim": 64,
                                "attention_heads": 4, "total_search_simulations": 64,
                                "compatible_worlds": 4, "rollout_step_cap": 2000,
                                "batch_size": (512 if paid else 128) if phase == "training" else None,
                                "max_epochs": 10, "validation_fraction": .1},
                   "deployment": {"mode": inputs.inference_mode, "device": inputs.inference_device,
                                  "search_simulations": inputs.inference_search_simulations,
                                  "p95_ms_limit": inputs.inference_p95_ms, "rss_gib_limit": inputs.inference_rss_gib}}
        work_interval = None
        if work:
            work_interval, notes = capacities(binding, seconds, evidence)
            volumes.update(work_interval)
            issues.extend(f"{phase}: {note}" for note in notes)
        hours = seconds * hosts / 3600
        billed = billed_hours(resource, seconds, work)
        rows.append({"phase": phase, "resource": resource, "hardware": hardware,
                     "hosts": hosts, "workers_per_host": workers, "start_seconds": offset,
                     "wall_seconds": seconds, "host_hours": hours, "billed_host_hours": billed,
                     "cash_usd": [billed * low, billed * high], "work_interval": work_interval,
                     "profile_binding": binding if work else None,
                     "consumer": CONSUMERS.get(phase, "provision/download/transfer; no useful work credited")})
        offset += seconds
    # Do not promise more optimizer exposures than the newly accepted corpus permits.
    labels = volumes["accepted_training_rows"]
    exposures = volumes["training_exposures"]
    if labels is None:
        volumes["training_exposures"] = None
    elif exposures is not None:
        volumes["training_exposures"] = [min(exposures[i], labels[i] * 10)
                                         for i in (0, 1)]
    for row in rows:
        if row["phase"] == "training":
            row["work_interval"] = {"training_exposures": volumes["training_exposures"]}
    host_cost = [sum(row["cash_usd"][i] for row in rows) for i in (0, 1)]
    rental_cost = [sum(row["cash_usd"][i] for row in rows if row["resource"] != "local") for i in (0, 1)]
    ancillary = ancillary if feasible else 0
    taxes = [(value + ancillary) * inputs.tax_rate for value in rental_cost] if paid else [0, 0]
    cash = [host_cost[i] + ancillary + taxes[i] for i in (0, 1)]
    return {"location": "local_plus_cloud" if paid else "local", "schedule_feasible": feasible,
            "status": "MEASUREMENT_REQUIRED" if feasible else "INFEASIBLE",
            "phases": rows, "scheduled_wall_seconds": offset,
            "host_hours": sum(row["host_hours"] for row in rows),
            "billed_host_hours": sum(row["billed_host_hours"] for row in rows),
            "local_active_hours": sum(row["host_hours"] for row in rows if row["resource"] == "local"),
            "cash_usd": cash, "storage_usd": inputs.storage_usd if paid and feasible else 0,
            "transfer_usd": inputs.transfer_usd if paid and feasible else 0,
            "tax_usd": taxes, "protected_failure_reserve_usd": reserve,
            "total_upper_with_reserve_usd": cash[1] + reserve,
            "unspent_above_reserve_usd": max(0, cap - reserve - cash[1]),
            "volumes": volumes, "strength_prediction": None,
            "issues": issues, "fixed_overhead_seconds": fixed,
            "next_measurement": "ETU-79 smallest complete train/export/play loop on the intended device; then three full profiles",
            "deliverable": "Attempt playable width-64 checkpoint and held-out complete-game evaluation; retain incumbent if unfinished",
            "recipe": "search_distillation", "model": {"hidden_dim": 64, "attention_heads": 4},
            "data": "Fresh admitted complete teacher games; immutable 90/10 whole-game split; reuse up to ten epochs. No free corpus assumed.",
            "training_batch": 512 if paid else 128,
            "followup": ("Compare scratch self-play and distillation+self-play at matched total cost after profiling; warm-start cost included"
                         if window == "1week" else "Repeat independent training seeds only if measured complete loops fit"),
            "concurrency": "Serial phases; only independent CPU games run across the declared hosts. No inferred host scaling."}


def make_cell(window, band, inputs, prices, evidence, deadline=None, cap=None):
    validate_inputs(inputs)
    if (not isinstance(evidence, dict) or not isinstance(evidence.get("profiles", []), list)
            or any(not isinstance(p, dict) for p in evidence.get("profiles", []))):
        raise ValueError("evidence must be an object with a profiles array of objects")
    deadline = WINDOWS[window] if deadline is None else number(deadline)
    cap = (9.99 if band == "near_zero" else 1000) if cap is None else number(cap)
    if deadline <= 0 or band not in ("near_zero", "paid") or not 0 <= cap <= 1000:
        raise ValueError("positive deadline and cash cap in [0, 1000] required")
    if band == "near_zero" and cap >= 10:
        raise ValueError("near_zero cash cap must be strictly less than $10")
    local = schedule(window, deadline, cap, False, inputs, prices, evidence)
    comparisons = [local]
    selected, reason = 0, "Local first; no measured benefit justifies rental spending"
    if band == "paid":
        cloud = schedule(window, deadline, cap, True, inputs, prices, evidence)
        comparisons.append(cloud)
        lv, cv = local["volumes"], cloud["volumes"]
        measured = all(lv[k] is not None and cv[k] is not None for k in lv)
        if (measured and cloud["schedule_feasible"] and not cloud["issues"]
                and all(cv[k][0] >= lv[k][1] for k in lv)
                and any(cv[k][0] > lv[k][1] for k in lv)):
            selected, reason = 1, "Paid lower-bound work exceeds local upper-bound work with no metric regression; strength still unknown"
    return {"window": window, "band": band, "deadline_seconds": deadline, "cash_cap_usd": cap,
            "selected": selected, "reason": reason, "comparisons": comparisons,
            "execution_authorized": False}


def read_prices(path):
    raw = path.read_bytes()
    snapshot = json.loads(raw)
    cpu = next(q for q in snapshot["quotes"] if q["class"].startswith("CCX33"))
    gpu = next(q for q in snapshot["quotes"] if q["class"].startswith("1x RTX A6000"))
    result = {"cpu_quote": number(cpu["usd_per_instance_hour"]),
              "gpu_quote": number(gpu["usd_per_instance_hour"]),
              "cpu_ceiling": number(snapshot["planning_hourly_ceilings"]["cpu"]),
              "gpu_ceiling": number(snapshot["planning_hourly_ceilings"]["gpu"]),
              "cpu_billing_seconds": number(snapshot["billing"]["cpu"]["increment_seconds"]),
              "gpu_billing_seconds": number(snapshot["billing"]["gpu"]["increment_seconds"])}
    if any(result[f"{r}_billing_seconds"] <= 0 for r in ("cpu", "gpu")):
        raise ValueError("billing increments must be positive")
    if any(result[f"{r}_quote"] > result[f"{r}_ceiling"] for r in ("cpu", "gpu")):
        raise ValueError("quote exceeds planning ceiling: replan")
    return result, {"sha256": hashlib.sha256(raw).hexdigest(), "observed_date": snapshot["observed_date"],
                    "sources": [cpu["source"], gpu["source"]], "billing": snapshot["billing"],
                    "requote_before_execution": True}


def build_report(inputs, prices_path, evidence, windows=WINDOWS, bands=("near_zero", "paid"),
                 deadline=None, cap=None):
    prices, reference = read_prices(prices_path)
    return {"inputs": asdict(inputs), "price_reference": reference,
            "evidence_sha256": digest(evidence),
            "cells": [make_cell(w, b, inputs, prices, evidence, deadline, cap)
                      for w in windows for b in bands]}


def markdown(report):
    lines = ["# Strongest playable manabot: planning grid", "",
             "Planning envelopes only. Volumes and strength remain unknown without matching ETU-79 receipts.",
             "Paid quotes are dated references; reserve is protected. No execution is authorized.", "",
             "| Window | Cash cap | Choice | Scheduled hours | Host-hours | Cash range | Reserve | Unspent above reserve |",
             "|---|---:|---|---:|---:|---:|---:|---:|"]
    for cell in report["cells"]:
        plan = cell["comparisons"][cell["selected"]]
        lines.append(f"| {cell['window']} | ${cell['cash_cap_usd']:.2f} | {plan['location']} | "
                     f"{plan['scheduled_wall_seconds']/3600:.3f} | {plan['host_hours']:.3f} | "
                     f"${plan['cash_usd'][0]:.3f}–{plan['cash_usd'][1]:.3f} | "
                     f"${plan['protected_failure_reserve_usd']:.3f} | ${plan['unspent_above_reserve_usd']:.3f} |")
    for cell in report["cells"]:
        lines += ["", f"## {cell['window']} / {cell['band']}", "", cell["reason"]]
        for plan in cell["comparisons"]:
            lines += ["", f"### {plan['location']}: {plan['status']}", "", plan["deliverable"],
                      f"Recipe: {plan['recipe']}, width 64 / 4 attention heads, batch {plan['training_batch']}.",
                      plan["data"], plan["concurrency"], plan["followup"],
                      f"Fixed overhead: {plan['fixed_overhead_seconds']} s. Cash: {plan['cash_usd']}; "
                      f"storage ${plan['storage_usd']:.3f}, transfer ${plan['transfer_usd']:.3f}, "
                      f"tax ${plan['tax_usd'][0]:.3f}–{plan['tax_usd'][1]:.3f} included.",
                      f"Reserve ${plan['protected_failure_reserve_usd']:.3f}; unspent above reserve ${plan['unspent_above_reserve_usd']:.3f}.",
                      "", "| Phase | Hardware × hosts × workers | Wall seconds | Host-hours | Billed host-hours | Work interval |",
                      "|---|---|---:|---:|---:|---|"]
            for row in plan["phases"]:
                lines.append(f"| {row['phase']} | {row['hardware']} × {row['hosts']} × {row['workers_per_host']} | "
                             f"{row['wall_seconds']:.2f} | {row['host_hours']:.4f} | {row['billed_host_hours']:.4f} | {row['work_interval']} |")
            lines += ["", f"Volumes: {plan['volumes']}. Strength: unknown.",
                      "Largest uncertainty: complete-loop feasibility and quality on the corrected world.",
                      f"Next measurement: {plan['next_measurement']}."]
            lines.append("")
            lines.extend(f"- {issue}" for issue in plan["issues"])
    lines += ["", "Inputs and dated reference:", "", "```json",
              json.dumps({k: v for k, v in report.items() if k != "cells"}, indent=2), "```", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", choices=WINDOWS, help="omit for all eight cells")
    parser.add_argument("--band", choices=("near_zero", "paid"))
    parser.add_argument("--deadline-seconds", type=float, help="exact override; requires --window and --band")
    parser.add_argument("--cash-cap", type=float, help="exact override; requires --window and --band")
    parser.add_argument("--inputs", type=Path, help="JSON object overriding Inputs defaults")
    parser.add_argument("--evidence", type=Path, help="ETU-79 receipt summaries, see experiments/planning/README.md")
    parser.add_argument("--prices", type=Path, default=ROOT / "experiments/planning/prices.json")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    args = parser.parse_args()
    if (args.cash_cap is not None or args.deadline_seconds is not None) and not (args.window and args.band):
        parser.error("exact overrides require --window and --band")
    try:
        inputs = Inputs(**json.loads(args.inputs.read_text())) if args.inputs else Inputs()
        evidence = json.loads(args.evidence.read_text()) if args.evidence else {}
        report = build_report(inputs, args.prices, evidence,
                              windows=[args.window] if args.window else WINDOWS,
                              bands=[args.band] if args.band else ("near_zero", "paid"),
                              deadline=args.deadline_seconds, cap=args.cash_cap)
        print(markdown(report) if args.format == "markdown" else json.dumps(report, indent=2, allow_nan=False))
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
