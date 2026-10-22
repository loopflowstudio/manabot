"""Bounded search for choice aliasing in native authored games.

Collect replayable legal states, group actions by public descriptors (never
instance IDs), and fit a few shared policies to opposite artificial preferences.
Structural violations and finite-budget fit misses are reported separately.
This diagnostic measures capacity, not strategy or held-out strength.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Literal

import torch

from manabot.env import ObservationSpace
from manabot.infra.hypers import AgentSpec
from manabot.model import Agent
from manabot.verify.action_cases import reorder_objects, reorder_offers
from manabot.verify.choice_corpus import (
    PACK,
    Case,
    _write_json,
    collect_choices as _collect,
)
import managym

Arm = Literal["generic", "semantic", "generic_no_attention"]
ARMS: tuple[Arm, ...] = ("generic", "semantic", "generic_no_attention")


def _spec(arm: Arm) -> AgentSpec:
    return AgentSpec(
        hidden_dim=32,
        num_attention_heads=2,
        attention_feedforward_dim=None if arm == "generic_no_attention" else 64,
        attention_on=arm != "generic_no_attention",
        semantic_pack=PACK if arm == "semantic" else None,
    )


def _fit(
    out: Path, cases: list[Case], arm: Arm, preference: int, seed: int, deadline: float
) -> dict[str, object]:
    started = time.perf_counter()
    torch.manual_seed(seed)
    spec = _spec(arm)
    agent = Agent(ObservationSpace(), spec)
    batch = {key: torch.cat([case.obs[key] for case in cases]) for key in cases[0].obs}
    selected = torch.zeros_like(batch["actions_valid"], dtype=torch.bool)
    pair = selected.clone()
    for row, case in enumerate(cases):
        selected[row, case.groups[preference]] = True
        pair[row, case.groups[0] + case.groups[1]] = True
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.003)
    with torch.no_grad():
        initial, _ = agent(batch)
    completed_updates = 0
    for _ in range(240):
        if time.perf_counter() >= deadline:
            return {
                "arm": arm,
                "preference": preference,
                "seed": seed,
                "status": "budget_exhausted",
                "updates": completed_updates,
            }
        logits, _ = agent(batch)
        # Conditional group likelihood isolates whether this pair is separable;
        # other legal actions remain in the forward pass but are not negatives.
        normalizer = logits.masked_fill(~pair, -torch.inf).logsumexp(-1)
        chosen = logits.masked_fill(~selected, -torch.inf).logsumexp(-1)
        loss = (normalizer - chosen).mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        completed_updates += 1
    agent.eval()
    path = out / f"{arm}-{preference}.pt"
    torch.save(
        {"state_dict": agent.state_dict(), "spec": spec.model_dump(), "seed": seed},
        path,
    )
    restored = Agent(ObservationSpace(), spec)
    restored.load_state_dict(torch.load(path, weights_only=True)["state_dict"])
    restored.eval()
    scores: list[dict[str, object]] = []
    with torch.no_grad():
        logits, values = agent(batch)
        reloaded, restored_values = restored(batch)
        reload_error = max(
            float((logits - reloaded).abs().max()),
            float((values - restored_values).abs().max()),
        )
        for row, case in enumerate(cases):
            obs = case.obs
            single, value = restored(obs)
            p = single.softmax(-1)
            group = case.groups[preference]
            union = case.groups[0] + case.groups[1]
            probability = float(p[0, group].sum() / p[0, union].sum())
            initial_row = initial[row].softmax(-1)
            before = float(initial_row[group].sum() / initial_row[union].sum())
            count = int(obs["actions_valid"].sum())
            order = torch.randperm(
                count, generator=torch.Generator().manual_seed(seed + row)
            )
            shuffled, _ = restored(reorder_offers(obs, order))
            objects, _ = restored(reorder_objects(obs))
            errors = {
                "batch": max(
                    float((single[0] - logits[row]).abs().max()),
                    float((value[0] - values[row]).abs()),
                ),
                "offer": float(
                    (shuffled.softmax(-1)[0, :count] - p[0, order]).abs().max()
                ),
                "object": float((objects.softmax(-1) - p).abs().max()),
                "padding_mass": float(p[obs["actions_valid"] == 0].sum()),
                "reload": reload_error,
            }
            scores.append(
                {
                    "case": case.name,
                    "family": case.family,
                    "initial": before,
                    "conditional_probability": probability,
                    "fit_passed": probability >= 0.95,
                    "errors": errors,
                    "invariants_passed": max(errors.values()) <= 1e-4,
                }
            )
    return {
        "arm": arm,
        "preference": preference,
        "seed": seed,
        "status": "completed",
        "updates": completed_updates,
        "seconds": time.perf_counter() - started,
        "checkpoint": path.name,
        "scores": scores,
    }


def run(
    out: Path, seconds: float, arms: tuple[Arm, ...] = ARMS, *, closest: bool = False
) -> None:
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.perf_counter()
    cpu_started = time.process_time()
    deadline = started + seconds
    rounds: list[dict[str, object]] = []
    result: dict[str, object] = {
        "protocol": "learnability-search-1",
        "pairing": "closest" if closest else "extremes",
        "source": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "world": managym.WORLD_VERSION,
        "torch": torch.__version__,
        "native_sha256": hashlib.sha256(
            Path(managym._managym.__file__).read_bytes()
        ).hexdigest(),
        "seconds_budget": seconds,
        "updates_per_fit": 240,
        "threshold": 0.95,
        "invariance_tolerance": 1e-4,
        "arms": arms,
        "rounds": rounds,
        "status": "running",
    }
    try:
        round_index = 0
        while time.perf_counter() < deadline:
            round_out = out / f"round-{round_index:03d}"
            cases, counts, findings = _collect(
                round_out, round_index, deadline, closest
            )
            fits: list[dict[str, object]] = []
            record: dict[str, object] = {
                "round": round_index,
                "counts": dict(counts),
                "cases": len(cases),
                "native_findings": len(findings),
                "fits": fits,
            }
            rounds.append(record)
            _write_json(out / "result.json", result)
            # Rotate starting configuration to balance coverage at the deadline.
            ordered = arms[round_index % len(arms) :] + arms[: round_index % len(arms)]
            for arm in ordered:
                for preference in (0, 1):
                    if not cases or time.perf_counter() >= deadline:
                        break
                    try:
                        fit = _fit(
                            round_out,
                            cases,
                            arm,
                            preference,
                            1000 + round_index,
                            deadline,
                        )
                    except Exception as error:
                        fit = {
                            "arm": arm,
                            "preference": preference,
                            "status": "error",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    fits.append(fit)
                    _write_json(out / "result.json", result)
                    score_rows = fit.get("scores", [])
                    assert isinstance(score_rows, list)
                    print(
                        json.dumps(
                            {
                                "round": round_index,
                                "cases": len(cases),
                                "arm": arm,
                                "preference": preference,
                                "status": fit["status"],
                                "error": fit.get("error"),
                                "fit_misses": sum(
                                    not s["fit_passed"] for s in score_rows
                                ),
                                "invariant_failures": sum(
                                    not s["invariants_passed"] for s in score_rows
                                ),
                                "elapsed": round(time.perf_counter() - started, 1),
                            }
                        ),
                        flush=True,
                    )
            round_index += 1
        result["status"] = "completed_budget"
    except BaseException as error:
        result["status"] = "error"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["wall_seconds"] = time.perf_counter() - started
        result["cpu_seconds"] = time.process_time() - cpu_started
        _write_json(out / "result.json", result)
        (out / "report.md").write_text(
            f"# Learnability exploration\n\nStatus: {result['status']}\n\n"
            f"Wall: {result['wall_seconds']:.1f}s; CPU: {result['cpu_seconds']:.1f}s.\n\n"
            "See result.json for per-case results; fit misses are candidates, not confirmed defects.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=1800)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--closest", action="store_true")
    args = parser.parse_args()
    if not 0 < args.seconds <= 1800:
        parser.error("seconds must be in (0, 1800]")
    run(
        args.out,
        args.seconds,
        tuple(arm for arm in ARMS if arm in args.arms),
        closest=args.closest,
    )


if __name__ == "__main__":
    main()
