"""Run this file in a pinned source archive with its own compiled extension.

The current case builder is passed explicitly. Only the missing old Python
selector fields are suppressed; native observations and policies belong wholly
to the historical runtime. One shared six-family fit and weight-independent pair
collisions qualify the daily information-removal controls. No old checkpoint is
relabeled. These are capacity/alias controls, not historical full-game training.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch

from manabot.env import ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.model import ActorCritic
import managym

PINS = {
    "w4": "9576dc40ef486966246e20d9fdf58e5d741f1c40",
    "w5": "78bf9429a628d1f4a2a0487199ccce1aa5ac7684",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--commit", choices=PINS.values(), required=True)
    args = parser.parse_args()
    if PINS.get(managym.WORLD_VERSION) != args.commit:
        raise ValueError("Historical world does not match pinned source")
    spec = importlib.util.spec_from_file_location("historical_cases", args.cases)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load case instrument")
    cases_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cases_module
    spec.loader.exec_module(cases_module)
    torch.set_num_threads(1)
    torch.manual_seed(17)
    agent = ActorCritic(
        ObservationSpace(), ModelConfig(hidden_dim=16, num_attention_heads=2)
    )
    cases = [
        cases_module.decision_case(family, legacy_selectors=True)
        for family in cases_module.FAMILIES
    ]
    started = time.perf_counter()
    gaps: dict[str, float] = {}
    with torch.no_grad():
        for case in cases:
            logits, _ = agent(case.observation)
            a, b = case.candidates[:2]
            gaps[case.family] = float((logits[0, a] - logits[0, b]).abs())
    # Fit all six families on one policy, not one policy per case. In a forced
    # tie neither candidate can exceed 0.5 regardless of optimizer or weights.
    batch = {
        key: torch.cat([c.observation[key] for c in cases])
        for key in cases[0].observation
    }
    labels = torch.tensor([c.preferred for c in cases])
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.01)
    for _ in range(120):
        logits, _ = agent(batch)
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(logits, labels).backward()
        optimizer.step()
    with torch.no_grad():
        logits, _ = agent(batch)
        probabilities = logits.softmax(-1)
    scores = {
        case.family: float(probabilities[i, case.preferred])
        for i, case in enumerate(cases)
    }
    expected_ties = set(cases_module.FAMILIES) - (
        {"attack"} if managym.WORLD_VERSION == "w5" else set()
    )
    passed = all(gaps[k] == 0 and scores[k] <= 0.500001 for k in expected_ties)
    if managym.WORLD_VERSION == "w5":
        passed = passed and scores["attack"] > 0.95
    receipt = {
        "source": args.commit,
        "case_builder_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_sha256": hashlib.sha256(
            Path(managym._managym.__file__).read_bytes()
        ).hexdigest(),
        "world": managym.WORLD_VERSION,
        "seed": 17,
        "updates": 120,
        "initial_logit_gaps": gaps,
        "trained_probabilities": scores,
        "wall_seconds": time.perf_counter() - started,
        "passed": passed,
        "ability": "synthetic equal-focus projection; old bindings omit selector fields",
        "native_consequences": "w6 qualification only; old permanent bindings omit definition identity",
    }
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
